#!/usr/bin/env python
"""PHASE J -- rescore a cached fold from its trunk npz and check it reproduces the stored scores.

WHAT THIS IS FOR. Two things at once:
  1. It proves the cache-rescoring path is sound. The FT3 evaluation rescores frozen poses with a
     changed head; if rescoring with the UNCHANGED head does not reproduce the ipTM already in
     results.csv, every later comparison is measuring our plumbing rather than the finetune.
  2. It measures what a validation pass costs, which until now was an estimate.

WHAT IT DOES NOT DO. It does not refold. Everything comes out of the npz the fold already wrote:
z_trunk, s_trunk, s_inputs, the N_sample poses and the feature masks. Nothing here touches the
diffusion module or the trunk.

TOLERANCE. bf16 trunk tensors are stored and reloaded, the confidence head runs under autocast, and
the fold itself is not bit-reproducible on this hardware -- so exact equality is not the bar. The bar
is that stored and recomputed ipTM agree to within --tol, and that the RANKING of poses within a seed
is identical, since selection is what the project measures.

Model loading replicates runner/inference.py's config assembly; copied from
diffusion_commit_clean_v2/steer_reps.py rather than imported, so extra_training does not depend on
that tree.
"""
import argparse
import collections
import csv
import functools
import json
import os
import struct
import sys
import time
import zipfile

import numpy as np
import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
print = functools.partial(print, flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_model(ckpt_dir, model_name, device):
    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("set LAYERNORM_TYPE=openfold (without it coordinates are silently NaN)")
    from collections.abc import Mapping
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from protenix.model.protenix import Protenix
    try:
        from runner.inference import update_gpu_compatible_configs, model_configs
    except ImportError:
        from runner.inference import model_configs
        update_gpu_compatible_configs = lambda c: c
    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    deep_update(base, model_configs[model_name])
    configs = parse_configs(configs=base, arg_str="", fill_required_with_null=True)
    configs.load_checkpoint_dir = ckpt_dir
    configs.model_name = model_name
    configs = update_gpu_compatible_configs(configs)
    model = Protenix(configs).to(device)
    ck = torch.load(os.path.join(ckpt_dir, f"{model_name}.pt"), map_location=device,
                    weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise SystemExit(f"checkpoint missing {len(res.missing_keys)} weights, e.g. "
                         f"{res.missing_keys[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, configs


def npy_memmap(path, key):
    """A copy-on-write mapping of ONE STORED npz member, or None if it cannot be mapped.

    WHY THIS IS POSSIBLE WITHOUT REWRITING THE 1.7 TB TRUNK CACHE. np.savez writes every member
    UNCOMPRESSED, and the trunk writer uses it: measured on three real trunks (66.4, 104.4 and
    191.8 MB) every member reports `compress_type 0`, and `z_trunk.npy` is the FIRST member at zip
    `header_offset 0` with its payload at a fixed 61-byte offset (30-byte local header + 11-byte
    name + 20-byte extra field, byte-identical in all three files). So the member's bytes are one
    contiguous run of the file and np.memmap can address them in place.

    WHAT IT BUYS. `z[key]` read()s the member into a fresh heap buffer and the device copy then
    reads that buffer -- two passes over ~243 MB per unit, ~1.1 TB per rank per epoch. A mapping
    lets the host-to-device copy read the page cache directly, so the heap pass disappears.

    NOTHING IS ASSUMED AND NOTHING IS REQUIRED. The payload offset is read from the member's own
    local header (the name and extra-field lengths live at fixed positions in it) and the shape,
    dtype and order from its own npy header. A deflated member, a fortran-ordered member, an object
    dtype, a short file or any OSError returns None and the caller falls back to np.load's own
    path, which is what every reader did before this existed.

    mode="c", NOT "r". A read-only mapping hands back a non-writeable array and torch.from_numpy
    then warns on every unit; copy-on-write is writeable, is never written here, and never writes
    back to the file.

    THE MAPPING IS UNALIGNED, AND THAT MAY COST THE WHOLE SAVING. The payload starts at 61 bytes
    (zip local header) plus a 128-byte npy header = byte 189, which is ODD -- so a uint16/bf16 array
    mapped here has `flags.aligned == False` and an odd `data_ptr()`, on EVERY trunk, structurally.
    That is bit-exact (verified) but a pageable cudaMemcpy from a 1-byte-aligned host pointer gives
    up the wide-load staging path, so the ~6 min this is supposed to save could shrink or invert.
    When it is finally measured on a GPU, compare `trunk_npz_read + trunk_h2d` SUMMED across the two
    arms, not `trunk_h2d` alone, and be prepared for no win. If it loses, the alternative is
    np.frombuffer on an mmap.mmap sliced to a 64-byte-aligned base plus a view offset.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            info = zf.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        total = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(info.header_offset)
            hdr = fh.read(30)
            if len(hdr) < 30 or hdr[:4] != b"PK\x03\x04":
                return None
            n_name, n_extra = struct.unpack("<HH", hdr[26:30])
            data0 = info.header_offset + 30 + n_name + n_extra
            fh.seek(data0)
            ver = np.lib.format.read_magic(fh)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(tuple(ver))
            if reader is None:
                return None
            shape, fortran, dtype = reader(fh)
            off = fh.tell()
        if fortran or dtype.hasobject:
            return None
        n = 1
        for d in shape:
            n *= int(d)
        # THE SPAN MUST EXHAUST THIS MEMBER EXACTLY, not merely fit inside the file. Checking only
        # `off + bytes <= filesize` would happily map across a member boundary and hand the head a
        # REINTERPRETATION of the wrong bytes -- which does not raise and is the worst failure mode
        # available here. The zip directory already knows the member's uncompressed length, so the
        # identity (npy header + payload == member length) is free and is the real check. Verified
        # on a real trunk: payload offset 61, npy header 128, so off-data0 = 128 and
        # 128 + 354*354*256*2 == info.file_size.
        if n == 0 or off + n * dtype.itemsize > total:
            return None
        if (off - data0) + n * dtype.itemsize != info.file_size:
            return None
        return np.memmap(path, dtype=dtype, mode="c", offset=off, shape=tuple(shape))
    except Exception:          # noqa: BLE001 -- THE CONTRACT IS "FALL BACK", NOT "RAISE".
        # A NARROW except LIST WAS WRONG HERE. A corrupt npy header raises SyntaxError out of
        # read_array_header_*, and a bad `descr` raises TypeError; neither is an OSError or a
        # ValueError, so the narrow list turned "map this if you can" into "kill the run". Every
        # caller has a working np.load path one line away.
        return None


# WHICH MEMBERS ARE WORTH MAPPING. z_trunk is ~99% of a trunk npz's bytes (187.4 MB of 191.8 on
# 9mnq); every other member is at or under ~1.5 MB, where the extra open()/mmap() syscalls cost
# more than the heap copy they would avoid.
MEMMAP_MEMBERS = ("z_trunk",)


# EVERY MEMBER load_unit CONSUMES, named once so the host read and the device build cannot drift.
# Adding one here without adding it below gets it read and never used, which costs time silently;
# the reverse raises a KeyError on the first unit, which is the better of the two failures.
TRUNK_MEMBERS = ("ci_feat_asym_id", "ci_feat_has_frame", "ci_feat_atom_to_token_idx",
                 "ci_feat_atom_to_tokatom_idx", "ci_feat_distogram_rep_atom_mask",
                 "ci_feat_is_ligand", "ci_s_inputs", "ci_s_trunk_f32", "z_trunk",
                 "ci_coordinate", "ci_N_cycle")


def read_unit_host(path, tick=None, memmap=False):
    """Every member load_unit consumes, as HOST numpy arrays, keyed by member name.

    SPLIT OUT SO A WORKER THREAD CAN DO IT. There is no torch and no device anywhere in here, which
    is what makes it safe to run off the main thread: only one thread may ever issue CUDA work, and
    this issues none. phase_k_train warms a step's trunks through this while the GPU is busy and
    hands the dicts to unit_from_host on the main thread.

    AND SO THE READ CAN BE DONE ONCE PER UNIT PER STEP INSTEAD OF TWICE. The relative-iTM objective
    runs two passes over every step's units -- a no_grad scoring pass, then the backward pass -- and
    each used to call load_unit, so each unit's ~243 MB was materialised TWICE per step: 4,600
    load_unit calls per rank per epoch for 2,300 units. A dict of host arrays is reusable between
    the two; the device tensors are not, since they are freed with the pose.

    HOST PEAK PER UNIT RISES, and it is the one behavioural difference from the old interleaved
    load_unit. That read one member, moved it to the device and dropped it, so its host peak was one
    member (~187 MB, z_trunk); this materialises all 11 before any tensor is built, so the peak is
    the whole unit (~243 MB). Deliberate -- the dict is the thing the cache and the warmer hand
    around -- but it is additive with phase_k_train's --trunk-cache-gb and is why that cap exists.

    `tick(key, t0)` is phase_k_train's accumulator; None is no instrumentation and no cost. A
    requested mapping that could not be made ticks `trunk_memmap_fallback`, because otherwise a run
    can honestly report `--trunk-memmap` while never mapping anything -- the same failure as a
    prefetch whose 100% miss rate looked identical to no prefetch at all.
    """
    _t0 = time.perf_counter()
    # CLOSED, unlike the old code. np.load's NpzFile holds an open fd and up to --prefetch-workers
    # warmer threads now call this at once; the lazy `z[key]` reads all happen inside the block, so
    # closing at the end is deterministic rather than at the whim of the collector.
    with np.load(path) as z:
        if tick is not None:
            tick("trunk_npz_open", _t0)
        out = {}
        for key in TRUNK_MEMBERS:
            _ta = time.perf_counter()
            want_mm = memmap and key in MEMMAP_MEMBERS
            a = npy_memmap(path, key) if want_mm else None
            if want_mm and a is None and tick is not None:
                tick("trunk_memmap_fallback", _ta)
            out[key] = z[key] if a is None else a
            if tick is not None:
                tick("trunk_npz_read", _ta)
    return out


def unit_from_host(host, device, tick=None):
    """The tuple run_confidence_head consumes, from read_unit_host's dict. THE ONLY CUDA HALF.

    EVERY MEMBER IS COPIED, NOT ALIASED, FOR A CUDA DEVICE -- `.to(cuda)` always copies -- which is
    what makes it safe for two passes to build their tensors from ONE shared host dict. For a CPU
    device `.to(cpu)` is a no-op and the tensors ALIAS the host arrays, so a caller that shares a
    host dict across passes on CPU would also share any in-place write. phase_k_train's cache is
    gated on a cuda device for exactly that reason.
    """
    def t(key, dtype=None):
        a = host[key]
        _tb = time.perf_counter()
        try:
            if a.dtype == np.uint16:            # bf16 stored as uint16
                return torch.from_numpy(a).to(device).view(torch.bfloat16)
            return (torch.from_numpy(a).to(device, dtype=dtype) if dtype
                    else torch.from_numpy(a).to(device))
        finally:
            if tick is not None:
                tick("trunk_h2d", _tb)
    feat = {
        "asym_id": t("ci_feat_asym_id"),
        "has_frame": t("ci_feat_has_frame"),
        "atom_to_token_idx": t("ci_feat_atom_to_token_idx"),
        "atom_to_tokatom_idx": t("ci_feat_atom_to_tokatom_idx"),
        "distogram_rep_atom_mask": t("ci_feat_distogram_rep_atom_mask"),
        "is_ligand": t("ci_feat_is_ligand"),
    }
    return (feat, t("ci_s_inputs").float(), t("ci_s_trunk_f32").float(), t("z_trunk"),
            t("ci_coordinate").float(), int(host["ci_N_cycle"]))


def load_unit(path, device, tick=None, memmap=False, host=None):
    """Rebuild what run_confidence_head consumes from one cached npz.

    TIMED IN THREE BUCKETS WHEN `tick` IS GIVEN, because this call carried no timer at all and is
    believed to hold essentially all of the in-unit time that `unit_setup` could not name: rank 0,
    epoch 3 of mqird6 spent 37.0 min in `unit_setup` with 0.2 min in named leaves, and 31.0 of the
    epoch's 35.9 min residual was INSIDE units.

    WHY THREE AND NOT ONE. np.load on an npz is LAZY -- the open parses the central directory and
    reads no member -- so `z[key]` is the disk read plus the numpy materialisation, and
    `.to(device)` is a separate SYNCHRONOUS PAGEABLE host-to-device copy of the same bytes. At a
    mean 243 MB per unit and 4,600 units per rank per epoch that is ~1.1 TB each way, and the two
    are limited by different things -- page cache and readahead for the first, PCIe and the absence
    of pinning for the second. Only the split says which of the two the 31 min is, and therefore
    whether prefetching the read (which hides only the first) or pinned staging (only the second)
    is the fix.

    `tick(key, t0)` is phase_k_train's own accumulator, so these land in the epoch table beside
    every other span; the default None is no instrumentation and no cost. `memmap=True` routes the
    members in MEMMAP_MEMBERS through npy_memmap, falling back silently per member. `host` is a
    dict read_unit_host returned earlier for THIS SAME path, and skips the read entirely.

    THE SPLIT IS ONLY HONEST WITH memmap OFF, which is the default and is why the measurement runs
    there first. A mapping is lazy: no byte is read until it is touched, so under `memmap=True` the
    disk read MOVES INTO `trunk_h2d` and `trunk_npz_read` collapses to the zip-directory parse.
    Read `trunk_npz_read + trunk_h2d` as the total either way; read them separately only at the
    default.
    """
    if host is None:
        host = read_unit_host(path, tick=tick, memmap=memmap)
    return unit_from_host(host, device, tick=tick)


def _np(v):
    """summary values come back as cuda tensors; the saved jsons are plain lists."""
    return v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=float)


def confidence(model, configs, feat, s_inputs, s, z, x, n_cycle):
    """ipTM / ranking score for poses x, exactly as protenix.py computes them."""
    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plddt, pae, pde, _res = model.run_confidence_head(
            input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_mask=None,
            x_pred_coords=x, triangle_multiplicative=configs.triangle_multiplicative,
            triangle_attention=configs.triangle_attention, inplace_safe=False, chunk_size=None)
        cp = autocasting_disable_decorator(True)(sample_confidence.compute_contact_prob)(
            distogram_logits=model.distogram_head(z),
            **sample_confidence.get_bin_params(configs.loss.distogram))
        summary, _full = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary)(
            configs=configs, pae_logits=pae, plddt_logits=plddt, pde_logits=pde, contact_probs=cp,
            token_asym_id=feat["asym_id"], token_has_frame=feat["has_frame"], atom_coordinate=x,
            atom_to_token_idx=feat["atom_to_token_idx"], atom_is_polymer=1 - feat["is_ligand"],
            N_recycle=n_cycle, interested_atom_mask=None, return_full_data=True,
            mol_id=None, elements_one_hot=None)
    return summary


def _chain_pair_itm_mean(pae_prob, has_frame, asym_id, token_is_ligand, min_bin, max_bin,
                         no_bins):
    """chain_pair_iptm's matrix, but the MEAN over anchor rows instead of the max.

    WHY THIS EXISTS. ipTM is max_i t_i, so a pose's whole score rests on one anchor row. Measured
    by run_logging/itm_ceiling.py on 738 interfaces at 50 candidates, ranking poses by the TRUE
    iTM recovers 58.6% of the oracle headroom under max pooling and 67.7% under mean, +0.0140 +-
    0.0034 paired (t=4.06). That is an ORACLE result and says nothing about a real head, which is
    what this computes so the question can be settled on actual predictions.

    SAME ARITHMETIC AS pae_row_weight.row_t, which the test suite pins against Protenix's own
    calculate_iptm, except that it takes PROBABILITIES rather than logits -- pair_iptm_only has
    already softmaxed, and softmaxing twice would be a different number.

    PER CHAIN PAIR, with that pair's own d0 and only its cross-chain targets. Using the crop-wide
    d0 or the union of pairs is what calculate_ptm does and is wrong for an interface.
    """
    from protenix.model import sample_confidence
    # PER POSE. pae_prob is [N_pose, N_tok, N_tok, bins] and chain_pair_iptm is [N_pose, C, C];
    # an earlier draft took pae_prob[0] and assigned that one pose's value to every pose, which
    # would have reported pose 0's score for all five and looked entirely plausible.
    pp_all = pae_prob if pae_prob.dim() == 4 else pae_prob.unsqueeze(0)
    hf = has_frame.reshape(-1).bool()
    asym = asym_id.reshape(-1).long()
    dev = pp_all.device
    chains = sorted(set(asym.tolist()))
    n_chain = max(chains) + 1
    bc = sample_confidence.get_bin_centers(min_bin=min_bin, max_bin=max_bin,
                                           no_bins=no_bins).to(dev).float()
    # LIGAND TOKENS ARE EXCLUDED exactly as calculate_chain_based_ptm excludes them; keeping them
    # would score rows Protenix does not score and make the two poolings incomparable.
    live = hf & (~token_is_ligand.reshape(-1).bool())
    arow = asym[live]
    out = torch.zeros((pp_all.shape[0], n_chain, n_chain), device=dev, dtype=torch.float32)
    # The target mask and d0 depend only on the chain layout, so they are built once and reused
    # across poses; only the probability tensor changes.
    masks = []
    for xi in range(len(chains)):
        for yi in range(xi + 1, len(chains)):
            cx, cy = chains[xi], chains[yi]
            tok = (asym == cx) | (asym == cy)
            nd = int(tok.sum())
            if nd == 0:
                continue
            d0 = 1.24 * (max(nd, 19) - 15) ** (1.0 / 3.0) - 1.8
            tm = (tok.unsqueeze(0) & (arow.unsqueeze(1) != asym.unsqueeze(0))
                  & ((arow == cx) | (arow == cy)).unsqueeze(1)).float()
            cnt = tm.sum(-1)
            if float(cnt.sum()) == 0:
                continue
            masks.append((cx, cy, tm, cnt, 1.0 / (1.0 + (bc / d0) ** 2)))
    for pi in range(pp_all.shape[0]):
        rows_p = pp_all[pi][live].float()                 # [N_frame, N_tok, bins]
        for cx, cy, tm, cnt, wb in masks:
            t = ((rows_p @ wb) * tm).sum(-1) / cnt.clamp_min(1.0)
            keep = cnt > 0
            if not bool(keep.any()):
                continue
            v = float(t[keep].mean())
            out[pi, cx, cy] = v
            out[pi, cy, cx] = v
    return out


def pair_iptm_only(model, configs, feat, s_inputs, s, z, x, n_cycle, want_pae=False,
                   want_mean=False):
    """chain_pair_iptm and nothing else, for callers that read only that.

    WHAT THIS SKIPS AND WHY IT IS SAFE. confidence() computes the whole Protenix summary: the
    distogram head, contact probabilities, atom pLDDT, token-pair PDE and PAE, clash and
    chain-based pLDDT. phase_k_eval reads exactly one field of it -- chain_pair_iptm -- so the rest
    is computed and discarded. calculate_chain_based_ptm needs only the PAE probabilities, the
    frames, asym_id and the ligand mask, so this goes straight there.

    MEASURED, 25 validation units, median 481 tokens, uncontended card:
      confidence head                 0.448 s   75.6% of the current path  (both paths pay it)
      distogram head + contact prob   0.001 s    0.1%                      (skipped here)
      compute_full_data_and_summary   0.097 s   16.3%                      (skipped here)
      softmax(PAE) + chain ptm        0.006 s                              (this path)
    so 0.592 -> 0.453 s per unit, 1.17x on the GPU half, 2.7 min off a 26.1 min pass.

    AND IT IS THE SAME NUMBER, not merely a close one: over those 25 units the largest absolute
    difference in chain_pair_iptm against confidence() was 0.000e+00. A faster number that differed
    would be a different metric, so that check gates adoption -- see
    phase_k_eval_stage_timing.py, which re-runs both paths side by side.

    Returned in confidence()'s shape -- a list of per-pose dicts -- so a caller can swap one for
    the other without touching the loop that consumes it.
    """
    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _plddt, pae, _pde, _res = model.run_confidence_head(
                input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_mask=None,
                x_pred_coords=x, triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention, inplace_safe=False, chunk_size=None)
        # PROTENIX OFFLOADS pae TO CPU ABOVE 2,000 TOKENS AND THIS PATH HAS TO UNDO IT.
        #
        # confidence.py:231-236 moves pae_pred and pde_pred to the CPU when z_trunk's token count
        # exceeds PROTENIX_CONF_OFFLOAD_TOKENS (default 2000) and the model is in eval mode, to
        # keep 50 samples of a [N_tok, N_tok, 64] map off the GPU. calculate_chain_based_ptm then
        # indexes that CPU tensor with chain masks built from feat, which are on the GPU, and
        # torch refuses: "indices should be either on cpu or on the same device as the indexed
        # tensor (cpu)". Measured 2026-11-12 on 9q5m, 2,354 tokens -- the largest structure in the
        # training pool and the only one over the threshold. It killed all three shards of a
        # 71,370-pose in-sample pass about 2,000 units in, an hour of GPU for no metric.
        #
        # MOVING IT BACK IS THE FIX, NOT RAISING THE THRESHOLD, because the threshold is a caller's
        # memory policy and this is a device-agreement bug in the reduction that follows. We score
        # one unit at a time on a 97 GB card, so the map fits; the alternative -- running the pTM
        # reduction on the CPU instead -- is the same arithmetic hundreds of times slower.
        _fdev = feat["asym_id"].device
        if pae.device != _fdev:
            pae = pae.to(_fdev)
        # fp32 for the softmax and the pTM sum: the bins are weighted and summed over N_token^2, and
        # bf16 accumulation there is not what the reference path does.
        pae_prob = torch.softmax(pae.float(), dim=-1)
        token_is_ligand = torch.zeros_like(feat["asym_id"]).scatter_add(
            0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0
        out = autocasting_disable_decorator(True)(sample_confidence.calculate_chain_based_ptm)(
            pae_prob, has_frame=feat["has_frame"].bool(), asym_id=feat["asym_id"],
            token_is_ligand=token_is_ligand,
            **sample_confidence.get_bin_params(configs.loss.pae))
        # EXPECTED ALIGNED ERROR IN ANGSTROMS, not logits and not probabilities. This is the same
        # arithmetic sample_confidence.logits_to_score does at line 106 of its own module --
        # prob @ get_bin_centers(**get_bin_params(configs.loss.pae)) -- which is what produces the
        # `token_pair_pae` the full summary reports. pae_prob is already computed above for the
        # chain pTM, so this costs one matmul and no extra forward.
        # OPT-IN, DEFAULT OFF: phase_k_eval children of a running trainer re-read this file, so the
        # existing return shape must not change underneath them.
        cim = None
        if want_mean:
            # OPT-IN, DEFAULT OFF, like want_pae: phase_k_eval children of a running trainer
            # re-read this file, so the return shape must not change underneath them.
            bp = sample_confidence.get_bin_params(configs.loss.pae)
            cim = _chain_pair_itm_mean(
                pae_prob, has_frame=feat["has_frame"], asym_id=feat["asym_id"],
                token_is_ligand=token_is_ligand, **bp)
        tpp = None
        if want_pae:
            bc = sample_confidence.get_bin_centers(
                **sample_confidence.get_bin_params(configs.loss.pae)).to(pae_prob.device)
            tpp = (pae_prob @ bc).float()
    cpi = out["chain_pair_iptm"]
    n_pose = cpi.shape[0] if cpi.dim() == 3 else 1
    res = [{"chain_pair_iptm": cpi[i] if cpi.dim() == 3 else cpi} for i in range(n_pose)]
    if tpp is not None:
        for i in range(n_pose):
            res[i]["token_pair_pae"] = tpp[i] if tpp.dim() == 3 else tpp
    if cim is not None:
        for i in range(n_pose):
            res[i]["chain_pair_itm_mean"] = cim[i] if cim.dim() == 3 else cim
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/validation/fold")
    ap.add_argument("--results", default=None, help="defaults to <fold>/results.csv")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--limit", type=int, default=0, help="0 = every unit")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth unit")
    ap.add_argument("--offset", type=int, default=0,
                    help="with --stride, which slice this worker takes; one worker per GPU")
    ap.add_argument("--tol", type=float, default=0.02, help="max |stored - recomputed| ipTM")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = a.results or os.path.join(a.fold, "results.csv")

    # COMPARE AGAINST THE FOLD'S OWN summary_confidence_sample_<k>.json, not against results.csv.
    # results.csv carries pair_iptm AFTER OST has mapped model chains to reference chain names; the
    # json is the same object the confidence head produced, so comparing to it isolates the rescoring
    # path from the chain-mapping layer, which is validated elsewhere and is not what is under test.
    import glob as _glob

    def stored_for(pdb, seed_dir):
        out = {}
        # BOTH ON-DISK LAYOUTS, as in phase_j_build_cache.sample_cifs: a handful of units kept
        # Protenix's native <unit>/<job>/seed_*/predictions tree instead of the flattened one.
        pats = [f"{seed_dir}/predictions/*summary_confidence_sample_*.json",
                f"{seed_dir}/*/seed_*/predictions/*summary_confidence_sample_*.json"]
        for f in sorted({g for pat in pats for g in _glob.glob(pat)}):
            k = int(f.split("sample_")[-1].split(".")[0])
            out[k] = json.load(open(f))
        return out

    units = sorted(f for f in __import__("glob").glob(f"{a.fold}/trunks/*/*.npz"))
    units = units[a.offset::a.stride]
    if a.limit:
        units = units[:a.limit]
    print(f"rescoring {len(units)} cached unit(s)")

    dev = torch.device("cuda")
    model, configs = build_model(a.ckpt_dir, a.model_name, dev)
    rows, t0 = [], time.time()
    dmax = dsum = dn = 0.0
    rank_bad = permuted = 0
    for i, p in enumerate(units, 1):
        pdb = os.path.basename(os.path.dirname(p))
        seed = os.path.basename(p).split("_")[-1][:-4]
        # EXACTLY ONE FINISHED DIRECTORY. glob(...)[0] took whichever the filesystem listed first,
        # and a worker that died mid-unit leaves its claim directory behind -- so this could read an
        # abandoned, empty one. Shared with the cache builder so the two cannot disagree.
        import phase_j_build_cache as _BCd
        seed_dir = _BCd.seed_dir(a.fold, pdb, seed)
        want = stored_for(pdb, seed_dir)
        feat, s_inputs, s_trunk, z, x, nc = load_unit(p, dev)
        summ = confidence(model, configs, feat, s_inputs, s_trunk, z, x, nc)
        if len(summ) != len(want):
            raise SystemExit(f"STOP: {pdb}/{seed} cache holds {len(summ)} poses, the fold wrote "
                             f"{len(want)} summaries")
        # MATCH POSES, DO NOT ASSUME ORDER. ci_coordinate is NOT stored in _sample_<k> order --
        # measured as a permutation ([1,2,0,3,4], [1,4,0,2,3], [4,1,3,2,0] on consecutive seeds of
        # 22gi). Comparing pose i to sample i therefore reports a difference of up to 0.01 and
        # scrambles the within-seed ranking, both artefacts of the pairing rather than the scoring.
        # Matched correctly, every pose reproduces its sample to 0.0000.
        ks = sorted(want)
        cost = np.zeros((len(summ), len(ks)))
        for ii in range(len(summ)):
            for jj, k in enumerate(ks):
                cost[ii, jj] = np.abs(_np(summ[ii]["chain_pair_iptm"])
                                      - np.asarray(want[k]["chain_pair_iptm"], float)).max()
        # OPTIMAL ASSIGNMENT, not a per-row argmin. Two poses can score IDENTICALLY -- 9fve/400073
        # has samples 3 and 4 with the same chain_pair_iptm to five decimals, so poses 0 and 3 each
        # match both at cost 0.00000 while differing by 89.6 A in coordinates. A row-wise argmin then
        # assigns both to the same sample and looks like a cache/fold disagreement. The poses are
        # simply indistinguishable BY THIS STATISTIC; the assignment below resolves the tie, and the
        # residual cost is what says whether the cache reproduces the fold.
        # IDENTITY FIRST, SCORES ONLY AS A FALLBACK. Matching on chain_pair_iptm cannot establish
        # which pose is which -- a whole-pose permutation of the confidence outputs would still pair
        # up perfectly, because the pairing follows the very statistic being checked. The cache
        # builder derives the mapping from COORDINATES against the `_sample_<k>.cif` files, which is
        # identity, so that is used whenever the poses are available and the assignment is kept only
        # for units whose cifs are gone.
        perm, how = None, "scores (Hungarian on chain_pair_iptm)"
        import phase_j_build_cache as _BC
        cifs = _BC.sample_cifs(seed_dir, pdb)
        if cifs:
            # THE WHOLE SAMPLE AXIS. This passed x[0], one pose of shape [N_atom, 3], where
            # sample_indices wants [N_sample, N_atom, 3]; len(poses) then counted ATOMS, the helper
            # raised "N cached poses but 5 sample cifs" every time, and a broad except fell back to
            # the score matching this was meant to replace -- so coordinate identity never once
            # governed the check it was added for.
            px = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            if px.ndim != 3:
                raise SystemExit(f"STOP: {pdb}/{seed} ci_coordinate has shape {px.shape}, expected "
                                 "[N_sample, N_atom, 3]")
            if len(cifs) != px.shape[0]:
                raise SystemExit(f"STOP: {pdb}/{seed} has {px.shape[0]} cached pose(s) but "
                                 f"{len(cifs)} sample cif(s)")
            # AND IT IS NOT ALLOWED TO FAIL QUIETLY. With the cifs present, a failed coordinate match
            # is a real disagreement about which pose is which; falling back to scores would hide it.
            perm = [int(v) for v in _BC.sample_indices(px, cifs, 0.01, 0.5)]
            how = "coordinates (identity, from the sample cifs)"
        else:
            print(f"  note: no sample cifs for {pdb}/{seed}; falling back to score matching, which "
                  "cannot establish identity")
        if perm is None:
            from scipy.optimize import linear_sum_assignment
            rows_i, cols_j = linear_sum_assignment(cost)
            match = np.empty(len(summ), dtype=int)
            match[rows_i] = cols_j
            perm = [int(ks[j]) for j in match]
        else:
            match = np.array([ks.index(k) for k in perm], dtype=int)
        if i == 1:
            print(f"  pose identity established by: {how}")
        got_r = [float(_np(summ[ii]["ranking_score"])) for ii in range(len(summ))]
        want_r = [float(want[k]["ranking_score"]) for k in perm]
        for ii in range(len(summ)):
            d = float(cost[ii, match[ii]])
            dmax = max(dmax, d); dsum += d; dn += 1
            rows.append({"pdb": pdb, "seed": seed, "cached_pose": ii, "stored_sample": perm[ii],
                         "max_abs_diff": round(d, 8),
                         "stored_iptm": round(float(want[perm[ii]]["iptm"]), 4),
                         "recomputed_iptm": round(float(_np(summ[ii]["iptm"])), 4)})
        # EVERY CHAIN PAIR SEPARATELY, not the matrix maximum. `.max()` over the whole
        # chain_pair_iptm matrix is not the ranking of any interface: the largest entry can be an
        # intrachain or unrelated pair, so a target pair can change order while the maxima are
        # identical. Requiring the ordering to hold for EVERY pair is strictly stronger than naming
        # one interface and needs no interface list.
        G = np.stack([np.asarray(_np(summ[ii]["chain_pair_iptm"]), dtype=float)
                      for ii in range(len(summ))])
        W = np.stack([np.asarray(want[k]["chain_pair_iptm"], dtype=float) for k in perm])
        if G.shape != W.shape:
            raise SystemExit(f"STOP: {pdb}/{seed} chain_pair_iptm {G.shape} vs stored {W.shape}")
        flat_g = G.reshape(G.shape[0], -1)
        flat_w = W.reshape(W.shape[0], -1)
        changed = [c for c in range(flat_g.shape[1])
                   if np.argsort(flat_g[:, c]).tolist() != np.argsort(flat_w[:, c]).tolist()]
        if changed:
            rank_bad += 1
            print(f"  PAIR-IPTM RANK CHANGED {pdb}/{seed} on {len(changed)} of "
                  f"{flat_g.shape[1]} chain pair(s), e.g. pair {changed[0]}: "
                  f"{np.argsort(flat_w[:, changed[0]]).tolist()} -> "
                  f"{np.argsort(flat_g[:, changed[0]]).tolist()}")
        elif np.argsort(got_r).tolist() != np.argsort(want_r).tolist():
            print(f"  note: global ranking_score order changed for {pdb}/{seed} "
                  f"({np.argsort(want_r).tolist()} -> {np.argsort(got_r).tolist()}) while the "
                  "per-interface pair ipTM order held; the endpoint uses the latter")
        if perm != list(range(len(summ))):
            permuted += 1
        if i % 10 == 0 or i == len(units):
            el = time.time() - t0
            print(f"  {i}/{len(units)}  {el/i:.2f} s/unit  max|diff| {dmax:.5f}  "
                  f"eta {(len(units)-i)*el/i/60:.1f} min")

    print(f"\nchain_pair_iptm: max |stored - recomputed| = {dmax:.6f}, mean {dsum/max(1,dn):.6f} "
          f"over {int(dn)} poses")
    print(f"pose ranking within a seed changed on {rank_bad} of {len(units)} unit(s) AFTER matching")
    print("NOTE: poses are matched by COORDINATES against the sample cifs wherever those exist, "
          "which is identity; the Hungarian assignment on chain_pair_iptm remains only as a "
          "fallback for units whose cifs are gone, and cannot establish identity on its own, since "
          "distinct poses can share scores (9fve/400073). The line above reports the ranking of "
          "per-interface pair ipTM, which is the statistic every selection we report uses.")
    print(f"units whose cached pose order is permuted vs _sample_<k>: {permuted} of {len(units)}")
    # A RANKING CHANGE IS A FAILURE. It was counted and printed but the script still exited 0, so a
    # caller chaining on success saw a pass. Selection is what this project measures: if rescoring a
    # cached unit reorders its poses, the cache cannot stand in for the fold.
    if rank_bad:
        raise SystemExit(
            f"STOP: the within-seed pose ranking changed on {rank_bad} of {len(units)} unit(s) after "
            "matching. Rescoring the cache does not reproduce the fold's ordering, so a selection "
            "measured on the cache would not be the selection the fold made.")
    if dmax > a.tol:
        raise SystemExit(f"STOP: recomputed ipTM differs from the fold's own summary by {dmax:.4f} "
                         f"(tolerance {a.tol}). Rescoring a cached unit does not reproduce the fold, "
                         "so a finetuned-head comparison would be measuring the plumbing.")
    print(f"\n{len(rows)} unit(s) in {(time.time()-t0)/60:.1f} min "
          f"({(time.time()-t0)/max(1,len(rows)):.2f} s/unit)")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
