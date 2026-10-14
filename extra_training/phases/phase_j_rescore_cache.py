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
import sys
import time

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


def load_unit(path, device):
    """Rebuild what run_confidence_head consumes from one cached npz."""
    z = np.load(path)
    def t(key, dtype=None):
        a = z[key]
        if a.dtype == np.uint16:            # bf16 stored as uint16
            return torch.from_numpy(a).to(device).view(torch.bfloat16)
        return torch.from_numpy(a).to(device, dtype=dtype) if dtype else torch.from_numpy(a).to(device)
    feat = {
        "asym_id": t("ci_feat_asym_id"),
        "has_frame": t("ci_feat_has_frame"),
        "atom_to_token_idx": t("ci_feat_atom_to_token_idx"),
        "atom_to_tokatom_idx": t("ci_feat_atom_to_tokatom_idx"),
        "distogram_rep_atom_mask": t("ci_feat_distogram_rep_atom_mask"),
        "is_ligand": t("ci_feat_is_ligand"),
    }
    return (feat, t("ci_s_inputs").float(), t("ci_s_trunk_f32").float(), t("z_trunk"),
            t("ci_coordinate").float(), int(z["ci_N_cycle"]))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/val2_fold")
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
        for f in sorted(_glob.glob(f"{seed_dir}/predictions/*summary_confidence_sample_*.json")):
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
        seed_dirs = _glob.glob(f"{a.fold}/poses/{pdb}/{seed}-*")
        if not seed_dirs:
            raise SystemExit(f"STOP: no pose directory for {pdb}/{seed}")
        want = stored_for(pdb, seed_dirs[0])
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
        from scipy.optimize import linear_sum_assignment
        rows_i, cols_j = linear_sum_assignment(cost)
        match = np.empty(len(summ), dtype=int)
        match[rows_i] = cols_j
        perm = [int(ks[j]) for j in match]
        got_r = [float(_np(summ[ii]["ranking_score"])) for ii in range(len(summ))]
        want_r = [float(want[k]["ranking_score"]) for k in perm]
        for ii in range(len(summ)):
            d = float(cost[ii, match[ii]])
            dmax = max(dmax, d); dsum += d; dn += 1
            rows.append({"pdb": pdb, "seed": seed, "cached_pose": ii, "stored_sample": perm[ii],
                         "max_abs_diff": round(d, 8),
                         "stored_iptm": round(float(want[perm[ii]]["iptm"]), 4),
                         "recomputed_iptm": round(float(_np(summ[ii]["iptm"])), 4)})
        # THE RANKING CHECK MUST ACTUALLY RUN. got_r/want_r were computed and never compared, so
        # rank_bad stayed 0 and the summary line reported "0 of N changed" unconditionally -- a
        # vacuous pass reported as a verification result.
        if np.argsort(got_r).tolist() != np.argsort(want_r).tolist():
            rank_bad += 1
            print(f"  RANK CHANGED {pdb}/{seed} after matching: "
                  f"{np.argsort(want_r).tolist()} -> {np.argsort(got_r).tolist()}")
        if perm != list(range(len(summ))):
            permuted += 1
        if i % 10 == 0 or i == len(units):
            el = time.time() - t0
            print(f"  {i}/{len(units)}  {el/i:.2f} s/unit  max|diff| {dmax:.5f}  "
                  f"eta {(len(units)-i)*el/i/60:.1f} min")

    print(f"\nchain_pair_iptm: max |stored - recomputed| = {dmax:.6f}, mean {dsum/max(1,dn):.6f} "
          f"over {int(dn)} poses")
    print(f"pose ranking within a seed changed on {rank_bad} of {len(units)} unit(s) AFTER matching")
    print("NOTE: poses are matched by confidence SCORES, which cannot establish identity -- distinct "
          "poses can share them (9fve/400073). This verifies that the cache reproduces the fold's "
          "scores; it does NOT license joining cached poses to DockQ or to labels by this mapping. "
          "The cache writer records the sample index instead.")
    print(f"units whose cached pose order is permuted vs _sample_<k>: {permuted} of {len(units)}")
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
