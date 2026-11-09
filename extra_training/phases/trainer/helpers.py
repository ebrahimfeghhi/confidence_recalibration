"""`trainer/helpers.py` -- the small functions `main()` defined inline.

EXTRACTED VERBATIM from `phases/phase_k_train.py`: 41 nested defs, 1741 lines
(source `d34baebc`), dedented one level. These are what `unit.py` and `loop.py` closed
over; collecting them here is what lets those two take an explicit context instead of
reaching into an enclosing scope.
"""
import collections
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time

import random
import threading
import queue
import numpy as np
import torch


# THE CROP PREFETCH'S HANDLES. These were locals of main() that the two _pf_* functions
# declared `nonlocal`; dedenting them to module level made that a syntax error. Module globals
# are equivalent here because each rank is its own process, so there is exactly one of each.
# NOTE the crop prefetch is INACTIVE on the live arm (--crop-tokens empty), so this whole
# pathway is a candidate for deletion -- see the bloat audit.
_pf_q = None
_pf_thread = None


def _spearman(rt, xs, ys):
    """Rank correlation without scipy: rank both, then Pearson. Ties get average ranks."""
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda k: v[k])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    sx = sum((v - mx) ** 2 for v in rx) ** 0.5
    sy = sum((v - my) ** 2 for v in ry) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a_ - mx) * (b_ - my) for a_, b_ in zip(rx, ry)) / (sx * sy)

def _pose_pairs(rt, pdb, lab_p, samp):
    """[(iface, ax, ay)] for ONE pose, from OST -- the SAME join phase_k_eval scores with.

    WHY NOT THE STATIC PAIRS. --rank-form rel's labels are computed on the pose's OST-resolved
    pair; scoring the static pair and looking the label up by interface name would pair
    predicted ipTM for one chain pair with true iTM for another. Measured: the two mappings
    disagree on 1,095 of 18,400 (structure, seed, pose) checks across 52 of 368 structures,
    and 7stf resolves 7stf_BC to (3,1) statically against (3,0) per pose -- a different chain,
    not a reordering. The first version of this arm had exactly that mismatch and no coverage
    check could see it, because the keys still matched. Caught in review.

    REFUSES rather than falling back. A fallback to the static pair is the defect itself.
    """
    import phase_j_build_cache as _BC
    key = (pdb, os.path.basename(lab_p), int(samp))
    if key in rt._ost_cache:
        return rt._ost_cache[key]
    sd = _rank_key(pdb, os.path.basename(lab_p)[:-4], (0, 0))[1]
    try:
        ud = _BC.seed_dir(rt.a.fold, pdb, int(sd))
    except Exception as _e:                              # noqa: BLE001
        raise SystemExit(
            f"STOP: --rank-form rel cannot resolve a unit directory for {pdb}/{sd} "
            f"({type(_e).__name__}), so this pose's chains cannot be mapped the way its "
            f"label was built.")
    got, why = rt._PKE.ost_chain_pairs(ud, pdb, int(samp), rt._rel_refs.get(pdb) or [])
    if got is None:
        raise SystemExit(
            f"STOP: --rank-form rel has no OST chain mapping for {pdb}/{sd} sample {samp} "
            f"({why}). Falling back to the static pair is the exact mismatch this resolution "
            f"exists to prevent.")
    rt._ost_cache[key] = sorted(got)
    return rt._ost_cache[key]

def _rel_labels(rt, pdb):
    """{(seed, sample, ax, ay): g} for one structure, from --rel-labels.

    A MISSING FILE IS FATAL, not empty. An empty dict would make every group "short", the loop
    would skip them all, and the run would train on PAE alone while its log and saved guard
    both said --rank-form rel -- the row-weight bool-cast failure in another costume.
    """
    if pdb in rt._rel_cache:
        return rt._rel_cache[pdb]
    if not rt.a.rel_labels:
        raise SystemExit("STOP: --rank-form rel needs --rel-labels; build it with "
                         "run_logging/build_itm_labels.py")
    f = os.path.join(rt.a.rel_labels, f"{pdb}.npz")
    if not os.path.exists(f):
        rt._rel_stat["missing_files"] += 1
        raise SystemExit(
            f"STOP: no relative-iTM label file for {pdb} at {f}. Every training structure "
            f"needs one; rebuild with run_logging/build_itm_labels.py and check its manifest "
            f"covers this manifest-selection.")
    z = np.load(f, allow_pickle=True)
    col = "g_max" if rt.a.rank_pool == "max" else "g_mean"
    # THE LABEL MUST MATCH THE POOLING. Training the mean-pooled score toward a max-pooled
    # label would regress onto a different quantity than the one being produced, and both
    # columns exist precisely so this is a lookup rather than a conversion.
    if "iface" not in z.files:
        raise SystemExit(
            f"STOP: {f} is an OLD label cache keyed by an asym pair. Those caches paired a "
            f"value computed on the pose's OST pair with the STATIC pair as its key, which is "
            f"the mismatch review caught. Rebuild with run_logging/build_itm_labels.py.")
    d = {(int(sd), int(sp), str(ifc)): float(g)
         for ifc, sd, sp, g in zip(z["iface"], z["seed"], z["sample"], z[col])}
    rt._rel_cache[pdb] = d
    return d

def _rank_key(rt, pdb, arm_seed_or_int, pair):
    """THE label key, built in exactly one place.

    A unit's identifier is the label/trunk basename -- "free_500000", not 500000 -- because
    that is what names the files. The guard built its keys from the manifest, where seeds are
    plain ints, while the lookup in run_unit did int(arm_seed) on the basename and raised
    ValueError on the first unit. Both sides were internally consistent and disagreed with
    each other, which is the same shape as the reference-vs-model chain bug: two independent
    key constructions. There is now one, so they cannot diverge again.
    """
    sd = arm_seed_or_int
    if isinstance(sd, str):
        sd = sd.rsplit("_", 1)[-1]
    return (pdb, int(sd), pair)

def _rank_check_coverage(rt, want_units):
    """Every (pdb, seed, interface) this run will train must carry --nsample finite DockQ.

    DRIVEN FROM THE EXPECTED SET, NOT FROM THE LABELS. Two earlier versions were weaker and
    both would have passed the run that was actually broken: one checked only that a structure
    appeared SOMEWHERE in the labels (a pdb-level test cannot see a per-pair hole), and one
    iterated over the labels and skipped keys absent from _scored_asym (so a scored pair with
    no labels at all was invisible). The reference-vs-model chain bug survived both. The
    expected set is built from the manifest and the scored pairs, and set difference does the
    work.
    """
    # NOT UNDER --loss-only. The held-out monitor is a subprocess on the VALIDATION fold
    # (seeds 400000+), it inherits rank_loss=True only so the resume guard matches, and the
    # ranking term never runs there -- a measurement takes no step. Demanding training-fold
    # labels for validation units is a check it cannot satisfy, and it killed the held-out
    # pass on every epoch of s7dh7o: "540 group(s) have NO labels, e.g. ('7qu2', 400000,
    # (0, 2))". FIFTH instance of this class -- --train-only, LoRA, the evaluator's
    # --pose-perblock-share, the ranking guard fields, now a check of my own that a
    # measurement cannot pass.
    # UNCHANGED BY --rank-cross-seed, and that is not an oversight. Cross-seed grouping
    # changes which poses are COMPARED, not which labels must exist: every (pdb, seed,
    # interface) the run touches still needs its --nsample DockQ values, and the groups are
    # built from the same manifest units this set is built from. A group missing a seed's
    # labels is therefore already caught here, before pass A ever runs.
    if not rt.a.rank_loss or rt.a.loss_only:
        return
    expected = {_rank_key(pdb, sd, pr) for pdb, sd in want_units
                for pr in (rt._scored_asym.get(pdb) or ())}
    if not expected:
        raise SystemExit("STOP: --rank-loss expected zero (pdb, seed, interface) groups. "
                         "Either the manifest or --pae-scored-pairs selects nothing.")
    missing = sorted(expected - set(rt._rank_lab))
    short, nonfinite = [], []
    for k in sorted(expected & set(rt._rank_lab)):
        m = rt._rank_lab[k]
        if sorted(m) != list(range(rt.a.nsample)):
            short.append((k, sorted(m)))
        if not all(math.isfinite(v) for v in m.values()):
            nonfinite.append(k)
    if missing or short or nonfinite:
        raise SystemExit(
            f"STOP: --rank-loss label coverage is incomplete for the {len(expected)} "
            f"(pdb, seed, interface) group(s) this run trains.\n"
            + (f"  {len(missing)} group(s) have NO labels, e.g. {missing[:6]}\n"
               if missing else "")
            + (f"  {len(short)} group(s) lack samples 0..{rt.a.nsample - 1}, e.g. {short[:4]}\n"
               if short else "")
            + (f"  {len(nonfinite)} group(s) carry a non-finite DockQ, e.g. "
               f"{nonfinite[:4]}\n" if nonfinite else "")
            + "  Those poses would pay a forward and contribute no ranking gradient.")
    if rt.is_main:
        print(f"  ranking label coverage: all {len(expected)} (pdb, seed, interface) group(s) "
              f"hold {rt.a.nsample} finite DockQ value(s)")

def _check_eval_pool(rt, ):
    """Stat the training-eval pool now, not at the first epoch boundary.

    eval_checkpoint shells out to phase_k_eval.py, which hard-stops on a missing trunk or a
    pool interface with no scored pose. That SystemExit lands on RANK 0 ONLY, while ranks 1
    and 2 sit in dist.barrier() until --dist-timeout-min, so a bad pool costs hours of
    training and then a four-hour zombie. The default fold pool really does contain a
    structure the manifest skips (9mx1 has no trunk at all), so this is one wrong path away.
    """
    if not rt.a.train_eval_pool:
        return
    # THE POOL'S OWN FOLD, matching what eval_checkpoint will pass. Checking against --fold
    # when --train-eval-fold overrides it rejected a perfectly good validation pool for
    # having no trunks in the TRAINING fold, which is exactly where it should not have any.
    _f = rt.a.train_eval_fold or rt.a.fold
    if "," in _f:
        raise SystemExit(f"STOP: --train-eval-pool needs a single fold; got {_f!r}. "
                         f"phase_k_eval treats --fold as one directory, so a comma-separated "
                         f"root would be read as a literal path.")
    need = [rt.a.train_eval_pool, os.path.join(_f, "results.csv"),
            os.path.join(_f, "chain_roles.csv")]
    gone = [f for f in need if not os.path.exists(f)]
    if gone:
        raise SystemExit(f"STOP: --train-eval-pool needs {gone}, which do not exist.")
    import csv as _csv2
    # --manifest-skip describes the TRAINING manifest, so it says nothing about a pool in
    # another fold; skipping a structure from training is not a reason it cannot be scored.
    skip = ({x.strip() for x in (rt.a.manifest_skip or "").split(",") if x.strip()}
            if not rt.a.train_eval_fold else set())
    pool_pdbs = {r["pdb"] for r in _csv2.DictReader(open(rt.a.train_eval_pool))}
    bad = sorted(pool_pdbs & skip)
    if bad:
        raise SystemExit(f"STOP: --train-eval-pool names {bad}, which --manifest-skip excludes "
                         f"from training. The evaluator hard-stops on a structure with no "
                         f"trunk, on rank 0, with the other ranks parked at a barrier.")
    notrunk = sorted(p for p in pool_pdbs
                     if not os.path.isdir(os.path.join(_f, "trunks", p)))
    if notrunk:
        raise SystemExit(f"STOP: --train-eval-pool names {len(notrunk)} structure(s) with no "
                         f"trunk directory, e.g. {notrunk[:4]}.")
    print(f"train-eval pool: {len(pool_pdbs)} structure(s), all present, none skipped")

def _pae_mask_coverage(rt, units):
    """Every structure this run trains on must be named by the pool, checked BEFORE training.

    The running guard below only fires while NOTHING has been masked yet, so once a single
    unit masks successfully an unlimited number of later units can fall back to the UNMASKED
    PAE term in silence. Review reproduced one masked call followed by 100 unmasked ones with
    no error. A per-structure coverage check at startup closes that: a structure missing from
    the pool is a manifest/pool mismatch, which is a setup error, and it is cheap to find here
    rather than impossible to notice later.
    """
    if not rt._scored_asym:
        return
    want = sorted({t[0] for t in units})
    gone = [p for p in want if not rt._scored_asym.get(p)]
    if not gone:
        print(f"  PAE mask: all {len(want)} training structure(s) are named by "
              f"{rt.a.pae_scored_pairs}")
        return
    if not rt.a.allow_unmasked_fallback:
        raise SystemExit(
            f"STOP: {len(gone)} of {len(want)} training structure(s) are NOT in "
            f"{rt.a.pae_scored_pairs}, so their poses would train the UNMASKED PAE term while "
            f"this run is labelled masked: {gone[:8]}. Fix the pool, drop them from "
            f"--manifest-selection, or pass --allow-unmasked-fallback to accept it "
            f"deliberately -- it is recorded in args.json either way.")
    print(f"  PAE mask: ACCEPTING UNMASKED FALLBACK for {len(gone)} of {len(want)} "
          f"structure(s) (--allow-unmasked-fallback): {gone[:8]}")

def _pae_mask_check(rt, ):
    """A mask that never fires is a baseline arm wearing the masked arm's label.

    Both fallbacks below are silent by design -- they keep training rather than dying on one
    odd structure. But if NOTHING is ever masked, the startup banner has announced an objective
    the run is not training. The classic way in is pointing --pae-scored-pairs at the wrong
    pool (the validation one, say): the banner reports 92 structures, every training unit then
    misses, and the run trains the unmasked term under the masked arm's name.
    """
    st = rt._pae_mask_stats
    if rt._scored_asym and st["masked"] == 0 and st["unmasked"] >= 20:
        raise SystemExit(
            f"STOP: --pae-scored-pairs matched NO unit in the first {st['unmasked']} seen. "
            f"The pool names {len(rt._scored_asym)} structure(s), none of which this run trains "
            f"on, so every pose fell back to the UNMASKED PAE term while the run is labelled "
            f"masked. Check that {rt.a.pae_scored_pairs} covers --manifest-selection.")

def _set_pae_mask(rt, pdb, asym, has_frame, unit=None, pairs_override=None):
    """[N_frame, N_token] over the CROPPED tokens: keep only scored cross-chain entries.

    Falls back to no mask -- never to an all-zero one. An empty mask would make PAELoss divide
    by eps and return a meaningless number rather than fail, which is the worst outcome.

    pairs_override EXISTS FOR --rank-form rel. _scored_asym is built once per structure from
    chain_roles, while rel resolves each interface's asym pair PER POSE through OST. On the 52
    structures where the two disagree, PAE would anchor a different interface from the one rel
    is ranking -- 7stf's (3,1) against (3,0) -- so the two terms would be supervising
    different objects on the same pose. That is objective misalignment, not gradient noise.
    Raised in review. The override is per pose because OST's answer is.
    """
    rt._pae_keep["pdb"], rt._pae_keep["unit"] = pdb, unit
    if not rt._scored_asym and pairs_override is None:
        return
    # COUNTED AND REPORTED, because "the override is wired" and "the override CHANGES
    # something" are different claims and only the second matters. Twice today a change to
    # this code path produced numbers bit-identical to the version it replaced -- once
    # because --ost-mapping was not forwarded to the workers, once here -- and in both cases
    # the identical numbers were the only evidence anything was wrong. So the run says how
    # many poses got an override and on how many it DIFFERED from the static pairs.
    if pairs_override is not None:
        _ov = rt._pae_keep.setdefault("ov", {"poses": 0, "differed": 0, "pdbs": set()})
        _ov["poses"] += 1
        if set(pairs_override) != set(rt._scored_asym.get(pdb) or ()):
            _ov["differed"] += 1
            _ov["pdbs"].add(pdb)
    pairs = pairs_override if pairs_override is not None else rt._scored_asym.get(pdb)
    if not pairs:
        rt._pae_keep["m"] = None
        rt._pae_keep["blocks"] = None
        rt._pae_mask_stats["unmasked"] += 1
        _pae_mask_check()
        return
    aa = asym.reshape(-1).long()
    fa = aa[has_frame.reshape(-1).bool()]
    keep = torch.zeros((fa.numel(), aa.numel()), dtype=torch.bool, device=aa.device)
    # PER-PAIR BLOCKS, KEPT SEPARATELY. Selection reads chain_pair_iptm[x, y] -- one max per
    # INTERFACE, each over just those two chains and with its own d0. 298 of the 368 training
    # structures score two or more pairs, so ranking rows on the union would rank a row in the
    # antigen by its heavy- and light-chain targets averaged together, which is a quantity no
    # ipTM reads. --pae-row-frac weights each block on its own t_i; the union is still what the
    # unweighted term uses, and the blocks sum back to it at uniform weights.
    blocks = []
    for x, y in pairs:
        blk = (((fa[:, None] == x) & (aa[None, :] == y))
               | ((fa[:, None] == y) & (aa[None, :] == x)))
        keep |= blk
        if bool(blk.any()):
            blocks.append((blk, int(((aa == x) | (aa == y)).sum())))
    if not bool(keep.any()):
        rt._pae_keep["m"] = None
        rt._pae_keep["blocks"] = None
        rt._pae_mask_stats["unmasked"] += 1
        _pae_mask_check()
        return
    rt._pae_keep["m"] = keep
    rt._pae_keep["blocks"] = blocks
    rt._pae_mask_stats["masked"] += 1

def _head_ckpt_tensors(rt, ):
    """The head checkpoint, ALWAYS in the shipped head's tensor names and shapes.

    Under LoRA the wrapper renames `blk.linear.weight` to `blk.linear.base.weight`, so
    saving named_parameters() straight would write a file phase_k_eval cannot load -- it
    knows nothing about adapters. Merging here keeps every downstream scorer unchanged and
    makes a LoRA checkpoint indistinguishable from a full-finetune one. Without LoRA this is
    exactly the dict that was saved before.
    """
    if rt.a.lora_rank:
        import lora as LORA
        _pfx = rt.prefixes[0]
        return {f"{_pfx}.{k}": v
                for k, v in LORA.merged_head_tensors(
                    dict(rt.model.named_modules())[_pfx]).items()}
    # EVERY HEAD TENSOR, NOT JUST THE TRAINABLE ONES. phase_k_eval requires SET EQUALITY
    # between the checkpoint's keys and the model's head parameters -- deliberately, so a
    # checkpoint from a different architecture cannot load with tensors left at their
    # initial values. Under --train-only the trainable set is three tensors, and writing
    # only those would be refused at the first eval. Identical to the old behaviour when the
    # whole head trains, which is every run so far.
    return {n: pp.detach().cpu() for n, pp in rt.model.named_parameters()
            if n in rt._head_names}

def _parse_seed_field(rt, txt, who):
    """'500000-500014' or '500000,500003' or a mix -> a set of ints."""
    out = set()
    for part in str(txt).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            try:
                lo, hi = (int(x) for x in part.split("-"))
            except ValueError:
                raise SystemExit(f"STOP: {who} seed field {part!r} is not lo-hi")
            if hi < lo:
                raise SystemExit(f"STOP: {who} seed range {part!r} runs backwards")
            out |= set(range(lo, hi + 1))
        else:
            try:
                out.add(int(part))
            except ValueError:
                raise SystemExit(f"STOP: {who} seed field {part!r} is not an integer")
    if not out:
        raise SystemExit(f"STOP: {who} has an empty `seeds` field; remove the column or name "
                         f"the seeds, but do not declare a structure with no depth")
    return out

def _crop_masks(rt, pdb_, zc_):
    """(keep_tok, keep_atom) from whichever crop rule is configured. One dispatch, two callers.

    A MISSING MASK IS FATAL, never a silent fall-through to the full structure: a run where
    some units were cropped and some were not would still write a complete trajectory and
    name itself after the crop.
    """
    kt_ = rt._CROPTOK.get(pdb_)
    if kt_ is None:
        raise SystemExit(f"STOP: --crop-tokens has no mask for {pdb_}. The file names "
                         f"{len(rt._CROPTOK)} structure(s); this manifest needs {pdb_} too. "
                         f"Rebuild the mask over this pool.")
    a2t_ = np.asarray(zc_["ci_feat_atom_to_token_idx"]).reshape(-1)
    ntok_ = int(np.asarray(zc_["ci_feat_asym_id"]).reshape(-1).shape[0])
    if kt_.shape[0] != ntok_:
        raise SystemExit(f"STOP: {pdb_}'s mask has {kt_.shape[0]} token(s) but its cached "
                         f"trunk has {ntok_}. The mask was built against a different cache "
                         f"and slicing with it would scramble the structure.")
    if rt._TOK_DROP:
        # ONE DRAW PER (STRUCTURE, EPOCH), seeded so it is reproducible and so every rank and
        # the prefetch worker agree without communicating. Per-epoch IS per-step here:
        # build_epoch_cross_seed puts every seed of a structure in a single step.
        # hashlib, NOT hash(). Python salts str hashing per process (PYTHONHASHSEED), so
        # hash() would give each RANK a different draw for the same structure -- and under
        # --rank-cross-seed the ranks hold different SEEDS OF THE SAME STRUCTURE inside one
        # step, so they must agree exactly or those poses are scored on different token sets
        # and the relative loss grades the disagreement. It also has to survive a resume.
        _sd_bytes = f"{int(rt.a.seed)}|{int(rt._crop_epoch[0])}|{pdb_}".encode()
        _rs = np.random.default_rng(
            int.from_bytes(hashlib.sha256(_sd_bytes).digest()[:8], "big"))
        _p = np.where(kt_, float(rt.a.token_keep_union), float(rt.a.token_keep_other))
        _keep = _rs.random(kt_.shape[0]) < _p
        # EVERY SUPERVISED PAIR MUST STAY SCOREABLE -- per pair, not "two chains somewhere".
        # ipTM for pair (x, y) pools over token pairs spanning x and y, so if either chain
        # loses its tokens that interface is scored on an EMPTY pool while its label stays in
        # the loss. A four-chain complex can keep two chains and still leave pair (3,1)
        # unscoreable, and 81% of this pool has 2+ scored pairs. This is NOT a requirement
        # that any ground-truth CONTACT survives -- the augmentation being tested is precisely
        # that contacts may vanish. Raised in review.
        _asym = np.asarray(zc_["ci_feat_asym_id"]).reshape(-1)
        _prs = rt._CROPPAIRS.get(pdb_)
        if _prs is None:
            raise SystemExit(
                f"STOP: --token-keep-* needs the scored pairs for {pdb_}, which this mask "
                f"does not carry (no {pdb_}__pairs entry). It was built before the builder "
                f"wrote them; rebuild with run_logging/build_union_crop_masks.py so the "
                f"per-pair scoreability check has something to check.")
        # REPAIRED, NOT REFUSED. The first version raised SystemExit here and killed the
        # run 2,240 poses into vxwvzr's first epoch, on 9iut: chain 2 drew 3 survivors against
        # a floor of 4. That was not bad luck. 98 scored-chain instances in this pool have an
        # EXPECTED survivor count under 12 and some scored chains hold only 6 tokens in
        # total -- short peptide antigens -- so a 0.70 draw dips under the floor routinely.
        #
        # Escalating a whole run over one small chain is the wrong response to a constraint
        # that is local: the floor exists so each supervised pair has a non-degenerate ipTM
        # pool, and that is restored by adding back a few of THAT CHAIN's tokens. Dropping
        # context is the augmentation; a 6-token chain has no context to drop. The top-up uses
        # the same generator in a deterministic order, so every rank still agrees exactly.
        _MIN_CH = 4
        for _x, _y in sorted({(int(_p0), int(_p1)) for _p0, _p1 in _prs}):
            for _c in (_x, _y):
                _idx_c = np.nonzero(_asym == _c)[0]
                _floor = min(_MIN_CH, _idx_c.shape[0])
                _have = int(_keep[_idx_c].sum())
                if _have >= _floor:
                    continue
                _missing = _idx_c[~_keep[_idx_c]]
                _add = _rs.permutation(_missing)[:_floor - _have]
                _keep[_add] = True
                rt._tok_drop_stat["topped"] += 1
                rt._tok_drop_stat["topped_tok"] += int(_add.shape[0])
        if int(_keep.sum()) < 16:
            raise SystemExit(
                f"STOP: the token draw for {pdb_} at epoch {rt._crop_epoch[0]} left only "
                f"{int(_keep.sum())} token(s) in total. Raise the keep probabilities.")
        rt._tok_drop_stat["kept"] += float(_keep.sum())
        rt._tok_drop_stat["tot"] += float(_keep.shape[0])
        rt._tok_drop_stat["n"] += 1
        kt_ = _keep
    inr_ = (a2t_ >= 0) & (a2t_ < ntok_)
    ka_ = np.zeros(a2t_.shape[0], dtype=bool)
    ka_[inr_] = kt_[a2t_[inr_]]
    return kt_, ka_

def _tick(rt, key, t0):
    rt._T[key] += time.perf_counter() - t0
    rt._TN[key] += 1

def _gpu_span(rt, key):
    if not torch.cuda.is_available():
        return None, None
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    rt._ev_pending.append((key, e0, e1))
    return e0, e1

def _drain_gpu_spans(rt, ):
    if not rt._ev_pending:
        return
    torch.cuda.synchronize()
    for key, e0, e1 in rt._ev_pending:
        rt._T[key] += e0.elapsed_time(e1) / 1000.0
        rt._TN[key] += 1
    rt._ev_pending.clear()

def _fp_sig(rt, pdb, trunk_p):
    out = []
    for path in (trunk_p,
                 os.path.join(rt.a.mmcif_dir, f"{pdb}.cif"),
                 os.path.join(rt.a.bio_dir, f"{pdb}.pkl.gz"),
                 rt.a.indices):
        try:
            st = os.stat(path)
            out.append(f"{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            out.append("absent")
    return "|".join(out)

def _unit_fp(rt, pdb, trunk_p, memo=None):
    """The COMPOSED provenance string for (pdb, trunk_p), memoised through its content half.

    THE ONE PLACE unit_fingerprint IS CALLED FROM. Both callers -- resolve_trunk's candidate
    loop and run_unit's per-unit re-assertion -- used to call BC.unit_fingerprint directly and
    each kept its own `if key not in cache` dance around it. Two call sites meant two chances
    for the memoisation and the timing to drift apart; this is one.

    WHAT IS MEMOISED AND WHAT IS NOT. The four SHAs (cif, bioassembly, indices rows, trunk) are
    memoised, in `_fp_content`, and that memo is what survives to disk. schema, code_fp and
    loss_fp are re-composed on every call from THIS process's values, so a cache written by a
    different checkout cannot smuggle its code_fp into a comparison -- the composed string is
    always this process's claim about this process's code. `_fp_cache` then holds the composed
    result so a repeat call is one dict lookup.

    `memo` OVERRIDES THE COMPOSED CACHE ONLY, deliberately: resolve_trunk takes a caller-
    supplied dict for the composed strings, and the content memo stays global because it is
    keyed on file identity and nothing else.
    """
    memo = rt._fp_cache if memo is None else memo
    key = (pdb, trunk_p)
    hit = memo.get(key)
    if hit is not None:
        return hit
    content = rt._fp_content.get(key)
    if content is None:
        _t0 = time.perf_counter()
        content = rt.BC.unit_content_fingerprint(rt.a, pdb, trunk_p)
        # A WORKER'S HASH GOES IN ITS OWN BUCKET, because `fingerprint_hash` is in _EXCL -- the
        # sum that is subtracted from the epoch's wall clock to get the residual. Several
        # warmer threads hashing concurrently with the main thread can put MORE than the
        # epoch's wall clock into that bucket and drive the printed residual negative. That is
        # precisely the double-count the `rank_pass_a` comment on _EXCL was written about; it
        # already applied to the cropped arm's _pf_worker and this makes it explicit for both.
        # fingerprint_hash_warm is reported and is in neither _EXCL nor _leaves.
        _tick("fingerprint_hash" if threading.current_thread() is threading.main_thread()
              else "fingerprint_hash_warm", _t0)
        # THE SIGNATURE IS TAKEN NOW, NOT AT SAVE TIME. _fp_cache_save used to re-stat at the
        # boundary, so a file rewritten between the hash and the save was persisted with a
        # signature describing the NEW file and content describing the OLD -- the one direction
        # a provenance cache must never be wrong in. Taking it beside the hash makes the pair
        # consistent by construction. (Version 1 had the same pattern; this is a strengthening,
        # not a regression fix.)
        rt._fp_sig_at_hash[key] = _fp_sig(pdb, trunk_p)
        rt._fp_content[key] = content
    fp = rt.BC.unit_fingerprint(rt.a, pdb, trunk_p, rt.code_fp, rt.loss_fp, content=content)
    memo[key] = fp
    return fp

def _fp_cache_load(rt, label="startup"):
    """Adopt every rank's CONTENT fingerprints whose stat signature still holds.

    CALLED AT EVERY EPOCH BOUNDARY, NOT ONLY AT STARTUP, so the three ranks pool their work.
    Each rank hashes only the units it was dealt and the three jointly cover all 6,903 units in
    epoch 1 -- but a rank that never loaded its siblings' files re-hashed, from epoch 2 onward,
    every unit the shuffle moved onto it. Reloading after _fp_cache_save at the boundary makes
    epoch 2 free for all three instead of one.

    IT CANNOT UNDO A HASH THIS PROCESS COMPUTED. Entries already in `_fp_content` are skipped
    before the signature is even looked at, so an adopted entry only ever fills a hole.
    """
    kept = dropped = 0
    for fp_path in sorted(glob.glob(os.path.join(rt.roots[0], "unit_fingerprints.r*.json"))):
        try:
            blob = json.load(open(fp_path))
        except (OSError, ValueError):
            continue
        # A STALE SCHEMA INVALIDATES THE WHOLE FILE: version 1 stored composed fingerprints
        # under the key "fp" and there is no way to recover the content half from one, since
        # the prefix it was composed with is whatever that process held. code_fp and loss_fp
        # are NOT checked -- that was the bug, see the block comment above.
        if blob.get("ver") != rt._FP_CACHE_VER:
            dropped += len(blob.get("entries", {}))
            continue
        for key, ent in blob.get("entries", {}).items():
            pdb, _, tp = key.partition("\t")
            if not tp or (pdb, tp) in rt._fp_content:
                continue
            if ent.get("content") and ent.get("sig") == _fp_sig(pdb, tp):
                rt._fp_content[(pdb, tp)] = ent["content"]
                # THE SIGNATURE JUST VERIFIED, carried forward so a later save re-writes what
                # was actually checked rather than re-statting at the boundary.
                rt._fp_sig_at_hash[(pdb, tp)] = ent["sig"]
                kept += 1
            else:
                dropped += 1
    # AT A BOUNDARY, ONLY ADOPTIONS ARE WORTH A LINE. `dropped` is sticky -- a save only ever
    # adds, so an entry whose inputs really did change is re-reported at every boundary for the
    # rest of the run -- and one such line per epoch per rank is noise, not information.
    if kept or (dropped and label == "startup"):
        print(f"fingerprint cache ({label}): {kept} content entry(ies) reused from "
              f"{rt.roots[0]}, {dropped} invalidated by a changed input or an older cache "
              f"version", flush=True)
    return kept

def _fp_cache_save(rt, ):
    """Atomic, and only ever ADDS to what this rank already had on disk."""
    try:
        prev = json.load(open(rt._fp_disk))
        entries = prev.get("entries", {}) if prev.get("ver") == rt._FP_CACHE_VER else {}
    except (OSError, ValueError):
        entries = {}
    # list(), BECAUSE A WARMER THREAD CAN STILL BE WRITING IT. _tr_worker calls resolve_trunk,
    # which fills _fp_content, and a warmer that outlived _tr_end's join would make this
    # "RuntimeError: dictionary changed size during iteration" -- a crash at an epoch boundary,
    # after the epoch's work, which is the worst possible moment. A snapshot costs one list of
    # ~11,040 tuples. The signature comes from the hash, not from a fresh stat; see _unit_fp.
    for (pdb, tp), content in list(rt._fp_content.items()):
        entries[f"{pdb}\t{tp}"] = {"content": content,
                                   "sig": rt._fp_sig_at_hash.get((pdb, tp)) or _fp_sig(pdb, tp)}
    tmp = rt._fp_disk + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump({"ver": rt._FP_CACHE_VER, "code_fp": rt.code_fp, "loss_fp": rt.loss_fp,
                       "entries": entries}, fh)
        os.replace(tmp, rt._fp_disk)
    except OSError as e:
        # A CACHE IS NOT WORTH A RUN. Reported and dropped.
        print(f"  WARNING: could not write {rt._fp_disk} ({e}); the next process re-hashes",
              flush=True)

def _cpu_crop(rt, pdb, lab_p, trunk_p):
    """The expensive CPU half, with no device in sight. Returns None if no crop is definable.

    TIMED IN FOUR STAGES, because this was a black box and every number anyone quoted about it
    had to be DERIVED from the epoch total. That is not good enough to tell the two
    explanations of a shrinking prefetch wait apart: the trunk read genuinely getting faster
    (a real win) and the same work merely moving from the main thread into this one (no win).
    Stage `cpucrop_trunk` falling while `cpucrop_tokens` and `cpucrop_slice` hold constant is
    the signature of the first. It also settles a gap nobody could close: a warm-cache bench
    on 10 units put the whole crop at 0.291 s while the live worker appeared to use ~0.485 s.

    The counters are shared with the main thread, which runs this inline on a prefetch miss.
    `+=` on a defaultdict is not atomic, so a simultaneous miss can lose one sample out of
    thousands. That is acceptable for a diagnostic and is NOT acceptable for anything the run
    branches on; nothing branches on these.
    """
    _t = time.perf_counter()
    lab_l = np.load(lab_p, allow_pickle=True)
    _tick("cpucrop_label", _t)
    _t = time.perf_counter()
    zc = np.load(trunk_p)
    _tick("cpucrop_trunk", _t)
    _t = time.perf_counter()
    kt, ka = (_crop_masks(pdb, zc) if rt.a.crop_tokens
              else rt.BC.interface_crop_tokens(zc, lab_l, rt.a.crop_contact, rt.a.crop_margin,
                                            all_poses=rt.a.crop_all_poses))
    _tick("cpucrop_tokens", _t)
    if kt is None:
        return ("nocrop", None, None)
    _t = time.perf_counter()
    sl = rt.BC.slice_cached_unit(zc, lab_l, kt, ka, torch.device("cpu"), torch)
    _tick("cpucrop_slice", _t)
    return ("ok", kt, sl)

def _pf_worker(rt, units):
    # ROUND-ROBIN SLICES, so worker w takes units[w::W] and the queue is fed in roughly the
    # order the main loop consumes. Roughly, not exactly: with W workers the arrival order
    # can run ahead by up to W units, which is what _pf_pending absorbs.
    for _k, t in enumerate(units):
        if rt._pf_stop.is_set():
            break
        pdb_, lab_p_, cands_ = t
        try:
            trunk_p_ = resolve_trunk(pdb_, lab_p_, cands_)
            # THE READAHEAD BELONGS TO THE THREAD THAT DOES THE READING. _prefetch() is also
            # issued by the MAIN loop for unit i+1, but with --prefetch-units 2 this worker is
            # already 2-3 units past the main loop, so that hint lands on a trunk this thread
            # has ALREADY read -- it is issued behind the only consumer it exists to help, and
            # its own timer confirms it is free and useless there (prefetch_issue 0.0m/1460).
            # Measured on j583i8: the worker sits at 49% of one core with the device at
            # 553 MB/s and ~250 MB of z_trunk per unit, i.e. ~0.505 s/unit blocked in read()
            # against 0.715 s/unit of main-thread work -- so the WORKER sets the epoch length
            # and 6.7 of 24.1 min is the main thread waiting on it.
            if _k + 1 < len(units):
                _n = units[_k + 1]
                try:
                    _prefetch(resolve_trunk(_n[0], _n[1], _n[2]))
                except BaseException:         # noqa: BLE001 -- a bad NEXT unit must not kill
                    pass                      # THIS one; it reports itself when reached.
            res = _cpu_crop(pdb_, lab_p_, trunk_p_)
        except BaseException as e:            # noqa: BLE001 -- see below
            # THE WORKER NEVER KILLS THE RUN. Anything it fails on is handed back as a miss
            # and the main thread redoes that unit inline, where the real error surfaces with
            # its own diagnostics. A prefetcher that can crash training is worse than none.
            res = ("error", type(e).__name__, str(e)[:120])
        while not rt._pf_stop.is_set():
            try:
                _pf_q.put((lab_p_, res), timeout=1.0)
                break
            except queue.Full:
                continue

def _pf_take(rt, lab_p, budget_s=120.0):
    """The prefetched result for this unit, WAITING for the worker if it is not ready yet.

    IT MUST BLOCK, AND get_nowait() IS WHY THE FIRST VERSION DID NOTHING. Measured on the
    smoke test: crop_slice 0.70 s/unit with the prefetch on against 0.71 s/unit with it off.
    A non-blocking take gives up the instant the queue is empty, which at a cold start is a
    PERMANENT MISS CASCADE:

      unit 0  queue empty (worker just started) -> None -> main thread crops unit 0 inline,
              0.70 s, while the worker crops the same unit 0
      unit 1  queue holds unit 0's result; k != lab_p so it is stashed and the loop goes round;
              queue now empty -> None -> main thread crops unit 1 inline
      ...     the worker stays exactly one unit behind forever, every take misses, and a core
              is burned recomputing what the main thread just computed

    The main thread never waited even once, so the worker never got the head start it needed
    to pull ahead. Blocking costs nothing against the alternative -- the worker is already
    partway through this very unit, and doing it inline means doing the whole thing -- and
    after one wait the main thread is on the GPU for ~0.84 s/unit while the worker fills the
    queue to --prefetch-units and every later take is a hit.

    THE BUDGET IS A DEADLOCK GUARD, NOT A TUNABLE. The worker emits exactly one result per
    unit, in the main loop's own order, and hands back ("error", ...) rather than dying, so
    the awaited result always comes. 120 s against a 0.7 s crop means the worker is gone;
    then this returns None and the unit is cropped inline exactly as before.

    IN ORDER, WITH A SMALL HOLDING BUFFER. Results that arrive early are held by lab_p rather
    than discarded.
    """
    if lab_p in rt._pf_pending:
        rt._pf_stat["hit"] += 1
        return rt._pf_pending.pop(lab_p)
    if _pf_q is None:
        return None
    # ALREADY DELIVERED AND EVICTED: it will never arrive again, so do not wait for it.
    if lab_p in rt._pf_dropped:
        rt._pf_dropped.discard(lab_p)
        rt._pf_stat["dropped"] += 1
        rt._pf_stat["miss"] += 1
        return None
    _t_w = time.perf_counter()
    deadline = _t_w + budget_s
    while True:
        # EVERY WORKER HAS FINISHED ITS STRIDE AND THE QUEUE IS EMPTY, so nothing more can
        # ever arrive. Observed in the reproduction: with W=4 the workers exit while the main
        # loop is still asking for units, after which EVERY remaining take cost the whole
        # budget. Waiting here is waiting on no one.
        if (_pf_thread is not None
                and not any(_t.is_alive() for _t in _pf_thread)
                and _pf_q.empty()):
            rt._pf_stat["exhausted"] += 1
            rt._pf_stat["miss"] += 1
            rt._pf_stat["wait_s"] += time.perf_counter() - _t_w
            return None
        try:
            k, v = _pf_q.get(timeout=max(0.05, deadline - time.perf_counter()))
        except (queue.Empty, AttributeError):
            rt._pf_stat["miss"] += 1
            rt._pf_stat["wait_s"] += time.perf_counter() - _t_w
            return None
        if k == lab_p:
            rt._pf_stat["hit"] += 1
            rt._pf_stat["wait_s"] += time.perf_counter() - _t_w
            return v
        rt._pf_pending[k] = v
        # SIZED FOR THE WORKERS, not a flat 8. W workers racing can deliver up to ~W units
        # ahead of the one wanted, and evicting a result that has not been consumed yet
        # turns into a miss and an inline recrop -- the very work the prefetch exists to
        # avoid. Each held slice is ~30 MB, so this is tens of MB per rank.
        if len(rt._pf_pending) > max(16, 8 * max(1, rt.a.prefetch_workers)):
            # RECORD IT. Dropping silently is what turned one eviction into a cascade of
            # full-budget waits; the set is units, not slices, so it costs nothing.
            _dk = next(iter(rt._pf_pending))
            rt._pf_pending.pop(_dk)
            rt._pf_dropped.add(_dk)

def _pf_start(rt, units):
    global _pf_q, _pf_thread
    rt._pf_started[0] = False
    if not rt.a.prefetch_units:
        return
    # NOT ON AN UNCROPPED ARM, where every result is thrown away. _pf_take is reached only
    # inside `if a.crop_margin > 0 or a.crop_tokens`, so with the crop off these workers ran a
    # full np.load plus interface_crop_tokens plus slice_cached_unit per unit and nobody ever
    # consumed one -- then blocked forever on a full put() once --prefetch-units x
    # --prefetch-workers results had piled up, holding those slices in RSS for the whole epoch.
    # The default arm passes --prefetch-units 2 --prefetch-workers 4, so that is 8 wasted
    # crops and 8 retained slices on every uncropped run to date. The uncropped path has its
    # own warmer (_tr_step_start) and does not use this queue.
    if not (rt.a.crop_margin > 0 or rt.a.crop_tokens):
        return
    rt._pf_stop.clear(); rt._pf_pending.clear(); rt._pf_dropped.clear()
    rt._pf_stat.update(hit=0, miss=0, wait_s=0.0, dropped=0, exhausted=0)
    _w = max(1, rt.a.prefetch_workers)
    # DEEP ENOUGH FOR EVERY WORKER TO BE IN FLIGHT. At maxsize 1 with W workers, W-1 of them
    # block on put() and the extra threads buy nothing.
    _pf_q = queue.Queue(maxsize=max(1, rt.a.prefetch_units) * _w)
    _pf_thread = [threading.Thread(target=_pf_worker, args=(units[i::_w],), daemon=True)
                  for i in range(_w)]
    for _t in _pf_thread:
        _t.start()
    rt._pf_started[0] = True

def _pf_end(rt, ):
    global _pf_q, _pf_thread
    if _pf_thread is None:
        return
    rt._pf_stop.set()
    for _t in _pf_thread:
        # DRAIN BETWEEN JOINS, not once before them. A worker blocked on a full put() never
        # sees the stop flag, so draining once and then joining W threads deadlocks on the
        # second one as soon as the first refills the queue.
        while True:
            try:
                _pf_q.get_nowait()
            except queue.Empty:
                break
        _t.join(timeout=30)
    _pf_thread = None; _pf_q = None; rt._pf_pending.clear(); rt._pf_dropped.clear()

def _to_dev(rt, x):
    """Move a slice_cached_unit payload onto the device, structure preserved."""
    if torch.is_tensor(x):
        return x.to(rt.dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_dev(v) for v in x)
    return x

def _prefetch(rt, path):
    """Ask the kernel to read a trunk ahead, so its I/O overlaps the current unit's compute.

    posix_fadvise(WILLNEED) rather than a thread and a queue: it is one syscall, it cannot
    race anything, and it adds no Python-side buffer. The bytes land in PAGE CACHE, which is
    reclaimable -- so this does not raise this process's RSS against --max-rss-gb and does not
    lower MemAvailable in the way that matters to other users of the box. Under memory pressure
    the kernel simply drops the readahead and the subsequent read is a normal cold read.

    BOUNDED AT ONE UNIT AHEAD, deliberately. A deeper queue would have several ~205 MB trunks
    in flight per rank for a loader that consumes them strictly in order, which buys nothing
    and evicts cache that the three ranks are competing for.
    """
    if not hasattr(os, "posix_fadvise") or not path:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
    except OSError:
        pass
    finally:
        os.close(fd)

def _tr_get(rt, trunk_p):
    """The cached host arrays for this trunk, or None. LOOKS UP, NEVER POPS, NEVER BLOCKS."""
    if not rt._tr_on:
        return None
    with rt._tr_lock:
        hit = rt._tr_cache.get(trunk_p)
        rt._tr_stat["hit" if hit is not None else "miss"] += 1
    return hit

def _tr_put(rt, trunk_p, host, gen=None):
    """Offer host arrays to the cache. Refused when full, which costs only the saving.

    `gen` is the warmer's generation; a put from a step that is over is dropped rather than
    charged to the current step's cap.
    """
    if not rt._tr_on or host is None:
        return
    nb = sum(int(v.nbytes) for v in host.values() if hasattr(v, "nbytes"))
    with rt._tr_lock:
        if gen is not None and gen != rt._tr_gen[0]:
            return
        if trunk_p in rt._tr_cache:
            return
        # THIS PATH'S OWN RESERVATION IS DISCOUNTED, because the bytes it reserved before the
        # read are the very bytes it is now offering -- charging both would refuse a put that
        # the reservation had already made room for.
        _mine = rt._tr_inflight.get(trunk_p, 0)
        if rt._tr_bytes[0] + (rt._tr_resv[0] - _mine) + nb > rt._tr_cap:
            rt._tr_stat["full"] += 1
            return
        rt._tr_cache[trunk_p] = host
        rt._tr_bytes[0] += nb
        rt._tr_stat["put"] += 1

def _tr_worker(rt, units, gen):
    """Materialise a step's trunks into the cache. NO TORCH, NO CUDA, NO EXCEPTION ESCAPES.

    RESOLVES THE TRUNK ITSELF, in the thread, exactly as _pf_worker does -- resolve_trunk
    raises SystemExit on an unresolvable fingerprint and that must surface on the main thread
    when the unit is really reached, with its own diagnostics, not here where it would kill a
    warmer and nothing else. A swallowed SystemExit here leaves _trunk_choice unwritten, so the
    main thread re-resolves and refuses properly.

    NEVER memmap. A mapping is lazy, so reading one here would transfer the work it is supposed
    to absorb straight back to the main thread's device copy. The warmer exists to spend host
    time, in a heap buffer, while the GPU is busy.

    `gen` IS CHECKED THREE TIMES: at the top of each unit, under the lock before reserving, and
    again inside _tr_put. A warmer whose step is over does no read and caches nothing.
    """
    for _t in units:
        if gen != rt._tr_gen[0]:
            return
        tp = None
        try:
            tp = resolve_trunk(*_t)
            # THE SIZE IS CHECKED BEFORE THE READ, not after. Reserving _tr_inflight and then
            # refusing in _tr_put means the whole ~243 MB is already on the heap when the
            # refusal happens, so the cap bounded the CACHE and not the PROCESS: with four
            # warmers that is ~1 GB over the cap at any instant. The npz's file size is a
            # faithful proxy for the host dict's bytes, since every member is stored
            # uncompressed at the dtype it is read back as.
            _sz = os.path.getsize(tp)
            with rt._tr_lock:
                if gen != rt._tr_gen[0]:
                    return
                # ADMITTED PLUS ALREADY-RESERVED, which is the whole point of the fix. With
                # only _tr_bytes[0] here, every concurrent warmer saw the same pre-read total.
                if (tp in rt._tr_cache or tp in rt._tr_inflight
                        or rt._tr_bytes[0] + rt._tr_resv[0] + _sz > rt._tr_cap):
                    rt._tr_stat["dup"] += 1
                    continue
                rt._tr_inflight[tp] = _sz
                rt._tr_resv[0] += _sz
            try:
                _tr_put(tp, rt.RC.read_unit_host(tp), gen=gen)
            finally:
                with rt._tr_lock:
                    rt._tr_resv[0] -= rt._tr_inflight.pop(tp, 0)
        except BaseException:          # noqa: BLE001 -- see the docstring
            # INSIDE THE LOCK, because `+=` on a dict entry is not atomic and several warmers
            # reach this. The same reason `dup` is counted under the lock above.
            with rt._tr_lock:
                rt._tr_stat["err"] += 1
                if tp is not None:
                    # BOTH, OR THE CAP LEAKS. An exception between the reservation and the
                    # `finally` -- resolve_trunk raising, a read error -- would otherwise leave
                    # the bytes reserved forever and the cache would refuse everything after a
                    # few failures.
                    rt._tr_resv[0] -= rt._tr_inflight.pop(tp, 0)

def _tr_reset(rt, ):
    """Retire the current generation and drop its arrays. One lock, no joins."""
    with rt._tr_lock:
        rt._tr_gen[0] += 1
        rt._tr_cache.clear()
        # THE RESERVATION IS NOT CLEARED HERE, only the admitted bytes. A warmer from the
        # retired generation may still be inside its read; its `finally` will pop its own entry
        # and decrement, and clearing the dict underneath it would make that pop a no-op and
        # leave _tr_resv permanently above zero. The generation check stops NEW reservations,
        # which is what retiring a step needs to do.
        rt._tr_bytes[0] = 0
        return rt._tr_gen[0]

def _tr_step_start(rt, step_units):
    """Drop the previous step's arrays and start warming this step's. Called per step."""
    if not rt._tr_on:
        return
    gen = _tr_reset()
    _w = max(1, min(rt.a.prefetch_workers, len(step_units)))
    # EVERY GENERATION'S THREADS ARE TRACKED, not just the latest. _tr_threads[0] was
    # OVERWRITTEN here each step, so a warmer from an earlier generation became unreachable and
    # _tr_end could never join it -- it could outlive the epoch boundary and run while
    # _fp_cache_save iterated shared state. Retired threads are reaped first so the list cannot
    # grow across 231 steps.
    rt._tr_live[:] = [t for t in rt._tr_live if t.is_alive()]
    rt._tr_threads[0] = [threading.Thread(target=_tr_worker, args=(step_units[i::_w], gen),
                                       daemon=True) for i in range(_w)]
    for _t in rt._tr_threads[0]:
        _t.start()
    # REGISTERED AFTER START, so _tr_end can join this generation even once _tr_threads[0]
    # has been replaced by the next step's.
    rt._tr_live.extend(rt._tr_threads[0])

def _tr_end(rt, ):
    """At the EPOCH boundary: retire, then join, so nothing races _fp_cache_save.

    THE RESET IS UNCONDITIONAL, NOT GATED ON THERE BEING THREADS. Returning early when
    _tr_threads[0] is falsy left the cache holding up to --trunk-cache-gb across the evaluator
    subprocesses in the case where the main thread had filled it but no warmer ever started.
    """
    _tr_reset()
    # JOIN EVERY GENERATION STILL ALIVE, not only the newest. The generation counter already
    # guarantees a stale warmer cannot reach the cache; this guarantees it is not still RUNNING
    # when _fp_cache_save walks shared state at the boundary.
    # DEDUPED. The current generation is registered in BOTH _tr_live and _tr_threads[0], so
    # joining the concatenation gave every current thread two 30 s windows and doubled the
    # worst-case boundary stall to 30 s x 2 x prefetch_workers. Threads are hashable, and
    # dict.fromkeys keeps the order.
    for _t in dict.fromkeys(list(rt._tr_live) + list(rt._tr_threads[0] or ())):
        _t.join(timeout=30)
    _still = [t for t in rt._tr_live if t.is_alive()]
    if _still:
        print(f"  trunk warmer: {len(_still)} thread(s) still alive after a 30s join; their "
              f"generation is retired so they cannot reach the cache, but they are reading.",
              flush=True)
    # A STRAGGLER STAYS REGISTERED. Clearing the list outright dropped the only reference to
    # a thread that outlived its join, so it was never joined again and the next boundary
    # could not wait for it either. _tr_start already reaps the ones that have since died.
    rt._tr_live[:] = _still
    rt._tr_threads[0] = None

def resolve_trunk(rt, pdb, lab_p, cands, want_fp_cache=None):
    _ck = (pdb, lab_p)
    if _ck in rt._trunk_choice:
        return rt._trunk_choice[_ck]
    want_fp_cache = rt._fp_cache if want_fp_cache is None else want_fp_cache
    """Pick the trunk whose PROVENANCE matches this label, not the first filename that exists.

    THE FIRST MATCHING FILENAME IS NOT THE RIGHT TRUNK. The multi-root lookup took it, so a
    label built from one root's trunk could be paired with another root's trunk of the same name
    -- and since the label is a PER-POSE permutation of its own trunk's poses, that pairs a pose
    with another run's permuted label and trains on it. Review round 8 reproduced exactly that
    with a fold_regen label and a fold trunk.

    The stored fingerprint names its trunk by content hash, so the trunk is chosen BY it. Two
    candidates matching is ambiguous and refused rather than resolved arbitrarily.
    """
    stored = str(np.asarray(np.load(lab_p, allow_pickle=True)["fingerprint"]))
    ok = []
    for tp in cands:
        if _unit_fp(pdb, tp, memo=want_fp_cache) == stored:
            ok.append(tp)
    if len(ok) == 1:
        rt._trunk_choice[_ck] = ok[0]
        return ok[0]
    if not ok:
        # WHICH FIELD DIFFERS IS THE ACTIONABLE PART. "stale" with no detail cannot be acted on,
        # and the fields are not equally serious: cif, bio, idx and trunk say the label was
        # built from DIFFERENT DATA, while code says only that the builder's source has changed.
        def fields(fp):
            out = {}
            for part in fp.split("|"):
                if "=" in part:
                    k, v = part.split("=", 1); out[k] = v
            return out
        want_f = fields(_unit_fp(pdb, cands[0], memo=want_fp_cache))
        got_f = fields(stored)
        diff = sorted(k for k in set(want_f) | set(got_f) if want_f.get(k) != got_f.get(k))
        if diff == ["code"] and rt.a.allow_code_drift:
            if not rt.code_drift["warned"]:
                print(f"  ACCEPTING CODE DRIFT (--allow-code-drift): labels were built by "
                      f"code={got_f.get('code')}, this run is code={want_f.get('code')}. Every "
                      f"data field matches -- cif, bio, idx, trunk, loss, schema -- so the "
                      f"labels are correct but not reproducible from this checkout.")
                rt.code_drift["warned"] = True
            rt.code_drift["n"] += 1
            return cands[0]
        raise SystemExit(
            f"STOP: {lab_p} does not match its inputs. No trunk among {cands} reproduces its "
            f"fingerprint.\n  field(s) differing: {diff}\n"
            + "".join(f"    {k}: label {got_f.get(k)} vs now {want_f.get(k)}\n" for k in diff)
            + (f"  Only `code` differs, so the DATA provenance is intact; pass "
               f"--allow-code-drift to accept that explicitly.\n" if diff == ["code"] else "")
            + f"  Otherwise rebuild with phase_j_build_cache.py --only {pdb}.")
    raise SystemExit(f"STOP: {len(ok)} trunks match {lab_p}'s fingerprint: {ok}. Ambiguous "
                     f"provenance is refused rather than resolved arbitrarily.")

def _xs_prepare(rt, step_units, ep):
    """PASS A for one optimiser step: score every pose, share the scalars, build dL/ds.

    Fills _xs_coef for exactly the poses of `step_units` across ALL ranks, and sets _xs_div to
    the number of interface-groups the step contains. Called once per step, before that step's
    first backward.

    WHY A WHOLE STEP AT A TIME. With --rank-cross-seed a pose's coefficient contains
    sigmoid(-m_ij/tau) against every other pose of its interface, and build_epoch_cross_seed
    puts all of a structure's seeds inside one step. So the unit of scoring is the step, not
    the unit -- which is the whole reason the step boundary had to become explicit.

    THE TRUNK IS READ TWICE PER STEP, once here and once in pass B. That is deliberate: the
    alternative is holding every unit of the step resident, and at ~205 MB of trunk per unit
    and 20 units per step that is ~4 GB of NON-reclaimable RSS per rank against a box that
    other people are using. The second read comes from page cache, which is reclaimable and
    does not count against --max-rss-gb or MemAvailable in the way that matters.

    NO OPTIMISER STEP MAY HAPPEN BETWEEN THIS AND PASS B, or the scores were computed at
    different weights than the gradient is taken at. That is guaranteed by construction: this
    runs at a step boundary, immediately after the previous step was applied.

    THE RNG IS SAVED AND RESTORED because the confidence head has dropout, so pass A would
    otherwise advance the generator and pass B would evaluate a different function than the one
    the coefficients describe.
    """
    rt._xs_coef.clear()
    rt._xs_div[0] = 1
    if not (rt.a.rank_cross_seed and rt.a.rank_loss) or rt.a.loss_only:
        return
    import rank_loss as RKL
    _rng_cpu = torch.get_rng_state()
    _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    _t_a = time.perf_counter()
    local = {}
    # pose -> TRUE sample index, carried separately so _xs_coef can stay keyed by the POSE
    # index pass B looks up while the LABEL lookup uses the sample index.
    local_samp = {}
    try:
        for _pdb, _lab, _cands in step_units:
            _tp = resolve_trunk(_pdb, _lab, _cands)
            _sd = _rank_key(_pdb, os.path.basename(_lab)[:-4], (0, 0))[1]
            for _m in rt.run_unit(_pdb, _lab, _tp, score_only=True):
                local_samp[(_pdb, _sd, int(_m["pose"]))] = int(_m["sample"])
                for _pair, _v in _m["iptm"].items():
                    local[(_pdb, _sd, int(_m["pose"]), _pair)] = float(_v)
    finally:
        torch.set_rng_state(_rng_cpu)
        if _rng_cuda is not None:
            torch.cuda.set_rng_state_all(_rng_cuda)
    _tick("rank_pass_a", _t_a)

    # PASS A MUST SCORE EVERYTHING IT WAS ASKED TO, AND THIS ASSERTS IT RATHER THAN HOPING.
    #
    # RKL.iptm returns None when a requested asym index falls outside the pose's
    # chain_pair_iptm -- which cropping can cause, not only a bad roles join. Without this
    # check that None is silently dropped from `local`, group_coeffs then INTERSECTS the
    # surviving scores with the labels, and a group left with fewer than two poses is skipped
    # entirely. The objective would quietly stop being the full-group ranking loss it is
    # documented as, and in a ranking-only arm (--alpha-pae 0) the affected poses would
    # receive NO gradient at all while the run looked healthy. The label-coverage guard cannot
    # see this: it checks that DockQ exists, not that a score was produced.
    #
    # ASSERTED PER RANK, BEFORE THE GATHER. Every rank verifies its own units, so the merged
    # set is complete by construction -- there is no need to know the other ranks' units here.
    # Found by review of f01dec8a.
    if rt.a.rank_form == "rel":
        # EXPECTED KEYS ARE (pdb, seed, pose, IFACE) for rel, because that is what pass A
        # emits. Built from the pool's interface names, which every pose must score.
        _exp = {(_pdb, _rank_key(_pdb, os.path.basename(_lab)[:-4], (0, 0))[1], _i, _ifc)
                for _pdb, _lab, _cands in step_units
                for _ifc in (rt._rel_ifaces.get(_pdb) or ())
                for _i in range(rt.a.nsample)}
    else:
        _exp = {(_pdb, _rank_key(_pdb, os.path.basename(_lab)[:-4], (0, 0))[1], _i, _pair)
                for _pdb, _lab, _cands in step_units
                for _pair in (rt._scored_asym.get(_pdb) or ())
                for _i in range(rt.a.nsample)}
    _got = set(local)
    if _exp != _got:
        _missing = sorted(_exp - _got)[:6]
        _extra = sorted(_got - _exp)[:6]
        raise SystemExit(
            f"STOP: pass A scored {len(_got)} of {len(_exp)} expected "
            f"(pdb, seed, pose, interface) key(s) for this step.\n"
            f"  missing e.g. {_missing}\n"
            f"  unexpected e.g. {_extra}\n"
            f"A missing score silently removes comparisons from the ranking objective, and "
            f"with --alpha-pae 0 it removes the affected poses' only gradient. The usual "
            f"cause is RKL.iptm returning None because an asym index falls outside the pose's "
            f"chain_pair_iptm -- check --pae-scored-pairs against the fold's chain_roles.csv "
            f"and whether the crop dropped a chain.")
    _nonfinite = sorted(k for k, v in local.items() if not math.isfinite(v))
    if _nonfinite:
        raise SystemExit(
            f"STOP: pass A produced {len(_nonfinite)} non-finite ipTM value(s), e.g. "
            f"{_nonfinite[:6]}. A non-finite score makes every coefficient in that group "
            f"non-finite, which would train on it rather than fail.")

    # EVERY RANK NEEDS THE WHOLE STEP. A few hundred floats, so the object collective's
    # pickling is irrelevant next to a 205 MB trunk read. All ranks then compute the SAME
    # coefficients from the SAME inputs, and each applies only its own poses -- so the
    # cross-rank gradient sum reconstructs one group contribution, not `world` of them.
    if rt.world > 1:
        _t_g = time.perf_counter()
        _bag = [None] * rt.world
        rt.dist.all_gather_object(_bag, local)
        # THE PERMUTATION IS GATHERED TOO. Without it a rank would hold scores for another
        # rank's poses and no way to key their labels, and the group would silently shrink.
        _bag_s = [None] * rt.world
        rt.dist.all_gather_object(_bag_s, local_samp)
        _tick("rank_gather", _t_g)
        allsc, allsamp = {}, {}
        for _d in _bag:
            allsc.update(_d or {})
        for _d in _bag_s:
            allsamp.update(_d or {})
    else:
        allsc = dict(local)
        allsamp = dict(local_samp)

    # GROUPED BY (pdb, interface), which is what an interface IS. Keyed by (seed, sample)
    # inside the group so group_coeffs can tell cross-seed pairs from within-seed ones.
    groups = {}
    for (_pdb, _sd, _i, _pair), _v in allsc.items():
        groups.setdefault((_pdb, _pair), {})[(_sd, _i)] = _v
    n_groups = 0
    for (_pdb, _pair), s_by in groups.items():
        dq_by = {}
        if rt.a.rank_form == "rel":
            # THE LABEL IS THE TRUE iTM, NOT DockQ. Keyed (seed, sample) to match s_by.
            _gl = _rel_labels(_pdb)
            for (_sd, _i) in s_by:
                # SAMPLE INDEX FOR THE LABEL, POSE INDEX FOR THE KEY. The cache is keyed by
                # the sample; s_by and _xs_coef are keyed by the pose. Using _i on both sides
                # trained pose 1 against pose 4's label on 99.3% of units.
                _sx = allsamp.get((_pdb, _sd, int(_i)))
                if _sx is None:
                    raise SystemExit(
                        f"STOP: no sample_index recorded for {_pdb}/{_sd} pose {_i}, so its "
                        f"true-iTM label cannot be keyed. Pass A must emit one per pose.")
                # _pair IS THE IFACE NAME under rel, not an asym tuple -- see pass A.
                _g = _gl.get((int(_sd), int(_sx), str(_pair)))
                if _g is not None:
                    dq_by[(_sd, _i)] = float(_g)
        else:
            for (_sd, _i) in s_by:
                _lab = rt._rank_lab.get(_rank_key(_pdb, _sd, _pair))
                if _lab is not None and _i in _lab:
                    dq_by[(_sd, _i)] = float(_lab[_i])
        if rt.a.rank_form == "rel" and len(dq_by) != len(s_by):
            # EXACT COVERAGE REQUIRED. The completeness guard elsewhere validates DockQ
            # labels, not this cache, so a partially covered group would quietly train on
            # fewer poses than it scored -- and a group centred on a SUBSET has a different
            # mean, so the surviving poses' targets would all be wrong, not merely fewer.
            raise SystemExit(
                f"STOP: {_pdb} interface {_pair} scored {len(s_by)} pose(s) but only "
                f"{len(dq_by)} have a true-iTM label. Centring on a subset changes the mean "
                f"for every surviving pose, so this is not a partial-data case. Rebuild the "
                f"cache with run_logging/build_itm_labels.py and check its manifest covers "
                f"this manifest-selection.")
        if len(dq_by) < 2:
            rt._rel_stat["short_groups"] += 1
            continue
        # THE FORM IS PASSED THROUGH, NOT RE-DERIVED. This read
        #     form=("softplus" if a.rank_form == "softplus" else "linear")
        # which sent any third form to the LINEAR loss while every log line and the saved
        # guard still said otherwise -- a run that looks right and is the wrong experiment.
        # rank_loss.group_coeffs raises on an unknown form, so a typo fails loudly instead.
        if rt.a.rank_form == "rel":
            # NOT group_coeffs. rel is a regression, not a pairwise family, so none of the
            # pairwise knobs apply and passing them would make flags look honoured.
            c, npair, lval = RKL.rel_coeffs(s_by, dq_by,
                                            standardize=rt.a.rel_standardize)
            rt._rel_stat["groups"] += 1
            rt._rel_stat["poses"] += npair
            # DIAGNOSTIC 1: WITHIN-INTERFACE SPEARMAN of the predicted mean-pooled ipTM
            # against DockQ -- the arm's actual prediction. The head measured +0.317 at
            # d32kqp epoch 66 against +0.690 for the TRUE iTM, and this is the quantity that
            # should move between them. It is far more sensitive than selected DockQ (t=7.6
            # against t=2.7) because it uses all 50 poses rather than just the argmax.
            #
            # DIAGNOSTIC 2: SCORE SPREAD, std of the predicted ipTM over the interface's
            # poses, against std of the TRUE iTM on the same poses. Below 1 the model is
            # collapsing its scores, above 1 it is exaggerating pose-to-pose differences, and
            # ~1 means it has the magnitude right. Every ranking arm in this repo lowered its
            # loss by margin collapse (x5frz6: within-seed std 0.00593 -> 0.00147 while the
            # loss fell), so this is the falsifier for that failure mode in one number.
            _ks = sorted(s_by)
            _sv = [float(s_by[_k2]) for _k2 in _ks]
            _gv = [float(dq_by[_k2]) for _k2 in _ks if _k2 in dq_by]
            _dk = rt._rel_dqkey.get(_pair)
            _dv = []
            if _dk is not None:
                for (_sd2, _i2) in _ks:
                    _lab2 = rt._rank_lab.get(_rank_key(_pdb, _sd2, _dk))
                    _sx2 = allsamp.get((_pdb, _sd2, int(_i2)))
                    if _lab2 is not None and _sx2 is not None and _sx2 in _lab2:
                        _dv.append(float(_lab2[_sx2]))
            if len(_dv) == len(_sv) and len(_sv) >= 3:
                _rho = _spearman(_sv, _dv)
                if _rho is not None:
                    rt._rel_stat.setdefault("rho", []).append(_rho)
            if len(_gv) == len(_sv) and len(_sv) >= 3:
                _ms = sum(_sv) / len(_sv)
                _mg = sum(_gv) / len(_gv)
                _ss = (sum((v - _ms) ** 2 for v in _sv) / len(_sv)) ** 0.5
                _sg = (sum((v - _mg) ** 2 for v in _gv) / len(_gv)) ** 0.5
                rt._rel_stat.setdefault("sd_s", []).append(_ss)
                rt._rel_stat.setdefault("sd_g", []).append(_sg)
                if _sg > 1e-9:
                    rt._rel_stat.setdefault("sd_ratio", []).append(_ss / _sg)
            if npair:
                rt._rel_stat["csum"] = max(rt._rel_stat["csum"],
                                        abs(sum(c.values())))
        else:
            c, npair, lval = RKL.group_coeffs(
             s_by, dq_by,
             form=rt.a.rank_form,
            # THE DEFAULT DEPENDS ON THE FORM. DEFAULT_TAU is a margin scale in ipTM units;
            # the expected form's tau is a softmax temperature. Falling back to 0.0305 for
            # the expected form would be a sharper selection than the 0.05 default intends,
            # with nothing in the log to reveal it.
            tau=(rt._rank_tau if rt._rank_tau is not None else
                 (RKL.DEFAULT_SOFTMAX_TAU if rt.a.rank_form == "expected" else RKL.DEFAULT_TAU)),
            weight_by_gap=rt.a.rank_weight_by_gap,
            pairs=rt.a.rank_pairs,
            within_weight=rt.a.rank_within_weight,
            rank_decay_k=rt.a.rank_topk_decay,
            straddle=rt.a.rank_straddle,
            # GROUP-FORM KNOBS. Ignored by every pairwise form, and the pairwise knobs above
            # are ignored by these two -- which is why the startup guard refuses the
            # combinations rather than letting a flag look honoured.
            spearman_tau=(rt.a.rank_spearman_tau if rt.a.rank_spearman_tau > 0
                          else RKL.DEFAULT_SPEARMAN_TAU),
             dist_weight=rt.a.rank_dist_weight,
             rel_scale=rt.a.rank_rel_scale)
        if npair == 0:
            continue
        n_groups += 1
        rt._xs_stat["pairs"] += npair
        rt._xs_stat["loss"] += lval
        for (_sd, _i), _cv in c.items():
            rt._xs_coef[(_pdb, _sd, _i, _pair)] = _cv
    rt._xs_div[0] = max(1, n_groups)
    # PER-STEP, not cumulative. Pass A is REPLICATED -- it fills _xs_coef for the poses of
    # step_units across ALL ranks -- so n_groups is identical on every rank and the skip
    # decision below needs no collective and cannot desynchronise them.
    rt._xs_stat["last_groups"] = n_groups
    rt._xs_stat["groups"] += n_groups
    rt._xs_stat["steps"] += 1

def outside_head_grads(rt, ):
    return sum(1 for n, p in rt.model.named_parameters()
               if n not in set(rt.trainable) and p.grad is not None
               and bool(p.grad.abs().sum() > 0))

def _grad_probe_report(rt, pr, names, a, step_taken, accum_local, world):
    """Report, then RAISE on failure. A safety gate that only prints is not a gate.

    WHAT IT CHECKS, and why each one exists:
      * every trainable tensor receives a FINITE, NON-ZERO ranking gradient. This is the only
        thing that establishes gradient flows through calculate_chain_based_ptm into the two
        matrices; every offline test validates the label side, which cannot. An earlier
        version computed finiteness and then left it out of the verdict, so a tensor holding
        both valid and non-finite values could pass on the non-zero check alone.
      * at least one optimiser step actually completed. The first probe ran 3 units = 15 poses
        against an accumulation window of 30, so it returned before any step was taken and
        still reported success.
      * the ACCUMULATED gradients, because that is what the optimiser sees. ||sum_i g_i|| is
        not sum_i ||g_i||, and the ranking coefficients sum to zero within a group, so per-pose
        contributions cancel. alpha is read off the accumulated pair; the per-pose spread is
        printed beside it for debugging only.

    Both points were raised in review of a3086ef and both were real.
    """
    bar = "=" * 78
    print(f"\n{bar}\n--grad-probe: {pr['poses']} pose(s), {pr['units']} unit(s), "
          f"alpha_rank {a.alpha_rank:g}, accumulation window {accum_local} pose(s)\n{bar}")
    fail = []

    for nm in ("pae", "rank"):
        got = pr["sum"].get(nm)
        if not got:
            fail.append(f"no {nm.upper()} gradient was measured at all")
            continue
        print(f"\n  {nm.upper()}, accumulated over the probe:")
        for n_ in names:
            g = got.get(n_)
            if g is None:
                print(f"    {n_:<60} ABSENT")
                continue
            print(f"    {n_:<60} |sum g| {float(g.norm()):.4e}  "
                  f"max|g| {float(g.abs().max()):.4e}")

    # THE TWO ACCUMULATED VECTORS, SUMMED ACROSS RANKS -- exactly what _apply_step does
    # before clipping and stepping. Each rank only ever saw its own units, so a per-rank
    # measurement is not the optimiser's gradient, and the ranking contributions cancel both
    # within a seed and across ranks. Found in review of 523013a.
    vecs = {}
    for nm in ("pae", "rank"):
        got = pr["sum"].get(nm) or {}
        fl = [got[n_].flatten() for n_ in names if got.get(n_) is not None]
        if not fl:
            continue
        v = torch.cat(fl)
        if world > 1:
            host = v.cpu()
            rt.dist.all_reduce(host, op=rt.dist.ReduceOp.SUM)
            v = host.to(v.device)
        vecs[nm] = v
    if world > 1:
        print(f"\n  (accumulated gradients SUMMED over {world} rank(s), as _apply_step does)")
    if "pae" in vecs and "rank" in vecs:
        vp, vr = vecs["pae"], vecs["rank"]
        np_, nr_ = float(vp.norm()), float(vr.norm())
        cos = float((vp @ vr) / (np_ * nr_)) if np_ > 0 and nr_ > 0 else float("nan")
        comb = float((vp + vr).norm())
        print(f"\n  ACCUMULATED (what the optimiser sees):")
        print(f"    |g_PAE|      {np_:.4e}")
        print(f"    |g_rank|     {nr_:.4e}   ratio {nr_ / max(1e-30, np_):.4f}")
        print(f"    cos(PAE,rank) {cos:+.4f}   -> " + (
            "they OPPOSE: alpha trades one objective for the other" if cos < -0.05 else
            "near-orthogonal: the ranking term is close to free" if abs(cos) <= 0.05 else
            "they AGREE: the ranking term reinforces the PAE direction"))
        print(f"    |g_PAE + g_rank| {comb:.4e}   "
              f"(clip threshold {a.clip_norm:g}"
              f"{', WOULD CLIP' if comb > a.clip_norm else ''})")
        if nr_ > 0:
            print(f"    for parity at the step: alpha_rank ~ "
                  f"{a.alpha_rank * np_ / nr_:.4g}")
        if pr["rows"]:
            pr_ratios = [r[4] / r[3] for r in pr["rows"] if r[3] > 0]
            pr_cos = [r[5] for r in pr["rows"] if r[5] == r[5]]
            if pr_ratios:
                print(f"\n  per-pose, for comparison only: ratio mean "
                      f"{sum(pr_ratios)/len(pr_ratios):.4f} "
                      f"[{min(pr_ratios):.4f}, {max(pr_ratios):.4f}]"
                      + (f", cos mean {sum(pr_cos)/len(pr_cos):+.4f}" if pr_cos else ""))
                print(f"  (a mean of per-pose ratios is NOT the accumulated ratio; the "
                      f"coefficients sum to zero per group so contributions cancel)")

        # ALIGNMENT: ||sum_i g_i|| / sum_i ||g_i||, THE ONE NUMBER THAT SEPARATES
        # CANCELLATION FROM LOST SENSITIVITY.
        #
        # A shrinking parameter gradient has two possible causes and they need different
        # fixes. Either the per-pose gradients still have real magnitude but point in
        # opposing directions and destroy each other when summed (CANCELLATION), or the poses
        # individually stopped producing gradient because the trainable parameters no longer
        # move ipTM much (SENSITIVITY). The accumulated norm alone cannot tell them apart,
        # which is exactly the ambiguity raised in review after the first plateau analysis:
        # 40% of pairs were still inverted and the score-level signal was GROWING while the
        # parameter gradient fell 8x, and I attributed that to cancellation without having
        # measured it.
        #
        # sum_i ||g_i|| is the total magnitude BEFORE any cancellation; ||sum_i g_i|| is what
        # the optimiser receives. Their ratio is 1 when every pose pulls the same way and
        # tends to ~1/sqrt(N) for N independent directions, so READ IT AGAINST THAT
        # BASELINE, not against 1. Tracking it across checkpoints answers the question:
        #   ratio falls while sum_i ||g_i|| holds up  -> CANCELLATION
        #   sum_i ||g_i|| itself collapses            -> SENSITIVITY
        #
        # Summed over ranks like everything else here, because each rank saw only its own
        # poses and a per-rank figure is not the optimiser's.
        # THE DECOMPOSITION, on the same poses and the same window:
        #     g_i = c_i * grad_theta s_i
        # so sum_i ||g_i|| / sum_i |c_i| is the realised mean sensitivity of ipTM to the
        # trainable weights. Tracked across checkpoints it separates the two causes of a
        # shrinking parameter gradient that the alignment ratio alone cannot:
        #     |c_i| falls, sensitivity flat  -> the loss stopped asking
        #     |c_i| flat, sensitivity falls  -> the weights lost leverage over ipTM
        # Previously the only coefficient numbers were val10 sum|c_i| against a training-set
        # gradient probe -- different poses, so the ratio was not formable. Raised in review.
        if pr["rows"] and len(pr["rows"][0]) > 6:
            _cl = sum(r[6] for r in pr["rows"])
            _gl = sum(r[4] for r in pr["rows"])
            _t = torch.tensor([_cl, _gl], dtype=torch.float64)
            if world > 1:
                rt.dist.all_reduce(_t, op=rt.dist.ReduceOp.SUM)
            _cl, _gl = float(_t[0]), float(_t[1])
            if _cl > 0:
                print(f"\n  rank DECOMPOSITION: sum_i |c_i| {_cl:.4e}   "
                      f"sum_i ||g_i|| {_gl:.4e}   "
                      f"mean sensitivity ||grad_theta s_i|| {_gl / _cl:.4e}")
                print(f"  (track across checkpoints: |c_i| falling = the loss stopped "
                      f"asking; sensitivity falling = the weights lost leverage over ipTM)")

        for nm, vec in (("PAE", vecs.get("pae")), ("rank", vecs.get("rank"))):
            col = 3 if nm == "PAE" else 4
            loc = sum(r[col] for r in pr["rows"]) if pr["rows"] else 0.0
            tot = torch.tensor([loc], dtype=torch.float64)
            if world > 1:
                rt.dist.all_reduce(tot, op=rt.dist.ReduceOp.SUM)
            tot = float(tot[0])
            if vec is None or tot <= 0:
                continue
            acc = float(vec.norm())
            # N IS THE ALL-REDUCED POSE COUNT, not this rank's count times the world size.
            # Ranks get different numbers of units when the split is uneven, so the latter
            # printed two different N (105 and 90) for the same accumulation window and so
            # two different 1/sqrt(N) baselines beside an identical ratio.
            _npt = torch.tensor([float(pr["poses"])], dtype=torch.float64)
            if world > 1:
                rt.dist.all_reduce(_npt, op=rt.dist.ReduceOp.SUM)
            n_poses = int(_npt[0])
            print(f"\n  {nm} ALIGNMENT over {n_poses} pose(s): "
                  f"||sum g_i|| {acc:.4e} / sum ||g_i|| {tot:.4e} = {acc / tot:.4f}"
                  f"   (1/sqrt(N) = {1.0 / math.sqrt(max(1, n_poses)):.4f} would be "
                  f"independent directions)")
    else:
        fail.append("could not form both accumulated gradient vectors")

    # THE ACCUMULATED NORM, NOT JUST THE PER-POSE ONES. Non-zero per-pose gradients can
    # cancel to nothing over the window -- the coefficients sum to zero inside each group --
    # and an alpha recommended off a zero accumulated norm is meaningless. Found in review
    # of 523013a.
    if "rank" in vecs:
        _rn = float(vecs["rank"].norm())
        if not math.isfinite(_rn):
            fail.append("the ACCUMULATED ranking gradient is non-finite")
        elif _rn == 0.0:
            fail.append("the ACCUMULATED ranking gradient is exactly zero: per-pose "
                        "contributions cancelled over the window, so no alpha can be read "
                        "from norm parity")
    if "pae" in vecs and "rank" in vecs:
        _vp2, _vr2 = vecs["pae"], vecs["rank"]
        if float(_vp2.norm()) == 0.0:
            fail.append("the ACCUMULATED PAE gradient is exactly zero")
        elif not math.isfinite(float((_vp2 @ _vr2) / max(1e-30, _vp2.norm() * _vr2.norm()))):
            fail.append("the PAE/ranking cosine is not finite")
    if not pr["finite"].get("rank", False):
        fail.append("the RANKING gradient contained non-finite values")
    if not pr["finite"].get("pae", False):
        fail.append("the PAE gradient contained non-finite values")
    # BOTH TERMS' COVERAGE, AND THE FAILURE IS "NEITHER", NOT "NOT THE RANKING TERM".
    #
    # The original check failed whenever a trainable tensor got no RANKING gradient, which
    # assumes the ranking term should reach everything PAE reaches. That is false for any term
    # that is a function of the PAE logits alone: on the first --rank-form rel probe it
    # reported 158 of 258 tensors with no ranking gradient and refused to launch, naming
    # confidence_head.plddt_weight, lower_bins, upper_bins and resolved -- parameters of the
    # pLDDT and resolved heads, which a PAE-derived score cannot touch and which --pae-only
    # switches off anyway.
    #
    # So the question the gate should ask is whether a tensor gets gradient from NEITHER term,
    # which means it is being optimised by nothing while counted as trainable. The per-term
    # counts are printed either way, because "158 of 258 tensors receive no PAE gradient
    # under --pae-only" is a fact about every arm in this project and should not stay hidden
    # behind a check that only looks at the other term.
    nz_r = pr["nonzero"].get("rank") or set()
    nz_p = pr["nonzero"].get("pae") or set()
    absent_r = [n_ for n_ in names if n_ not in nz_r]
    absent_p = [n_ for n_ in names if n_ not in nz_p]
    dead = [n_ for n_ in names if n_ not in nz_r and n_ not in nz_p]
    _ov = rt._pae_keep.get("ov")
    if _ov:
        print(f"\n  PAE MASK OVERRIDE (--rank-form rel): applied on {_ov['poses']} pose(s), "
              f"DIFFERED from the static chain_roles pairs on {_ov['differed']} "
              f"({100 * _ov['differed'] / max(1, _ov['poses']):.1f}%)"
              + (f" across {len(_ov['pdbs'])} structure(s), e.g. "
                 f"{sorted(_ov['pdbs'])[:4]}" if _ov["pdbs"] else
                 " -- NO structure in this window disagrees, so the fix is UNEXERCISED here "
                 "and the numbers are EXPECTED to match the pre-fix run"))
    elif a.rank_form == "rel":
        print("\n  PAE MASK OVERRIDE: NOT APPLIED on any pose, which under --rank-form rel "
              "means the per-pose pairs never reached _set_pae_mask -- the fix is not live.")
    print(f"\n  GRADIENT COVERAGE over {len(names)} trainable tensor(s): "
          f"PAE reaches {len(names) - len(absent_p)}, the {a.rank_form} term reaches "
          f"{len(names) - len(absent_r)}, NEITHER reaches {len(dead)}")
    if absent_r and not dead:
        print(f"    ({len(absent_r)} get no {a.rank_form} gradient but DO get PAE -- expected "
              f"for a term that is a function of the PAE logits alone, e.g. "
              f"{absent_r[:3]})")
    if dead:
        # EXPECTED UNDER --pae-only, AND SAID OUT LOUD RATHER THAN SWALLOWED.
        #
        # Measured on the first rel probe: of 258 tensors marked trainable, PAE reaches 100,
        # the rel term reaches 100, and NEITHER reaches 158. The 158 are pLDDT, PDE and
        # resolved head parameters, whose losses --pae-only switches off, so they sit at their
        # pretrained values for the whole run. That is true of EVERY --pae-only arm in this
        # project -- nskyne, d32kqp and the 45q96x lineage all print "258 trainable tensor(s),
        # 19.8M of 464.5M parameters (4.3%)" while only 100 tensors are optimised, and AdamW
        # carries moment state for the other 158 regardless.
        #
        # So it is not a reason to refuse a --pae-only launch, but it IS a reason the
        # trainable-count line should not be read as the effective capacity. Without
        # --pae-only a dead tensor has no such explanation and still fails.
        msg = (f"{len(dead)} of {len(names)} trainable tensor(s) get NO gradient from either "
               f"term, so they stay at their pretrained values: {dead[:6]}"
               + (f" (+{len(dead)-6} more)" if len(dead) > 6 else ""))
        if a.pae_only:
            print(f"    NOTE, expected under --pae-only: {msg}")
            print(f"    EFFECTIVE trainable set is {len(names) - len(dead)} tensor(s), not "
                  f"{len(names)}. The startup trainable line counts what is MARKED "
                  f"trainable, not what receives gradient.")
        else:
            fail.append(msg)
    if not step_taken:
        fail.append(f"no optimiser step completed: {pr['poses']} pose(s) probed against an "
                    f"accumulation window of {accum_local}. Raise --grad-probe.")

    print("\n  " + "-" * 74)
    if fail:
        print("  FAIL:")
        for f in fail:
            print(f"    - {f}")
        print(f"{bar}\n", flush=True)
        raise SystemExit("STOP: --grad-probe failed its checks (above). Do not launch. "
                         "A gate that only prints is not a gate.")
    # REPORTS WHAT WAS ACTUALLY COVERED, not "all of them". The previous wording claimed
    # every trainable tensor got a ranking gradient, which directly contradicted the coverage
    # line printed two lines above it once that line existed: on a --pae-only rel probe, 100
    # of 258 tensors are reached and the old string still said 258. A success message that
    # restates the check's premise rather than its result is how a known limitation gets read
    # as a guarantee.
    print(f"  PASS: the {len(names) - len(absent_r)} tensor(s) the {a.rank_form} term can "
          f"reach all received a finite, non-zero gradient, the PAE gradient is finite and "
          f"non-zero, the cosine is finite, and an optimiser step completed."
          + (f" {len(dead)} tensor(s) are reached by neither term (expected under "
             f"--pae-only)." if dead else ""))
    print(f"{bar}\n", flush=True)

def _apply_step(rt, step):
    """Clip, assert, step. Returns (grad_norm, n_outside_head, ok).

    A FINITE LOSS DOES NOT MEAN FINITE GRADIENTS -- the loss is checked per pose already, and
    that is not the same assertion. error_if_nonfinite makes clip_grad_norm_ raise rather than
    hand inf or nan to AdamW, which would silently poison every head parameter through the
    exponential averages and never recover. The gradients are dropped and the run continues,
    counted, rather than the step being taken.
    """
    if rt.a.loss_only:
        # THE WINDOW STILL CLOSES, so the trajectory keeps its step_taken==1 rows and the
        # pose-weighted epoch PAE is computed from them exactly as for a training run. What is
        # skipped is the clip, the all-reduce of gradients that do not exist, and AdamW.
        # grad_norm is reported as nan rather than 0.0: zero is a measurement, nan is "not
        # measured", and a 0.0 here would read as a vanished gradient in the usual summaries.
        return float("nan"), 0, True
    # A STEP WITH NO TASK GRADIENT MUST NOT BE TAKEN. --rank-straddle makes this reachable
    # for the first time: a group with no straddling pair is skipped at the `npair == 0`
    # continue above, and 52.3% of the training fold's structures have no straddling pair at
    # all, so with --rank-structs-per-step 2 about 27% of steps accumulate NOTHING from the
    # ranking term. With --alpha-pae 0 there is then no task gradient whatsoever, and the
    # L2-SP block below would be the step's ONLY gradient.
    #
    # THAT IS NOT A SMALL STEP, BECAUSE ADAMW IS SCALE-INVARIANT. Once exp_avg_sq has decayed
    # (beta2 0.95, ~20-step memory) a gradient 100x smaller does not give a step 100x smaller;
    # it gives a near-full lr-sized step straight back toward l2sp_anchor.pt. Measured on the
    # first attempt at this arm: 16 of 69 taken steps had grad_norm < 1e-3 against a typical
    # 0.01-0.20, the late ones clustered at ~2.5e-4 and rising as the weights drifted, i.e.
    # the anchor pulling harder the further the arm got. The lambda calibrated against
    # |g_rank| is simply not in force on those steps.
    #
    # Counted, not silent: the epoch summary prints how many steps were skipped this way.
    if (rt.a.rank_loss and rt.a.rank_cross_seed and not rt.a.alpha_pae
            and rt._xs_stat.get("last_groups", 1) == 0):
        rt._xs_stat["skipped"] += 1
        rt.opt.zero_grad(set_to_none=True)
        return float("nan"), 0, False
    params = [pp for _n, pp in rt.trainable_params]
    if rt.world > 1:
        # BEFORE THE CLIP, NOT AFTER. Clipping each rank's partial gradient to norm 1 and then
        # averaging is not the same function as clipping the summed gradient, so the multi-GPU
        # run would not reproduce the single-GPU one. A rank with no gradient for a parameter
        # still has to contribute a tensor, or the collective's shapes do not line up.
        for pp in params:
            if pp.grad is None:
                pp.grad = torch.zeros_like(pp)
        # ONE COLLECTIVE, NOT 248. The head's gradients are flattened into a single buffer, so
        # the per-call latency is paid once per step rather than once per tensor. gloo is a CPU
        # backend, so that buffer is staged through host memory and copied back.
        # ONE SYNC PER STEP, BEFORE THE TIMER STARTS. Without it the collective's timer
        # absorbs every queued kernel this rank has not finished, and the "collective" number
        # becomes mostly local compute. One sync per 290-step epoch is not the per-operation
        # synchronisation that would distort the measurement.
        _drain_gpu_spans()
        _t0 = time.perf_counter()
        flat = torch.cat([pp.grad.reshape(-1) for pp in params])
        if rt.a.dist_backend == "gloo":
            host = flat.cpu()
            rt.dist.all_reduce(host, op=rt.dist.ReduceOp.SUM)
            flat = host.to(flat.device)
        else:
            rt.dist.all_reduce(flat, op=rt.dist.ReduceOp.SUM)
        off = 0
        for pp in params:
            k = pp.grad.numel()
            pp.grad.copy_(flat[off:off + k].view_as(pp.grad))
            off += k
        # COLLECTIVE PLUS STRAGGLER WAIT, and the two cannot be separated from inside one rank:
        # a rank that arrives early blocks here until the slowest arrives. The per-rank
        # pose_total totals printed below are what separates them -- their spread IS the
        # straggler share.
        _tick("grad_collective_plus_wait", _t0)
    # L2-SP PENALTY, ADDED ONCE PER STEP. After the all-reduce, because that SUMS each rank's
    # gradient and adding before it would multiply the penalty by the world size; before the
    # clip, because the clip is meant to bound the total gradient actually applied.
    # d/dW [ lambda * ||W - W0||^2 ] = 2 * lambda * (W - W0).
    if rt.a.l2_sp and rt._l2sp_anchor is not None:
        with torch.no_grad():
            _sq = 0.0
            for _n_sp, _p_sp in rt._l2sp_params:
                _d_sp = _p_sp.detach() - rt._l2sp_anchor[_n_sp]
                _p_sp.grad.add_(_d_sp, alpha=2.0 * rt.a.l2_sp)
                _sq += float((_d_sp * _d_sp).sum())
            rt._l2sp_drift["sq"] = _sq
            rt._l2sp_drift["n"] += 1
    try:
        gn = float(torch.nn.utils.clip_grad_norm_(params, rt.a.clip_norm,
                                                  error_if_nonfinite=True))
    except RuntimeError as e:
        nonfinite["n"] += 1
        print(f"  NON-FINITE GRADIENT at step {step} ({e}); gradients dropped, no step taken "
              f"({nonfinite['n']} so far)")
        rt.opt.zero_grad(set_to_none=True)
        if nonfinite["n"] > rt.a.max_nonfinite:
            raise SystemExit(f"STOP: {nonfinite['n']} non-finite gradient(s), over "
                             f"--max-nonfinite {rt.a.max_nonfinite}. The learning rate or the data "
                             f"is wrong; continuing would train on noise.")
        return float("nan"), -1, False
    outside = -1
    if step % rt.a.grad_check_every == 0:
        outside = outside_head_grads()
        if outside:
            raise SystemExit(
                f"STOP: {outside} parameter(s) OUTSIDE the confidence head carry a non-zero "
                f"gradient at step {step}. FT3 freezes the trunk; this is not FT3 any more.")
    rt.opt.step()
    rt.opt.zero_grad(set_to_none=True)
    # THE DEAD-OP RE-CHECK, HERE AND NOT AT THE CROSS-SEED STEP BOUNDARY. The obvious home for
    # it was beside PFC.step_reset() in the unit loop -- but that site is inside
    # `if ui in _xs_at:`, and _xs_at is only populated under --rank-cross-seed, so on any other
    # arm the check would have run ZERO times for the whole run while the install line claimed
    # "re-checked at every step boundary". The prefix cache survives that because it has a
    # second, unconditional call at the epoch boundary; this has no equivalent. Immediately
    # after opt.step() is also the strictly better moment: an optimiser step is the only thing
    # that could have written a stubbed tensor, and this is the first instruction after it.
    if rt._dops is not None:
        rt.DOPS.step_check()
    return gn, outside, True

def _seed_lo_hi(rt, spec):
    """--seeds is `lo-hi` and nothing else, so parse it that way and say so.

    The first version accepted comma-separated ranges, which the rest of the file does not:
    the manifest builder does `lo, hi = (int(x) for x in a.seeds.split("-"))` and hard-fails
    on anything else. Worse, the paired --seed-base derivation took the first number of the
    first range, so `500000-500009,500020-500029` would have asked the evaluator for 20 seeds
    starting at 500000 -- half of them never trained on. Dead generality that implied a wrong
    answer is worse than no generality.
    """
    lo, hi = (int(x) for x in str(spec).split("-"))
    return lo, hi - lo + 1

def _record_val_loss(rt, epoch, step, bundle, mean, n_poses):
    """Append one held-out point to val_loss.csv. Shared by the sharded and unsharded paths.

    RECORDING THE MEASUREMENT MUST NOT BE ABLE TO END THE RUN. A NameError on this exact
    bookkeeping killed epoch 1 of x3k2m4 after the pass had already succeeded -- the mean was
    computed and 78 minutes of training died on the append. The subprocess trajectories are the
    durable record either way, so a failure here is reported and dropped.
    """
    try:
        vp = os.path.join(rt.a.out, "val_loss.csv")
        new_file = not os.path.exists(vp)
        with open(vp, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["epoch", "step", "n_poses", "val_pae", "bundle"])
            w.writerow([epoch, step, n_poses, f"{mean:.6f}", os.path.basename(bundle)])
    except Exception as e:
        print(f"  WARNING: could not append to val_loss.csv ({type(e).__name__}: {e}); the "
              f"number is in the line above", flush=True)

def build_epoch(rt, ep):
    """This epoch's unit sequence FOR THIS RANK: K structures per step, one seed of each.

    Every rank walks the SAME structures in the SAME order and differs only in WHICH seed of
    each it takes. Every seed of a structure crops to an identical token count -- the interface
    crop is defined on the ground-truth interface rather than the predicted pose, and 0 of 488
    structures disagree across their ten seeds -- so the ranks do equal work per step and the
    collective has nothing to wait for.

    THIS REPLACES --shuffle structure PLUS shard_by_structure, which handed each rank ten
    CONSECUTIVE seeds of ONE structure. Simulated on this run's own kept_tok values that order
    leaves a 111% median per-step spread (p90 186%) and spends 38% of the critical path in
    straggler wait -- epoch 6 of 2w5to5 burned 22.6 min of 65.4 in loss_stats_collective. This
    order measures 0.0% and 0.0%, and a step spans K structures rather than `world`, so the
    gradient is built from MORE structures than before, not fewer.

    `world` does not divide ten seeds, so floor(10/world) whole groups are used and the
    remainder -- one seed per structure at world 3 -- sits out. The rng is reseeded per epoch,
    so a different seed sits out each time rather than a fixed tenth of the data never being
    trained on.
    """
    by = {}
    for _p, _lp, _c in flat:
        by.setdefault(_p, []).append((_lp, _c))
    rng_e = random.Random(rt.a.seed + ep)
    # ASCENDING BY TOKENS when asked. The pdb tie-break keeps it deterministic across ranks
    # and epochs. A structure the manifest gave no numeric `tokens` for sorts first and is
    # named below rather than silently placed.
    _manifest_order = (rt.a.epoch_order in ("manifest", "tokens"))
    if rt.a.epoch_order == "tokens":
        order = sorted(by, key=lambda _q: (rt._sel_tokens.get(rt._q, 0), rt._q))
    elif _manifest_order:
        order = list(by)
    else:
        order = sorted(by)
        rng_e.shuffle(order)
    # A GROUP IS `world` SEEDS OF ONE STRUCTURE -- the unit of equal work.
    groups = []
    for _p in order:
        u = by[_p][:]
        if not _manifest_order:
            rng_e.shuffle(u)
        for g in range(len(u) // rt.world):
            groups.append([(_p, _lp, _c) for _lp, _c in u[g * rt.world:(g + 1) * rt.world]])
    if not _manifest_order:
        rng_e.shuffle(groups)
    # THE REMAINDER IS KEPT, as a short final step. Truncating to a whole number of steps
    # drops units, and for TRAINING that is invisible -- 1,464 groups over K=6 divides
    # exactly -- but --loss-only runs world=1 over 300 units at K=17 and 300//17*17 is 289,
    # so a held-out pass would have measured 1,445 poses instead of 1,500 and the epoch
    # would have been marked incomplete for missing its expected count. The accumulator
    # already flushes a short step with the gradient scaled to the poses that contributed,
    # and on a measurement pass no step is taken at all.
    mine = []
    for i in range(0, len(groups), rt.structs_per_step):
        for gi, grp in enumerate(groups[i:i + rt.structs_per_step]):
            # ROTATED BY POSITION so a rank is not always handed the same seed slot.
            mine.append(grp[(rt.rank + gi) % rt.world])
    return mine

def _flush(rt, path, cols, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        if new:
            wr.writeheader()
        wr.writerows(rows)
