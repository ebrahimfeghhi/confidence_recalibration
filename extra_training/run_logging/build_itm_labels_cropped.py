#!/usr/bin/env python
"""Rebuild the true-iTM labels under every crop in a structure's selected pool.

THIS STEP IS THE ONE THAT IS SILENTLY WRONG IF SKIPPED. `--rank-form rel` trains the head to
reproduce each pose's TRUE iTM, and that iTM is an average over the interface's cross-chain pairs.
Crop the input and the head's prediction is an average over a SUBSET, while the cached
itm_labels_extended target is still the full-structure value. The residual absorbs the difference,
nothing refuses, and the arm trains against a target it cannot reach.

OUTPUT, per structure, to <out>/<pdb>.npz -- the same schema itm_labels_extended uses, with
crop_index added to the key:

    crop_index [M] int16     which crop in the selected pool this row belongs to
    iface      [M] str
    seed       [M] int64
    sample     [M] int64
    g_mean     [M] float32   mean-pooled true iTM under that crop
    g_max      [M] float32   max-pooled, for the same reason the uncropped builder stores both

d0 IS PINNED AT THE UNCROPPED SCORED CHAIN-PAIR TOKEN COUNT, not recomputed from the crop. d0 is
TM-score's length normalisation; shrinking it with the crop steepens the kernel -- scoring the
same error worse -- for a task cropping does not make easier, and it would make cropped and
uncropped iTM differ by two things rather than one. The uncropped reference is written as
crop_index -1 so a consumer can check the pinning held.

WHY IT IS MINUTES, NOT HOURS. PAELoss.calculate_label runs ONCE per pose regardless of crop count
-- it is the expensive step. Scoring a crop is then re-averaging those cached bin indices over a
different pair subset, which is one sgemm for all crops at once.

    python -I run_logging/build_itm_labels_cropped.py --workers 20 --shard 0 \
        --pool training/crop_final --out training/fold/itm_labels_crop
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = os.environ.get("CROP_THREADS", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
sys.path.insert(0, os.path.join(ROOT, "phases"))

import crop_itm_rank as C                                          # noqa: E402
from phase_k_eval import (interface_chain_pairs, ost_chain_pairs,   # noqa: E402
                          pool_ref_pairs)
import phase_j_build_cache as BC                                    # noqa: E402
from protenix.model.loss import PAELoss                            # noqa: E402

SEEDS = list(range(500000, 500015))


def load_masks(pool_dir, pdb):
    """[n_crop] list of token-index arrays, plus the per-crop crop size."""
    p = os.path.join(pool_dir, f"{pdb}.npz")
    if not os.path.exists(p):
        return None, None
    d = np.load(p, allow_pickle=True)
    arrs = [d[k] for k in sorted((k for k in d.files if k.startswith("c")
                                  and k[1:].isdigit()), key=lambda k: int(k[1:]))]
    if not arrs:
        return None, None
    # Each rung stores its own [n, size] block; a crop's mask is one row, and rows from different
    # rungs have different lengths, which is why they cannot be a single rectangular array.
    masks = [row for a in arrs for row in a]
    return masks, d["sizes"]


def itm_both(W, M, keep, row_tokens):
    """(mean-pooled, max-pooled) iTM per pose per crop. Max is the shipped readout."""
    P, F, N = W.shape
    WM = (W * M[None]).reshape(P * F, N)
    num = (WM @ keep).reshape(P, F, -1)
    den = M @ keep
    t = num / np.clip(den, 1.0, None)[None]
    live = ((den > 0) & (keep[row_tokens] > 0))
    n_live = live.sum(0)
    tm = np.where(live[None], t, 0.0)
    mean = tm.sum(1) / np.clip(n_live, 1.0, None)[None]
    mx = np.where(live[None], t, -np.inf).max(1)
    return mean, np.where(n_live[None] > 0, mx, np.nan)


def one_structure(pdb, pairs, pool_dir, pl, a, refs):
    """PAIRS ARE RESOLVED PER POSE THROUGH OST, never from the static chain_roles join.

    chain_roles assigns each model chain one reference chain for the whole structure, which is
    unanswerable when two reference chains are chemically identical -- a structure with two
    antigen copies of the same sequence has no sequence-determined "first antigen". OST decides it
    per pose. The uncropped builder records that the static join was wrong on 3,154 of 36,900 rows
    (8.55%) across 52 structures, and REFUSES a ceiling not built with --ost-mapping. This builder
    reintroduced the same defect by taking (ax, ay) from interface_chain_pairs and reusing
    poses[0]'s pair mask for every pose; caught in external review before any cropped run.
    """
    masks, sizes = load_masks(pool_dir, pdb)
    if masks is None:
        return None, "not in pool"
    N = None
    out = collections.defaultdict(list)
    keep = None
    for seed in SEEDS:
        got = C.unit_tensors(a.fold, pdb, seed)
        if got is None:
            continue
        asym, hf, fai, rep, tr, lb = got
        try:
            ud = BC.seed_dir(a.fold, pdb, seed)
        except Exception:                                          # noqa: BLE001
            return None, "no unit dir for OST mapping"
        if N is None:
            N = int(asym.numel())
            row_tokens = np.asarray(torch.nonzero(hf, as_tuple=True)[0])
            keep = np.zeros((N, len(masks) + 1), dtype=np.float32)
            keep[:, 0] = 1.0                           # column 0 is the UNCROPPED reference
            for c, idx in enumerate(masks):
                keep[np.asarray(idx, dtype=np.int64), c + 1] = 1.0
        for p in range(int(lb["n_pose"])):
            samp = int(lb["sample_index"][p])
            use, why = ost_chain_pairs(ud, pdb, samp, refs or [])
            if use is None:
                return None, f"no OST mapping ({why})"
            idx_p, pm_p = C.pose_bins(pl, tr, lb, p, hf, fai, rep)
            for iface, ax, ay in use:
                tok_in = (asym == ax) | (asym == ay)
                npt = int(tok_in.sum())
                if npt < 4:
                    continue
                M = (pm_p & (tok_in[hf].unsqueeze(1) & tok_in.unsqueeze(0))
                     & (asym[hf].unsqueeze(1) != asym.unsqueeze(0))).float().numpy()
                if M.sum() == 0:
                    continue
                # d0 FROM THE UNCROPPED PAIR, for every crop. This is the pinning.
                wb = C.weight_vector(npt, {}).numpy()
                W = wb[idx_p.numpy()][None]
                for lo in range(0, keep.shape[1], 400):
                    mean, mx = itm_both(W, M, keep[:, lo:lo + 400], row_tokens)
                    for j in range(mean.shape[1]):
                        out["crop_index"].append(lo + j - 1)
                        out["iface"].append(iface)
                        out["seed"].append(int(seed))
                        out["sample"].append(samp)
                        out["g_mean"].append(mean[0, j])
                        out["g_max"].append(mx[0, j])
    if N is None:
        return None, "no tensors"
    if not out:
        return None, "no scorable interface"
    return {"crop_index": np.array(out["crop_index"], np.int16),
            "iface": np.array(out["iface"]),
            "seed": np.array(out["seed"], np.int64),
            "sample": np.array(out["sample"], np.int64),
            "g_mean": np.array(out["g_mean"], np.float32),
            "g_max": np.array(out["g_max"], np.float32),
            "n_crop": np.int32(len(masks)),
            "crop_sizes": np.asarray(sizes, np.int16)}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--pool-csv", default="training/combined/pool.csv")
    ap.add_argument("--roles", default="training/fold/chain_roles_all.csv")
    ap.add_argument("--pool", default="training/crop_final")
    ap.add_argument("--out", default="training/fold/itm_labels_crop")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--manifest-only", action="store_true",
                    help="write manifest.json from whatever is already in --out and exit. Run "
                         "this AFTER every shard has finished; a shard cannot know that it did.")
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(int(os.environ.get("CROP_THREADS", "1")))
    os.makedirs(a.out, exist_ok=True)

    if a.manifest_only:
        write_manifest(a)
        return
    pairs_by_pdb, _ = interface_chain_pairs(a.pool_csv, a.roles)
    # THE REFERENCE-LETTER POOL, which ost_chain_pairs joins against per pose.
    refs = pool_ref_pairs(a.pool_csv)
    todo = sorted(f[:-4] for f in os.listdir(a.pool) if f.endswith(".npz"))
    todo = [p for p in todo if p in pairs_by_pdb]
    todo = todo[a.shard::a.workers] if a.workers > 1 else todo
    if a.limit:
        todo = todo[:a.limit]

    pl = PAELoss()
    t0, done, skipped = time.time(), 0, collections.Counter()
    for i, pdb in enumerate(todo):
        if a.skip_existing and os.path.exists(os.path.join(a.out, f"{pdb}.npz")):
            skipped["already built"] += 1
            continue
        out, why = one_structure(pdb, pairs_by_pdb[pdb], a.pool, pl, a, refs.get(pdb))
        if out is None:
            skipped[why] += 1
            continue
        np.savez_compressed(os.path.join(a.out, f"{pdb}.npz"), **out)
        done += 1
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [shard {a.shard}] {i+1}/{len(todo)}, {done} written, {el/60:.1f} min, "
                  f"eta {el/(i+1)*(len(todo)-i-1)/60:.1f} min", flush=True)
    print(f"[shard {a.shard}] wrote {done} to {a.out}   skipped {dict(skipped)}", flush=True)

    # THE MANIFEST IS A SEPARATE STEP, not a shard's parting act. The first version let shard
    # workers-1 write it on the theory that it finishes last. It does not: shard 19 finished while
    # 38 structures were still being written by other shards, and the manifest went out claiming
    # 791 structures with 829 on disk -- a partial build asserting completeness, which is the one
    # thing the manifest exists to rule out. Run with --manifest-only after every shard exits.
    if a.workers > 1:
        print(f"  (no manifest from a shard -- run --manifest-only once every shard has exited)",
              flush=True)
        return
    write_manifest(a)


def write_manifest(a):
    _files = sorted(glob.glob(os.path.join(a.out, "*.npz")))
    _rows = _crops = 0
    _h = hashlib.sha256()
    for _f in _files:
        _z = np.load(_f, allow_pickle=True)
        _rows += int(_z["g_mean"].shape[0])
        _crops += int(_z["n_crop"])
        _h.update(os.path.basename(_f).encode())
        _h.update(np.asarray(_z["g_mean"], np.float32).tobytes())
    try:
        _commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
    except Exception:                                               # noqa: BLE001
        _commit = "unknown"
    json.dump({
        "structures": len(_files),
        "rows": _rows,
        "total_crops": _crops,
        "digest": _h.hexdigest(),
        "pool": a.pool,
        "pool_csv": a.pool_csv,
        "roles": a.roles,
        "keys": "(crop_index, seed, sample, iface) -> g_mean, g_max",
        "key_note": "crop_index -1 is the UNCROPPED reference, kept so a consumer can verify that "
                    "d0 was pinned at the uncropped scored chain-pair count rather than "
                    "recomputed from the crop",
        "chain_mapping": "OST per pose (ost_chain_pairs), NOT the static chain_roles join -- the "
                         "static join was wrong on 3,154 of 36,900 rows across 52 structures in "
                         "the uncropped builder, and 7stf's pair swaps between samples of the "
                         "same structure",
        "d0": "pinned at the uncropped scored chain-pair token count",
        "builder": "build_itm_labels_cropped.py",
        "commit": _commit,
    }, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"  manifest: {len(_files)} structure(s), {_rows} row(s), {_crops} crop(s) total",
          flush=True)


if __name__ == "__main__":
    main()
