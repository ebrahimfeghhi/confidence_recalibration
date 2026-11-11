#!/usr/bin/env python
"""Sample a pool of pair-anchored crops per structure, score every one, and SAVE BOTH.

WHY SAMPLING AND SELECTION ARE SEPARATE. The tail rule is still being argued about -- drop the
worst 5%, drop anything >0.05 worse than uncropped, floor the smaller surviving side. Re-sampling
to try another one would be minutes of CPU and, worse, would change the crops themselves so two
rules could not be compared on the same pool. So this writes every sampled mask AND a per-crop
score table; `select_crop_pool.py` applies a rule over that table and writes the final pool. A new
rule is then a re-selection, not a re-sample.

WHAT IS WRITTEN, per structure, to <out>/<pdb>.npz:
    crops      [n_crops, crop_size] uint16   token indices, sorted
    n_token    scalar                        the uncropped assembly size
    valid      [n_crops] bool                every scored interface keeps >=1 cross-chain pair
    ifaces     [n_iface] str                 interface ids, in the order the per-crop arrays use
    regret     [n_iface, n_crops] float32    oracle_dockq - DockQ(this crop's label's pick)
    rho        [n_iface, n_crops] float32    Spearman(cropped true iTM, DockQ) over the poses
    side_min   [n_iface, n_crops] int16      smaller surviving side of the scored pair
    ret15      [n_iface, n_crops] float32    fraction of 15 A interface tokens kept
    regret_full / rho_full / side_min_full  [n_iface]   the uncropped references
    __meta     json                          anchor rule, crop size, seed, digests, commit

STRUCTURES WITHOUT ENOUGH POSES STILL GET MASKS. 416 of 846 have a single cached trunk seed, so
regret and rho over 5 poses are degenerate -- they are written as NaN rather than omitted, and a
selection rule must decide what to do with them rather than silently dropping half the manifest.

    python -I run_logging/build_crop_pool.py --crops 1000 --workers 16 --shard 0 \
        --out training/crop_pool_256
"""
import argparse
import binascii
import collections
import csv
import json
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = os.environ.get("CROP_THREADS", "4")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
sys.path.insert(0, os.path.join(ROOT, "phases"))

import crop_itm_rank as C                                          # noqa: E402
from phase_k_eval import interface_chain_pairs                     # noqa: E402
from phase_k_eval import ost_chain_pairs                           # noqa: E402
from phase_k_eval import pool_ref_pairs as _PKE_pool_ref_pairs     # noqa: E402
import phase_j_build_cache as BC                                   # noqa: E402
from protenix.model.loss import PAELoss                            # noqa: E402

SEEDS = list(range(500000, 500015))


def one_structure(pdb, pairs, dq, pl, a):
    got_any = None
    _uniq_ifaces = sorted({_i for _i, _x, _y in pairs})
    poses, keys = [], []
    dockq_by_iface = collections.defaultdict(list)
    for seed in SEEDS:
        got = C.unit_tensors(a.fold, pdb, seed)
        if got is None:
            continue
        asym, hf, fai, rep, tr, lb = got
        if got_any is None:
            xyz, resolved = C.token_geometry(lb, rep)
            got_any = (asym, hf, xyz, resolved)
        for p in range(int(lb["n_pose"])):
            samp = int(lb["sample_index"][p])
            poses.append(C.pose_bins(pl, tr, lb, p, hf, fai, rep))
            keys.append((int(seed), samp))
            # ONE DockQ PER POSE PER *INTERFACE NAME*, not per (interface, asym pair) row.
            # `pairs` now carries the same interface once per distinct asym pair OST resolves it
            # to across poses, so iterating it here appended the same pose's DockQ two or three
            # times and dockq_by_iface grew longer than `poses` -- the scoring loop then indexed
            # poses[j] past its end. DockQ is a property of the INTERFACE; which asym pair
            # represents it is exactly the detail these duplicate rows exist to vary.
            for iface in _uniq_ifaces:
                dockq_by_iface[iface].append(dq.get(iface, {}).get((seed, samp), float("nan")))
    if got_any is None or not poses:
        return None, "no tensors"
    asym, hf, xyz, resolved = got_any
    N = int(asym.numel())

    crops, n_elig = C.sample_crops_pair_anchor(
        xyz, resolved, asym, pairs, a.crop_size, a.crops, C.crop_seed(a.seed, pdb))
    if not crops:
        return None, "no eligible anchor"
    keep = np.zeros((N, len(crops)), dtype=np.float32)
    for c, idx in enumerate(crops):
        keep[idx, c] = 1.0

    row_tokens = np.asarray(torch.nonzero(hf, as_tuple=True)[0])
    out = {"crops": np.stack([np.pad(c, (0, a.crop_size - len(c)), constant_values=c[-1])
                              if len(c) < a.crop_size else c for c in crops]).astype(np.uint16),
           "n_token": np.int32(N), "ifaces": np.array([i for i, _x, _y in pairs])}
    n_if, n_cr = len(pairs), len(crops)
    reg = np.full((n_if, n_cr), np.nan, np.float32)
    rho = np.full((n_if, n_cr), np.nan, np.float32)
    smin = np.zeros((n_if, n_cr), np.int16)
    ret = np.full((n_if, n_cr), np.nan, np.float32)
    usable = np.zeros((n_if, n_cr), bool)
    reg_full = np.full(n_if, np.nan, np.float32)
    rho_full = np.full(n_if, np.nan, np.float32)
    smin_full = np.zeros(n_if, np.int16)

    for ii, (iface, ax, ay) in enumerate(pairs):
        d = np.asarray(dockq_by_iface[iface], dtype=np.float64)
        ok_pose = ~np.isnan(d)
        tok_in = ((asym == ax) | (asym == ay))
        npt = int(tok_in.sum())
        if npt < 4 or ok_pose.sum() < 2:
            continue
        sel = [poses[j] for j in np.nonzero(ok_pose)[0]]
        kk = [keys[j] for j in np.nonzero(ok_pose)[0]]
        dq_i = d[ok_pose]
        pm0 = sel[0][1]
        M = (pm0 & (tok_in[hf].unsqueeze(1) & tok_in.unsqueeze(0))
             & (asym[hf].unsqueeze(1) != asym.unsqueeze(0))).float().numpy()
        if M.sum() == 0:
            continue
        wb = C.weight_vector(npt, {}).numpy()
        W = np.stack([wb[p[0].numpy()] for p in sel])

        in_x, in_y = (asym == ax).numpy(), (asym == ay).numpy()
        fx, fy = C.side_counts(M, np.ones((N, 1), np.float32), row_tokens, in_x, in_y)
        smin_full[ii] = int(min(fx[0], fy[0]))
        itm_f, _ = C.itm_over_subsets(W, M, np.ones((N, 1), np.float32), row_tokens)
        oracle = float(dq_i.max())
        kf, _ = C.pick_pose(itm_f[:, 0], kk)
        reg_full[ii] = oracle - float(dq_i[kf])
        rho_full[ii] = C.spearman(itm_f[:, 0], dq_i)

        invol = np.zeros(N, bool)
        invol[row_tokens[M.any(1)]] = True
        invol |= M.any(0) > 0
        xa, xb = (asym == ax) & resolved, (asym == ay) & resolved
        if xa.any() and xb.any():
            dxy = torch.cdist(xyz[xa], xyz[xb]).numpy()
            cont = np.concatenate([np.nonzero(xa.numpy())[0][(dxy < 15.0).any(1)],
                                   np.nonzero(xb.numpy())[0][(dxy < 15.0).any(0)]])
        else:
            cont = np.zeros(0, int)

        for lo in range(0, n_cr, 250):
            kch = keep[:, lo:lo + 250]
            blk, nlive = C.itm_over_subsets(W, M, kch, row_tokens)
            nx, ny = C.side_counts(M, kch, row_tokens, in_x, in_y)
            smin[ii, lo:lo + kch.shape[1]] = np.minimum(nx, ny)
            usable[ii, lo:lo + kch.shape[1]] = nlive > 0
            if len(cont):
                ret[ii, lo:lo + kch.shape[1]] = kch[cont].mean(0)
            for j in range(blk.shape[1]):
                if nlive[j] <= 0:
                    continue
                k, _t = C.pick_pose(blk[:, j], kk)
                reg[ii, lo + j] = oracle - dq_i[k]
                rho[ii, lo + j] = C.spearman(blk[:, j], dq_i)
    out.update(valid=usable.all(0), regret=reg, rho=rho, side_min=smin, ret15=ret,
               usable=usable, regret_full=reg_full, rho_full=rho_full,
               side_min_full=smin_full, n_poses=np.int32(len(poses)),
               n_eligible_anchors=np.int32(n_elig))
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--pool", default="training/combined/pool.csv")
    ap.add_argument("--roles", default="training/fold/chain_roles_all.csv")
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--manifest", default="train_val_splits/cache_train_structures_extended.csv")
    ap.add_argument("--skip", default="4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m")
    ap.add_argument("--crops", type=int, default=1000)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="training/crop_pool_256")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a structure whose npz is already written, so a build can be "
                         "re-sharded without redoing finished work")
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(int(os.environ.get("CROP_THREADS", "4")))
    os.makedirs(a.out, exist_ok=True)

    skip = set(x for x in a.skip.split(",") if x)
    want = [r["pdb"] for r in csv.DictReader(open(a.manifest)) if r["pdb"] not in skip]
    # THE PAIRS MUST BE THE ONES TRAINING SCORES, WHICH ARE OST'S, NOT chain_roles'.
    #
    # THIS IS THE BUG THAT KILLED 3pdv7i 21 MINUTES INTO EPOCH 1. interface_chain_pairs reads
    # chain_roles, which assigns each model chain one reference chain for the whole structure.
    # phase_k_train scores the PER-POSE OST join instead (_pose_pairs -> ost_chain_pairs), and on
    # 90 of 846 structures the two name DIFFERENT CHAINS. 9o4o is the one that fired: static says
    # its interfaces are asym (0,3) and (1,3), OST says (0,2) and (1,2). The pool therefore
    # certified a 256-token crop that preserves chains {0,1,3} and DELETES chain 2 -- the chain
    # training actually needs -- so iptm_pooled returned None for both interfaces on every pose
    # and the pass-A completeness guard stopped the run.
    #
    # A CROP IS ONE PER STRUCTURE, SHARED BY EVERY POSE, so validating the pair of a single pose
    # is not enough: OST resolves the SAME interface to different asym pairs across poses of the
    # same structure (10op_BM is (0,2) in some poses and (0,3) in others). The crop has to keep
    # every pair ANY pose might ask for, so the union is what gets validated -- each distinct
    # (iface, asym pair) becomes its own row, and `valid = usable.all(0)` then requires all of
    # them to survive.
    _refs = _PKE_pool_ref_pairs(a.pool)
    pairs_by_pdb = {}
    for _pdb in sorted(set(r["pdb"] for r in csv.DictReader(open(a.manifest)))):
        _seen = set()
        for _seed in SEEDS:
            try:
                _ud = BC.seed_dir(a.fold, _pdb, int(_seed))
            except Exception:                                    # noqa: BLE001
                continue
            for _samp in range(5):
                _got, _ = ost_chain_pairs(_ud, _pdb, _samp, _refs.get(_pdb) or [])
                for _i, _x, _y in (_got or []):
                    _seen.add((_i, min(_x, _y), max(_x, _y)))
        if _seen:
            pairs_by_pdb[_pdb] = sorted(_seen)
    _nst, _ = interface_chain_pairs(a.pool, a.roles)
    _diff = sum(1 for _p, _v in pairs_by_pdb.items()
                if {(i, min(x, y), max(x, y)) for i, x, y in (_nst.get(_p) or [])} != set(_v))
    print(f"  chain pairs from the PER-POSE OST join: {len(pairs_by_pdb)} structure(s); "
          f"{_diff} differ from the static chain_roles join", flush=True)
    dq = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.results)):
        if r.get("arm") != "free":
            continue
        try:
            dq[r["iface"]][(int(r["seed"]), int(r["sample"]))] = float(r["dockq"])
        except (TypeError, ValueError):
            continue

    todo = [p for p in want if p in pairs_by_pdb]
    todo = todo[a.shard::a.workers] if a.workers > 1 else todo
    if a.limit:
        todo = todo[:a.limit]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:                                               # noqa: BLE001
        commit = "unknown"
    meta = dict(anchor="pair_restricted_15A", crop_size=a.crop_size, n_crops=a.crops,
                seed=a.seed, d0="uncropped scored chain-pair", commit=commit,
                builder="build_crop_pool.py", fold=a.fold, pool=a.pool, roles=a.roles)

    pl = PAELoss()
    t0, done, skipped = time.time(), 0, collections.Counter()
    for i, pdb in enumerate(todo):
        if a.skip_existing and os.path.exists(os.path.join(a.out, f"{pdb}.npz")):
            skipped["already built"] += 1
            continue
        out, why = one_structure(pdb, pairs_by_pdb[pdb], dq, pl, a)
        if out is None:
            skipped[why] += 1
            continue
        out["__meta"] = json.dumps(meta)
        np.savez_compressed(os.path.join(a.out, f"{pdb}.npz"), **out)
        done += 1
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [shard {a.shard}] {i+1}/{len(todo)} structures, {done} written, "
                  f"{el/60:.1f} min, eta {el/(i+1)*(len(todo)-i-1)/60:.1f} min", flush=True)
    print(f"[shard {a.shard}] wrote {done} structure(s) to {a.out}   skipped {dict(skipped)}",
          flush=True)


if __name__ == "__main__":
    main()
