#!/usr/bin/env python
"""iTM: the interface TM-score from GROUND-TRUTH PAE -- the ceiling any PAE-only method can reach.

NAMED iTM, NOT "ground-truth ipTM", because the p in ipTM is PREDICTED. Computed from the true
errors there is nothing predicted left, so the quantity is the interface TM-score itself. The
CSV columns still read gt_iptm_* for the run already on disk and are accepted under either
spelling.

THE QUESTION THIS ANSWERS. Zero PAE loss would give a perfect predicted PAE, hence a perfect
predicted ipTM. But ipTM is itself a lossy summary of a pose -- a pooling over anchor rows of a
saturating transform of predicted error -- so a PERFECT ipTM is not a perfect pose ranker. This
scores every candidate pose by the ipTM it WOULD receive if the head emitted the true error bin
exactly, selects on that, and reports how much of the oracle headroom it recovers.

That number bounds every PAE-only approach, the row weighting included:
  - if it recovers most of the headroom, PAE is the right target and the problem is optimisation;
  - if it recovers little, no amount of PAE perfection gets there, and a loss that targets the
    ordering directly is not a shortcut around imperfect optimisation but the only route.

BOTH POOLINGS ARE REPORTED. Shipped ipTM is the MAX over anchor rows, which rests a pose's whole
score on one row; the MEAN cannot be decided by a single row. Comparing them AT THE CEILING
separates two different losses of information: what the pooling throws away, and what imperfect
prediction throws away. If the mean beats the max here, the max is discarding reachable headroom
by itself, independently of how well PAE is learned.

WHY ONE-HOT ON THE TRUE BIN rather than the raw continuous distance: the head can only ever emit
a distribution over 64 bins, so a point mass on the correct bin is the best output available to
it. Scoring the continuous distance would give a slightly HIGHER ceiling that no binned head can
reach, which would overstate what is attainable.

TWO FACTS ABOUT PAELoss.calculate_label THAT COST TIME TO LEARN, recorded so they are not
rediscovered:
  1. It ALREADY returns one-hot, shape [1, N_tok, N_tok, 64]. It is the label, not a bin index.
     Encoding it again with F.one_hot raises a shape error rather than giving a wrong number.
  2. Feeding log(one-hot) to pae_row_weight.row_t works but is ~64x more work than needed AND
     repeats it per chain pair: row_t softmaxes a [N, N, 64] tensor for every pair. At N=1500 that
     is ~600 MB per pose per pair. The fast path here argmaxes the one-hot ONCE per pose and
     gathers per-bin weights, which is exact because softmax(log(one-hot)) IS the one-hot.
     --verify asserts the fast path equals row_t to 1e-6 rather than assuming it.

d0 IS PER CHAIN PAIR, so the bin weights cannot be gathered once per pose: d0 = 1.24 *
(max(N,19)-15)^(1/3) - 1.8 uses the token count of the TWO CHAINS, which differs between an
interface and the whole crop. The weights are cached by that token count, not computed once.

CPU ONLY, NO GPU, NO MODEL. It reads the trunk's predicted coordinates (ci_coordinate) and the
label cache's aligned true coordinates (label_coordinate) straight off disk, so it is safe to run
beside a training job. Each worker is pinned to ONE torch thread -- without that, nine workers on
a 288-core box each spawn 288 threads and thrash.

THE CHAIN JOIN IS THE STATIC ONE, AND THAT IS A REAL LIMITATION. phase_k_eval resolves each
interface's model chains PER POSE through OST, because chain_roles.csv is known to be swapped
against OST for some structures; this uses chain_roles directly via phase_k_eval's own
interface_chain_pairs. 298 of these 368 structures have more than two pooled chains, so a swap
there scores the wrong pair for that interface and corrupts its ranking. Acceptable for a ceiling
ESTIMATE; flagged rather than hidden, and the count of unresolved pairs is printed.

REPRODUCIBILITY. No randomness: the structure list comes from --pool, seeds are the fixed training
fold 500000-500009, iteration is sorted, and sharding is a deterministic stride (units[k::workers]).
Re-running with the same arguments rewrites the same per-shard CSVs. The candidate set and DockQ
come from the EVAL's own per-pose rows (--rows-dir), so the ceiling is measured over exactly the
poses the arms were compared on rather than a set re-derived here.

Usage:
  python -I run_logging/gt_iptm_ceiling.py                      # 9 workers, then report
  python -I run_logging/gt_iptm_ceiling.py --verify             # exactness check only, ~20 s
  python -I run_logging/gt_iptm_ceiling.py --report-only        # re-report from existing CSVs
  python -I run_logging/gt_iptm_ceiling.py --limit-units 40     # quick timing probe
"""
import argparse
import collections
import csv
import glob
import math
import os
import subprocess
import sys
import time

# PINNED BEFORE torch IS IMPORTED -- set afterwards it has no effect on the thread pools.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import pae_row_weight as rw  # noqa: E402
from phase_k_eval import interface_chain_pairs  # noqa: E402
from protenix.model.loss import PAELoss  # noqa: E402

SEEDS = tuple(range(500000, 500010))     # the training fold's own seeds, fixed
N_BINS = 64
MIN_BIN, MAX_BIN = 0.0, 32.0

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=9)
ap.add_argument("--shard", type=int, default=-1, help="internal: set by the parent per worker")
ap.add_argument("--rows-dir", default="training/runs/rowweight_compare/rowweight_ep55",
                help="the eval's per-pose rows: the authority for candidates and DockQ")
ap.add_argument("--pool", default="training/full_train/pool_winnable.csv")
ap.add_argument("--roles", default="training/fold/chain_roles.csv")
ap.add_argument("--fold", default="training/fold")
ap.add_argument("--out", default="training/runs/gt_iptm_ceiling")
ap.add_argument("--limit-units", type=int, default=0, help="stop after N units; timing probe only")
ap.add_argument("--verify", action="store_true", help="assert the fast path equals row_t, then exit")
ap.add_argument("--report-only", action="store_true")
a = ap.parse_args()
OUT = os.path.join(ROOT, a.out)


def unit_tensors(pdb, seed):
    """(asym, has_frame, frame_atom_index, rep_atom_mask, trunk, label) or None if absent."""
    tp = os.path.join(ROOT, a.fold, "trunks", pdb, f"free_{seed}.npz")
    lp = os.path.join(ROOT, a.fold, "labels", pdb, f"free_{seed}.npz")
    if not (os.path.exists(tp) and os.path.exists(lp)):
        return None
    tr = np.load(tp, allow_pickle=True)
    lb = np.load(lp, allow_pickle=True)
    return (torch.as_tensor(tr["ci_feat_asym_id"]).long(),
            torch.as_tensor(lb["lf_has_frame"]).bool(),
            torch.as_tensor(lb["lf_frame_atom_index"]).long(),
            torch.as_tensor(lb["lf_pae_rep_atom_mask"]).bool(),
            tr, lb)


def pose_label(pl, tr, lb, p, hf, fai, rep):
    """(bin_index [N_tok, N_tok] int64, pair_mask [N_frame, N_tok] bool) for one pose."""
    _, bins, pm = pl.calculate_label(
        pred_coordinate=torch.as_tensor(tr["ci_coordinate"][p]).float().unsqueeze(0),
        true_coordinate=torch.as_tensor(lb["label_coordinate"][p]).float(),
        coordinate_mask=torch.as_tensor(lb["label_coordinate_mask"][p]).float(),
        rep_atom_mask=rep, frame_atom_index=fai, has_frame=hf)
    # ARGMAX ONCE, then the [N, N, 64] one-hot is freed. Everything downstream is [N, N].
    return bins[0].argmax(-1).contiguous(), pm


def t_from_label(idx, hf, tm, n_pair_tokens, wcache):
    """Every row's t_i, from the TRUE bin index. Exact equivalent of row_t(log(one-hot), ...).

    softmax(log(one-hot)) is the one-hot, so the per-bin weighted sum collapses to a lookup of the
    true bin's weight -- no 64-wide tensor and no softmax. d0, hence the weight vector, depends on
    the PAIR's token count, so weights are cached by that count rather than computed once.
    """
    nd = int(n_pair_tokens)
    wb = wcache.get(nd)
    if wb is None:
        wb = rw._tm_weights(N_BINS, rw.calculate_normalization(nd),
                            idx.device, torch.float32, MIN_BIN, MAX_BIN)
        wcache[nd] = wb
    w = wb[idx[hf]]                               # [N_frame, N_token]
    tmf = tm.to(torch.float32)
    cnt = tmf.sum(-1)
    return (w * tmf).sum(-1) / cnt.clamp_min(1.0), cnt


def pair_mask_for(pm, asym, hf, ax, ay):
    """The cross-chain targets of ONE interface: both chains, excluding same-chain pairs."""
    tok = (asym == ax) | (asym == ay)
    return pm & (tok[hf].unsqueeze(1) & tok.unsqueeze(0)) \
              & (asym[hf].unsqueeze(1) != asym.unsqueeze(0)), int(tok.sum())


def gt_for_unit(pdb, seed, pairs, pl):
    """{(iface, sample): (gt_iptm_max, gt_iptm_mean)} for one (structure, seed)."""
    got = unit_tensors(pdb, seed)
    if got is None:
        return {}
    asym, hf, fai, rep, tr, lb = got
    out, wcache = {}, {}
    for p in range(int(lb["n_pose"])):
        idx, pm = pose_label(pl, tr, lb, p, hf, fai, rep)
        samp = int(lb["sample_index"][p])
        for iface, ax, ay in pairs:
            tm, ntok = pair_mask_for(pm, asym, hf, ax, ay)
            if int(tm.sum()) == 0:
                continue
            t, cnt = t_from_label(idx, hf, tm, ntok, wcache)
            # ROWS WITH NO TARGETS IN THIS PAIR ARE EXCLUDED FROM BOTH POOLINGS. Their t_i is 0 by
            # construction, which does not disturb a max but would drag a mean toward zero by an
            # amount that depends on crop size rather than on pose quality.
            live = cnt > 0
            if not bool(live.any()):
                continue
            tv = t[live]
            out[(iface, samp)] = (float(tv.max()), float(tv.mean()))
        del idx, pm
    return out


def verify():
    """Assert the fast path equals pae_row_weight.row_t on real data, not synthetic."""
    pairs_by_pdb, _ = interface_chain_pairs(os.path.join(ROOT, a.pool),
                                            os.path.join(ROOT, a.roles))
    pl = PAELoss()
    worst, checked = 0.0, 0
    for pdb in sorted(pairs_by_pdb)[:3]:
        got = unit_tensors(pdb, SEEDS[0])
        if got is None:
            continue
        asym, hf, fai, rep, tr, lb = got
        idx, pm = pose_label(pl, tr, lb, 0, hf, fai, rep)
        logits = torch.log(
            torch.nn.functional.one_hot(idx, num_classes=N_BINS).float() + 1e-9).unsqueeze(0)
        wcache = {}
        for iface, ax, ay in pairs_by_pdb[pdb]:
            tm, ntok = pair_mask_for(pm, asym, hf, ax, ay)
            if int(tm.sum()) == 0:
                continue
            fast, _ = t_from_label(idx, hf, tm, ntok, wcache)
            slow, _ = rw.row_t(logits, hf, tm, n_pair_tokens=ntok,
                               min_bin=MIN_BIN, max_bin=MAX_BIN)
            d = float((fast - slow).abs().max())
            worst = max(worst, d)
            checked += 1
            print(f"    {iface}: max|fast-row_t| = {d:.3e}   "
                  f"(max t {float(fast.max()):.4f}, {fast.numel()} rows)")
    ok = checked > 0 and worst < 1e-6
    print(f"\n  {'PASS' if ok else 'FAIL'}  {checked} pair(s) checked, "
          f"worst deviation {worst:.3e} (tolerance 1e-6)")
    return 0 if ok else 1


def run_shard(k):
    pairs_by_pdb, unres = interface_chain_pairs(os.path.join(ROOT, a.pool),
                                                os.path.join(ROOT, a.roles))
    units = [(p, s) for p in sorted(pairs_by_pdb) for s in SEEDS]
    mine = units[k::a.workers]
    if a.limit_units:
        mine = mine[:a.limit_units]
    pl = PAELoss()
    rows, t0 = [], time.time()
    for i, (pdb, seed) in enumerate(mine):
        for (iface, samp), (vmax, vmean) in gt_for_unit(
                pdb, seed, pairs_by_pdb[pdb], pl).items():
            rows.append((iface, seed, samp, f"{vmax:.6f}", f"{vmean:.6f}"))
        if i and i % 25 == 0:
            el = time.time() - t0
            print(f"  shard {k}: {i}/{len(mine)} units, {len(rows)} rows, {el:.0f}s "
                  f"(eta {el/i*(len(mine)-i)/60:.1f} min)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"gt_iptm_s{k}.csv")
    tmp = f + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iface", "seed", "sample", "gt_iptm_max", "gt_iptm_mean"])
        w.writerows(rows)
    os.replace(tmp, f)
    print(f"  shard {k}: {len(rows)} row(s) -> {os.path.relpath(f, ROOT)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if k == 0 and unres:
        print(f"  NOTE: {len(unres)} pool pair(s) unresolved in chain_roles, e.g. {unres[:5]}")
    return 0


def dockq_table():
    """{(iface, seed, sample): (dockq, shipped_iptm)} from the eval's own per-pose rows."""
    d = {}
    for f in sorted(glob.glob(os.path.join(ROOT, a.rows_dir, "rows_b*_s*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                d[(r["iface"], int(r["seed"]), int(r["sample"]))] = (
                    float(r["dockq"]), float(r["stored_iptm"]))
            except (ValueError, TypeError, KeyError):
                continue
    return d


def report():
    gt = {}
    for f in sorted(glob.glob(os.path.join(OUT, "gt_iptm_s*.csv"))):
        for r in csv.DictReader(open(f)):
            kmax = "itm_max" if "itm_max" in r else "gt_iptm_max"
            kmean = "itm_mean" if "itm_mean" in r else "gt_iptm_mean"
            gt[(r["iface"], int(r["seed"]), int(r["sample"]))] = (
                float(r[kmax]), float(r[kmean]))
    if not gt:
        sys.exit(f"  no gt_iptm_s*.csv under {a.out}")
    dq = dockq_table()
    by = collections.defaultdict(list)
    for k, (vmax, vmean) in gt.items():
        if k in dq:
            d, ship = dq[k]
            by[k[0]].append((vmax, vmean, ship, d))
    ifc = sorted(by)
    if not ifc:
        sys.exit("  nothing joined -- check --rows-dir matches --pool")
    # ONLY INTERFACES WITH THE FULL CANDIDATE SET, so a selector is never credited for choosing
    # from fewer poses than its rivals saw.
    full = max(collections.Counter(len(by[i]) for i in ifc).items(),
               key=lambda kv: kv[1])[0]
    keep = [i for i in ifc if len(by[i]) == full]
    n = len(keep)
    pick = lambda j: np.array([max(by[i], key=lambda t: t[j])[3] for i in keep])  # noqa: E731
    s_max, s_mean, base = pick(0), pick(1), pick(2)
    orac = np.array([max(t[3] for t in by[i]) for i in keep])
    hr = orac.mean() - base.mean()
    print(f"\n  iTM CEILING (ipTM from true PAE) -- {n} interface(s) at {full} candidates each"
          + (f"  ({len(ifc)-n} dropped for a short candidate set)" if len(ifc) != n else ""))
    print(f"\n    {'selector':>27} {'mean DockQ':>11} {'frac':>8} {'vs base':>10} "
          f"{'t':>7} {'% headroom':>11}")
    print(f"    {'shipped ipTM (baseline)':>27} {base.mean():>11.4f} "
          f"{(base >= 0.23).mean():>8.4f} {'--':>10} {'--':>7} {'--':>11}")
    for nm, sel in (("iTM, MAX over rows", s_max), ("iTM, MEAN over rows", s_mean)):
        d = sel - base
        se = d.std(ddof=1) / math.sqrt(n)
        print(f"    {nm:>27} {sel.mean():>11.4f} {(sel >= 0.23).mean():>8.4f} "
              f"{d.mean():>+10.4f} {d.mean()/se:>+7.2f} {100*d.mean()/hr:>10.1f}%")
    print(f"    {'oracle (best DockQ)':>27} {orac.mean():>11.4f} "
          f"{(orac >= 0.23).mean():>8.4f} {hr:>+10.4f} {'--':>7} {'100.0%':>11}")
    d = s_mean - s_max
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"\n    MEAN minus MAX, paired on the same poses: {d.mean():+.4f} +- {se:.4f} "
          f"(t={d.mean()/se:+.2f})")
    print(f"    Headroom {hr:.4f}. For reference, on this population at epoch 55 the "
          f"row-weighted head\n    recovered 8.5% of it and the unweighted control 4.4%.")


if __name__ == "__main__":
    if a.verify:
        sys.exit(verify())
    if a.report_only:
        report()
    elif a.shard >= 0:
        sys.exit(run_shard(a.shard))
    else:
        t0 = time.time()
        ps = [subprocess.Popen(
            [sys.executable, "-I", os.path.abspath(__file__), "--shard", str(k),
             "--workers", str(a.workers), "--rows-dir", a.rows_dir, "--pool", a.pool,
             "--roles", a.roles, "--fold", a.fold, "--out", a.out]
            + (["--limit-units", str(a.limit_units)] if a.limit_units else []))
            for k in range(a.workers)]
        rc = [p.wait() for p in ps]
        print(f"\n  {a.workers} worker(s) finished in {(time.time()-t0)/60:.1f} min, rc={rc}")
        if any(rc):
            sys.exit(f"STOP: shard(s) failed with {rc}")
        report()
