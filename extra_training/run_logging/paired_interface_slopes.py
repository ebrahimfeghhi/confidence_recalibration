#!/usr/bin/env python
"""WITHIN-SEED vs ACROSS-SEED, PAIRED INSIDE EACH CHAIN PAIR. Replaces the pooled regression.

WHY THIS REPLACES IT. The pooled fit (run_logging/pair_gap_regression.py) estimates one within
slope and one across slope over ~11M pairs at once, with no chain-pair fixed effect. Pairs are
built inside a single chain pair, so interface DIFFICULTY cannot separate the two populations --
but interface WEIGHT can: inside a chain pair with S seeds there are 10*S within-seed pose pairs
and ~12.5*S^2 across-seed ones, so a heavily sampled chain pair dominates the across estimate far
more than the within estimate. If the ipTM-vs-DockQ slope varies by chain pair -- it does -- the
difference between the two pooled slopes is then part real and part reweighting, and the pooled fit
cannot tell you which part. Measured on the master CSV: pxm is immune (every chain pair has >=100
seeds, so the ratio is identical for all 292), but in sabdab the top decile of chain pairs by seed
count holds 19.2% of within pairs against 24.1% of across pairs, and single-seed chain pairs
contribute within pairs and NO across pairs at all.

THE DESIGN. One number per chain pair per cell, then a PAIRED test across chain pairs. Each chain
pair is its own control, every chain pair counts once regardless of how many seeds it has, and a
chain pair enters a row only if it has BOTH cells there -- so composition and weighting drop out
algebraically rather than being adjusted for.

GAP-MATCHED BY STRATUM, because the two cells have different gap distributions inside a chain pair
too (within-seed gaps are smaller), and the relation is curved -- the pooled fit put a large d^2 on
it. A slope fitted on small gaps therefore differs from one fitted on large gaps with no
within/across effect present at all. So both cells are restricted to the same |DockQ| gap stratum
before anything is compared, and the strata are reported separately rather than pooled.

THE PER-CELL STATISTIC is the through-origin slope b = sum(d*y) / sum(d^2) of the signed ipTM gap y
on the DockQ gap d, pairs oriented so d > 0. Through the origin because a pair with no quality
difference should have no expected ipTM difference; d^2-weighted rather than mean(y/d) because the
latter divides by a gap that can be arbitrarily small at a bin's lower edge. Alongside it,
P(correct) = mean 1[y>0] with an ipTM tie counting 0.5 -- the decision-relevant outcome, which a
slope does not determine on its own.

NO SEED CAP. The pooled fit sampled 20 seeds per chain pair, which was a cost choice and nothing
more. Here each chain pair yields one number per cell, so the cost is in pair construction, not in
rows handed to a regression -- every seed is used.

SABDAB USES THE ORIGINAL 5 POSES PER SEED ONLY, so all three datasets have the same 5 poses per
seed and the within cell means the same thing in each. The regenerated poses (10 per seed) are a
separate replication, selectable with --pose-set all.
"""
import argparse
import collections
import csv
import functools
import os
import sys

print = functools.partial(print, flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]


def cell_stats(d, y):
    """(slope through origin, P(correct)) for one cell, or None if empty."""
    import numpy as np
    if d.size == 0:
        return None
    den = float((d * d).sum())
    b = float((d * y).sum() / den) if den > 0 else float("nan")
    pc = float(np.where(y > 0, 1.0, np.where(y == 0, 0.5, 0.0)).mean())
    return b, pc, int(d.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=f"{ROOT}/training/fold/abag_all_results.csv")
    ap.add_argument("--pose-set", default="orig", choices=["orig", "all"])
    ap.add_argument("--min-pairs", type=int, default=5,
                    help="a chain pair needs this many pairs in BOTH cells of a stratum to count")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import numpy as np
    from scipy import stats as sps

    # chain pair -> list of (dockq, iptm, seed)
    poses = collections.defaultdict(list)
    for r in csv.DictReader(open(a.master)):
        if a.pose_set == "orig" and r["pose_set"] != "orig":
            continue
        try:
            poses[(f'{r["dataset"]} {r["split"]}'.strip(), r["interface"])].append(
                (float(r["dockq"]), float(r["iptm"]), r["seed"]))
        except (ValueError, KeyError, TypeError):
            continue
    print(f"chain pairs read {len(poses)}   pose rows "
          f"{sum(len(v) for v in poses.values())}")

    # per chain pair, per stratum, per cell: the two statistics
    rec = collections.defaultdict(dict)   # (dataset, bin) -> key -> (bw, ba, pw, pa, nw, na)
    n_pairs_built = 0
    skipped_one_seed = 0
    for key, v in poses.items():
        seeds = {s for _, _, s in v}
        if len(seeds) < 2:
            skipped_one_seed += 1
            continue
        dq = np.array([x[0] for x in v])
        ip = np.array([x[1] for x in v])
        sd = np.array([x[2] for x in v])
        ii, jj = np.triu_indices(len(v), k=1)
        gap = dq[ii] - dq[jj]
        live = gap != 0.0
        ii, jj, gap = ii[live], jj[live], gap[live]
        if ii.size == 0:
            continue
        flip = gap < 0
        hi = np.where(flip, jj, ii)
        lo = np.where(flip, ii, jj)
        d = dq[hi] - dq[lo]
        y = ip[hi] - ip[lo]
        same = sd[hi] == sd[lo]
        n_pairs_built += d.size
        for b0, b1 in zip(EDGES[:-1], EDGES[1:]):
            m = (d >= b0) & (d < b1)
            if not m.any():
                continue
            w = cell_stats(d[m & same], y[m & same])
            c = cell_stats(d[m & ~same], y[m & ~same])
            if w is None or c is None:
                continue
            if w[2] < a.min_pairs or c[2] < a.min_pairs:
                continue
            rec[(key[0], (b0, b1))][key] = (w[0], c[0], w[1], c[1], w[2], c[2])
        # the unstratified row, for continuity with the pooled fit
        w = cell_stats(d[same], y[same])
        c = cell_stats(d[~same], y[~same])
        if w is not None and c is not None and w[2] >= a.min_pairs and c[2] >= a.min_pairs:
            rec[(key[0], None)][key] = (w[0], c[0], w[1], c[1], w[2], c[2])

    print(f"chain pairs with >=2 seeds used   "
          f"{len({k for v in rec.values() for k in v})}"
          f"   (single-seed chain pairs dropped {skipped_one_seed})")
    print(f"pose pairs built {n_pairs_built}")
    print()
    print("PAIRED ACROSS CHAIN PAIRS -- each chain pair contributes one within and one across "
          "number per row, both from the SAME |DockQ| gap stratum.")
    print("slope = signed dipTM per unit DockQ gap, through the origin.  "
          "P(correct) = ipTM picked the better pose, tie 0.5.")
    print()
    datasets = sorted({k[0] for k in rec})
    rows_out = []
    for ds in datasets:
        print(f"=== {ds}")
        print(f"{'|dDockQ| stratum':>17} {'chain pairs':>11} "
              f"{'slope within':>12} {'slope across':>12} {'difference':>11} {'t':>7} {'p':>9}   "
              f"{'P(ok) within':>12} {'P(ok) across':>12} {'difference':>11} {'t':>7} {'p':>9}")
        for b in [None] + list(zip(EDGES[:-1], EDGES[1:])):
            v = rec.get((ds, b))
            if not v:
                continue
            bw = np.array([x[0] for x in v.values()])
            ba = np.array([x[1] for x in v.values()])
            pw = np.array([x[2] for x in v.values()])
            pa = np.array([x[3] for x in v.values()])
            ok = np.isfinite(bw) & np.isfinite(ba)
            bw, ba = bw[ok], ba[ok]
            if len(bw) < 3:
                continue
            tb = sps.ttest_rel(bw, ba)
            tp = sps.ttest_rel(pw, pa)
            lab = "ALL" if b is None else f"[{b[0]:.2f},{b[1]:.2f})"
            print(f"{lab:>17} {len(bw):11d} "
                  f"{bw.mean():12.4f} {ba.mean():12.4f} {bw.mean() - ba.mean():+11.4f} "
                  f"{tb.statistic:+7.2f} {tb.pvalue:9.2e}   "
                  f"{pw.mean():12.4f} {pa.mean():12.4f} {pw.mean() - pa.mean():+11.4f} "
                  f"{tp.statistic:+7.2f} {tp.pvalue:9.2e}")
            rows_out.append({"dataset": ds, "stratum": lab, "chain_pairs": len(bw),
                             "slope_within": bw.mean(), "slope_across": ba.mean(),
                             "slope_diff": bw.mean() - ba.mean(),
                             "slope_t": tb.statistic, "slope_p": tb.pvalue,
                             "pok_within": pw.mean(), "pok_across": pa.mean(),
                             "pok_diff": pw.mean() - pa.mean(),
                             "pok_t": tp.statistic, "pok_p": tp.pvalue})
        print()

    if a.out and rows_out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"rows -> {a.out}")


if __name__ == "__main__":
    main()
