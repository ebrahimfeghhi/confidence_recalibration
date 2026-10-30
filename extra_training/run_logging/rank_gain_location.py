#!/usr/bin/env python
"""WHERE in each interface's pose list does a ranking gain sit? The answer selection cares about.

WHY THIS EXISTS. The mean-centred iTM arm raised within-interface Spearman and did not raise
selected DockQ -- in-sample, on the data it trained on, where there is no generalisation gap to
blame. "Better correlation, same argmax" is not a paradox but it is not self-explanatory either,
and the two candidate explanations call for different next experiments:

  (a) the ordering improved everywhere, including the top, and selection is simply noisy
  (b) the ordering improved only among poses no selector would ever pick

Spearman over all 50 poses cannot distinguish them: ~40 of those poses are ones nobody picks, and
they dominate the statistic. So this restricts the same correlation to the TOP of the TRUE
distribution and reads the selector's rank of the actually-best pose, which is the quantity
selection consumes.

MEASURED ON THE EVAL'S OWN ROWS, not on a fresh forward pass: rows_*.csv already carries
(iface, seed, sample, iptm, dockq, stored_iptm, iptm_mean) for every pose, so this is CPU-only and
costs no cards. stored_iptm is the SHIPPED head's recorded ipTM over the identical poses, which is
what makes this a paired comparison rather than two separate measurements.

TIES ARE HANDLED EXACTLY, and they are not a corner case: stored_iptm is recorded at 4 decimal
places. Spearman uses average ranks; the rank of the best pose and the top-1 rate are computed
from counts of strictly-better and tied poses rather than from a sort, because a sort resolves
ties by row order -- reading these same rows from an unsorted glob moved the shipped head's median
rank from 21 to 22. A collapsed prediction would make this the dominant effect.

Usage:
  python -I run_logging/rank_gain_location.py <rows_dir> [--top 10]
  python -I run_logging/rank_gain_location.py training/runs/rowweight_compare/rel27_ep27
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("rows_dir", help="a directory of rows_*_s*.csv from phase_k_eval_sharded")
ap.add_argument("--top", type=int, default=10,
                help="how many poses, ranked by TRUE DockQ, count as 'the top' (default 10)")
a = ap.parse_args()


def spearman(xs, ys):
    """Spearman with AVERAGE ranks. Hand-written: scipy is not in this venv."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


by = defaultdict(list)
files = sorted(glob.glob(os.path.join(a.rows_dir, "rows_*_s*.csv")))
if not files:
    sys.exit(f"  no rows_*_s*.csv under {a.rows_dir}")
for f in files:
    for r in csv.DictReader(open(f)):
        try:
            by[r["iface"]].append((float(r["iptm_mean"]), float(r["stored_iptm"]),
                                   float(r["iptm"]), float(r["dockq"])))
        except (ValueError, TypeError, KeyError):
            continue
# COMPLETE CANDIDATE SETS ONLY. An interface missing poses would be compared against a different
# denominator for "rank of the best pose", which is the one number here that depends on the count.
full = max((len(v) for v in by.values()), default=0)
keep = sorted(k for k, v in by.items() if len(v) == full)
if not keep:
    sys.exit("  no interface has a complete candidate set")
print(f"  {len(keep)} interface(s) at the full {full} candidates, from {len(files)} shard file(s)")
print(f"  'the top' = the {a.top} poses with the highest TRUE DockQ\n")

SELECTORS = [("ours, MEAN-pooled", 0), ("ours, MAX-pooled", 2), ("shipped ipTM (MAX)", 1)]
print(f"  {'selector':>20}  {'rho all':>8} {'rho top':>8} {'rank of best':>13} "
      f"{'picks best':>11} {'best in top5':>13}")
for name, j in SELECTORS:
    allr, topr, br, hit, in5 = [], [], [], 0, 0
    for k in keep:
        v = by[k]
        s = [p[j] for p in v]
        d = [p[3] for p in v]
        allr.append(spearman(s, d))
        idx = sorted(range(len(v)), key=lambda i: -d[i])[:a.top]
        topr.append(spearman([s[i] for i in idx], [d[i] for i in idx]))
        # TIES ARE COUNTED, NOT SORTED THROUGH. stored_iptm is recorded at 4 decimal places, so
        # several poses routinely share a score, and `sorted(...).index(best)` then returns
        # whichever one the file iteration order happened to put first -- reading the same rows
        # from an unsorted glob moved the shipped head's median rank from 21 to 22 and its top-1
        # rate with it. These are computed from counts instead, which no row order can change:
        #   n_gt  poses strictly better than the best pose    -> its best possible rank
        #   n_eq  poses tied with it                          -> spread over the next n_eq places
        # The reported rank is the expectation under random tie-breaking, and the top-1 rate is
        # the exact probability 1/(1+n_eq) of a uniform tie-break landing on it.
        best = max(range(len(v)), key=lambda i: d[i])
        n_gt = sum(1 for i in range(len(v)) if s[i] > s[best])
        n_eq = sum(1 for i in range(len(v)) if i != best and s[i] == s[best])
        br.append(n_gt + 1 + n_eq / 2)
        hit += (1.0 / (1 + n_eq)) if n_gt == 0 else 0.0
        in5 += min(1.0, max(0.0, (5 - n_gt) / (1 + n_eq))) if n_gt < 5 else 0.0
    print(f"  {name:>20}  {st.median(allr):>+8.4f} {st.median(topr):>+8.4f} "
          f"{st.median(br):>9.1f}/{full:<3} {100*hit/len(keep):>10.2f}% "
          f"{100*in5/len(keep):>12.1f}%")
print(f"\n  A random selector picks the best pose {100/full:.1f}% of the time and puts it inside "
      f"its top 5 {100*5/full:.1f}%.")
print("  rho all   -- median within-interface Spearman against DockQ over every candidate")
print(f"  rho top   -- the same, restricted to the {a.top} poses worth picking")
print("  THE TEST: a gain in 'rho all' that does NOT appear in 'rho top' is a gain among poses")
print("  no selector reads, and cannot move selected DockQ.")
