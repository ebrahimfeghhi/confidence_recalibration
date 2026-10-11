#!/usr/bin/env python
"""On interfaces with top pair ipTM > 0.8 whose free-50 pool has both acceptable (DockQ >= 0.23) and wrong poses:
(1) per-interface AUC of ipTM for acceptable vs wrong over all poses; (2) share acceptable by within-interface ipTM rank
band; (3) pooled calibration: share acceptable per ipTM bin. Written 2026-10-02.
Usage: python analysis/iptm_calibration_high_iptm.py > experimental_results/iptm_calibration_high_iptm.txt"""
import os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import mannwhitneyu
import score_experiment as se, paths

BANDS = [(1, 1), (2, 10), (11, 25), (26, 50), (51, 100), (101, 150), (151, 250)]
BINS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.01]
for name in ("foldbench", "pxm2025"):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    aucs, band = [], {b: [0, 0] for b in BANDS}
    allp = []
    n_hi = 0
    for i in pool:
        keep = [r for r in pool[i] if r[0] in first[i.split("_")[0]]]
        if max(r[3] for r in keep) <= 0.8:
            continue
        n_hi += 1
        ok = [r[3] for r in keep if r[2] >= .23]; bad = [r[3] for r in keep if r[2] < .23]
        if not ok or not bad:
            continue
        aucs.append(mannwhitneyu(ok, bad).statistic / (len(ok) * len(bad)))
        ranked = sorted(keep, key=lambda r: -r[3])
        for k, r in enumerate(ranked, 1):
            for b in BANDS:
                if b[0] <= k <= b[1]:
                    band[b][0] += r[2] >= .23; band[b][1] += 1
        allp += [(r[3], r[2] >= .23) for r in keep]
    a = np.array(aucs)
    print(f"\n== {name}: {n_hi} interfaces with top ipTM > 0.8; mixed pools (both acceptable and wrong poses): {len(a)}")
    print(f"(1) AUC acceptable vs wrong, all poses: median {np.median(a):.2f}  IQR {np.percentile(a,25):.2f}-{np.percentile(a,75):.2f}  "
          f"AUC>=0.9 {sum(a>=.9)}  0.7-0.9 {sum((a>=.7)&(a<.9))}  0.5-0.7 {sum((a>=.5)&(a<.7))}  <0.5 {sum(a<.5)}")
    print("(2) share acceptable by ipTM rank within interface (mixed pools):")
    for b in BANDS:
        g, n = band[b]; print(f"    rank {b[0]:>3}-{b[1]:<3}  {g/n:6.1%}  ({g}/{n})")
    print("(3) pooled calibration (mixed pools): ipTM bin -> share acceptable")
    for lo, hi in zip(BINS, BINS[1:]):
        v = [o for p, o in allp if lo <= p < hi]
        if v: print(f"    {lo:.2f}-{min(hi,1):.2f}  {sum(v)/len(v):6.1%}  (n={len(v)})")
