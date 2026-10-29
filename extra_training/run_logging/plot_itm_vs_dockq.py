#!/usr/bin/env python
"""True mean-row iTM against DockQ: pooled, and within each interface.

WHY TWO PANELS. Selection compares poses WITHIN one interface, so the within-interface
relationship is the only one it uses. Pooled across interfaces the correlation is dominated by
easy-vs-hard targets -- a complex whose poses are all good has both high iTM and high DockQ -- and
that is a different, much stronger relationship that says nothing about ranking. Measured on this
data: pooled Spearman is far above the per-interface median, and conflating them is how a score
gets called a good ranker when it is only a good difficulty detector.

The right panel removes each interface's own mean from both axes, so every point is "how much
better than this interface's average". That is what the argmax sees.

iTM HERE IS THE TRUE ONE, from run_logging/itm_ceiling.py -- the score a perfect PAE would
produce. So this is the CEILING's scatter, not a trained model's: it shows how well the metric
itself can rank, before any question of prediction error.

Usage:  python -I run_logging/plot_itm_vs_dockq.py [out.png]
"""
import csv
import collections
import glob
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import provenance as PROV  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/itm_vs_dockq.png"
GT = os.path.join(ROOT, "training", "runs", "gt_iptm_ceiling", "gt_iptm_s*.csv")
RD = os.path.join(ROOT, "training", "runs", "rowweight_compare", "rowweight_ep55",
                  "rows_b10_*_s*.csv")

itm = {}
for f in sorted(glob.glob(GT)):
    for r in csv.DictReader(open(f)):
        itm[(r["iface"], int(r["seed"]), int(r["sample"]))] = float(r["gt_iptm_mean"])
dq = {}
for f in sorted(glob.glob(RD)):
    for r in csv.DictReader(open(f)):
        try:
            dq[(r["iface"], int(r["seed"]), int(r["sample"]))] = float(r["dockq"])
        except (ValueError, TypeError, KeyError):
            continue
by = collections.defaultdict(list)
for k, v in itm.items():
    if k in dq:
        by[k[0]].append((v, dq[k]))
ifc = sorted(by)
x = np.concatenate([np.array(by[i])[:, 0] for i in ifc])
y = np.concatenate([np.array(by[i])[:, 1] for i in ifc])

# WITHIN-INTERFACE: each interface centred on its own mean, which is what the argmax compares.
xc, yc, per = [], [], []
for i in ifc:
    a = np.array(by[i])
    xc.append(a[:, 0] - a[:, 0].mean())
    yc.append(a[:, 1] - a[:, 1].mean())
    if a[:, 0].std() > 1e-9 and a[:, 1].std() > 1e-9:
        per.append(stats.spearmanr(a[:, 0], a[:, 1]).statistic)
xc, yc, per = np.concatenate(xc), np.concatenate(yc), np.array(per)
sp_pool = stats.spearmanr(x, y).statistic
sp_within = stats.spearmanr(xc, yc).statistic

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#dddddd", "legend.frameon": False,
                     "figure.facecolor": "white"})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.6))
ax.scatter(x, y, s=2, alpha=0.06, color="#2f6fb0", edgecolor="none", rasterized=True)
ax.axhline(0.23, color="#b03a3a", lw=1, ls="--")
ax.annotate("DockQ 0.23 (acceptable)", (0.98, 0.235), fontsize=8.5, color="#b03a3a",
            ha="right", va="bottom")
ax.set_xlabel("true iTM (mean over anchor rows)")
ax.set_ylabel("DockQ")
ax.set_title(f"pooled over all {len(x):,} poses\nSpearman {sp_pool:+.3f}  "
             f"-- dominated by easy vs hard targets")
bx.scatter(xc, yc, s=2, alpha=0.06, color="#3a8a5f", edgecolor="none", rasterized=True)
bx.axhline(0, color="#888888", lw=0.8)
bx.axvline(0, color="#888888", lw=0.8)
bx.set_xlabel("true iTM minus this interface's mean")
bx.set_ylabel("DockQ minus this interface's mean")
bx.set_title(f"WITHIN interface -- what selection actually uses\n"
             f"Spearman {sp_within:+.3f}   per-interface median {np.median(per):+.3f}")
for c in (ax, bx):
    c.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, os.path.dirname(OUT)), exist_ok=True)
fig.savefig(os.path.join(ROOT, OUT), dpi=150, bbox_inches="tight")
print(f"  {len(x):,} poses over {len(ifc)} interface(s) -> {OUT}")
print(f"  pooled Spearman        {sp_pool:+.3f}")
print(f"  within-interface       {sp_within:+.3f}   (per-interface median {np.median(per):+.3f})")
print(f"  per-interface Spearman: {(per < 0).sum()} of {len(per)} are NEGATIVE "
      f"({100*(per<0).mean():.0f}%)")
for lo, hi, nm in ((-1, 0, "negative"), (0, .3, "0-0.3"), (.3, .7, "0.3-0.7"), (.7, 1.01, ">0.7")):
    m = (per >= lo) & (per < hi)
    print(f"    {nm:>9}: {m.sum():>4} ({100*m.mean():.0f}%)")
PROV.write(os.path.join(ROOT, os.path.dirname(OUT)), ROOT,
           inputs=[os.path.relpath(GT, ROOT), os.path.relpath(RD, ROOT)],
           outputs_glob=os.path.basename(OUT), name="itm_vs_dockq.provenance.json")
