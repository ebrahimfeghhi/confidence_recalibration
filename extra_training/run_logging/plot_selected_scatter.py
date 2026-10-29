#!/usr/bin/env python
"""Selected DockQ per interface: baseline head on x, trained head on y.

WHAT A POINT IS. One interface. Both coordinates are the DockQ of a POSE ACTUALLY PICKED out of
the same candidate set -- x is the pose the shipped head's ipTM ranked first, y is the pose the
trained head's ipTM ranked first. So the comparison is selection and nothing else: identical
candidates, identical oracle, one scoring function swapped.

WHY THE DIAGONAL CARRIES THE WHOLE STORY. A point ON the line is an interface where the two heads
picked poses of equal DockQ -- usually the same pose. Above the line the trained head picked
better, below it picked worse. The mean improvement is the mean vertical distance from the line,
so a cloud that sits tight to the diagonal with a few scattered points means a metric moved by a
handful of interfaces rather than a broad shift, which is exactly what a mean alone hides.

POINTS ON THE DIAGONAL ARE DRAWN FAINTLY AND COUNTED SEPARATELY. On the in-sample pass 92 of 738
interfaces had identical selections; plotted at full strength they are 12% of the ink arguing for
"no change" at the one place where no change is invisible anyway.

Usage:  python -I run_logging/plot_selected_scatter.py <rows_dir> [out.png] [--label NAME]
"""
import csv
import glob
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [x for x in sys.argv[1:] if not x.startswith("--")]
if not args:
    sys.exit("need a rows directory, e.g. training/runs/rowweight_compare/rowweight_ep55")
D = args[0]
OUT = args[1] if len(args) > 1 else os.path.join(
    "figures", "selected_scatter_" + os.path.basename(D.rstrip("/")) + ".png")
LABEL = "trained head"
for i, x in enumerate(sys.argv):
    if x == "--label" and i + 1 < len(sys.argv):
        LABEL = sys.argv[i + 1]

files = sorted(glob.glob(os.path.join(ROOT, D, "rows_b*_s*.csv")))
if not files:
    sys.exit(f"  no rows_b*_s*.csv under {D}")
cand = {}
for f in files:
    for r in csv.DictReader(open(f)):
        try:
            cand.setdefault(r["iface"], []).append(
                (float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"])))
        except (ValueError, TypeError, KeyError):
            continue
if not cand:
    sys.exit(f"  parsed no usable rows under {D}")

ifaces = sorted(cand)
x = np.array([max(cand[i], key=lambda t: t[1])[2] for i in ifaces])   # shipped ipTM picks
y = np.array([max(cand[i], key=lambda t: t[0])[2] for i in ifaces])   # trained ipTM picks
d = y - x
same = np.abs(d) < 1e-9
n = len(ifaces)
se = d.std(ddof=1) / math.sqrt(n)

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#dddddd", "legend.frameon": False,
                     "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(6.8, 6.6))
ax.plot([0, 1], [0, 1], color="#444444", lw=1.2, ls="--", zorder=1,
        label="equal selection")
# ACCEPTABILITY AT 0.23 IS DRAWN because frac acceptable is the other reported metric: the two
# lines cut the square into the quadrants that metric actually counts.
for v in (0.23,):
    ax.axhline(v, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(v, color="#bbbbbb", lw=0.8, zorder=0)
ax.scatter(x[same], y[same], s=14, color="#bbbbbb", alpha=0.55, zorder=2,
           label=f"same pose picked ({same.sum()})")
up, dn = (~same) & (d > 0), (~same) & (d < 0)
ax.scatter(x[up], y[up], s=26, color="#2f6fb0", alpha=0.75, edgecolor="none", zorder=3,
           label=f"{LABEL} better ({up.sum()})")
ax.scatter(x[dn], y[dn], s=26, color="#b03a3a", alpha=0.75, edgecolor="none", zorder=3,
           label=f"{LABEL} worse ({dn.sum()})")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.set_xlabel("selected DockQ -- baseline (shipped ipTM)")
ax.set_ylabel(f"selected DockQ -- {LABEL}")
ax.set_title(f"{n} interfaces, paired on the same candidates\n"
             f"mean {y.mean():.4f} vs {x.mean():.4f}   "
             f"delta {d.mean():+.4f} +- {se:.4f}")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, os.path.dirname(OUT)) or ".", exist_ok=True)
fig.savefig(os.path.join(ROOT, OUT), dpi=150, bbox_inches="tight")
print(f"  {n} interface(s) -> {OUT}")
print(f"  baseline mean {x.mean():.4f}   {LABEL} mean {y.mean():.4f}   "
      f"delta {d.mean():+.4f} +- {se:.4f} (t={d.mean()/se:.2f})")
print(f"  frac acceptable: baseline {(x >= 0.23).mean():.4f}   {LABEL} {(y >= 0.23).mean():.4f}")
print(f"  changed: {up.sum()} better, {dn.sum()} worse, {same.sum()} identical")
print(f"  biggest gains:  {sorted(zip(d, ifaces), reverse=True)[:3]}")
print(f"  biggest losses: {sorted(zip(d, ifaces))[:3]}")
