#!/usr/bin/env python
"""Selected DockQ per interface: shipped on x, the cropped arm on y, matched GT crop.

Both passes are cropped by the SAME ground-truth rule, so a point's offset from y=x is the HEAD
and nothing else. MAX pooling, because that is where section 9b's +0.0376 lives, and the eval's
own tie-break -- max by (score, -seed, -sample) -- so the points reproduce the headline number.

A point is ONE SCORED CHAIN PAIR, not one complex: valfull lists H-ag and L-ag separately, so
7qu2_AC and 7qu2_BC are the same structure twice and the cloud has fewer independent points than
it has markers. Same-complex pairs are joined by a faint line to make that visible.
"""
import argparse, csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ours", default="training/runs/cropped_eval/valfull_ours_ep30")
ap.add_argument("--shipped", default="training/runs/cropped_eval/valfull_shipped")
ap.add_argument("--col", default="iptm", choices=("iptm", "iptm_mean"),
                help="iptm = MAX pooled (9b's headline), iptm_mean = MEAN pooled")
ap.add_argument("--out", default="figures/scatter_ours_vs_shipped.png")
ap.add_argument("--label-thresh", type=float, default=0.10,
                help="annotate interfaces whose |delta| exceeds this")
a = ap.parse_args()

ACC = 0.23


def load(d, col):
    by = defaultdict(list)
    fs = sorted(glob.glob(os.path.join(ROOT, d, "rows_b30_*.csv")))
    if not fs:
        raise SystemExit(f"STOP: no rows_b30_*.csv under {d}")
    for f in fs:
        for r in csv.DictReader(open(f)):
            by[r["iface"]].append((float(r[col]), -int(r["seed"]), -int(r["sample"]),
                                   float(r["dockq"])))
    return by


O, S = load(a.ours, a.col), load(a.shipped, a.col)
if set(O) != set(S):
    raise SystemExit(f"STOP: interface sets differ -- ours {len(O)}, shipped {len(S)}, "
                     f"symmetric difference {sorted(set(O) ^ set(S))[:5]}")

ifs = sorted(O)
rec = []
for i in ifs:
    if len(O[i]) != len(S[i]):
        raise SystemExit(f"STOP: {i} has {len(O[i])} poses in ours, {len(S[i])} in shipped")
    best = max(p[3] for p in O[i])
    rec.append((i, max(S[i])[3], max(O[i])[3], best, float(np.mean([p[3] >= ACC for p in O[i]]))))

x = np.array([r[1] for r in rec])
y = np.array([r[2] for r in rec])
best = np.array([r[3] for r in rec])
d = y - x

fig, ax = plt.subplots(figsize=(7.6, 7.2))
ax.plot([0, 1], [0, 1], color="0.55", lw=1.0, zorder=1)
for t in (ACC,):
    ax.axhline(t, color="0.8", lw=0.8, ls=":", zorder=0)
    ax.axvline(t, color="0.8", lw=0.8, ls=":", zorder=0)

# join the chain pairs of one complex, so the non-independence is visible
bypdb = defaultdict(list)
for k, (i, xi, yi, _, _) in enumerate(rec):
    bypdb[i.split("_")[0]].append((xi, yi))
for v in bypdb.values():
    if len(v) > 1:
        ax.plot([p[0] for p in v], [p[1] for p in v], color="0.75", lw=0.7, zorder=1)

# size by how much of the pool is acceptable: small marker = needle in a haystack
frac = np.array([r[4] for r in rec])
sz = 26 + 150 * frac
win, los, tie = d > 1e-9, d < -1e-9, np.abs(d) <= 1e-9
ax.scatter(x[win], y[win], s=sz[win], c="#1a7f37", alpha=0.80, lw=0.5,
           edgecolor="white", zorder=3, label=f"ours better  (n={win.sum()})")
ax.scatter(x[los], y[los], s=sz[los], c="#b4232c", alpha=0.80, lw=0.5,
           edgecolor="white", zorder=3, label=f"shipped better  (n={los.sum()})")
ax.scatter(x[tie], y[tie], s=sz[tie], c="0.55", alpha=0.75, lw=0.5,
           edgecolor="white", zorder=2, label=f"tie  (n={tie.sum()})")

for i, xi, yi, _, _ in rec:
    if abs(yi - xi) >= a.label_thresh:
        ax.annotate(i, (xi, yi), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.0, color="0.25")

pool = "MAX" if a.col == "iptm" else "MEAN"
ax.set_xlabel(f"shipped head, selected DockQ  ({pool}-pooled ipTM)")
ax.set_ylabel(f"cropped arm ep30, selected DockQ  ({pool}-pooled ipTM)")
ax.set_title(f"Selected DockQ per interface, IDENTICAL ground-truth crop\n"
             f"valfull, {len(rec)} scored chain pairs, 150 candidates each", fontsize=10.5)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
ax.set_aspect("equal")
ax.grid(alpha=0.18, lw=0.6)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
top5_share = 100 * sum(sorted(d)[-5:]) / len(d) / d.mean()
ax.text(0.03, 0.97,
        f"mean $\\Delta$ {d.mean():+.4f}  (t={t:+.2f})\n"
        f"ours {y.mean():.4f}   shipped {x.mean():.4f}\n"
        f"frac acceptable {(y >= ACC).mean():.3f} vs {(x >= ACC).mean():.3f}\n"
        f"top 5 points carry {top5_share:.0f}% of the mean $\\Delta$",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.95))
# the caption goes UNDER the axes: at lower-right it sat on top of the legend
fig.text(0.5, 0.012, "marker size = fraction of the pool that is acceptable   |   "
                     "thin grey line joins chain pairs of the same complex",
         ha="center", va="bottom", fontsize=7.5, color="0.4")

op = os.path.join(ROOT, a.out)
os.makedirs(os.path.dirname(op), exist_ok=True)
fig.tight_layout(rect=(0, 0.028, 1, 1))
fig.savefig(op, dpi=165)
print(f"  {len(rec)} interfaces -> {a.out}")
print(f"  mean delta {d.mean():+.4f}  t {t:+.2f}  wins {win.sum()} losses {los.sum()} "
      f"ties {tie.sum()}")
