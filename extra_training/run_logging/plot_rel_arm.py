#!/usr/bin/env python
"""Both training loss components of the rel-iTM arm, per epoch, on one figure.

TWO AXES, NOT ONE, and that is the point. The PAE term runs order 2.0 and the relative term order
1e-2 -- roughly 180x apart -- so on a shared axis the rel curve is a flat line on the floor and
its shape, which is the thing being watched, is invisible. Two y-axes make both readable; the
colours match the axis labels so nobody reads one curve against the other's scale.

WHERE EACH COMES FROM. pae_loss is per POSE in trajectory.csv and is averaged over the epoch's
poses here. The rel term is NOT in trajectory.csv -- it is a per-interface-group quantity, not a
per-pose one -- so it is parsed from run.log's per-epoch "rank-loss (rel, ...)" line, which is
rank 0's view of the groups that rank saw. Those are different denominators and the script says
so rather than implying the two curves are commensurable.

THE RATIO IS THE THIRD CURVE, and it is the one that decides whether beta is right. Over seven
pre-launch windows the parity alpha ranged 30 to 1316 -- a 44x spread -- because rel's
coefficients sum to exactly zero so its accumulated gradient is a residual after near-complete
cancellation. beta 75 is a starting ratio, not a derived value, so what matters is whether the
realised |g_rel| / |g_PAE| stays in a usable band as training proceeds, not what it was at init.

THE PARTIAL EPOCH IS DRAWN HOLLOW: trajectory.csv is appended per pose, so the newest epoch is
usually incomplete and its mean is over fewer poses.

Usage:  python -I run_logging/plot_rel_arm.py [run_id] [out.png]
"""
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RID = sys.argv[1] if len(sys.argv) > 1 else "6t9h7d"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/rel_arm_{RID}.png"
EP_POSES = 5520

traj = os.path.join(ROOT, "training", "runs", RID, "trajectory.csv")
log = os.path.join(ROOT, "training", "runs", RID, "run.log")
if not os.path.exists(traj):
    sys.exit(f"  no trajectory.csv for {RID} yet")

pae = defaultdict(list)
for r in csv.DictReader(open(traj)):
    try:
        e, v = r.get("epoch"), r.get("pae_loss")
        if e in (None, "") or v in (None, ""):
            continue
        f = float(v)
        if np.isfinite(f):
            pae[int(float(e))].append(f)
    except (ValueError, TypeError):
        continue

# rank 0's per-epoch rel loss and the two diagnostics, from the log
rel, rho, sdr = {}, {}, {}
if os.path.exists(log):
    _t = open(log, errors="replace").read()
    for m in re.finditer(r"\[rank 0\] epoch (\d+) rank-loss \(rel[^)]*\): ([+-][0-9.]+)", _t):
        rel[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"\[rank 0\] epoch (\d+) WITHIN-INTERFACE SPEARMAN[^:]*: median "
                         r"([+-][0-9.]+)", _t):
        rho[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"\[rank 0\] epoch (\d+) SCORE SPREAD:.*?median ratio ([0-9.]+)", _t):
        sdr[int(m.group(1))] = float(m.group(2))

eps = sorted(pae)
if not eps:
    sys.exit(f"  trajectory.csv for {RID} has no pae_loss rows yet")
pv = [float(np.mean(pae[e])) for e in eps]
pn = [len(pae[e]) for e in eps]
full = max(pn[:-1]) if len(pn) > 1 else 0
partial = len(eps) > 1 and pn[-1] < full * 0.98
k = len(eps) - 1 if partial else len(eps)

# FOUR SINGLE-AXIS PANELS, NOT TWO WITH TWIN AXES. The first version put PAE and rel on one
# panel with twin y-axes and the two diagnostics on another the same way: four axes, four
# colour-coded tick sets, two legends and reference annotations that collided with the curves.
# Each quantity has its own scale and its own question, so each gets its own panel; the only
# thing they share is the epoch axis.
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
                     "legend.frameon": False, "figure.facecolor": "white",
                     "xtick.labelsize": 9, "ytick.labelsize": 9})
fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), sharex=True)
(ax_pae, ax_rel), (ax_rho, ax_sd) = axes


def draw(ax, xs, ys, colour, title, last_fmt="{:.4f}", hollow_last=False):
    if not xs:
        ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center",
                color="#999999", fontsize=9)
        ax.set_title(title, loc="left")
        return
    k = len(xs) - 1 if hollow_last else len(xs)
    ax.plot(xs[:k], ys[:k], color=colour, lw=1.8, marker="o", ms=3.5)
    if hollow_last:
        ax.plot(xs[k:], ys[k:], color=colour, lw=1.2, ls=":", marker="o", ms=5.5,
                mfc="white", mew=1.4)
    ax.annotate(last_fmt.format(ys[-1]), (xs[-1], ys[-1]), color=colour, fontsize=9,
                fontweight="bold", xytext=(5, 2), textcoords="offset points")
    ax.set_title(title, loc="left", color=colour)


draw(ax_pae, eps, pv, "#b03a3a", "PAE loss  (mean over the epoch's poses)",
     hollow_last=partial)
if partial:
    ax_pae.annotate(f"epoch {eps[-1]} partial, n={pn[-1]}", (0.98, 0.04),
                    xycoords="axes fraction", ha="right", fontsize=8, color="#999999")

_re = sorted(rel)
draw(ax_rel, _re, [rel[e] for e in _re], "#2f6fb0",
     "relative iTM term  (mean over interface-groups)", "{:.5f}")

# SPEARMAN gets a shaded band between where the head was and what the true iTM reaches, which
# says "this is the distance to cover" without two more annotated lines crossing the data.
_rhe = sorted(rho)
# THE REFERENCE IS THE MEAN-POOLED TRAINED HEAD, +0.2994, not +0.3171. The first version used
# 0.317, which is d32kqp ep66 under MAX pooling -- comparing this arm's mean-pooled diagnostic
# against a max-pooled reference. Recomputed from that run's own rows: MAX +0.3171, MEAN +0.2994,
# and the true iTM reaches +0.6898 under mean pooling. The shipped head is +0.1931 (max-pooled;
# there is no mean-pooled shipped number without a dedicated eval pass).
ax_rho.axhspan(0.2994, 0.6898, color="#3a8a5f", alpha=0.07, lw=0)
for _y, _lab in ((0.2994, "d32kqp ep66, mean-pooled  +0.2994"),
                 (0.6898, "true iTM  +0.6898")):
    ax_rho.axhline(_y, color="#3a8a5f", lw=0.9, ls="--", alpha=0.55)
    ax_rho.annotate(_lab, (0.99, _y), xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=8, color="#3a8a5f", alpha=0.9)
draw(ax_rho, _rhe, [rho[e] for e in _rhe], "#3a8a5f",
     "within-interface Spearman vs DockQ", "{:+.4f}")
ax_rho.set_ylim(0.15, 0.78)

_sde = sorted(sdr)
# TWO REFERENCES: 1.0 is matching the truth, and 0.450 is where the row-weight arm sat at
# epoch 66, so "is this arm less collapsed than the last one" is readable directly.
ax_sd.axhspan(0.450, 1.0, color="#8a5f3a", alpha=0.07, lw=0)
for _y, _lab in ((1.0, "matches the truth"), (0.450, "d32kqp ep66  0.450")):
    ax_sd.axhline(_y, color="#8a5f3a", lw=0.9, ls="--", alpha=0.55)
    ax_sd.annotate(_lab, (0.99, _y), xycoords=("axes fraction", "data"), ha="right",
                   va="bottom", fontsize=8, color="#8a5f3a", alpha=0.9)
draw(ax_sd, _sde, [sdr[e] for e in _sde], "#8a5f3a",
     "score spread  std(predicted) / std(true iTM)", "{:.3f}")
if _sde:
    ax_sd.set_ylim(0, max(1.15, max(sdr.values()) * 1.15))

for _a in (ax_rho, ax_sd):
    _a.set_xlabel("epoch")
for _a in axes.ravel():
    _a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
fig.suptitle(f"{RID}   relative-iTM arm, beta 75, mean pooling", x=0.012, ha="left",
             fontsize=11.5, fontweight="bold")
fig.text(0.012, 0.945, "top row: the two training terms   |   bottom row: the arm's prediction",
         fontsize=8.5, color="#777777")
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(os.path.join(ROOT, os.path.dirname(OUT)), exist_ok=True)
fig.savefig(os.path.join(ROOT, OUT), dpi=150, bbox_inches="tight")
print(f"  {len(eps)} epoch(s) -> {OUT}")
print(f"  {'ep':>3} {'PAE':>9} {'rel':>10} {'rho':>8} {'sd_ratio':>9} {'poses':>7}")
for e, v, n in zip(eps, pv, pn):
    r, rh, sr = rel.get(e), rho.get(e), sdr.get(e)
    print(f"  {e:>3} {v:>9.4f} {('%10.5f' % r) if r is not None else '         -'}"
          f" {('%+8.4f' % rh) if rh is not None else '       -'}"
          f" {('%9.3f' % sr) if sr is not None else '        -'} {n:>7}"
          + ("  PARTIAL" if partial and e == eps[-1] else ""))
