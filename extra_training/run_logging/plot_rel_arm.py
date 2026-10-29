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

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "grid.color": "#dddddd",
                     "legend.frameon": False, "figure.facecolor": "white"})
fig, (ax, cx) = plt.subplots(1, 2, figsize=(15.0, 5.4))
C_PAE, C_REL = "#b03a3a", "#2f6fb0"
ax.plot(eps[:k], pv[:k], color=C_PAE, lw=2, marker="o", ms=4.5, label="PAE (left axis)")
if partial:
    ax.plot(eps[k:], pv[k:], color=C_PAE, lw=1.2, ls=":", marker="o", ms=7, mfc="white",
            mew=1.6, label=f"epoch {eps[-1]} PARTIAL (n={pn[-1]} of ~{full})")
ax.set_xlabel("epoch")
ax.set_ylabel("PAE loss (training, masked)", color=C_PAE)
ax.tick_params(axis="y", labelcolor=C_PAE)
ax.grid(alpha=0.4)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

bx = ax.twinx()
bx.spines["top"].set_visible(False)
if rel:
    re_, rv = zip(*sorted(rel.items()))
    bx.plot(re_, rv, color=C_REL, lw=2, marker="s", ms=4.5,
            label="relative iTM term (right axis)")
    bx.annotate(f"{rv[-1]:.4f}", (re_[-1], rv[-1]), fontsize=9, color=C_REL,
                xytext=(5, 3), textcoords="offset points")
else:
    bx.text(0.5, 0.5, "no 'rank-loss (rel, ...)' line in run.log yet",
            transform=bx.transAxes, ha="center", color=C_REL, fontsize=9)
bx.set_ylabel("relative iTM loss (per interface-group)", color=C_REL)
bx.tick_params(axis="y", labelcolor=C_REL)

ax.set_title(f"{RID}: the two training terms\n"
             f"DIFFERENT DENOMINATORS -- PAE is a mean over the epoch's poses, "
             f"rel a mean over rank 0's interface-groups")

# RIGHT PANEL: the two diagnostics, which are the arm's actual prediction rather than its loss.
# Spearman should move off +0.317 toward the true iTM's +0.690; the spread ratio should sit near
# 1, and a fall below it is the margin-collapse shape every ranking arm here showed.
C_RHO, C_SD = "#3a8a5f", "#8a5f3a"
if rho:
    _re, _rv = zip(*sorted(rho.items()))
    cx.plot(_re, _rv, color=C_RHO, lw=2, marker="o", ms=4.5,
            label="within-interface Spearman (left)")
    cx.annotate(f"{_rv[-1]:+.4f}", (_re[-1], _rv[-1]), fontsize=9, color=C_RHO,
                xytext=(5, 3), textcoords="offset points")
cx.axhline(0.317, color="#999999", lw=1, ls="--")
cx.annotate("head at d32kqp ep66: +0.317", (0.02, 0.317), xycoords=("axes fraction", "data"),
            fontsize=8.5, color="#777777", va="bottom")
cx.axhline(0.690, color="#b03a3a", lw=1, ls="--")
cx.annotate("TRUE iTM ceiling: +0.690", (0.02, 0.690), xycoords=("axes fraction", "data"),
            fontsize=8.5, color="#b03a3a", va="bottom")
cx.set_xlabel("epoch")
cx.set_ylabel("within-interface Spearman vs DockQ", color=C_RHO)
cx.tick_params(axis="y", labelcolor=C_RHO)
cx.set_ylim(-0.05, 0.80)
cx.grid(alpha=0.4)
cx.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
dx = cx.twinx()
dx.spines["top"].set_visible(False)
if sdr:
    _se, _sv2 = zip(*sorted(sdr.items()))
    dx.plot(_se, _sv2, color=C_SD, lw=1.8, ls="-.", marker="s", ms=4,
            label="std(predicted) / std(true iTM) (right)")
dx.axhline(1.0, color=C_SD, lw=0.8, alpha=0.5)
dx.set_ylabel("score-spread ratio   [<1 collapsing, >1 exaggerating]", color=C_SD)
dx.tick_params(axis="y", labelcolor=C_SD)
cx.set_title("THE ARM'S PREDICTION, not its loss\n"
             "Spearman should rise off +0.317 toward +0.690; the ratio should stay near 1")
h3, l3 = cx.get_legend_handles_labels()
h4, l4 = dx.get_legend_handles_labels()
if h3 or h4:
    cx.legend(h3 + h4, l3 + l4, loc="lower right", fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = bx.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, os.path.dirname(OUT)), exist_ok=True)
fig.savefig(os.path.join(ROOT, OUT), dpi=140, bbox_inches="tight")
print(f"  {len(eps)} epoch(s) -> {OUT}")
print(f"  {'ep':>3} {'PAE':>9} {'rel':>10} {'rho':>8} {'sd_ratio':>9} {'poses':>7}")
for e, v, n in zip(eps, pv, pn):
    r, rh, sr = rel.get(e), rho.get(e), sdr.get(e)
    print(f"  {e:>3} {v:>9.4f} {('%10.5f' % r) if r is not None else '         -'}"
          f" {('%+8.4f' % rh) if rh is not None else '       -'}"
          f" {('%9.3f' % sr) if sr is not None else '        -'} {n:>7}"
          + ("  PARTIAL" if partial and e == eps[-1] else ""))
