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

# THE WHOLE CHAIN, oldest first. A resume writes to a FRESH run directory while continuing the
# parent's epoch numbering, so reading only RID would plot epochs 28-60 of a 60-epoch arm and drop
# everything before the pause -- the exact bug that made the control look like a 35-epoch run.
# plot_three.run_chain resolves it from args.json's recorded --resume path, not from the name.
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import plot_three                                                            # noqa: E402
RIDS = list(plot_three.run_chain(RID))
_present = [r for r in RIDS
            if os.path.exists(os.path.join(ROOT, "training", "runs", r, "trajectory.csv"))]
if not _present:
    sys.exit(f"  no trajectory.csv for {RID} yet")

# A LATER RUN WINS on any epoch both wrote. A resume re-runs the epoch it was interrupted in, so
# the parent's partial rows for that epoch would otherwise be averaged in with the full ones.
pae = defaultdict(list)
for _r in _present:
    _ep = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ROOT, "training", "runs", _r,
                                              "trajectory.csv"))):
        try:
            e, v = r.get("epoch"), r.get("pae_loss")
            if e in (None, "") or v in (None, ""):
                continue
            f = float(v)
            if np.isfinite(f):
                _ep[int(float(e))].append(f)
        except (ValueError, TypeError):
            continue
    pae.update(_ep)

# rank 0's per-epoch rel loss and the two diagnostics, from the log of every run in the chain
rel, rho, sdr = {}, {}, {}
for _r in RIDS:
    log = os.path.join(ROOT, "training", "runs", _r, "run.log")
    if not os.path.exists(log):
        continue
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
# THE CURVE AND THE REFERENCES ARE MEASURED ON DIFFERENT INPUTS, and the band used to hide it.
# This curve comes from the TRAINING loop, so every pose is CROPPED -- crop 3400 with a
# 10-residue interface margin. Every reference number in this project (d32kqp ep66, the true-iTM
# ceiling, the shipped head) comes from the evaluator, which has no crop flags and always runs the
# FULL structure. Shading from 0.2994 to 0.6898 under a cropped curve therefore drew the arm
# crossing most of a gap it was not measured against: the same epoch-27 checkpoint reads +0.5392
# cropped here and +0.2819 uncropped on the identical 738 interfaces.
#
# SO THE BAND IS GONE and the references are drawn as what they are -- uncropped, on their own
# axis -- with the ONE matched uncropped pair this arm has, measured by
# run_logging/rank_gain_location.py at epoch 27. Read the curve for its SHAPE; read the uncropped
# markers for where the arm actually stands.
_UNCROPPED = ((0.6898, "true iTM, uncropped  +0.6898", "#3a8a5f"),
              (0.2994, "d32kqp ep66, uncropped  +0.2994", "#3a8a5f"),
              (0.2819, "THIS ARM ep27, uncropped  +0.2819", "#b03a3a"),
              (0.1931, "shipped head, uncropped  +0.1931", "#8a8a8a"))
for _y, _lab, _c in _UNCROPPED:
    ax_rho.axhline(_y, color=_c, lw=0.9, ls="--", alpha=0.6)
    ax_rho.annotate(_lab, (0.99, _y), xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=7.5, color=_c, alpha=0.95)
draw(ax_rho, _rhe, [rho[e] for e in _rhe], "#3a8a5f",
     "within-interface Spearman vs DockQ  (curve CROPPED, dashed refs UNCROPPED)", "{:+.4f}")
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
