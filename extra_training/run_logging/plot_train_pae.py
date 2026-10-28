#!/usr/bin/env python
"""Training PAE for ONE run, per epoch, from its trajectory.csv.

WHY A SEPARATE SCRIPT. plot_train_loss.py draws a fixed roster of arms on one axis; this one takes
a run id, so it works for an arm that has no eval pass at all and whose only signal is the
training curve.

THE PARTIAL EPOCH IS DRAWN HOLLOW. trajectory.csv is appended per pose, so the newest epoch is
almost always incomplete and its mean is over fewer poses than the others -- early in an epoch it
is a mean over a handful of structures and moves a lot. Drawing it filled, identical to a complete
epoch, is how a partial mean gets read as a trend; the hollow marker and the n= label say what it
is instead.

ONE AXIS, NO HELD-OUT CURVE, deliberately: this run computes no val metrics, so there is nothing
to pair the training curve with, and a single falling line is not evidence of anything on its own.
The caption says so rather than leaving it implied.

Usage:  python -I run_logging/plot_train_pae.py <run_id> [out.png]
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("need a run id")
out = sys.argv[2] if len(sys.argv) > 2 else f"figures/train_pae_{rid}.png"
traj = os.path.join(ROOT, "training", "runs", rid, "trajectory.csv")
if not os.path.exists(traj):
    sys.exit(f"  no trajectory.csv for {rid} yet")

per = defaultdict(list)
with open(traj, newline="") as fh:
    for r in csv.DictReader(fh):
        e, v = r.get("epoch"), r.get("pae_loss")
        if e in (None, "") or v in (None, ""):
            continue
        try:
            per[int(float(e))].append(float(v))
        except ValueError:
            continue
if not per:
    sys.exit(f"  trajectory.csv for {rid} has no finished poses yet")

eps = sorted(per)
mean = [float(np.mean(per[e])) for e in eps]
n = [len(per[e]) for e in eps]
# THE LAST EPOCH IS PARTIAL unless its pose count matches the completed epochs' mode.
full = max(n[:-1]) if len(n) > 1 else 0
partial = len(eps) > 1 and n[-1] < full * 0.98

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#dddddd", "legend.frameon": False,
                     "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(8.4, 5.0))
k = len(eps) - 1 if partial else len(eps)
ax.plot(eps[:k], mean[:k], color="#b03a3a", lw=2, marker="o", ms=5, label="train PAE (complete)")
if partial:
    ax.plot(eps[k:], mean[k:], color="#b03a3a", lw=1.2, ls=":", marker="o", ms=7,
            mfc="white", mew=1.6, label=f"epoch {eps[-1]} PARTIAL (n={n[-1]} of ~{full})")
ax.annotate(f"{mean[-1]:.4f}", (eps[-1], mean[-1]), fontsize=10, color="#b03a3a",
            xytext=(6, 4), textcoords="offset points")
if len(eps) > 1:
    ax.annotate(f"epoch {eps[0]}: {mean[0]:.4f}", (eps[0], mean[0]), fontsize=9, color="#777777",
                xytext=(6, -14), textcoords="offset points")
ax.set_xlabel("epoch")
ax.set_ylabel("masked PAE loss (training)")
ax.set_title(f"{rid} -- training PAE\nno anchor, no eval pass: this run has NO held-out curve "
             f"to pair this against")
ax.grid(alpha=0.4)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, os.path.dirname(out)) or ".", exist_ok=True)
fig.savefig(os.path.join(ROOT, out), dpi=140, bbox_inches="tight")
print(f"  {len(eps)} epoch(s) -> {out}")
for e, m, c in zip(eps, mean, n):
    print(f"    epoch {e:>3}  train PAE {m:.4f}  ({c} pose(s))"
          + ("  PARTIAL" if partial and e == eps[-1] else ""))
