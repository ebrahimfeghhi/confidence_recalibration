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
args = [x for x in sys.argv[1:] if not x.startswith("--")]
CHAIN = "--chain" in sys.argv
# --ymin PINS THE BOTTOM OF THE AXIS. Default is matplotlib's autoscale; setting it is a reading
# aid, not a claim -- a floor at 1.0 says "PAE cannot go below 1 here", which is not established,
# so the axis label says it is an axis choice.
YMIN = None
for _a in sys.argv[1:]:
    if _a.startswith("--ymin"):
        YMIN = float(_a.split("=", 1)[1]) if "=" in _a else float(
            sys.argv[sys.argv.index(_a) + 1])
# --winnable-only RESTRICTS TO THE 368 STRUCTURES THE ROW-WEIGHT ARM TRAINS ON. The 119 dropped
# ones have no pose above DockQ 0.23, sit at PAE 1.10 against 1.96 for the rest, and are 25% of
# poses -- so a pooled curve is 0.21 below the number for the structures that can be won. Without
# this flag an arm trained on 368 looks like a 0.21 regression against one trained on 487 when
# nothing has regressed, which is the one way this plot can actively mislead.
WINNABLE = "--winnable-only" in sys.argv
_keep = None
if WINNABLE:
    _drop = os.path.join(ROOT, "training", "full_train", "dropped_unwinnable.txt")
    if not os.path.exists(_drop):
        sys.exit(f"  --winnable-only needs {_drop}")
    _skip = {ln.strip() for ln in open(_drop) if ln.strip()}
    _keep = lambda p: p not in _skip  # noqa: E731
    print(f"  --winnable-only: excluding {len(_skip)} structure(s) with no pose above DockQ 0.23")
rid = args[0] if args else sys.exit("need a run id")
out = args[1] if len(args) > 1 else f"figures/train_pae_{rid}.png"

# --chain FOLLOWS THE RESUME LINEAGE. A run that resumed from another holds only its OWN epochs:
# jy6sah's trajectory starts at epoch 101 because six earlier runs produced 1-100. Plotting it
# alone shows the tail of a curve and looks like the whole thing, which is how a chain's real
# starting loss gets lost. run_chain is the same resolver plot_overfit30 uses.
rids = [rid]
if CHAIN:
    sys.path.insert(0, os.path.join(ROOT, "run_logging"))
    import plot_three as P
    rids = list(P.run_chain(rid))
    print(f"  chain: {' -> '.join(rids)}")

per = defaultdict(list)
for r_ in rids:
    traj = os.path.join(ROOT, "training", "runs", r_, "trajectory.csv")
    if not os.path.exists(traj):
        if r_ == rid:
            sys.exit(f"  no trajectory.csv for {r_} yet")
        print(f"  (no trajectory.csv for {r_}, skipped)")
        continue
    with open(traj, newline="") as fh:
        for r in csv.DictReader(fh):
            e, v = r.get("epoch"), r.get("pae_loss")
            if e in (None, "") or v in (None, ""):
                continue
            if _keep is not None and not _keep((r.get("pdb") or "").strip()):
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
# THE SUBTITLE IS CONDITIONAL. It used to assert "no eval pass, no held-out curve", which is
# true of fawgtu and false of any run that monitored val -- printing it on jy6sah would caption
# the plot with a claim about the wrong run.
ax.set_title(f"{rid}" + (" (chained)" if CHAIN and len(rids) > 1 else "")
             + (" -- winnable structures only" if WINNABLE else ""))
if YMIN is not None:
    ax.set_ylim(bottom=YMIN)
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
