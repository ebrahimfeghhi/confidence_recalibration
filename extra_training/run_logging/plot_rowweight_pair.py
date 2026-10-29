#!/usr/bin/env python
"""Both of nskyne's PAE columns on one axis, against the unweighted control.

WHY ONE AXIS. The two columns answer different questions and the gap between them IS the quantity
of interest: pae_loss is the row-weighted term the arm trains, pae_loss_unweighted is the same
poses scored with every row equal. Their separation is what the weighting costs on the 80% of rows
ipTM does not read. Plotted apart, that separation has to be held in the head.

THE CONTROL IS A CHAIN, NOT ONE RUN. fawgtu holds only epochs 1-36; the unweighted 60-epoch
lineage continues fawgtu -> j9pspo -> tvden2 -> j8axco -> 45q96x. Reading fawgtu alone silently
truncated the control at 36 and drew nothing past it, which is how a 60-epoch control gets
reported as a 35-epoch one. CONTROL_HEAD is resolved through plot_three.run_chain, the same
resolver plot_train_pae.py --chain uses.

BOUNDARY EPOCHS ARE DEDUPED BY POSE COUNT, not pooled. Each resume re-enters at the epoch it
stopped on, so epochs 36, 48 and 51 appear in two runs -- once as a partial tail (830, 78, 82
poses) and once complete. Pooling them averages two different weight states into one point; this
takes whichever run holds the most poses for that epoch and drops the other.

TWO PANELS BECAUSE THE EPOCH AXIS IS NOT MATCHED COMPUTE. nskyne trains on the 368 winnable
structures and the control trained on 487, so an nskyne epoch is 5,520 poses and a control epoch
is 7,306 -- 1.32x larger. Plotted against epoch index the control has seen 32% more training
poses at every x, and nskyne's whole 60-epoch budget (331,200 poses) only reaches control epoch
45. The right panel puts cumulative training poses on x so the two curves are read at equal
compute; the control's line simply runs further right because it did more work in total.

The pose counts are taken WITHOUT the winnable filter, because the control did train on all 487 --
the filter belongs on the y value (what we score) and not on the x value (what it saw).

THE CONTROL'S LR IS NOT nskyne'S. The chain ran 1e-2 to epoch 40, 1e-3 for 41-50, 1e-4 from 51;
nskyne runs a cosine 1e-2 -> 1e-4 over epochs 20-50. Both land at 1e-4 but the paths differ, so
past about epoch 20 the two arms differ in schedule as well as in loss and the delta is not
attributable to the weighting alone. Restricted to the same 368 winnable structures, since the
control trained on 487.
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
RID = sys.argv[1] if len(sys.argv) > 1 else "nskyne"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/rowweight_pair_{RID}.png"
EP_MIN = 5400
CONTROL_HEAD = "45q96x"
SKIP = {l.strip() for l in open(os.path.join(ROOT, "training", "full_train",
                                             "dropped_unwinnable.txt")) if l.strip()}


def curve(rid, col, win=False):
    d = defaultdict(list)
    p = os.path.join(ROOT, "training", "runs", rid, "trajectory.csv")
    for r in csv.DictReader(open(p)):
        try:
            e = int(float(r["epoch"])); v = r.get(col)
            if v in (None, "") or (win and (r.get("pdb") or "").strip() in SKIP):
                continue
            f = float(v)
            if np.isfinite(f):
                d[e].append(f)
        except (ValueError, KeyError, TypeError):
            continue
    return d


def chained_curve(head, col, win=False):
    """Union of the chain's per-epoch pose lists, keeping the longest list per epoch."""
    sys.path.insert(0, os.path.join(ROOT, "run_logging"))
    import plot_three as P
    rids = list(P.run_chain(head))
    print(f"  control chain: {' -> '.join(rids)}")
    best = {}
    for rid in rids:
        if not os.path.exists(os.path.join(ROOT, "training", "runs", rid, "trajectory.csv")):
            print(f"    (no trajectory.csv for {rid}, skipped)")
            continue
        for e, vals in curve(rid, col, win=win).items():
            if e not in best or len(vals) > len(best[e][1]):
                best[e] = (rid, vals)
    return {e: v for e, (_r, v) in best.items()}


def mean_series(d):
    eps = [e for e in sorted(d) if len(d[e]) >= EP_MIN]
    return eps, [float(np.mean(d[e])) for e in eps]

w_e, w_v = mean_series(curve(RID, "pae_loss"))
u_e, u_v = mean_series(curve(RID, "pae_loss_unweighted"))
c_e, c_v = mean_series(chained_curve(CONTROL_HEAD, "pae_loss", win=True))
if not u_e:
    sys.exit("  no complete epochs yet")

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#dddddd", "legend.frameon": False,
                     "figure.facecolor": "white"})
POSES_PER_EPOCH = {RID: 5520, "control": 7306}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.4, 5.4))
# THE CONTROL IS DRAWN TO ITS OWN END, NOT CLIPPED TO nskyne'S CURRENT EPOCH. It is finished, so
# its later epochs are the target nskyne has to reach, and clipping them hid that target. The part
# past nskyne's newest epoch is drawn lighter so it reads as "not yet matched" rather than as a
# comparison already made.
lim = max(u_e)
_cn = [(e, v) for e, v in zip(c_e, c_v) if e <= lim]
_cf = [(e, v) for e, v in zip(c_e, c_v) if e >= lim]
ax.plot([e for e, _ in _cn], [v for _, v in _cn],
        color="#888888", lw=2, ls="--", marker="o", ms=3.5,
        label=f"unweighted control (chain, {min(c_e)}-{max(c_e)})")
if len(_cf) > 1:
    ax.plot([e for e, _ in _cf], [v for _, v in _cf], color="#bbbbbb", lw=1.4, ls="--",
            marker="o", ms=3, label=f"control, ahead of {RID} (ends {_cf[-1][1]:.4f})")
    ax.annotate(f"{_cf[-1][1]:.4f}", (_cf[-1][0], _cf[-1][1]), fontsize=9, color="#888888",
                xytext=(4, -11), textcoords="offset points")
ax.plot(u_e, u_v, color="#b03a3a", lw=2, marker="o", ms=4.5,
        label=f"{RID} all rows equal")
ax.plot(w_e, w_v, color="#2f6fb0", lw=2, marker="o", ms=4.5,
        label=f"{RID} row-weighted (trained)")
ax.set_xlabel("epoch")
ax.set_ylabel("masked PAE loss (training)")
ax.set_title(f"{RID}: what the row weighting costs on the rows ipTM does not read")
ax.grid(alpha=0.4)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
ax.legend(loc="upper right", fontsize=9)

# RIGHT PANEL -- x IS CUMULATIVE TRAINING POSES, the same y values rescaled on x only.
_pc, _pa = POSES_PER_EPOCH["control"], POSES_PER_EPOCH[RID]
bx.plot([e * _pc / 1000 for e in c_e], c_v, color="#888888", lw=2, ls="--", marker="o", ms=3.5,
        label="unweighted control")
bx.plot([e * _pa / 1000 for e in u_e], u_v, color="#b03a3a", lw=2, marker="o", ms=4.5,
        label=f"{RID} all rows equal")
bx.plot([e * _pa / 1000 for e in w_e], w_v, color="#2f6fb0", lw=2, marker="o", ms=4.5,
        label=f"{RID} row-weighted (trained)")
_budget = 60 * _pa / 1000
bx.axvline(_budget, color="#444444", lw=1, ls=":")
bx.annotate(f"{RID} 60-epoch budget\n({_budget:.0f}k poses = control ep {60 * _pa / _pc:.0f})",
            (_budget, max(c_v)), fontsize=8, color="#444444", ha="right", va="top",
            xytext=(-5, -2), textcoords="offset points")
bx.set_xlabel("cumulative training poses (thousands)")
bx.set_ylabel("masked PAE loss (training)")
bx.set_title("same curves at equal compute")
bx.grid(alpha=0.4)
bx.legend(loc="upper right", fontsize=9)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, os.path.dirname(OUT)) or ".", exist_ok=True)
fig.savefig(os.path.join(ROOT, OUT), dpi=140, bbox_inches="tight")
print(f"  {len(u_e)} epoch(s) -> {OUT}")
print(f"  {'ep':>3}  {'weighted':>9} {'all-equal':>10} {'gap':>8}  {'control':>8} {'delta':>8}")
for e in u_e:
    g = (w_v[w_e.index(e)] - u_v[u_e.index(e)]) if e in w_e else float("nan")
    c = c_v[c_e.index(e)] if e in c_e else float("nan")
    print(f"  {e:>3}  {w_v[w_e.index(e)]:>9.4f} {u_v[u_e.index(e)]:>10.4f} {g:>+8.4f}  "
          f"{c:>8.4f} {u_v[u_e.index(e)]-c:>+8.4f}")
