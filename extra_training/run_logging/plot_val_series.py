#!/usr/bin/env python
"""Held-out metrics against epoch, one panel each, for an arm evaluated every epoch.

WHY A SERIES AND NOT BARS. plot_rel_bars shows ONE epoch against the shipped head in detail; with
an eval after every epoch the question changes to "is this going anywhere", which is a shape over
epochs and unreadable as a row of bar charts.

THREE PANELS, ONE COLUMN, SHARED X. Mean selected DockQ, fraction acceptable, and median
within-interface Spearman. They do not share a y scale and must not be forced onto one -- DockQ
moves in thousandths here and Spearman in tenths, so a shared axis would flatten one of them into
a straight line.

THE SHIPPED HEAD IS A HORIZONTAL LINE, not a series, because it is a fixed head scored on the same
candidates at every epoch; drawing it as a curve would imply it changes. It is recomputed from
each epoch's own rows rather than assumed constant, and the spread across epochs is printed -- if
that is not ~0 the pools are not identical and nothing below is paired.

EVERYTHING HERE IS UNCROPPED on both sides, unlike paper_results section 9, so these numbers are
comparable to the shipped head as it will actually be used.

Usage:  python -I run_logging/plot_val_series.py <run_id> [out.png]
"""
import csv
import glob
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
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import eval_dirs                                                             # noqa: E402
import plot_three                                                            # noqa: E402

RID = sys.argv[1] if len(sys.argv) > 1 else "erb7nd"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/val_series_{RID}.png"
ACC = 0.23


def spearman(xs, ys):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                out[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


dirs = eval_dirs.chain_shard_dirs(plot_three.run_chain(RID),
                                  os.path.join(ROOT, "training", "runs"))
if not dirs:
    sys.exit(f"  no eval shard directories for {RID} yet")

eps, S = [], {k: [] for k in ("mean_dq", "max_dq", "ship_dq", "mean_fa", "max_fa", "ship_fa",
                              "mean_rho", "max_rho", "ship_rho", "n")}
for ep in sorted(dirs):
    g = defaultdict(list)
    for f in glob.glob(os.path.join(dirs[ep], "rows_*_s*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                g[r["iface"]].append((float(r["iptm"]), float(r["iptm_mean"]),
                                      float(r["stored_iptm"]), float(r["dockq"])))
            except (TypeError, ValueError, KeyError):
                continue
    if not g:
        continue
    full = max(len(v) for v in g.values())
    keep = [i for i in sorted(g) if len(g[i]) == full]
    if not keep:
        continue
    eps.append(ep)
    S["n"].append(len(keep))
    for tag, j in (("max", 0), ("mean", 1), ("ship", 2)):
        dq = np.array([max(g[i], key=lambda t: t[j])[3] for i in keep])
        S[f"{tag}_dq"].append(float(dq.mean()))
        S[f"{tag}_fa"].append(float((dq >= ACC).mean()))
        S[f"{tag}_rho"].append(float(np.median(
            [spearman([p[j] for p in g[i]], [p[3] for p in g[i]]) for i in keep])))

if not eps:
    sys.exit(f"  no complete eval epoch for {RID} yet")

# THE BASELINE MUST BE CONSTANT, or the epochs are not scored on the same candidates.
drift = max(max(S[f"ship_{k}"]) - min(S[f"ship_{k}"]) for k in ("dq", "fa", "rho"))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})
fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
PANELS = (("mean selected DockQ", "dq", "{:.4f}"),
          ("fraction acceptable  (DockQ >= 0.23)", "fa", "{:.3f}"),
          ("median within-interface Spearman vs DockQ", "rho", "{:+.3f}"))
for ax, (title, key, fmt) in zip(axes, PANELS):
    ship = float(np.mean(S[f"ship_{key}"]))
    ax.axhline(ship, color="#8a8a8a", lw=1.1, ls="--", alpha=0.8)
    ax.annotate(f"shipped  {fmt.format(ship)}", (0.995, ship), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#8a8a8a")
    for tag, colour, label in (("mean", "#2f6fb0", "ours, MEAN-pooled"),
                               ("max", "#b03a3a", "ours, MAX-pooled")):
        y = S[f"{tag}_{key}"]
        ax.plot(eps, y, color=colour, lw=1.8, marker="o", ms=4, label=label)
        # THE LAST VALUE, OFFSET, so two series ending close together do not overprint.
        ax.annotate(fmt.format(y[-1]), (eps[-1], y[-1]), color=colour, fontsize=8.5,
                    fontweight="bold", xytext=(6, -3 if tag == "max" else 4),
                    textcoords="offset points")
    ax.set_title(title, loc="left", fontsize=9.5, pad=6)
    # INTEGER EPOCHS, EXPLICITLY. MaxNLocator(integer=True) still produced 0.945..1.050 with a
    # single point, because it fits ticks to a degenerate range before rounding them.
    if len(eps) <= 12:
        ax.set_xticks(eps)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(min(eps) - 0.5, max(eps) + 0.9)
    # ROOM AT TOP AND BOTTOM, so a point never sits ON the spine and its label never leaves the
    # axes -- both happened with the default margins at one epoch.
    ax.margins(y=0.28)
# THE LEGEND GOES ABOVE THE AXES, not inside them: at one epoch "lower left" landed on top of the
# MAX series, and with more epochs it would sit wherever the curves happen to be.
axes[0].legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left",
               bbox_to_anchor=(0, 1.12), borderaxespad=0)
axes[-1].set_xlabel("epoch")
# TWO SHORT LINES RATHER THAN ONE LONG ONE. The single-line title ran off the right edge of the
# figure and lost the words that said what the pool was.
# THE WORD "HELD OUT" GOES IN THE BOLD LINE. It was in the grey subtitle, which is the line a
# reader skips -- two figures from the same arm then looked interchangeable and the only thing
# telling them apart was the filename.
fig.suptitle(f"{RID}   HELD-OUT metrics  (valfull, never trained on)", x=0.012, y=0.990,
             ha="left", fontsize=11.5, fontweight="bold", va="top")
fig.text(0.012, 0.9555, f"uncropped arm, beta 75   {S['n'][-1]} interfaces at budget 30   "
         f"both sides uncropped = deployment-comparable"
         + (f"   [baseline drift {drift:.4f}]" if drift > 1e-6 else ""),
         ha="left", va="top", fontsize=8, color="#777777")
fig.tight_layout(rect=[0, 0, 1, 0.938])
fig.savefig(os.path.join(ROOT, OUT), dpi=150)
print(f"  {len(eps)} eval epoch(s) -> {OUT}")
print(f"  {'ep':>4} {'ours MEAN':>10} {'ours MAX':>9} {'shipped':>8} {'dMEAN':>8} {'dMAX':>8}")
for i, e in enumerate(eps):
    print(f"  {e:>4} {S['mean_dq'][i]:>10.4f} {S['max_dq'][i]:>9.4f} {S['ship_dq'][i]:>8.4f} "
          f"{S['mean_dq'][i]-S['ship_dq'][i]:>+8.4f} {S['max_dq'][i]-S['ship_dq'][i]:>+8.4f}")
if drift > 1e-6:
    print(f"  WARNING: the shipped baseline moved by {drift:.4f} across epochs; the pools differ")
