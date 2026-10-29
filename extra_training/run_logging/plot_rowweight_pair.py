#!/usr/bin/env python
"""Both of nskyne's PAE columns on one axis, against the unweighted control.

WHY ONE AXIS. The two columns answer different questions and the gap between them IS the quantity
of interest: pae_loss is the row-weighted term the arm trains, pae_loss_unweighted is the same
poses scored with every row equal. Their separation is what the weighting costs on the 80% of rows
ipTM does not read. Plotted apart, that separation has to be held in the head.

fawgtu is the matched control -- fresh, unweighted, same lr 1e-2 through its first 20 epochs --
restricted to the same 368 winnable structures, since it trained on 487.
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


def mean_series(d):
    eps = [e for e in sorted(d) if len(d[e]) >= EP_MIN]
    return eps, [float(np.mean(d[e])) for e in eps]

w_e, w_v = mean_series(curve(RID, "pae_loss"))
u_e, u_v = mean_series(curve(RID, "pae_loss_unweighted"))
c_e, c_v = mean_series(curve("fawgtu", "pae_loss", win=True))
if not u_e:
    sys.exit("  no complete epochs yet")

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#dddddd", "legend.frameon": False,
                     "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(9.0, 5.2))
lim = max(u_e)
ax.plot([e for e in c_e if e <= lim], [v for e, v in zip(c_e, c_v) if e <= lim],
        color="#888888", lw=2, ls="--", marker="o", ms=3.5,
        label="fawgtu, unweighted control")
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
