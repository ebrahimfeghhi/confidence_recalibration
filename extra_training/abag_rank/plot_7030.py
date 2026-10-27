#!/usr/bin/env python
"""Three panels for a held-out-structure ABAG-Rank run: training loss, mean selected DockQ, frac
acceptable.

EVERY BASELINE HERE IS PAIRED. The ipTM, random and oracle lines come from val_baseline.json,
computed over the SAME K=10 candidate poses per interface that the training loop scores, verified
by reproducing its oracle to four decimals. A baseline quoted from another file -- a different
budget, a different population, all 50 poses instead of 10 -- would not be the same comparison.

The random-pick line is drawn because it is the floor that separates selection from pose quality:
a model landing between random and ipTM is ordering poses, just worse than the metric already
shipped.

frac acceptable needs `val frac` in the log, which train.py only began printing on 2026-10-27;
an older run's third panel says so rather than being left blank and read as zero.

Usage:  python -I abag_rank/plot_7030.py [run dir] [out png]
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PAT = re.compile(r"epoch\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)\s+soft-rho ([+\-\d.]+)\s+"
                 r"val top-1 DockQ ([\d.]+)\s+\(oracle ([\d.]+)\)")
FRAC = re.compile(r"val frac ([\d.]+)\s+\(oracle ([\d.]+)\)")

run = sys.argv[1] if len(sys.argv) > 1 else "training/abag/run_7030"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/abag_rank_7030.png"
ep, tr, t1, fr = [], [], [], []
for line in open(os.path.join(run, "train.log")):
    m = PAT.search(line)
    if not m:
        continue
    ep.append(int(m.group(1))); tr.append(float(m.group(2))); t1.append(float(m.group(5)))
    f = FRAC.search(line)
    fr.append(float(f.group(1)) if f else np.nan)
if not ep:
    sys.exit(f"  no epoch lines in {run}/train.log")
bl = json.load(open(os.path.join(run, "val_baseline.json")))
sp = json.load(open(os.path.join(run, "split.json")))

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#dddddd", "legend.frameon": False,
                     "figure.facecolor": "white"})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

ax[0].plot(ep, tr, color="#b03a3a", lw=1.8)
ax[0].set_title("training loss\nsmooth-L1 + ranking + distance")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")

def panel(axis, y, key_model, iptm, rnd, orac, title, ylab):
    axis.axhline(orac, color="#777777", ls=":", lw=1.6, label=f"oracle {orac:.4f}")
    axis.axhline(iptm, color="#1f7a1f", ls="--", lw=1.8, label=f"ipTM (shipped) {iptm:.4f}")
    axis.axhline(rnd, color="#999999", ls="-.", lw=1.3, label=f"random pick {rnd:.4f}")
    if np.all(np.isnan(y)):
        axis.text(0.5, 0.5, "not logged for this run\n(train.py began printing\n'val frac' later)",
                  ha="center", va="center", transform=axis.transAxes, color="#777777")
    else:
        axis.plot(ep, y, color="#2f6fb0", lw=1.8, label="ABAG-Rank")
        best = np.nanmax(y)
        axis.axhline(best, color="#2f6fb0", lw=0.8, alpha=0.5)
        axis.annotate(f"best {best:.4f}", (ep[int(np.nanargmax(y))], best), fontsize=9,
                      color="#2f6fb0", xytext=(3, 4), textcoords="offset points")
    axis.set_title(title); axis.set_xlabel("epoch"); axis.set_ylabel(ylab)
    axis.legend(loc="lower right", fontsize=9)

panel(ax[1], np.array(t1), None, bl["iptm_dockq"], bl["random_dockq"], bl["oracle_dockq"],
      "mean selected DockQ\nheld-out structures", "mean DockQ")
panel(ax[2], np.array(fr), None, bl["iptm_frac"], bl["random_frac"], bl["oracle_frac"],
      "fraction acceptable (DockQ >= 0.23)\nheld-out structures", "fraction")

for x in ax:
    x.grid(alpha=0.4); x.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
fig.suptitle(f"ABAG-Rank trained on 70% of training structures, evaluated on the held-out 30% "
             f"-- {sp['n_val_structures']}/{sp['n_structures']} structures, "
             f"{bl['n_interfaces']} interfaces, K={bl['k']} poses each, seed {sp['seed']}  "
             f"| epoch {ep[-1]}", fontsize=11.5, y=1.015)
fig.tight_layout()
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out, dpi=135, bbox_inches="tight")
print(f"  {len(ep)} epoch(s) -> {out}")
print(f"  best mean DockQ {max(t1):.4f} at epoch {ep[t1.index(max(t1))]} "
      f"vs ipTM {bl['iptm_dockq']:.4f} (delta {max(t1) - bl['iptm_dockq']:+.4f})")
