#!/usr/bin/env python
"""Training PAE loss for every arm on one axis -- the capacity question, directly.

TRAIN LOSS IS COMPARABLE ACROSS THESE ARMS; HELD-OUT LOSS IS NOT. They all minimise the same
masked PAE loss over the same training fold at the same crop (ocqtyt carries one extra structure,
9dzj, of 488), so the training curves sit on one scale. Their HELD-OUT curves do not: ocqtyt
monitored val10's 18 interfaces while every other arm used valfull's 92, and val10 and valfull are
never comparable. So held-out is drawn dashed and ocqtyt's is drawn in grey with a note, rather
than being left to look like the same measurement.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_three as P  # noqa: E402

ARMS = [("ip8ga4", "share, 10,240 params", "#2f6fb0"),
        ("dntxmn", "copy, 40,960 params", "#e07b39"),
        ("tv6hue", "full head 19.8M + L2-SP 1e-3", "#2e9e83"),
        ("jer222", "full head 19.8M + L2-SP 1e-2", "#b03a3a"),
        ("ocqtyt", "full head 19.8M, NO L2, no pose", "#7f5fb0")]

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "grid.color": "#dddddd",
                     "legend.frameon": False, "figure.facecolor": "white"})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
for rid, label, c in ARMS:
    try:
        e, tr, vl = P.pae_curves(rid)
    except Exception:
        continue
    if not e:
        continue
    ax[0].plot(e, tr, color=c, lw=2, marker="o", ms=3.5, label=f"{label}  (n={len(e)})")
    ax[1].plot(e, vl, color=c, lw=2, marker="o", ms=3.5, ls="--",
               label=f"{label}{'  [val10, NOT comparable]' if rid == 'ocqtyt' else ''}")
ax[0].set_title("TRAINING PAE loss -- comparable across every arm\n"
                "same objective, same fold, same crop")
ax[1].set_title("held-out PAE loss -- NOT comparable across every arm\n"
                "ocqtyt monitored val10 (18 ifaces); the rest valfull (92)")
for a in ax:
    a.set_xlabel("epoch")
    a.set_ylabel("PAE loss")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.grid(alpha=0.6, lw=0.7)
    a.legend(fontsize=9, loc="best")
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "figures/train_loss.png"
fig.savefig(out, dpi=150)
print("wrote", out)
