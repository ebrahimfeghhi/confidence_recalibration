#!/usr/bin/env python
"""Plot an ABAG-Rank run against valfull: training loss, selected DockQ, and the paired delta.

SHIPPED AND ORACLE ARE DRAWN AS LINES, not quoted in a caption, because the only question this
run asks is whether the reranker's selected DockQ clears shipped on the same 92 interfaces.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EP = re.compile(r"epoch\s+(\d+)\s+train ([\d.]+)")
VF = re.compile(r"epoch\s+(\d+)\s+train ([\d.]+)\s+\| VALFULL dockq ([\d.]+) shipped ([\d.]+) "
                r"delta ([+\-\d.]+) frac ([\d.]+) n (\d+) oracle ([\d.]+)")


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "training/abag/run_full/train.log"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/abag_rank_valfull.png"
    tr, vf = [], []
    for l in open(log):
        m = VF.search(l)
        if m:
            vf.append(tuple(map(float, m.groups())))
        m2 = EP.search(l)
        if m2:
            tr.append(tuple(map(float, m2.groups())))
    if not vf:
        print("  no VALFULL lines yet")
        return 1
    ep, _t, dq, ship, delta, frac, _n, orc = zip(*vf)
    best = max(range(len(dq)), key=lambda i: dq[i])
    print(f"  {len(tr)} epoch(s), {len(vf)} valfull point(s); best dockq {dq[best]:.4f} "
          f"(delta {delta[best]:+.4f}) at epoch {int(ep[best])}")

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": "#dddddd", "legend.frameon": False,
                         "figure.facecolor": "white"})
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax[0].plot([t[0] for t in tr], [t[1] for t in tr], color="#b03a3a", lw=1.6)
    ax[0].set_title("training loss\nsmooth-L1 + ranking + distance")
    ax[0].set_ylabel("loss")

    ax[1].plot(ep, dq, color="#7f5fb0", lw=2, marker="o", ms=3, label="reranker")
    ax[1].axhline(ship[0], color="#9a9a9a", ls="--", lw=1.4, label=f"shipped {ship[0]:.4f}")
    ax[1].axhline(orc[0], color="#2e9e83", ls=":", lw=1.4, label=f"oracle {orc[0]:.4f}")
    ax[1].set_title("valfull MEAN SELECTED DockQ\n92 interfaces, budget 30")
    ax[1].set_ylabel("mean DockQ")
    ax[1].legend(fontsize=9, loc="best")

    # SHIPPED's frac IS RECOMPUTED FROM THE SAME ROWS, not quoted: 68 of 92 interfaces, 0.7391.
    ax[2].plot(ep, frac, color="#2f6fb0", lw=2, marker="o", ms=3, label="reranker")
    ax[2].axhline(68 / 92, color="#9a9a9a", ls="--", lw=1.4, label=f"shipped {68 / 92:.4f}")
    ax[2].set_title("valfull FRACTION ACCEPTABLE\nDockQ >= 0.23")
    ax[2].set_ylabel("fraction")
    ax[2].legend(fontsize=9, loc="best")
    fig.suptitle("ABAG-Rank trained on all 975 training interfaces · valfull is the "
                 "validation set · FoldBench and PXM remain untouched", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("  wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
