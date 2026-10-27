#!/usr/bin/env python
"""Plot an ABAG-Rank run's training curves from its log.

FOUR PANELS BECAUSE THE WHOLE POINT IS THAT THEY DISAGREE. This project has already measured a
model improving its loss and its rank correlation while selected DockQ did not move -- the
confidence-head arms went rho 0.2301 -> ~0.255 for +0.0024 selected DockQ. So loss, rank
correlation and TOP-1 are drawn separately rather than averaged into one "it is learning" story,
and the top-1 panel carries the oracle so the headroom is visible.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PAT = re.compile(r"epoch\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)\s+soft-rho ([+\-\d.]+)\s+"
                 r"val top-1 DockQ ([\d.]+)\s+\(oracle ([\d.]+)\)")


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "training/abag/run1/train.log"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/abag_rank_training.png"
    rows = [tuple(map(float, m.groups())) for m in (PAT.search(l) for l in open(log)) if m]
    if not rows:
        print(f"  no epoch lines in {log}")
        return 1
    ep, tr, vl, rho, t1, orc = zip(*rows)
    print(f"  {len(ep)} epoch(s); best top-1 {max(t1):.4f} at epoch {int(ep[t1.index(max(t1))])}")

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": "#dddddd", "legend.frameon": False,
                         "figure.facecolor": "white"})
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    ax[0].plot(ep, tr, color="#b03a3a", lw=2, marker="o", ms=3)
    ax[0].set_title("training loss\nsmooth-L1 + ranking + distance")
    ax[1].plot(ep, vl, color="#2f6fb0", lw=2, marker="o", ms=3)
    ax[1].set_title("internal-val regression loss\nNOT the selection criterion")
    ax[2].plot(ep, rho, color="#2e9e83", lw=2, marker="o", ms=3)
    ax[2].set_title("internal-val soft Spearman\nordering across a complex's poses")
    ax[3].plot(ep, t1, color="#7f5fb0", lw=2, marker="o", ms=3, label="selected")
    ax[3].axhline(orc[0], color="#2e9e83", ls=":", lw=1.4, label=f"oracle {orc[0]:.3f}")
    ax[3].set_title("internal-val TOP-1 DockQ\nthe criterion checkpoints are chosen on")
    ax[3].legend(fontsize=9, loc="best")
    for a, lab in zip(ax, ("loss", "loss", "soft rho", "mean DockQ")):
        a.set_xlabel("epoch")
        a.set_ylabel(lab)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.6, lw=0.7)
    fig.suptitle("ABAG-Rank on our interfaces · internal split (875 train / 100 val "
                 "interfaces, held out BY STRUCTURE) · valfull untouched", fontsize=13,
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("  wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
