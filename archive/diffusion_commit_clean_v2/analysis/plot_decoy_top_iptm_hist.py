#!/usr/bin/env python
"""Histogram of the top pair ipTM of every confident-decoy interface (score_experiment.classes_at == "decoy": a pose
with DockQ >= 0.23 exists in the free-50 pool but the top-ipTM pose is below 0.23), one panel per dataset.
x = the interface's top pair ipTM (over its ~250 free-50 poses), y = number of decoy interfaces in that 0.05 bin.
Same pool rule as analysis/plot_scatter_high_iptm.py. Written 2026-10-02.
Usage: python analysis/plot_decoy_top_iptm_hist.py -> experimental_results/decoy_top_iptm_hist.{png,pdf}"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import score_experiment as se, paths

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experimental_results")
BLUE, INK, MUTED, GRID, BG = "#2a78d6", "#1f1f1e", "#6b6b68", "#e6e5e1", "#fcfcfb"
CUTS = (0.7, 0.8, 0.85, 0.9)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
fig.patch.set_facecolor(BG)
print("dataset\tn_decoy\t" + "\t".join(f"top>{c}" for c in CUTS))
for ax, name in zip(axes, ("foldbench", "pxm2025")):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    cls = se.classes_at(pool, first)
    top = np.array([max(r[3] for r in pool[i] if r[0] in first[i.split("_")[0]]) for i, c in cls.items() if c == "decoy"])
    above = [int((top > c).sum()) for c in CUTS]
    print(f"{name}\t{len(top)}\t" + "\t".join(map(str, above)))
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.hist(top, bins=np.arange(0, 1.0001, 0.05), color=BLUE, edgecolor=BG, linewidth=1.5, zorder=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_xlim(0, 1)
    ax.set_xlabel("top pair ipTM of the interface (highest over its free-50 poses)", fontsize=9, color=INK)
    ax.set_ylabel("number of confident-decoy interfaces", fontsize=9, color=INK)
    ax.set_title(f"{name}: n = {len(top)} confident-decoy interfaces", fontsize=10, color=INK, loc="left")
    ax.text(0.02, 0.97, "\n".join(f"top ipTM > {c}: {a} of {len(top)}" for c, a in zip(CUTS, above)),
            transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(f"{OUT}/decoy_top_iptm_hist.png", dpi=170, facecolor=BG)
fig.savefig(f"{OUT}/decoy_top_iptm_hist.pdf", facecolor=BG)
print(f"-> {OUT}/decoy_top_iptm_hist.png")
