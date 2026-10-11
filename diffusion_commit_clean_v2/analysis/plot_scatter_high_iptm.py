#!/usr/bin/env python
"""ipTM vs DockQ, one scatter per interface, for every interface whose top pair ipTM > T (default 0.9). Every pose
of the free-50 pool (~250 poses), sorted by top ipTM (highest first); ringed = the top-ipTM pose; dashed = DockQ 0.23.
One figure per dataset. Same pool rule as analysis/plot_dockq_range_high_iptm.py. Written 2026-10-02.
Usage: python analysis/plot_scatter_high_iptm.py [T] -> experimental_results/scatter_high_iptm_<T>_<dataset>.{png,pdf}"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import score_experiment as se, paths

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experimental_results")
BLUE, INK, MUTED, GRID, BG = "#2a78d6", "#1f1f1e", "#6b6b68", "#e6e5e1", "#fcfcfb"

for name in ("foldbench", "pxm2025"):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    panels = []
    for i, rows in pool.items():
        keep = [r for r in rows if r[0] in first[i.split("_")[0]]]
        if keep and max(r[3] for r in keep) > T:
            panels.append((i, keep, max(keep, key=lambda r: r[3])))
    panels.sort(key=lambda p: -p[2][3])
    nc = math.ceil(math.sqrt(len(panels)))
    nr = math.ceil(len(panels) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.2 * nr), sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    print(f"{name}: {len(panels)} interfaces with top ipTM > {T}")
    print("interface\tn_poses\tn_acceptable\ttop_iptm\ttop_dockq")
    for ax, (i, keep, top) in zip(axes.flat, panels):
        acc = sum(r[2] >= se.ACCEPTABLE for r in keep)
        ax.set_facecolor(BG)
        ax.grid(color=GRID, linewidth=0.5, zorder=0)
        ax.axhline(se.ACCEPTABLE, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.scatter([r[3] for r in keep], [r[2] for r in keep], s=8, color=BLUE, alpha=0.5, edgecolors="none", zorder=2)
        ax.scatter([top[3]], [top[2]], s=60, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{i}  n={len(keep)}, acc={acc}", fontsize=7.5, color=INK, loc="left")
        ax.tick_params(labelsize=6.5, colors=MUTED)
        print(f"{i}\t{len(keep)}\t{acc}\t{top[3]:.3f}\t{top[2]:.3f}")
    fig.supxlabel("pair ipTM", fontsize=9, color=INK)
    fig.supylabel("DockQ vs crystal", fontsize=9, color=INK)
    fig.suptitle(f"{name}: {len(panels)} interfaces with top pair ipTM > {T}; every pose of the free-50 pool "
                 f"(ringed: top-ipTM pose; dashed: DockQ {se.ACCEPTABLE}; acc = poses >= {se.ACCEPTABLE})",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = f"{OUT}/scatter_high_iptm_{T}_{name}"
    fig.savefig(out + ".png", dpi=170, facecolor=BG)
    fig.savefig(out + ".pdf", facecolor=BG)
    print(f"-> {out}.png\n")
