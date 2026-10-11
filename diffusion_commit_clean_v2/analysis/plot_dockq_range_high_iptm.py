#!/usr/bin/env python
"""DockQ range across the free-50 pool (~250 poses) for every interface whose top pair ipTM > 0.8, sorted by top ipTM
(highest left). Thin line = min-max, thick bar = interquartile range, dot = the top-ipTM pose's DockQ. One panel per
dataset. Same pool rule as analysis/high_iptm_solved.py. Written 2026-10-02.
Usage: python analysis/plot_dockq_range_high_iptm.py -> experimental_results/dockq_range_high_iptm.{png,csv}"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import score_experiment as se, paths

T = 0.8
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experimental_results")
BLUE, ORANGE, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b68", "#e6e5e1", "#fcfcfb"

rows = []
for name in ("foldbench", "pxm2025"):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    cls = se.classes_at(pool, first)
    for i in cls:
        keep = [r for r in pool[i] if r[0] in first[i.split("_")[0]]]
        top = max(keep, key=lambda r: r[3])
        if top[3] <= T:
            continue
        d = np.array([r[2] for r in keep])
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        rows.append((name, i, cls[i], len(keep), top[3], top[2], d.min(), q1, med, q3, d.max()))

with open(os.path.join(OUT, "dockq_range_high_iptm.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "interface", "class", "poses", "top_iptm", "top_pose_dockq",
                "dockq_min", "dockq_q1", "dockq_median", "dockq_q3", "dockq_max"])
    w.writerows(rows)

fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor=BG)
for ax, name in zip(axes, ("foldbench", "pxm2025")):
    r = sorted([x for x in rows if x[0] == name], key=lambda x: -x[4])
    xs = np.arange(len(r))
    ax.set_facecolor(BG)
    ax.vlines(xs, [x[6] for x in r], [x[10] for x in r], color=BLUE, lw=1.2, alpha=0.6, zorder=2, label="min to max DockQ")
    ax.vlines(xs, [x[7] for x in r], [x[9] for x in r], color=BLUE, lw=4.5, zorder=3, label="middle 50% of poses")
    ax.scatter(xs, [x[5] for x in r], s=22, color=ORANGE, edgecolor=BG, linewidth=1, zorder=4, label="top-ipTM pose")
    ax.axhline(0.23, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.text(len(r) - 0.5, 0.235, "acceptable (0.23)", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(xs, [f"{x[1]}  {x[4]:.2f}" for x in r], rotation=90, fontsize=6.5)
    ax.set_xlim(-1, len(r))
    ax.set_ylim(0, 1)
    ax.set_ylabel("DockQ across the interface's poses", color=MUTED)
    ax.set_title(f"{'FoldBench' if name == 'foldbench' else 'pxm2025'}: {len(r)} interfaces with top ipTM > {T} "
                 f"(x labels: interface, top ipTM; sorted high to low)", color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    print(f"{name}: {len(r)} interfaces; top pose acceptable {sum(x[5] >= .23 for x in r)}; "
          f"min DockQ < 0.23 in {sum(x[6] < .23 for x in r)}; IQR > 0.1 in {sum(x[9]-x[7] > .1 for x in r)}")
axes[0].legend(frameon=False, loc="lower left", fontsize=9, ncol=3, labelcolor=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dockq_range_high_iptm.png"), dpi=150)
