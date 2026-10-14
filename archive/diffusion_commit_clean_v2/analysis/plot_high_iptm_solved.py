#!/usr/bin/env python
"""Bar chart: for top-ipTM thresholds 0.95 down to 0.60 (step 0.05), how many interfaces have top free-50 pair ipTM
above the threshold (orange, total) and how many of those are solved (blue, drawn over the total). FoldBench and
pxm2025 side by side. Same pool and class rule as analysis/high_iptm_solved.py (score_experiment.pool_by_seed /
classes_at). Written 2026-10-02.
Usage: python analysis/plot_high_iptm_solved.py   -> experimental_results/high_iptm_solved.png + .csv"""
import collections, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import score_experiment as se, paths

THRESHOLDS = [round(0.95 - 0.05 * k, 2) for k in range(8)]  # 0.95 ... 0.60
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experimental_results")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b68", "#e6e5e1"
LABEL = {"foldbench": "FoldBench", "pxm2025": "pxm2025"}

rows = []
for name in ("foldbench", "pxm2025"):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    cls = se.classes_at(pool, first)
    top = {}
    for i in cls:
        keep = [r for r in pool[i] if r[0] in first[i.split("_")[0]]]
        top[i] = max(keep, key=lambda r: r[3])
        assert (top[i][2] >= se.ACCEPTABLE) == (cls[i] == "solved")
    for t in THRESHOLDS:
        hi = [i for i in cls if top[i][3] > t]
        rows.append((name, t, len(hi), sum(cls[i] == "solved" for i in hi), len(cls)))

with open(os.path.join(OUT, "high_iptm_solved.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "top_iptm_threshold", "interfaces_above", "solved_above", "interfaces_in_dataset"])
    w.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#fcfcfb")
for ax, name in zip(axes, ("foldbench", "pxm2025")):
    r = [x for x in rows if x[0] == name]
    xs = range(len(r))
    ax.set_facecolor("#fcfcfb")
    ax.bar(xs, [x[2] for x in r], width=0.62, color=ORANGE, label="all interfaces above threshold", zorder=2)
    ax.bar(xs, [x[3] for x in r], width=0.62, color=BLUE, label="solved (top-ipTM pose DockQ ≥ 0.23)", zorder=3)
    ymax = max(x[2] for x in r)
    for k, x in enumerate(r):
        ax.text(k, x[2] + ymax * 0.015, f"{x[3]}/{x[2]}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(xs), [f"> {x[1]:.2f}" for x in r])
    ax.set_xlabel("Top pair ipTM of the interface (free 50 seeds)", color=MUTED)
    ax.set_ylabel("Number of interfaces", color=MUTED)
    ax.set_title(f"{LABEL[name]} ({r[0][4]} interfaces)", color=INK, loc="left", fontsize=12)
    ax.set_ylim(0, ymax * 1.12)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
axes[0].legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
fig.suptitle("Interfaces whose top ipTM clears a threshold: how many are solved", color=INK, fontsize=13, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "high_iptm_solved.png"), dpi=160)
for x in rows:
    print(*x)
