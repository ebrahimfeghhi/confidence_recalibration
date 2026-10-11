#!/usr/bin/env python
"""Within-interface Spearman correlation of pair ipTM vs DockQ across the free-50 pool (~250 poses), grouped into
buckets by the interface's top pair ipTM. Bar = median over the bucket's interfaces, whiskers = interquartile range.
Also reports each bucket's median DockQ spread (IQR of DockQ within the pool), since a pool with near-identical
poses has little to correlate. Same pool rule as analysis/high_iptm_solved.py. Written 2026-10-02.
Usage: python analysis/plot_iptm_dockq_corr_by_bucket.py -> experimental_results/iptm_dockq_corr_by_bucket.{png,csv}"""
import csv, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import score_experiment as se, paths

EDGES = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]  # bucket k = (EDGES[k], EDGES[k-1]], top bucket > 0.95
LABELS = ["> 0.95"] + [f"{lo:.2f}–{hi:.2f}" for hi, lo in zip(EDGES, EDGES[1:])] + ["≤ 0.60"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experimental_results")
BLUE, INK, MUTED, GRID, BG = "#2a78d6", "#1f1f1e", "#6b6b68", "#e6e5e1", "#fcfcfb"


def bucket(t):
    for k, e in enumerate(EDGES):
        if t > e:
            return k
    return len(EDGES)


rows, per = [], {}
for name in ("foldbench", "pxm2025"):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    cls = se.classes_at(pool, first)
    for i in cls:
        keep = [r for r in pool[i] if r[0] in first[i.split("_")[0]]]
        d = np.array([r[2] for r in keep]); p = np.array([r[3] for r in keep])
        rho = spearmanr(p, d).statistic if d.std() > 0 and p.std() > 0 else float("nan")
        q1, q3 = np.percentile(d, [25, 75])
        rows.append((name, i, cls[i], len(keep), float(p.max()), bucket(float(p.max())), rho, float(q3 - q1)))

with open(os.path.join(OUT, "iptm_dockq_corr_by_bucket.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "interface", "class", "poses", "top_iptm", "bucket", "spearman_iptm_dockq", "dockq_iqr"])
    w.writerows([r[:5] + (LABELS[r[5]],) + r[6:] for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
for ax, name in zip(axes, ("foldbench", "pxm2025")):
    ax.set_facecolor(BG)
    med, lo, hi, ns, spread = [], [], [], [], []
    for k in range(len(LABELS)):
        v = [r[6] for r in rows if r[0] == name and r[5] == k and not np.isnan(r[6])]
        s = [r[7] for r in rows if r[0] == name and r[5] == k]
        ns.append(len(v)); spread.append(st.median(s) if s else float("nan"))
        if v:
            a, m, b = np.percentile(v, [25, 50, 75]); med.append(m); lo.append(m - a); hi.append(b - m)
        else:
            med.append(0); lo.append(0); hi.append(0)
    xs = np.arange(len(LABELS))
    ax.bar(xs, med, width=0.62, color=BLUE, zorder=3)
    ax.errorbar(xs, med, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    for k in xs:
        if ns[k]:
            y = med[k] + hi[k] if med[k] >= 0 else med[k] - lo[k]
            ax.text(k, y + (0.03 if med[k] >= 0 else -0.03), f"n={ns[k]}", ha="center",
                    va="bottom" if med[k] >= 0 else "top", fontsize=8, color=MUTED)
        print(f"{name:9s} {LABELS[k]:10s} n={ns[k]:3d} median rho={med[k]:+.2f} median DockQ IQR={spread[k]:.3f}")
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xticks(xs, LABELS, rotation=35, ha="right")
    ax.set_ylim(-0.6, 1.1)
    ax.set_xlabel("Top pair ipTM of the interface (free 50 seeds)", color=MUTED)
    ax.set_ylabel("Spearman correlation, ipTM vs DockQ, within interface", color=MUTED)
    ax.set_title(f"{'FoldBench' if name == 'foldbench' else 'pxm2025'}", color=INK, loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
fig.suptitle("How well ipTM ranks poses within an interface, by the interface's top ipTM (bar = median, whiskers = middle 50%)",
             color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iptm_dockq_corr_by_bucket.png"), dpi=160)
nan = sum(np.isnan(r[6]) for r in rows)
print(f"interfaces {len(rows)}; correlation undefined (constant DockQ or ipTM) for {nan}")
