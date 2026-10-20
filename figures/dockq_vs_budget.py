#!/usr/bin/env python
"""Mean oracle DockQ against mean ipTM-selected DockQ, per candidate budget.

THE CONTINUOUS COMPANION to selection_vs_budget.py, which thresholds at DockQ >= 0.23. A
thresholded curve cannot tell a pose that crossed 0.23 by a thousandth from one that reached 0.8,
and it is silent about whether the ranking deficit persists above the cutoff. This answers both:

  - if the means converge while the acceptable fractions still differ, the gap is concentrated at
    the threshold and is largely marginal crossings
  - if they stay apart across the range, ipTM is leaving pose QUALITY on the table everywhere

Shares its pool definition, selection rule, bootstrap and cluster-aware error bars with
selection_vs_budget.py through figures/pool_sweep.py, so the two cannot disagree about what a pool
or a pick is.

  figures/dockq_vs_budget.py --out figures/dockq_vs_budget.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_sweep as PS
from selection_vs_budget import BENCH

COLORS = ["#1f77b4", "#444444"]
LABELS = ["ipTM-selected", "oracle (best pose in pool)"]
MARKERS = ["o", "^"]
MARKEVERY = [(0, 10), (5, 10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-budget", type=int, default=100)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="cluster bootstrap replicates; percentile intervals need more than a standard deviation does")
    ap.add_argument("--bands", action="store_true",
                    help="draw the 95%% confidence ribbons; off for the main figure, on for the "
                         "supplementary one. Same reasoning as selection_vs_budget.py.")
    ap.add_argument("--out", default="figures/dockq_vs_budget.png")
    a = ap.parse_args()

    series = []
    for name, rel, ic, sc, smc, dc, qc, pc in BENCH:
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print(f"{name}: {path} absent, skipped")
            continue
        series.append((name, PS.add_gap(PS.sweep(path, ic, sc, smc, dc, qc, a.max_budget,
                                                 a.draws, pdb_col=pc))))

    for name, sw in series:
        xs = sw["budgets"]
        sel, orc = PS.mean_curve(sw, "sel_dq"), PS.mean_curve(sw, "orc_dq")
        sel_lo, sel_hi = PS.cluster_ci(sw, "sel_dq", a.n_boot)
        orc_lo, orc_hi = PS.cluster_ci(sw, "orc_dq", a.n_boot)
        # PAIRED, for the same reason as the thresholded figure: the two means are measured on the
        # same interfaces and move together.
        gap = PS.mean_curve(sw, "gap_dq")
        gap_lo, gap_hi = PS.cluster_ci(sw, "gap_dq", a.n_boot)
        seed_sd = PS.mean_seed_sd(sw, "seed_sd_dq")
        print(f"\n{name}: {len(sw['ifaces'])} interfaces over {len(set(sw['pdb']))} complexes, "
              f"{a.n_boot} replicates, 95% percentile CIs")
        print(f"{'N':>4}  {'selected':>22} {'oracle':>22}  {'paired gap':>22} {'seedSD':>7}")
        for n in (1, 5, 10, 25, 50, 75, 100):
            if n > len(xs):
                continue
            j = n - 1
            print(f"{n:>4}  {sel[j]:>6.3f} [{sel_lo[j]:>5.3f},{sel_hi[j]:>5.3f}] "
                  f"{orc[j]:>6.3f} [{orc_lo[j]:>5.3f},{orc_hi[j]:>5.3f}]  "
                  f"{gap[j]:>6.3f} [{gap_lo[j]:>5.3f},{gap_hi[j]:>5.3f}] {seed_sd[j]:>7.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, sw) in zip(axes, series):
        xs = sw["budgets"]
        curves = [(PS.mean_curve(sw, "sel_dq"), PS.cluster_ci(sw, "sel_dq", a.n_boot)),
                  (PS.mean_curve(sw, "orc_dq"), PS.cluster_ci(sw, "orc_dq", a.n_boot))]
        if a.bands:
            for (ms, (lo_, hi_)), c in zip(curves, COLORS):
                ax.fill_between(xs, lo_, hi_, color=c, alpha=0.15, linewidth=0, zorder=1)
        for i, ((ms, _ci), c, lbl) in enumerate(zip(curves, COLORS, LABELS)):
            ax.plot(xs, ms, color=c, label=lbl, lw=1.9, ls="-", marker=MARKERS[i], ms=4.0,
                    markevery=MARKEVERY[i], zorder=5 - i)
        if a.bands:
            ylo = min(v for _ms, (lo_, _h) in curves for v in lo_)
            yhi = max(v for _ms, (_l, hi_) in curves for v in hi_)
        else:
            ylo = min(m for ms, _ci in curves for m in ms)
            yhi = max(m for ms, _ci in curves for m in ms)
        pad = max(0.005, (yhi - ylo) * 0.08)
        ax.set_ylim(max(0.0, ylo - pad), min(1.0, yhi + pad))
        ax.set_title(f"{name}  (n={len(sw['ifaces'])} interfaces, "
                     f"{len(set(sw['pdb']))} complexes)", fontsize=9.5)
        ax.set_xlabel("# of seeds")
        ax.set_xlim(1, a.max_budget)
        ax.set_ylabel("mean DockQ")
        g = PS.mean_curve(sw, "gap_dq")
        glo, ghi = PS.cluster_ci(sw, "gap_dq", a.n_boot)
        j50 = min(len(xs), 50) - 1
        ax.annotate(f"quality gap @ 50 seeds\n{g[j50]:.3f}  [{glo[j50]:.3f}, {ghi[j50]:.3f}]",
                    xy=(0.96, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", lw=0.7))
        ax.grid(True, alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=2, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    tmp = a.out + ".tmp.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    os.replace(tmp, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
