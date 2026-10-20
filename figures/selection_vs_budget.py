#!/usr/bin/env python
"""Oracle vs ipTM-selected acceptable rate against candidate budget, on three benchmarks.

  oracle   = an acceptable pose (DockQ >= 0.23) is somewhere in the pool of N seeds
  selected = the pose ipTM actually picks is acceptable

The distance between them is the SELECTION GAP: interfaces the folding solved and the ranking then
lost. It is the entire headroom available to a confidence head; everything below the oracle curve
is a generation failure no reranking can reach.

ERROR BARS ARE CLUSTER BOOTSTRAP OVER COMPLEXES, not a binomial SEM over interfaces -- see
figures/pool_sweep.py for why, and --binomial-band to draw the naive one for comparison.

  figures/selection_vs_budget.py --out figures/selection_vs_budget.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_sweep as PS

# name, table, interface col, seed col, sample col, dockq col, iptm col, complex col
BENCH = [
    # NOT "Validation": that reads like a generic split, when it is specifically the held-out
    # partition of SAbDab2 this project trains against -- 53 structures, 92 interfaces -- and the
    # other two panels are named benchmarks.
    ("SAbDab2 partition", "extra_training/validation/fold/results.csv", "iface", "seed", "sample",
     "dockq", "iptm", None),
    ("FoldBench", "experimental_data/protenix/foldbench/foldbench_baseline_results.csv", "iface",
     "seed", "sample", "dockq", "pair_iptm", "pdb"),
    ("PXM-2025", "experimental_data/protenix/pxm2025/pxm2025_baseline_results.csv", "iface",
     "seed", "sample", "dockq", "pair_iptm", "pdb"),
]
COLORS = ["#1f77b4", "#444444"]          # selected, oracle
LABELS = ["ipTM-selected", "oracle (best pose in pool)"]
MARKERS = ["o", "^"]
MARKEVERY = [(0, 10), (5, 10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-budget", type=int, default=100)
    ap.add_argument("--draws", type=int, default=40,
                    help="seed subsets drawn per interface; the curve is an expectation over "
                         "WHICH N seeds you have, not over the first N by seed number")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="cluster bootstrap replicates. 2,000 rather than 400 because the\n                         bands are PERCENTILE intervals, which depend on the tails and need\n                         far more replicates than a standard deviation does.")
    ap.add_argument("--bands", action="store_true",
                    help="draw the 95%% confidence ribbons. OFF for the main figure and ON for the "
                         "supplementary one: the ribbons around the two CURVES are much wider than "
                         "the interval on their paired DIFFERENCE, because complex difficulty moves "
                         "both curves together and cancels in the difference. Overlapping ribbons "
                         "therefore read as 'the gap is uncertain' when the paired gap's 95%% "
                         "interval excludes zero on every benchmark. The uncertainty is not hidden "
                         "-- the gap's interval is annotated on every panel and the supplementary "
                         "figure carries the ribbons.")
    ap.add_argument("--binomial-band", action="store_true",
                    help="also draw the naive per-interface binomial SEM, for comparison")
    ap.add_argument("--out", default="figures/selection_vs_budget.png")
    a = ap.parse_args()

    series = []
    for name, rel, ic, sc, smc, dc, qc, pc in BENCH:
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print(f"{name}: {path} absent, skipped")
            continue
        sw = PS.add_gap(PS.sweep(path, ic, sc, smc, dc, qc, a.max_budget, a.draws,
                                 pdb_col=pc))
        series.append((name, sw))

    for name, sw in series:
        xs = sw["budgets"]
        sel, orc = PS.mean_curve(sw, "sel_acc"), PS.mean_curve(sw, "orc_acc")
        sel_lo, sel_hi = PS.cluster_ci(sw, "sel_acc", a.n_boot)
        orc_lo, orc_hi = PS.cluster_ci(sw, "orc_acc", a.n_boot)
        # THE GAP IS THE CLAIM, SO IT CARRIES ITS OWN PAIRED INTERVAL. Differencing two aggregate
        # curves and eyeballing their separate bands ignores that the two are measured on the SAME
        # interfaces and move together; the paired bootstrap keeps that pairing.
        gap = PS.mean_curve(sw, "gap_acc")
        gap_lo, gap_hi = PS.cluster_ci(sw, "gap_acc", a.n_boot)
        sel_b = PS.binomial_sem(sw, "sel_acc")
        seed_sd = PS.mean_seed_sd(sw, "seed_sd_acc")
        print(f"\n{name}: {len(sw['ifaces'])} interfaces over {len(set(sw['pdb']))} complexes, "
              f"{a.draws} seed draws, {a.n_boot} cluster-bootstrap replicates, 95% percentile CIs")
        print(f"{'N':>4}  {'selected':>22} {'oracle':>22}  {'paired gap':>22}  {'binomSEM':>8} "
              f"{'seedSD':>7}")
        for n in (1, 5, 10, 25, 50, 75, 100):
            if n > len(xs):
                continue
            j = n - 1
            print(f"{n:>4}  {sel[j]*100:>6.1f} [{sel_lo[j]*100:>5.1f},{sel_hi[j]*100:>5.1f}] "
                  f"{orc[j]*100:>6.1f} [{orc_lo[j]*100:>5.1f},{orc_hi[j]*100:>5.1f}]  "
                  f"{gap[j]*100:>6.1f} [{gap_lo[j]*100:>5.1f},{gap_hi[j]*100:>5.1f}]  "
                  f"{sel_b[j]*100:>8.1f} {seed_sd[j]*100:>7.1f}")

    # PAIRED GAPS, computed once, so the annotation and the printed table cannot disagree.
    gap_all = {}
    for name, sw in series:
        g = PS.mean_curve(sw, "gap_acc")
        glo, ghi = PS.cluster_ci(sw, "gap_acc", a.n_boot)
        gap_all[name] = (g, glo, ghi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, sw) in zip(axes, series):
        xs = sw["budgets"]
        curves = [(PS.mean_curve(sw, "sel_acc"), PS.cluster_ci(sw, "sel_acc", a.n_boot)),
                  (PS.mean_curve(sw, "orc_acc"), PS.cluster_ci(sw, "orc_acc", a.n_boot))]
        if a.bands:
            for (ms, (lo, hi)), c in zip(curves, COLORS):
                ax.fill_between(xs, [v * 100 for v in lo], [v * 100 for v in hi],
                                color=c, alpha=0.15, linewidth=0, zorder=1)
        if a.binomial_band:
            bs = PS.binomial_sem(sw, "sel_acc")
            ms = curves[0][0]
            ax.plot(xs, [(m - e) * 100 for m, e in zip(ms, bs)], color=COLORS[0], lw=0.8,
                    ls=":", zorder=2)
            ax.plot(xs, [(m + e) * 100 for m, e in zip(ms, bs)], color=COLORS[0], lw=0.8,
                    ls=":", zorder=2, label="binomial SEM (naive)")
        for i, ((ms, _ci), c, lbl) in enumerate(zip(curves, COLORS, LABELS)):
            ax.plot(xs, [m * 100 for m in ms], color=c, label=lbl, lw=1.9, ls="-",
                    marker=MARKERS[i], ms=4.0, markevery=MARKEVERY[i], zorder=5 - i)
        if a.bands:
            ylo = min(v * 100 for _ms, (lo_, _hi) in curves for v in lo_)
            yhi = max(v * 100 for _ms, (_lo, hi_) in curves for v in hi_)
        else:
            ylo = min(m * 100 for ms, _ci in curves for m in ms)
            yhi = max(m * 100 for ms, _ci in curves for m in ms)
        pad = max(1.0, (yhi - ylo) * 0.08)
        ax.set_ylim(max(0.0, ylo - pad), min(100.0, yhi + pad))
        ax.set_title(f"{name}  (n={len(sw['ifaces'])} interfaces, "
                     f"{len(set(sw['pdb']))} complexes)", fontsize=9.5)
        ax.set_xlabel("# of seeds")
        ax.set_xlim(1, a.max_budget)
        # >= rather than >: ACCEPTABLE is 0.23 and the comparison is dq >= ACCEPTABLE.
        ax.set_ylabel("DockQ ≥ 0.23 (%)")
        # THE PAIRED GAP AND ITS INTERVAL, on every panel. This is the quantity the paper is
        # about, and it is better determined than either curve: complex difficulty moves oracle
        # and selected together and cancels in their difference.
        g, glo, ghi = gap_all[name]
        j50 = min(len(xs), 50) - 1
        ax.annotate(f"selection gap @ 50 seeds\n{g[j50]*100:.1f} pp  "
                    f"[{glo[j50]*100:.1f}, {ghi[j50]*100:.1f}]",
                    xy=(0.96, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", lw=0.7))
        ax.grid(True, alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    tmp = a.out + ".tmp.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    os.replace(tmp, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
