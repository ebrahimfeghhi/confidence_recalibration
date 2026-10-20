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
    ap.add_argument("--n-boot", type=int, default=400, help="cluster bootstrap replicates")
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
        sw = PS.sweep(path, ic, sc, smc, dc, qc, a.max_budget, a.draws, pdb_col=pc)
        series.append((name, sw))

    for name, sw in series:
        xs = sw["budgets"]
        sel, orc = PS.mean_curve(sw, "sel_acc"), PS.mean_curve(sw, "orc_acc")
        sel_e = PS.cluster_sem(sw, "sel_acc", a.n_boot)
        orc_e = PS.cluster_sem(sw, "orc_acc", a.n_boot)
        sel_b = PS.binomial_sem(sw, "sel_acc")
        seed_sd = PS.mean_seed_sd(sw, "seed_sd_acc")
        print(f"\n{name}: {len(sw['ifaces'])} interfaces over {len(set(sw['pdb']))} complexes, "
              f"{a.draws} seed draws, {a.n_boot} cluster-bootstrap replicates")
        print(f"{'N':>4}  {'selected':>18} {'oracle':>18}  {'gap':>6}  {'binomSEM':>8} "
              f"{'seedSD':>7}")
        for n in (1, 5, 10, 25, 50, 75, 100):
            if n > len(xs):
                continue
            j = n - 1
            print(f"{n:>4}  {sel[j]*100:>8.1f}+/-{sel_e[j]*100:<7.1f} "
                  f"{orc[j]*100:>8.1f}+/-{orc_e[j]*100:<7.1f}  {(orc[j]-sel[j])*100:>6.1f}  "
                  f"{sel_b[j]*100:>8.1f} {seed_sd[j]*100:>7.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, sw) in zip(axes, series):
        xs = sw["budgets"]
        curves = [(PS.mean_curve(sw, "sel_acc"), PS.cluster_sem(sw, "sel_acc", a.n_boot)),
                  (PS.mean_curve(sw, "orc_acc"), PS.cluster_sem(sw, "orc_acc", a.n_boot))]
        for (ms, es), c in zip(curves, COLORS):
            ax.fill_between(xs, [(m - e) * 100 for m, e in zip(ms, es)],
                            [(m + e) * 100 for m, e in zip(ms, es)],
                            color=c, alpha=0.15, linewidth=0, zorder=1)
        if a.binomial_band:
            bs = PS.binomial_sem(sw, "sel_acc")
            ms = curves[0][0]
            ax.plot(xs, [(m - e) * 100 for m, e in zip(ms, bs)], color=COLORS[0], lw=0.8,
                    ls=":", zorder=2)
            ax.plot(xs, [(m + e) * 100 for m, e in zip(ms, bs)], color=COLORS[0], lw=0.8,
                    ls=":", zorder=2, label="binomial SEM (naive)")
        for i, ((ms, es), c, lbl) in enumerate(zip(curves, COLORS, LABELS)):
            ax.plot(xs, [m * 100 for m in ms], color=c, label=lbl, lw=1.9, ls="-",
                    marker=MARKERS[i], ms=4.0, markevery=MARKEVERY[i], zorder=5 - i)
        lo = min((m - e) * 100 for ms, es in curves for m, e in zip(ms, es))
        hi = max((m + e) * 100 for ms, es in curves for m, e in zip(ms, es))
        pad = max(1.0, (hi - lo) * 0.08)
        ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad))
        ax.set_title(f"{name}  (n={len(sw['ifaces'])} interfaces, "
                     f"{len(set(sw['pdb']))} complexes)", fontsize=9.5)
        ax.set_xlabel("# of seeds")
        ax.set_xlim(1, a.max_budget)
        # >= rather than >: ACCEPTABLE is 0.23 and the comparison is dq >= ACCEPTABLE.
        ax.set_ylabel("DockQ ≥ 0.23 (%)")
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
