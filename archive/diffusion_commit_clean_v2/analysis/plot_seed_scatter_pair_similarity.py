#!/usr/bin/env python
"""How similar are the poses one seed makes? Share of pose pairs within DockQ >= N, for N = 0.95 down to 0.05.

SUMMARY
    Every scored fold x interface of the seed_scatter_50 experiment has a 50 x 50 pose-to-pose DockQ
    matrix over one seed's 50 draws (seed_scatter_score.py; symmetrised, crystal-mapped chains). For each
    threshold N the plot shows the share of the 1,225 pose pairs with DockQ >= N, averaged over fold x
    interface rows (every row has the same number of pairs, so this equals the pooled share). Three lines:
      all seeds         every fold x interface row
      committed seeds   rows whose largest group (connected at DockQ >= 0.5) holds >= 50% of the draws
      uncommitted       rows whose largest group holds < 50%
    Pairs are WITHIN one seed only: poses from different seeds are not compared here.

INPUTS
    --exp:  the experiment. Default experimental_data/seed_scatter_50.
    --out:  png path (pdf and csv beside it). Default experimental_results/seed_scatter_50_pair_similarity.png.

OUTPUTS
    the figure; a csv with one row per threshold (share for each line, n rows); stdout: the same numbers

EXAMPLE
    analysis/plot_seed_scatter_pair_similarity.py

FUNCTIONS
    main():  read the matrices, tabulate, plot
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from seed_scatter_report import exp_dir, groups  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
COLORS = {"all seeds": "#2a78d6", "committed seeds": "#eb6834", "uncommitted seeds": "#1baf7a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=exp_dir("pxm2025"))
    ap.add_argument("--out", default=f"{ROOT}/experimental_results/seed_scatter_50_pair_similarity.png")
    A = ap.parse_args()

    thr = np.round(np.arange(0.95, 0.0, -0.05), 2)
    rows = {k: [] for k in COLORS}
    for f in sorted(glob.glob(f"{A.exp}/matrices/*.npz")):
        d = np.load(f)
        n = np.where(d["source"] == "new")[0]
        S = d["dockq"][np.ix_(n, n)]
        pairs = S[np.triu_indices(len(n), 1)]
        frac = np.array([(pairs >= t).mean() for t in thr])
        committed = np.bincount(groups(S)).max() / len(n) >= 0.5
        rows["all seeds"].append(frac)
        rows["committed seeds" if committed else "uncommitted seeds"].append(frac)
    mean = {k: np.mean(v, axis=0) for k, v in rows.items()}
    nrow = {k: len(v) for k, v in rows.items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for k, c in COLORS.items():
        ax.plot(thr, mean[k], color=c, lw=2, marker="o", ms=5, zorder=3, label=f"{k} (n = {nrow[k]})")
        ax.text(0.035, mean[k][-1], f"{k}: {mean[k][-1]:.0%}", color="#0b0b0b", ha="left", va="center",
                fontsize=9)
    ax.set_xlim(1.0, -0.22)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(thr[::2])
    ax.set_xlabel("N: pose-to-pose DockQ threshold (two poses count as similar if DockQ >= N)")
    ax.set_ylabel("share of pose pairs within one seed with DockQ >= N")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("How similar are the 50 poses one seed makes? (pxm2025 re-runs)\none point = share of a "
                 "seed's 1,225 pose pairs, averaged over fold x interface rows;\ncommitted = its largest group "
                 "of near-copies holds >= 50% of the draws", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    fig.savefig(A.out, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(os.path.splitext(A.out)[0] + ".pdf", facecolor=fig.get_facecolor())
    with open(os.path.splitext(A.out)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N"] + [f"{k} (n={nrow[k]})" for k in COLORS])
        for j, t in enumerate(thr):
            w.writerow([f"{t:.2f}"] + [f"{mean[k][j]:.4f}" for k in COLORS])
    print(f"  rows: {nrow}")
    for j, t in enumerate(thr):
        print(f"  N={t:.2f}  " + "  ".join(f"{k}: {mean[k][j]:.1%}" for k in COLORS))
    print(f"  wrote {A.out} (+ .pdf, .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
