#!/usr/bin/env python
"""Plot run_logging/pose_pair_dockq.py: is a same-seed pose pair structurally closer than a
different-seed pose pair of the SAME quality gap?

The claim this tests is that within-seed selection fails because there is nothing to see -- two
poses of one seed being near-copies of each other. Pose-to-pose DockQ measures exactly that, with
no score involved, and the pairs are matched on |dDockQ| inside each chain pair so the two columns
differ only in whether the poses share a seed.

ROW A is the raw comparison, one box per condition. ROW B is the same thing paired: one
(within - across) difference per chain pair, which removes the between-chain-pair spread that
dominates row A and is the number the claim actually lives or dies on. A claim of "much more
similar" needs row B's box to sit well above zero.
"""
import argparse
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_WI, C_AC = "#2c7fb8", "#d95f0e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=f"{ROOT}/docs/pose_pair_dockq.csv")
    ap.add_argument("--min-pairs", type=int, default=4,
                    help="chain pairs need this many of BOTH kinds to enter the paired row")
    ap.add_argument("--out", default=f"{ROOT}/figures/pose_pair_dockq.png")
    a = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    rows = list(csv.DictReader(open(a.rows)))
    dss = list(dict.fromkeys(r["dataset"] for r in rows))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["kind"])].append(float(r["dockq_sym"]))
    gaps = collections.defaultdict(list)
    for r in rows:
        gaps[(r["dataset"], r["kind"])].append(float(r["gap"]))

    per_cp = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per_cp[(r["dataset"], r["structure"], r["interface"])][r["kind"]].append(
            float(r["dockq_sym"]))

    fig, axes = plt.subplots(2, len(dss), figsize=(5.6 * len(dss), 9.2), squeeze=False)
    out = []
    for j, ds in enumerate(dss):
        ax = axes[0][j]
        data = [np.array(by[(ds, k)]) for k in ("within", "across")]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for p, c in zip(bp["boxes"], (C_WI, C_AC)):
            p.set_facecolor(c)
            p.set_alpha(0.75)
        for m in bp["medians"]:
            m.set_color("black")
        ax.set_xticklabels(
            [f"same seed\nn={len(data[0])}\nmean |dDockQ| {np.mean(gaps[(ds,'within')]):.3f}",
             f"different seeds\nn={len(data[1])}\nmean |dDockQ| {np.mean(gaps[(ds,'across')]):.3f}"],
            fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{ds}\nA. pose-to-pose DockQ (symmetrised)", fontsize=15)
        if j == 0:
            ax.set_ylabel("DockQ of one pose against the other", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        for i, v in enumerate(data, 1):
            ax.text(i, 1.0, f"mean {v.mean():.3f}", ha="center", va="top", fontsize=12)

        ax = axes[1][j]
        d = [np.mean(v["within"]) - np.mean(v["across"])
             for k, v in per_cp.items()
             if k[0] == ds and len(v["within"]) >= a.min_pairs
             and len(v["across"]) >= a.min_pairs]
        d = np.array(d)
        bp = ax.boxplot([d], patch_artist=True, showfliers=False, widths=0.45)
        bp["boxes"][0].set_facecolor("#737373")
        bp["boxes"][0].set_alpha(0.7)
        bp["medians"][0].set_color("black")
        ax.axhline(0, color="black", lw=1.2)
        t = sps.ttest_1samp(d, 0.0) if d.size > 2 else None
        ax.set_xticklabels([f"chain pairs n={d.size}"], fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("B. paired, per chain pair\n(same seed - different seeds)", fontsize=15)
        if j == 0:
            ax.set_ylabel("difference in pose-to-pose DockQ", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        lab = f"mean {d.mean():+.4f}"
        if t is not None:
            lab += f"   t {t.statistic:+.2f}   p {t.pvalue:.1e}"
        ax.text(0.5, 0.97, lab, transform=ax.transAxes, ha="center", va="top", fontsize=12)
        out.append((ds, data[0].mean(), data[1].mean(), d.mean(),
                    t.statistic if t is not None else float("nan"),
                    t.pvalue if t is not None else float("nan"), d.size))

    fig.suptitle("Are same-seed poses more similar to each other, at an equal true-DockQ gap?",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"figure -> {a.out}\n")
    print(f"{'dataset':>12s} {'same seed':>10s} {'diff seeds':>11s} {'paired diff':>12s} "
          f"{'t':>7s} {'p':>10s} {'chain pairs':>12s}")
    for r in out:
        print(f"{r[0]:>12s} {r[1]:10.4f} {r[2]:11.4f} {r[3]:+12.4f} {r[4]:7.2f} "
              f"{r[5]:10.2e} {r[6]:12d}")


if __name__ == "__main__":
    main()
