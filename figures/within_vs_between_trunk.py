#!/usr/bin/env python
"""Figure 2: ipTM ranks poses across trunks far better than within one.

THE MECHANISM BEHIND THE SELECTION GAP. Figure 1 shows ipTM leaves 14-28 points of acceptable
interfaces on the table. This asks where that failure lives.

A seed produces ONE trunk representation, and diffusion then draws 5 poses from it. So:

  within-trunk  pairs = two of the 5 samples of one seed -- same representation, different pose
  between-trunk pairs = poses from different seeds -- different representation

The confidence head sees the pose only through a cdist over representative atoms, added into
z_pair once before the pairformer stack. Everything else it reads is the trunk. If that single
injection is weak relative to the trunk signal, the head should rank across trunks well and within
one badly -- which is what this measures.

THE CONDITIONING IS THE POINT. Raw within-trunk correlation is low partly because within-trunk
DockQ variance is itself small (0.002-0.005 against 0.016-0.029 between), so there is less to
rank. Splitting by |dDockQ| removes that confound: among pairs that differ by 0.4+ DockQ, is the
higher-ipTM pose the better one?

WHAT IT SHOWS, AND WHAT IT DOES NOT. At |dDockQ| >= 0.4 the head is right 90-97% of the time
across trunks and 66-81% within one. That is a large, consistent deficit -- but 66-81% is well
above chance, so the head DOES use the pose. The supportable claim is that its pose sensitivity is
much weaker than its trunk sensitivity, not that it ignores coordinates. At small |dDockQ| the
within-trunk accuracy is slightly HIGHER than between-trunk, because between-trunk pairs are
confounded by trunk-level ipTM offsets that do not track small quality differences.

  figures/within_vs_between_trunk.py --out figures/within_vs_between_trunk.png
"""
import argparse
import collections
import csv
import itertools
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection_vs_budget import BENCH

BINS = [(0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]
LABELS = ["0.05–0.1", "0.1–0.2", "0.2–0.4", "≥0.4"]
COLORS = ["#c0392b", "#1f77b4"]       # within-trunk, between-trunk
SERIES = ["within trunk (same seed)", "between trunks (different seeds)"]


def accuracy(path, ic, sc, dc, qc, max_between_pairs=3000, rng_seed=0):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                by[r[ic]][r[sc]].append((float(r[qc]), float(r[dc])))
            except (ValueError, KeyError):
                continue
    W = collections.defaultdict(lambda: [0, 0])
    B = collections.defaultdict(lambda: [0, 0])
    rng = random.Random(rng_seed)
    for ifc, seeds in by.items():
        seeds = {k: v for k, v in seeds.items() if len(v) >= 2}
        if len(seeds) < 2:
            continue
        for sd, poses in seeds.items():
            for a, b in itertools.combinations(poses, 2):
                d = abs(a[1] - b[1])
                for k in BINS:
                    if k[0] <= d < k[1]:
                        W[k][1] += 1
                        W[k][0] += (a[0] > b[0]) == (a[1] > b[1])
                        break
        # ONE POSE PER SEED, so a between-trunk pair cannot share a representation. Taking all
        # poses would mix the two populations and the comparison would mean nothing.
        reps = [v[0] for v in seeds.values()]
        pairs = list(itertools.combinations(reps, 2))
        if len(pairs) > max_between_pairs:
            pairs = rng.sample(pairs, max_between_pairs)
        for a, b in pairs:
            d = abs(a[1] - b[1])
            for k in BINS:
                if k[0] <= d < k[1]:
                    B[k][1] += 1
                    B[k][0] += (a[0] > b[0]) == (a[1] > b[1])
                    break
    return W, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures/within_vs_between_trunk.png")
    a = ap.parse_args()

    series = []
    for name, rel, ic, sc, smc, dc, qc, pc in BENCH:
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            continue
        series.append((name,) + accuracy(path, ic, sc, dc, qc))

    for name, W, B in series:
        print(f"\n=== {name}: is the higher-ipTM pose the better one? ===")
        print("  %-12s %-24s %-24s" % ("|dDockQ|", "WITHIN trunk", "BETWEEN trunks"))
        for k, lbl in zip(BINS, LABELS):
            w, b = W[k], B[k]
            print("  %-12s %-24s %-24s" % (
                lbl,
                "%.3f (n=%6d)" % (w[0] / w[1], w[1]) if w[1] > 50 else "--",
                "%.3f (n=%6d)" % (b[0] / b[1], b[1]) if b[1] > 50 else "--"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    x = range(len(BINS))
    for ax, (name, W, B) in zip(axes, series):
        for src, c, lbl in ((W, COLORS[0], SERIES[0]), (B, COLORS[1], SERIES[1])):
            ys = [src[k][0] / src[k][1] if src[k][1] > 50 else float("nan") for k in BINS]
            # WILSON-ish binomial SE on each point; these are pair counts in the tens of
            # thousands, so the bars are small and the comparison is not close.
            es = [((p * (1 - p) / src[k][1]) ** 0.5 if src[k][1] > 50 else 0.0)
                  for p, k in zip(ys, BINS)]
            ax.errorbar(x, ys, yerr=es, color=c, label=lbl, lw=1.9, marker="o", ms=5,
                        capsize=3)
        ax.axhline(0.5, ls=":", lw=1.0, color="#888888")
        ax.annotate("chance", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
                    fontsize=7.5, color="#888888", va="bottom")
        ax.set_xticks(list(x))
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("|ΔDockQ| between the two poses")
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("P(higher ipTM = higher DockQ)")
    axes[0].set_ylim(0.45, 1.0)
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
