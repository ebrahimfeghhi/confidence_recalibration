#!/usr/bin/env python
"""Figure 3: a seed that commits scores higher ipTM at the SAME true quality.

A seed produces one trunk; diffusion draws 5 poses from it. Group seeds by how tightly those 5
agree, then compare ipTM at matched DockQ. If ipTM were measuring correctness, the two groups
would score the same. They do not: a committed seed scores far higher, including when it is wrong.

WHY THIS MATTERS. Figure 2 shows the head is near-blind within a trunk -- 66-81% pairwise accuracy
at large |dDockQ| against 90-97% across trunks -- so what it actually scores is a TRUNK-level
property. Commitment is such a property, and it is rewarded regardless of correctness. That is a
generator for confident decoys: not random errors, but trunks that are certain and wrong, which is
the worst possible failure mode for an argmax selector.

THE CONTROL IS THE FIGURE. Comparing committed against diffuse seeds overall would be confounded
by quality, so every comparison here is WITHIN a narrow band of true mean DockQ. In the SAbDab2
0.05-0.10 band the diffuse group is slightly BETTER in DockQ (0.074 vs 0.066) and scores 0.40
LOWER in ipTM.

=============================================================================================
THE MEASURE IS A PROXY AND IT IS THE WRONG ONE. "Commitment" here is the standard deviation of
the five poses' DockQ -- and DockQ is pose-versus-REFERENCE. So this measures whether the samples
MISS THE TARGET BY SIMILAR AMOUNTS, not whether they are near each other. Five poses in two
distinct wrong basins that happen to score alike register as committed when they are not.

The right measure is pose-to-pose distance, never touching the reference: pose-vs-pose DockQ,
epitope Jaccard or FCC over the 5 samples. metrics.py on the reorg/declutter branch implements
all three on a common signature, but it needs the pose files rather than this score table.

A FLOOR EFFECT compounds it: in the 0.00-0.05 band spread cannot be large, so low spread is partly
forced, and that band holds 16,334 of PXM's seeds. What argues against the whole thing being an
artefact is that the effect is LARGEST away from the floor -- PXM's 0.10-0.16 band shows +0.55,
bigger than any near-zero band. But until the pose-to-pose version is run this figure is
suggestive, not settled.
=============================================================================================

  figures/commitment_vs_iptm.py --out figures/commitment_vs_iptm.png
"""
import argparse
import collections
import csv
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection_vs_budget import BENCH

BANDS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.16), (0.16, 0.23),
         (0.23, 0.35), (0.35, 0.49), (0.49, 0.80), (0.80, 1.01)]
TICKS = ["0–.05", ".05–.1", ".1–.16", ".16–.23", ".23–.35", ".35–.49", ".49–.8", ".8–1"]
COLORS = ["#c0392b", "#1f77b4"]
LABELS = ["committed (lowest-spread quartile)", "diffuse (highest-spread quartile)"]
ACCEPTABLE = 0.23


def units_for(path, ic, sc, dc, qc):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                by[r[ic]][r[sc]].append((float(r[qc]), float(r[dc])))
            except (ValueError, KeyError):
                continue
    out = []
    for ifc, seeds in by.items():
        for sd, poses in seeds.items():
            if len(poses) < 4:
                continue
            out.append((statistics.mean(p[0] for p in poses),      # mean ipTM
                        statistics.mean(p[1] for p in poses),      # mean DockQ (true quality)
                        statistics.pstdev([p[1] for p in poses]))) # spread PROXY -- see caveat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--min-band", type=int, default=80)
    ap.add_argument("--out", default="figures/commitment_vs_iptm.png")
    a = ap.parse_args()

    series = []
    for name, rel, ic, sc, smc, dc, qc, pc in BENCH:
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            continue
        series.append((name, units_for(path, ic, sc, dc, qc)))

    for name, units in series:
        print(f"\n=== {name}: {len(units)} (interface, seed) units ===")
        print("  %-12s %6s  %-26s %-26s %s" % ("DockQ band", "n", "committed", "diffuse", "d ipTM"))
        for (lo, hi) in BANDS:
            sub = [u for u in units if lo <= u[1] < hi]
            if len(sub) < a.min_band:
                continue
            sp = sorted(u[2] for u in sub)
            q1, q3 = sp[len(sp) // 4], sp[3 * len(sp) // 4]
            c = [u for u in sub if u[2] <= q1]
            d = [u for u in sub if u[2] >= q3]
            ci, cq = statistics.mean(u[0] for u in c), statistics.mean(u[1] for u in c)
            di, dq = statistics.mean(u[0] for u in d), statistics.mean(u[1] for u in d)
            print("  %-12s %6d  ipTM %.4f DockQ %.3f   ipTM %.4f DockQ %.3f   %+0.4f"
                  % (f"{lo:.2f}-{hi:.2f}", len(sub), ci, cq, di, dq, ci - di))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (name, units) in zip(axes, series):
        xs, cm, df = [], [], []
        for i, (lo, hi) in enumerate(BANDS):
            sub = [u for u in units if lo <= u[1] < hi]
            if len(sub) < a.min_band:
                continue
            sp = sorted(u[2] for u in sub)
            q1, q3 = sp[len(sp) // 4], sp[3 * len(sp) // 4]
            xs.append(i)
            cm.append(statistics.mean(u[0] for u in sub if u[2] <= q1))
            df.append(statistics.mean(u[0] for u in sub if u[2] >= q3))
        ax.plot(xs, cm, color=COLORS[0], lw=1.9, marker="o", ms=5, label=LABELS[0])
        ax.plot(xs, df, color=COLORS[1], lw=1.9, marker="s", ms=5, label=LABELS[1])
        # THE ACCEPTABLE BOUNDARY. Everything left of it is a wrong pose, and the vertical gap
        # there is confidence bought by commitment rather than by correctness.
        bi = next((i for i, (lo, hi) in enumerate(BANDS) if lo >= ACCEPTABLE), None)
        if bi is not None:
            ax.axvline(bi - 0.5, ls="--", lw=1.0, color="#888888")
            ax.annotate("acceptable →", xy=(bi - 0.45, 0.02), xycoords=("data", "axes fraction"),
                        fontsize=7.5, color="#888888", va="bottom")
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(TICKS, fontsize=7.5, rotation=45, ha="right")
        ax.set_xlabel("true mean DockQ of the seed's 5 poses")
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean ipTM")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=2, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    tmp = a.out + ".tmp.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    os.replace(tmp, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
