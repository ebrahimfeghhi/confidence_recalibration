#!/usr/bin/env python
"""Per-interface plot: how tight is the decoy seed vs the good seed, on interfaces still decoys after refolding?

SUMMARY
    Interfaces: pxm2025 decoy interfaces (top-ipTM pose of the first 50 seeds wrong, an acceptable pose
    exists) where, pooling every refolded seed's 50 draws, a wrong draw still out-scores every acceptable
    draw on ipTM. Per interface, two seeds:
      decoy seed  the seed of the original top-ipTM (wrong) pose, refolded with 50 draws
      good seed   among the interface's other refolded seeds, the one with the most acceptable draws
    Three panels, one dot per interface, x = good seed, y = decoy seed, dashed line = equal:
      agreement            mean pairwise DockQ over the seed's 50 draws
      largest group share  share of the 50 draws in the largest group (connected at DockQ >= 0.5)
      share correct        share of the 50 draws with crystal DockQ >= 0.23
    A dot above the line: the decoy seed is higher on that metric. EVALUATION ONLY.

INPUTS
    --exp:  the experiment. Default experimental_data/seed_scatter_50.
    --out:  png path (a pdf is written beside it). Default experimental_results/seed_scatter_50_decoy_vs_good.png.

OUTPUTS
    the figure; a csv beside it with one row per interface (the plotted values); stdout: n and counts

EXAMPLE
    analysis/plot_seed_scatter_decoy_vs_good.py

FUNCTIONS
    tight():  one (interface, seed) -> agreement, largest group share
    main():   pick interfaces and seeds, plot
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from seed_scatter_report import exp_dir, groups  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
DECOY, GOOD = "#eb6834", "#2a78d6"          # reference palette slots 2 and 1, validated together


def tight(exp, iface, seed):
    """in: experiment, interface, seed.  out: (agreement, largest group share) over its 50 new draws"""
    d = np.load(f"{exp}/matrices/{iface.split('_')[0]}_{seed}_{iface}.npz")
    n = np.where(d["source"] == "new")[0]
    S = d["dockq"][np.ix_(n, n)]
    return float(S[np.triu_indices(len(n), 1)].mean()), float(np.bincount(groups(S)).max() / len(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=exp_dir("pxm2025"))
    ap.add_argument("--out", default=f"{ROOT}/experimental_results/seed_scatter_50_decoy_vs_good.png")
    A = ap.parse_args()

    dec = {r["iface"]: r["seed"] for r in csv.DictReader(open(f"{A.exp}/selection.csv"))
           if r["pick"] == "decoy_top_iptm"}
    res = {}
    for r in csv.DictReader(open(f"{A.exp}/results.csv")):
        if r["dockq"] and r["iptm"]:
            res.setdefault((r["iface"], r["seed"]), []).append((float(r["iptm"]), float(r["dockq"])))
    rows = []
    for i, ds in sorted(dec.items()):
        seeds = [s for (j, s) in res if j == i]
        draws = [x for s in seeds for x in res[(i, s)]]
        acc = [p for p, q in draws if q >= 0.23]
        bad = [p for p, q in draws if q < 0.23]
        if not (acc and bad and max(bad) > max(acc)) or ds not in seeds:
            continue
        gs = max((s for s in seeds if s != ds), key=lambda s: sum(q >= 0.23 for _, q in res[(i, s)]))
        da, dg = tight(A.exp, i, ds)
        ga, gg = tight(A.exp, i, gs)
        dc = sum(q >= 0.23 for _, q in res[(i, ds)]) / len(res[(i, ds)])
        gc = sum(q >= 0.23 for _, q in res[(i, gs)]) / len(res[(i, gs)])
        rows.append(dict(iface=i, decoy_seed=ds, good_seed=gs, decoy_agreement=da, good_agreement=ga,
                         decoy_largest_group=dg, good_largest_group=gg, decoy_correct=dc, good_correct=gc))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    panels = [("agreement", "agreement among the seed's 50 poses\n(mean pairwise DockQ)"),
              ("largest_group", "share of the 50 poses in the seed's largest group\n(near-copies, DockQ >= 0.5)"),
              ("correct", "share of the 50 poses that are correct\n(crystal DockQ >= 0.23)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.3), facecolor="#fcfcfb")
    n = len(rows)
    for ax, (k, label) in zip(axes, panels):
        x = np.array([r[f"good_{k}"] for r in rows])
        y = np.array([r[f"decoy_{k}"] for r in rows])
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#a3a29d", zorder=1)
        ax.scatter(x, y, s=46, color=DECOY, edgecolor="#fcfcfb", linewidth=1.2, alpha=0.85, zorder=2)
        up, down = int((y > x).sum()), int((y < x).sum())
        ax.set_title(f"above the line (decoy seed higher): {up}    below (good seed higher): {down}",
                     fontsize=9.5, color="#52514e")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_xlabel(f"good seed: {label}")
        ax.set_ylabel("decoy seed: same measure")
        ax.grid(color="#e8e7e3", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Decoy seed vs good seed, per interface (pxm2025, n = {n} interfaces still decoys after "
                 "refolding; one dot = one interface; each seed refolded with 50 draws)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    fig.savefig(A.out, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(os.path.splitext(A.out)[0] + ".pdf", facecolor=fig.get_facecolor())
    with open(os.path.splitext(A.out)[0] + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print(f"  n = {n} interfaces; wrote {A.out} (+ .pdf, .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
