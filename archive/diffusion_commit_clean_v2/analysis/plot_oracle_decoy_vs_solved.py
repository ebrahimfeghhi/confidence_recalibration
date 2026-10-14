#!/usr/bin/env python
"""Oracle generation on decoy vs solved interfaces: how good is the best pose the model generates?

SUMMARY
    From the crystal-scored free pool, first 50 seeds x 5 samples per complex (classes as
    score_experiment.classes_at): decoy = an acceptable pose exists but the top-ipTM pose is not; solved =
    the top-ipTM pose is acceptable. Per interface:
      oracle DockQ      the best crystal DockQ among its 250 poses (evaluation only: reads the crystal)
      share acceptable  the share of its 250 poses with DockQ >= 0.23
    Two panels; in each, one bar per class at the median, one dot per interface (jittered sideways).

INPUTS
    --dataset:  pxm2025 or foldbench. Default pxm2025.
    --out:      png path (pdf and csv beside it). Default experimental_results/oracle_decoy_vs_solved_<dataset>.png.

OUTPUTS
    the figure; a csv with one row per interface (class, oracle DockQ, share acceptable); stdout: medians, n

EXAMPLE
    analysis/plot_oracle_decoy_vs_solved.py

FUNCTIONS
    main():  read the pool, tabulate, plot
"""
import argparse
import csv
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, pool_by_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
CLASSES = [("decoy", "decoy\n(top-ipTM pose wrong,\nan acceptable pose exists)", "#eb6834"),
           ("solved", "solved\n(top-ipTM pose acceptable)", "#2a78d6")]
DEAD = {"9r10"}


def interface_rows(dataset):
    """out: [(iface, class, oracle DockQ, share acceptable, n poses, bestseed range, bestseed n)]

    Shared by both figures so the two cannot drift apart: same pool, same classes, same 50 seeds.
    Only decoy and solved are returned -- `nocov` interfaces (no acceptable pose anywhere in the
    250) have no oracle worth plotting against these two and are counted separately by the caller.

    TWO MEASURES OF WHETHER THAT FOLD COMMITTED, and they point in OPPOSITE directions:

    bestseed range   max minus min DockQ-against-the-crystal among the fold's 5 samples.
                     HIGH = the fold straddled both basins = hedged.
    bestseed p2p     mean pairwise POSE-TO-POSE DockQ among the same 5 samples, read from the stored
                     crystal-frame matrices (paths.DATASETS[name].matrices, never rebuilt, and never
                     the old_crystal_iptm set which is wrong by up to 0.83).
                     HIGH = the five poses resemble each other = committed. So the two metrics should
                     ANTI-correlate, and p2p is the honest one: DockQ-to-crystal spread cannot tell
                     two poses on opposite antigen faces apart, both scoring ~0.05.

    9n6e_GF seed 300046 is the motivating case: three samples at 0.024 and two at 0.69, a crystal
    range of 0.67, so the fold found the answer and hedged -- and its pair_iptm was flat to within
    0.004 across the whole split, which is why the selector never saw it.

    Matrix rows are labelled, so poses are matched by (seed, sample) and never by row order; poses
    with no antibody-antigen contact were dropped when the matrix was built, so a fold can contribute
    fewer than 5 and is skipped below 2. Those skips are counted, never silently dropped.
    """
    import glob as _glob
    pool, first = pool_by_seed(50, str(paths.DATASETS[dataset].pool_csv))
    cls = classes_at(pool, first)
    mdir = str(paths.DATASETS[dataset].matrices)
    rows, nocov, no_matrix = [], 0, []
    for iface, c in sorted(cls.items()):
        pdb = iface.split("_")[0]
        if pdb in DEAD:
            continue
        if c == "nocov":
            nocov += 1
            continue
        if c not in ("decoy", "solved"):
            continue
        keep = [r for r in pool[iface] if r[0] in first[pdb]]
        dq = [r[2] for r in keep]
        by_seed = {}
        for r in keep:
            by_seed.setdefault(r[0], []).append(r[2])
        best_seed = max(keep, key=lambda r: r[2])[0]
        sib = by_seed[best_seed]

        p2p, n_p2p = float("nan"), 0
        allp2p, n_seed_used, allpairs = float("nan"), 0, float("nan")
        mp = f"{mdir}/{iface}.npz"
        if os.path.exists(mp):
            z = np.load(mp, allow_pickle=True)
            seed_arr = [str(s) for s in z["seed"]]
            idx = [i for i, s in enumerate(seed_arr) if s == str(best_seed)]
            n_p2p = len(idx)
            if n_p2p >= 2:
                sub = z["dockq"][np.ix_(idx, idx)]
                iu = np.triu_indices(n_p2p, 1)
                p2p = float(sub[iu].mean())
            # UNCONDITIONAL version: every seed, not only the one that happened to succeed. The
            # best-pose seed was chosen post-hoc for having found the answer, which selects for folds
            # that straddled two basins; this averages over all 50 folds and has no such selection.
            per = []
            for s in sorted(set(seed_arr)):
                ii = [i for i, x in enumerate(seed_arr) if x == s]
                if len(ii) < 2:
                    continue
                sb = z["dockq"][np.ix_(ii, ii)]
                per.append(float(sb[np.triu_indices(len(ii), 1)].mean()))
            if per:
                allp2p, n_seed_used = float(np.mean(per)), len(per)
            # ALL PAIRS, ignoring fold boundaries: the whole 250x250 upper triangle. Comparing this
            # with the within-fold number says how much of the diversity is WITHIN a fold versus
            # BETWEEN folds.
            N = z["dockq"].shape[0]
            if N >= 2:
                allpairs = float(z["dockq"][np.triu_indices(N, 1)].mean())
        else:
            no_matrix.append(iface)
        rows.append((iface, c, max(dq), sum(d >= ACCEPTABLE for d in dq) / len(dq), len(dq),
                     max(sib) - min(sib), len(sib), p2p, n_p2p, allp2p, n_seed_used,
                     allpairs))
    if no_matrix:
        print(f"  {dataset}: NO stored matrix for {len(no_matrix)} interface(s): {no_matrix[:6]}")
    return rows, nocov


def main_hist(A):
    """Overlaid histograms of one metric, decoy vs solved, one panel per benchmark.

    DENSITY, not counts: solved outnumbers decoy 119:28 and 142:89, so raw counts would hide the
    decoy distribution entirely. Each class's bars sum to 1, so the shapes are comparable and the
    OVERLAP is what the eye reads -- which is the point, since the gap between means says nothing
    about whether the quantity could be used as a per-interface flag. The AUC printed on each panel
    is that question answered numerically: P(a random solved interface scores above a random decoy).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL, XLAB, TITLE = {
        "oracle": (2, "oracle DockQ (best of the interface's 250 free poses)",
                   "Oracle generation quality"),
        "bestseed_range": (5, "within-fold DockQ-to-crystal range, best-pose fold",
                           "Did the best-pose fold commit?"),
        "bestseed_p2p": (7, "within-fold pose-to-pose DockQ, best-pose fold",
                         "Did the best-pose fold commit?"),
        "allseed_p2p": (9, "within-fold pose-to-pose DockQ, averaged over all 50 folds",
                        "How self-consistent is a typical fold?"),
        "allpairs_p2p": (11, "all-pairs pose-to-pose DockQ over all 250 poses",
                         "How much does the whole pool agree?"),
    }[A.metric]
    A.out = A.out or f"{ROOT}/experimental_results/{A.metric}_hist_decoy_vs_solved.png"
    DSETS = ["foldbench", "pxm2025"]
    data = {d: interface_rows(d) for d in DSETS}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "xtick.color": "#0b0b0b",
                         "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), facecolor="#fcfcfb", sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, dset in zip(axes, DSETS):
        ax.set_facecolor("#fcfcfb")
        vals = {}
        for c, lab, color in CLASSES:
            v = np.array([r[COL] for r in data[dset][0] if r[1] == c], dtype=float)
            v = v[~np.isnan(v)]
            vals[c] = v
            w = np.ones_like(v) / len(v)                      # each class sums to 1
            ax.hist(v, bins=bins, weights=w, color=color, alpha=0.55, edgecolor=color,
                    linewidth=1.1, label=f"{c}  n = {len(v)}")
            ax.axvline(float(np.median(v)), color=color, ls="--", lw=1.4, zorder=5)
        a, b = vals["decoy"], vals["solved"]
        gt = (b[:, None] > a[None, :]).sum() + 0.5 * (b[:, None] == a[None, :]).sum()
        auc = gt / (len(a) * len(b))
        ax.set_title(f"{dset}\nAUC(solved > decoy) = {auc:.3f}", fontsize=10.5)
        ax.set_xlabel(XLAB, fontsize=9)
        ax.grid(axis="y", color="#e8e7e3", lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        print(f"  {dset}: AUC {auc:.3f}   decoy median {np.median(a):.3f} (n {len(a)})   "
              f"solved median {np.median(b):.3f} (n {len(b)})")
    axes[0].set_ylabel("share of that class's interfaces", fontsize=9.5)
    fig.suptitle(f"{TITLE}  --  x = {XLAB},  y = share of the class's interfaces\n"
                 "dashed line = that class's median;  each class normalised to sum to 1",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    fig.savefig(A.out, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(os.path.splitext(A.out)[0] + ".pdf", facecolor=fig.get_facecolor())
    print(f"  wrote {A.out} (+ .pdf)")
    return 0


def main_both(A):
    """One panel, four bars: each benchmark's decoy and solved side by side, bars at the MEAN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL, YLAB, TITLE, SHORT = {
        "oracle": (2, "oracle DockQ\n(best of the interface's 250 free poses)",
                   "Oracle generation quality, decoy vs solved interfaces", "oracle DockQ"),
        "bestseed_range": (5, "DockQ-to-crystal range (max - min) across the 5 samples\nof the fold "
                              "that made the interface's best pose\n(HIGH = the fold hedged)",
                           "Did the fold that found the best pose COMMIT to it?",
                           "within-fold DockQ-to-crystal range"),
        "allpairs_p2p": (11, "mean pairwise POSE-TO-POSE DockQ over ALL 250 poses\n(fold "
                             "boundaries ignored)\n(HIGH = the whole pool agrees)",
                         "How much do the interface's 250 poses agree overall?",
                         "all-pairs pose-to-pose DockQ"),
        "allseed_p2p": (9, "mean pairwise POSE-TO-POSE DockQ within a fold,\naveraged over ALL 50 folds "
                           "of the interface\n(HIGH = folds are internally consistent)",
                        "How self-consistent is a typical fold?", "within-fold pose-to-pose DockQ"),
        "bestseed_p2p": (7, "mean pairwise POSE-TO-POSE DockQ among the 5 samples\nof the fold that "
                            "made the interface's best pose\n(HIGH = the five poses agree = committed)",
                         "Did the fold that found the best pose COMMIT to it?",
                         "within-fold pose-to-pose DockQ"),
    }[A.metric]
    STAT = {"mean": np.nanmean, "median": np.nanmedian}[A.stat]
    A.out = A.out or f"{ROOT}/experimental_results/{A.metric}_{A.stat}_decoy_vs_solved_both.png"
    DSETS = ["foldbench", "pxm2025"]
    data = {d: interface_rows(d) for d in DSETS}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "xtick.color": "#0b0b0b",
                         "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    rng = np.random.default_rng(0)
    # x layout: the two classes sit adjacent WITHIN a benchmark and the benchmarks are separated by a
    # gap, so the eye compares decoy-to-solved first (the contrast the figure is about) and
    # benchmark-to-benchmark second.
    xs, ticks, labels, out = [], [], [], []
    for gi, dset in enumerate(DSETS):
        rows, nocov = data[dset]
        for ci, (c, _, color) in enumerate(CLASSES):
            x = gi * 2.6 + ci * 0.85
            v = np.array([r[COL] for r in rows if r[1] == c], dtype=float)
            v = v[~np.isnan(v)]                         # interfaces whose best fold had <2 matrix rows
            mean = float(STAT(v))
            ax.bar(x, mean, width=0.68, color=color, alpha=0.25, edgecolor=color, linewidth=1.5,
                   zorder=1)
            ax.scatter(x + rng.uniform(-0.24, 0.24, len(v)), v, s=20, color=color,
                       edgecolor="#fcfcfb", linewidth=0.7, alpha=0.9, zorder=3)
            ax.text(x, mean + 0.035, f"{mean:.2f}", ha="center", color="#0b0b0b", fontsize=9.5,
                    fontweight="bold", zorder=4)
            ticks.append(x)
            labels.append(f"{c}\nn = {len(v)}")
            xs.append(x)
            out.append((dset, c, len(v), mean, float(np.median(v)), nocov))
        ax.text(gi * 2.6 + 0.425, -0.145, dset, ha="center", va="top", fontsize=11,
                fontweight="bold", color="#0b0b0b", transform=ax.get_xaxis_transform())
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.65, DSETS.index(DSETS[-1]) * 2.6 + 1.5)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(YLAB, fontsize=10)
    if A.metric == "oracle":
        ax.axhline(ACCEPTABLE, ls="--", lw=1, color="#a3a29d", zorder=0)
        ax.text(ax.get_xlim()[1] - 0.05, ACCEPTABLE + 0.012, "acceptable (0.23)", ha="right",
                va="bottom", fontsize=8, color="#52514e")
    ax.grid(axis="y", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle(f"{TITLE}\nx = interface class within benchmark,  y = {SHORT}\n"
                 f"one dot = one interface,  bar = {A.stat.upper()} across interfaces", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    fig.savefig(A.out, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(os.path.splitext(A.out)[0] + ".pdf", facecolor=fig.get_facecolor())
    with open(os.path.splitext(A.out)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "iface", "class", "oracle_dockq", "share_acceptable", "n_poses",
                    "bestseed_dockq_range", "bestseed_n_poses", "bestseed_p2p", "bestseed_p2p_n",
                    "allseed_p2p", "allseed_n_folds"])
        for dset in DSETS:
            w.writerows([dset, r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", r[4], f"{r[5]:.4f}", r[6],
                         f"{r[7]:.4f}", r[8], f"{r[9]:.4f}", r[10]] for r in data[dset][0])
    print(f"{'dataset':11}{'class':8}{'n iface':>9}{'mean':>10}{'median':>10}"
          f"{'n nocov excluded':>18}   metric={A.metric} bar={A.stat}")
    for dset, c, n, mean, med, nocov in out:
        print(f"{dset:11}{c:8}{n:>9}{mean:>10.3f}{med:>10.3f}{nocov:>18}")
    print(f"  wrote {A.out} (+ .pdf, .csv)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stat", choices=("mean", "median"), default="mean",
                    help="--both only: what the BAR shows. The dots are always every interface.")
    ap.add_argument("--metric", choices=("oracle", "bestseed_range", "bestseed_p2p",
                                        "allseed_p2p", "allpairs_p2p"), default="oracle",
                    help="--both only. oracle = best pose in the 250. bestseed_range = max-min "
                         "DockQ across the 5 samples of the fold that produced that best pose, i.e. "
                         "whether that fold committed to its answer or hedged. bestseed_p2p = "
                         "mean pairwise pose-to-pose DockQ among those same 5 samples, from the "
                         "stored matrices; HIGH means they agree, the opposite sense to the range.")
    ap.add_argument("--hist", action="store_true",
                    help="overlaid decoy/solved histograms of --metric, one panel per benchmark, "
                         "each class normalised to sum to 1, with the AUC printed. Use this to read "
                         "the OVERLAP; --both shows the gap between class averages.")
    ap.add_argument("--both", action="store_true",
                    help="ONE panel, both benchmarks side by side: four bars (foldbench decoy, "
                         "foldbench solved, pxm2025 decoy, pxm2025 solved). Bar = MEAN across "
                         "interfaces, one dot = one interface's oracle DockQ. The default two-panel "
                         "per-dataset figure is unchanged and bars there stay at the median.")
    A = ap.parse_args()
    if A.hist:
        return main_hist(A)
    if A.both:
        return main_both(A)
    A.out = A.out or f"{ROOT}/experimental_results/oracle_decoy_vs_solved_{A.dataset}.png"

    pool, first = pool_by_seed(50, str(paths.DATASETS[A.dataset].pool_csv))
    cls = classes_at(pool, first)
    rows = []
    for iface, c in sorted(cls.items()):
        pdb = iface.split("_")[0]
        if c not in ("decoy", "solved") or pdb in DEAD:
            continue
        dq = [r[2] for r in pool[iface] if r[0] in first[pdb]]
        rows.append((iface, c, max(dq), sum(d >= ACCEPTABLE for d in dq) / len(dq), len(dq)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "xtick.color": "#0b0b0b",
                         "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), facecolor="#fcfcfb")
    rng = np.random.default_rng(0)
    panels = [(2, "oracle DockQ:\nbest pose among the interface's 250 poses", False),
              (3, "share of the interface's 250 poses\nthat are acceptable (DockQ >= 0.23)", True)]
    for ax, (col, ylab, pct) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for x, (c, lab, color) in enumerate(CLASSES):
            v = np.array([r[col] for r in rows if r[1] == c])
            med = float(np.median(v))
            ax.bar(x, med, width=0.6, color=color, alpha=0.25, edgecolor=color, linewidth=1.5, zorder=1)
            ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=22, color=color, edgecolor="#fcfcfb",
                       linewidth=0.8, alpha=0.9, zorder=3)
            ax.text(x + 0.33, med, f"median {med:.0%}" if pct else f"median {med:.2f}", va="center",
                    color="#0b0b0b", fontsize=9)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([f"{lab}\nn = {sum(r[1] == c for r in rows)}" for c, lab, _ in CLASSES], fontsize=9)
        ax.set_xlim(-0.6, 1.9)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(ylab, fontsize=9.5)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.axhline(ACCEPTABLE, ls="--", lw=1, color="#a3a29d", zorder=0)
            ax.text(1.85, ACCEPTABLE + 0.01, "acceptable (0.23)", ha="right", va="bottom", fontsize=8,
                    color="#52514e")
        ax.grid(axis="y", color="#e8e7e3", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"What the model generates on decoy vs solved interfaces ({A.dataset}, first 50 seeds x 5 "
                 "poses per interface; bar = median, one dot = one interface)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    fig.savefig(A.out, dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(os.path.splitext(A.out)[0] + ".pdf", facecolor=fig.get_facecolor())
    with open(os.path.splitext(A.out)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iface", "class", "oracle_dockq", "share_acceptable", "n_poses"])
        w.writerows([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", r[4]] for r in rows)
    for c, _, _ in CLASSES:
        v = [r for r in rows if r[1] == c]
        print(f"  {c}: n={len(v)} interfaces, {len({r[0].split('_')[0] for r in v})} complexes | oracle DockQ "
              f"median {st.median(r[2] for r in v):.2f} | share acceptable median {st.median(r[3] for r in v):.1%}"
              f" | oracle >= 0.49 (medium): {sum(r[2] >= 0.49 for r in v)}")
    print(f"  wrote {A.out} (+ .pdf, .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
