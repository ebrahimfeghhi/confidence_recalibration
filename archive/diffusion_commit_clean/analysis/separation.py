#!/usr/bin/env python
"""Do good poses look like each other? One number per interface per metric, with no knobs.

THE IDEA. Label every pose good or bad by ground-truth DockQ, then split all pairs into three buckets
-- good-good, good-bad, bad-bad -- and average the similarity in each. If good poses form a coherent
group, good-good is far above good-bad. This is a ceiling measurement, not a selector: it uses the
answer to define the buckets. It says whether the signal a clustering could exploit is there at all,
which is worth establishing before choosing a clustering algorithm, a K, or a threshold -- any of which
could manufacture or destroy an effect.

WHY SPLIT BY INTERFACE CLASS. On a solved interface the model already found the right answer
repeatedly, so of course its good poses agree; separation there is nearly guaranteed and tells us
little. The case that decides whether clustering is useful is the confident decoy, where a good pose
exists and pooled ipTM picks a bad one. A metric that separates only on solved interfaces would be
useless exactly where it is needed, and an average over both would hide that.

    separation.py [--matrices matrices] [--out sep]

Writes <out>.csv (one row per interface per metric) and <out>.png.
"""
import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402
from fmt import table  # noqa: E402

ACC = 0.23
METRICS = ("epitope", "fcc", "dockq")


def pool_classes():
    """{interface: solved|decoy|nocov} by what POOLED ipTM selects.

    Pooled across seeds, not per seed: a confident decoy is defined by the most confident pose in the
    whole pool being wrong while a good pose exists, which per-seed selection cannot see.
    """
    best = collections.defaultdict(lambda: (-1.0, 0.0))
    anygood = collections.defaultdict(bool)
    with open(str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            k = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
            ip, dq = float(r["pair_iptm"]), float(r["dockq"])
            if ip > best[k][0]:
                best[k] = (ip, dq)
            anygood[k] |= dq >= ACC
    return {k: ("solved" if v[1] >= ACC else "decoy" if anygood[k] else "nocov")
            for k, v in best.items()}


def buckets(M, good):
    """(good-good, good-bad, bad-bad) mean similarity, diagonal excluded."""
    M = M.astype(float).copy()
    np.fill_diagonal(M, np.nan)
    f = lambda a, b: float(np.nanmean(M[np.ix_(a, b)]))
    return f(good, good), f(good, ~good), f(~good, ~good)


def auc(M, good):
    """P(a good-good pair scores above a good-bad pair), ties counted as half.

    The mean gap is NOT comparable across metrics: epitope runs higher on both buckets, so it has more
    dynamic range to spend and a gap comparison rewards range rather than discrimination. This is
    invariant to any monotone rescaling of the similarity, so it compares the metrics on the only thing
    that matters for clustering -- the ORDER they put pairs in.
    """
    n = len(good)
    iu = np.triu_indices(n, 1)
    same = good[iu[0]] & good[iu[1]]            # good-good pairs
    cross = good[iu[0]] ^ good[iu[1]]           # good-bad pairs
    v = M[iu].astype(float)
    a, b = v[same], v[cross]
    if not len(a) or not len(b):
        return float("nan")
    # Mann-Whitney U via ranks: handles ties correctly, and O(n log n) rather than O(n^2) pairs
    r = np.concatenate([a, b]).argsort().argsort().astype(float) + 1
    ranks = np.empty_like(r)
    order = np.concatenate([a, b]).argsort()
    ranks[order] = np.arange(1, len(r) + 1)
    # average ranks within tie groups
    vals = np.concatenate([a, b])
    srt = np.sort(vals)
    _, first, counts = np.unique(srt, return_index=True, return_counts=True)
    avg = {}
    for f0, c in zip(first, counts):
        avg[srt[f0]] = f0 + (c + 1) / 2.0
    ranks = np.array([avg[x] for x in vals])
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return float(u / (len(a) * len(b)))


def seed_buckets(M, seeds):
    """Mean and spread of similarity for SAME-seed pairs vs DIFFERENT-seed pairs.

    Asks whether a seed is a coherent unit. Each seed contributes 5 poses; if its 5 poses are much
    more alike than poses from other seeds, then collapsing poses to seeds keeps the structure and
    seed-level selection is meaningful. If within and across are similar, a seed is just 5 arbitrary
    draws and treating it as one object throws away the thing that actually varies.

    Both mean AND standard deviation are returned: a seed could be tight on average while some seeds
    scatter badly, and the mean alone would hide that.
    """
    n = len(seeds)
    iu = np.triu_indices(n, 1)
    same = seeds[iu[0]] == seeds[iu[1]]
    v = M[iu].astype(float)
    w, a = v[same], v[~same]
    f = lambda x: (float(np.mean(x)), float(np.std(x))) if len(x) else (float("nan"),) * 2
    return f(w), f(a), int(same.sum()), int((~same).sum())


def agreement(z, good):
    """Spearman between each pair of metrics, over the same pair vector.

    Answers whether two metrics with a similar gap are actually measuring the same thing.
    """
    n = len(good)
    iu = np.triu_indices(n, 1)
    rk = {m: np.argsort(np.argsort(z[m][iu].astype(float))).astype(float) for m in METRICS}
    out = {}
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1:]:
            out[f"{a}-{b}"] = float(np.corrcoef(rk[a], rk[b])[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--out", default="sep")
    A = ap.parse_args()

    cls = pool_classes()
    rows = []
    for f in sorted(glob.glob(f"{A.matrices}/*.npz")):
        iface = os.path.basename(f)[:-4]
        z = np.load(f)
        good = z["dockq_gt"] >= ACC
        # an interface with only one class has no good-bad pairs at all; skip rather than emit nan
        if good.sum() < 3 or (~good).sum() < 3:
            continue
        agr = agreement(z, good)
        for m in METRICS:
            gg, gb, bb = buckets(z[m], good)
            rows.append(dict(interface=iface, klass=cls.get(iface, "?"), metric=m,
                             n=len(good), n_good=int(good.sum()), gg=gg, gb=gb, bb=bb, gap=gg - gb,
                             auc=auc(z[m], good), **agr))

    with open(f"{A.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: mean gap per metric per class, n on every row
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["klass"], r["metric"])].append(r)
    out = []
    for (k, m), rs in sorted(by.items()):
        gaps = [r["gap"] for r in rs]
        aucs = [r["auc"] for r in rs if r["auc"] == r["auc"]]
        out.append([k, m, len(rs), round(float(np.mean([r["gg"] for r in rs])), 3),
                    round(float(np.mean([r["gb"] for r in rs])), 3),
                    round(float(np.mean(gaps)), 3),
                    round(float(np.mean(aucs)), 3), round(float(np.median(aucs)), 3),
                    sum(1 for a in aucs if a > 0.9)])
    print(table(["interface class", "metric", "interfaces (n)", "mean good-good", "mean good-bad",
                 "mean gap", "mean AUC", "median AUC", "AUC > 0.9"], out, align="l l r r r r r r r"))
    # do two metrics with a similar gap actually rank pairs the same way?
    ag = [[p, len(rows) // 3, round(float(np.mean([r[p] for r in rows if r["metric"] == "epitope"])), 3)]
          for p in ("epitope-fcc", "epitope-dockq", "fcc-dockq")]
    print("\nrank agreement between metrics, Spearman over pairs, averaged across interfaces:")
    print(table(["metric pair", "interfaces (n)", "mean Spearman"], ag, align="l r r"))

    plot(rows, f"{A.out}.png")
    print(f"\n  wrote {A.out}.csv and {A.out}.png")
    return 0


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"solved": "#2c7fb8", "decoy": "#d95f0e", "nocov": "#999999"}
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for i, m in enumerate(METRICS):
        rs = [r for r in rows if r["metric"] == m]
        for k in ("solved", "decoy", "nocov"):
            s = [r for r in rs if r["klass"] == k]
            if s:
                ax[i].scatter([r["gb"] for r in s], [r["gg"] for r in s], s=26, alpha=.75,
                              c=COL[k], edgecolor="none", label=f"{k} (n={len(s)})")
        ax[i].plot([0, 1], [0, 1], lw=1, ls="--", c="k", alpha=.4)
        ax[i].set(xlim=(-.02, 1), ylim=(-.02, 1), xlabel="mean similarity, good-bad pairs",
                  ylabel="mean similarity, good-good pairs" if i == 0 else "", title=m)
        ax[i].legend(fontsize=8, loc="lower right", frameon=False)
    # panel 4: the gap, paired across metrics, split by class -- the comparison we actually want
    pos, labels = [], []
    for j, k in enumerate(("solved", "decoy")):
        for i, m in enumerate(METRICS):
            g = [r["gap"] for r in rows if r["metric"] == m and r["klass"] == k]
            if not g:
                continue
            x = j * 4 + i
            ax[3].boxplot(g, positions=[x], widths=.7, showfliers=False,
                          patch_artist=True, medianprops=dict(color="k"),
                          boxprops=dict(facecolor=COL[k], alpha=.55, edgecolor="none"))
            ax[3].scatter(np.full(len(g), x) + np.linspace(-.18, .18, len(g)), g, s=7,
                          c="k", alpha=.35, zorder=3)
            pos.append(x)
            labels.append(f"{m}\n{k}\nn={len(g)}")
    ax[3].set(xticks=pos, ylabel="separation gap  (good-good  -  good-bad)",
              title="separation by metric and interface class")
    ax[3].set_xticklabels(labels, fontsize=7)
    ax[3].axhline(0, lw=1, c="k", alpha=.4)
    fig.suptitle("Do good poses resemble each other? Ground-truth-labelled bucket means, "
                 "one point per interface", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
