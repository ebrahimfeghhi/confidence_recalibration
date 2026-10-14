#!/usr/bin/env python
"""If we can afford K self-templating runs, is one of them guaranteed to be a good donor set?

THE QUESTION, operationally. Cut the poses into K clusters and self-template each cluster. A run is
useful if its cluster contains ONLY good poses -- a pure cluster, whatever its size. So the quantity
that matters is the SMALLEST K at which a pure cluster appears. K = 2 means two runs and one of them
lands; K = 10 means the budget is ten runs; no pure cluster by K = 10 means the metric cannot isolate
the good poses at any affordable budget.

NO MINIMUM CLUSTER SIZE, deliberately. A cluster holding a single good pose is a legitimate win: you
self-template on that pose and it is a good template. The cost of this is that an OUTLIER good pose --
one unlike everything else, including other good poses -- splits off as a singleton early and counts as
a success. That is not a measurement artefact, it is the method working for an unglamorous reason, but
the cluster size is recorded so such wins are visible rather than hidden inside a rate.

RECALL IS NOT THE TARGET. Good poses stranded in mixed clusters do not count against anything. One good
donor set is all the downstream step needs.

Reported per interface: min K for a fully pure cluster, min K for an 80%-pure one, and for the winning
cluster its size and best DockQ -- because five poses all at DockQ 0.25 are "pure" but a weak donor set,
and that should be visible rather than assumed away.

    cluster.py [--matrices matrices] [--kmax 10] [--out clus]
"""
import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")

import paths  # noqa: E402
from fmt import table  # noqa: E402
from separation import METRICS, pool_classes  # noqa: E402

ACC = 0.23


def cut(M, kmax):
    """Average-linkage clusterings for K = 1..kmax, as a list of label arrays.

    Distance is 1 - similarity. Average linkage because the similarities are bounded [0, 1] and not a
    metric: single linkage would chain two basins together through one intermediate pose, complete
    linkage would shatter a basin over one dissimilar member.
    """
    D = 1.0 - np.clip(M.astype(float), 0, 1)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                       # squareform demands exact symmetry
    Z = linkage(squareform(D, checks=False), method="average")
    return [fcluster(Z, k, criterion="maxclust") for k in range(1, kmax + 1)]


def first_pure(labels_by_k, good, dq, thresh):
    """(K, cluster size, best DockQ in it) for the first K with a cluster >= thresh pure, else None."""
    for k, lab in enumerate(labels_by_k, 1):
        best = None
        for c in np.unique(lab):
            sel = lab == c
            frac = good[sel].mean()
            if frac >= thresh:
                cand = (int(sel.sum()), float(dq[sel].max()))
                if best is None or cand > best:     # prefer the bigger / better cluster at this K
                    best = cand
        if best:
            return k, best[0], best[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--out", default="clus")
    A = ap.parse_args()

    cls = pool_classes()
    rows = []
    for f in sorted(glob.glob(f"{A.matrices}/*.npz")):
        iface = os.path.basename(f)[:-4]
        z = np.load(f)
        dq = z["dockq_gt"]
        good = dq >= ACC
        if not good.any():
            continue                            # nothing to find; not a failure of the metric
        for m in METRICS:
            labs = cut(z[m], A.kmax)
            rec = dict(interface=iface, klass=cls.get(iface, "?"), metric=m, n=len(good),
                       n_good=int(good.sum()), prevalence=round(float(good.mean()), 3))
            # 0.5 because the template embedder AVERAGES its templates, so a majority-good donor set
            # can still pull toward the right basin; 1.0 and 0.8 kept for comparison
            for tag, th in (("pure", 1.0), ("80", 0.8), ("50", 0.5)):
                hit = first_pure(labs, good, dq, th)
                rec[f"minK_{tag}"] = "" if not hit else hit[0]
                rec[f"size_{tag}"] = "" if not hit else hit[1]
                rec[f"bestdq_{tag}"] = "" if not hit else round(hit[2], 3)
            rows.append(rec)

    with open(f"{A.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["klass"], r["metric"])].append(r)
    out = []
    for (k, m), rs in sorted(by.items()):
        ks = [r["minK_pure"] for r in rs if r["minK_pure"] != ""]
        out.append([k, m, len(rs),
                    sum(1 for x in ks if x <= 1), sum(1 for x in ks if x <= 5),
                    sum(1 for x in ks if x <= 10), len(rs) - len(ks),
                    round(float(np.median(ks)), 1) if ks else float("nan"),
                    round(float(np.median([r["size_pure"] for r in rs if r["minK_pure"] != ""])), 1)
                    if ks else float("nan")])
    print(table(["interface class", "metric", "interfaces (n)", "pure at K=1", "pure by K<=5",
                 "pure by K<=10", "never by K=10", "median min K", "median cluster size"],
                out, align="l l r r r r r r r"))
    plot(rows, A.kmax, f"{A.out}.png")
    print(f"\n  wrote {A.out}.csv and {A.out}.png")
    return 0


def plot(rows, kmax, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    COL = {"epitope": "#7570b3", "fcc": "#1b9e77", "dockq": "#d95f02"}
    for j, klass in enumerate(("decoy", "solved")):
        for m in METRICS:
            rs = [r for r in rows if r["metric"] == m and r["klass"] == klass]
            if not rs:
                continue
            ks = [r["minK_pure"] for r in rs if r["minK_pure"] != ""]
            # cumulative: fraction of interfaces solved with a budget of K runs
            y = [sum(1 for x in ks if x <= k) / len(rs) for k in range(1, kmax + 1)]
            ax[j].plot(range(1, kmax + 1), y, marker="o", ms=4, c=COL[m],
                       label=f"{m} (n={len(rs)})")
        ax[j].set(xlabel="budget K (number of self-templating runs)",
                  ylabel="fraction of interfaces with a pure good cluster" if j == 0 else "",
                  title=f"{klass} interfaces", ylim=(0, 1.02), xticks=range(1, kmax + 1))
        ax[j].grid(alpha=.25)
        ax[j].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("With a budget of K self-templating runs, how often is one cluster all good poses?",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
