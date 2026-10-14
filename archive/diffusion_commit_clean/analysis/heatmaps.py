#!/usr/bin/env python
"""Pose x pose heatmaps, one page per interface, to see which metric has visible cluster structure.

WHAT YOU ARE LOOKING AT. Each page is one interface: three heatmaps, one per metric, poses ordered by
that metric's own dendrogram so blocks of mutually-similar poses sit together. The strip under each
heatmap marks ground-truth quality, so a block that lines up with the good strip is a basin worth
templating and a block that straddles it is not.

EACH METRIC GETS ITS OWN LEAF ORDER, deliberately. A shared ordering taken from one metric would show
that metric's structure and scramble the others -- it would rig the comparison. The cost is that the
three panels are not row-aligned, so you cannot read "this pose" across panels; the CSV carries the
per-metric scores for that.

EYEBALLING 292 x 3 HEATMAPS IS NOT A MEASUREMENT, so each panel is also scored two ways:

  cophenetic r   how faithfully a tree represents the actual distances. High means the data really is
                 hierarchically clustered rather than forced into a tree by the algorithm.
  silhouette     at the best K in 2..10, how separated the clusters are. Independent of any tree.

Both are label-free -- they measure whether structure EXISTS, not whether it aligns with DockQ. The
alignment question is what cluster.py scores.

Poses with no antibody-antigen contact were already dropped upstream by build_matrices.py, so every
row here is a pose that actually formed the interface.

    heatmaps.py [--matrices matrices] [--out heat] [--only iface1,iface2]
"""
import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")

from fmt import table  # noqa: E402
from separation import METRICS, pool_classes  # noqa: E402

ACC = 0.23


def prep(M):
    """(distance matrix, condensed form, linkage) for one similarity matrix."""
    D = 1.0 - np.clip(M.astype(float), 0, 1)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    c = squareform(D, checks=False)
    return D, c, linkage(c, method="average")


def clarity(D, c, Z):
    """(cophenetic r, best silhouette, K at that silhouette) -- does structure exist at all?"""
    coph = float(cophenet(Z, c)[0])
    best, bestk = -1.0, 0
    for k in range(2, min(11, len(D))):
        lab = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        s = float(silhouette_score(D, lab, metric="precomputed"))
        if s > best:
            best, bestk = s, k
    return coph, best, bestk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--out", default="heat")
    ap.add_argument("--only", default=None)
    A = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    cls = pool_classes()
    files = sorted(glob.glob(f"{A.matrices}/*.npz"))
    if A.only:
        want = set(A.only.split(","))
        files = [f for f in files if os.path.basename(f)[:-4] in want]

    rows = []
    with PdfPages(f"{A.out}.pdf") as pdf:
        for f in files:
            iface = os.path.basename(f)[:-4]
            z = np.load(f)
            dq = z["dockq_gt"].astype(float)
            n = len(dq)
            if n < 10:
                continue
            k = cls.get(iface, "?")
            fig, ax = plt.subplots(2, 3, figsize=(15, 5.6),
                                   gridspec_kw=dict(height_ratios=[14, 1], hspace=.06))
            for i, m in enumerate(METRICS):
                D, c, Z = prep(z[m])
                order = leaves_list(Z)
                coph, sil, bk = clarity(D, c, Z)
                rows.append(dict(interface=iface, klass=k, metric=m, n=n,
                                 n_good=int((dq >= ACC).sum()),
                                 cophenetic_r=round(coph, 3), silhouette=round(sil, 3), best_k=bk))
                ax[0, i].imshow(z[m][np.ix_(order, order)], vmin=0, vmax=1, cmap="magma",
                                interpolation="nearest", rasterized=True)
                ax[0, i].set(xticks=[], yticks=[],
                             title=f"{m}   cophenetic r={coph:.2f}   silhouette={sil:.2f} (K={bk})")
                # quality strip in the SAME leaf order: bright = acceptable
                ax[1, i].imshow((dq[order] >= ACC)[None, :], aspect="auto", cmap="Greens",
                                vmin=0, vmax=1, interpolation="nearest", rasterized=True)
                ax[1, i].set(xticks=[], yticks=[], xlabel="poses, ordered by this metric's dendrogram")
            fig.suptitle(f"{iface}   [{k}]   n={n} poses, {int((dq>=ACC).sum())} acceptable "
                         f"(green strip), max DockQ {dq.max():.3f}", y=.99)
            pdf.savefig(fig, dpi=90, bbox_inches="tight")
            plt.close(fig)

    with open(f"{A.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["klass"], r["metric"])].append(r)
    out = []
    for (k, m), rs in sorted(by.items()):
        out.append([k, m, len(rs),
                    round(float(np.mean([r["cophenetic_r"] for r in rs])), 3),
                    round(float(np.mean([r["silhouette"] for r in rs])), 3),
                    round(float(np.median([r["best_k"] for r in rs])), 1),
                    sum(1 for r in rs if r["silhouette"] > 0.5)])
    print(table(["class", "metric", "interfaces (n)", "mean cophenetic r", "mean best silhouette",
                 "median best K", "silhouette > 0.5"], out, align="l l r r r r r"))
    print(f"\n  wrote {A.out}.pdf ({len(rows)//3} pages) and {A.out}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
