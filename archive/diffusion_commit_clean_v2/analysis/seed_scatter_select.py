#!/usr/bin/env python
"""Pick free seeds to refold with 50 samples: per interface, its most committed and most scattered seed.

SUMMARY
    Question: when a seed's 5 poses scatter, is the trunk a weak signal (poses land anywhere) or does it
    encode a few discrete docking sites? Refolding a seed with 50 samples draws 50 poses from the same
    trunk (the trunk is deterministic per seed), which shows the shape. This picks the seeds.

    A seed's AGREEMENT at an interface = mean pairwise DockQ over its 5 samples' 10 pairs, read from the
    existing pose-to-pose matrices (symmetrised, anchor chain letters, every antibody-antigen interface
    of the free-50 pool). Per interface: the seed with the highest agreement (committed) and the lowest
    (scattered), among seeds with all 5 samples in the matrix. A fold scores every interface of its
    complex, so the queue is the union of picks per complex -- each complex's seeds are folded once.

    CONFIDENT DECOYS FIRST. For every decoy interface (an acceptable pose exists in the first 50 seeds
    but the top-ipTM pose is not; classes as score_experiment.classes_at, reference chain letters) two
    more seeds go at the FRONT of the queue: the seed of its top-ipTM pose (decoy_top_iptm, the
    confident decoy) and the seed of its best-DockQ pose (decoy_best_dockq). Then, right after those,
    the seed most committed to a correct pose (decoy_committed_correct: most of its 5 poses acceptable,
    ties by mean DockQ) -- the best-DockQ seed is often a lucky one-off from a scattered seed, this one
    is not. Evaluation reads the crystal here only to choose what to fold.

INPUTS
    --dataset:     pxm2025 or foldbench. Default pxm2025. Picks the pool, the matrices and --out.
    --decoys-only: write only the decoy picks. Default off.
    --matrices:  pose-to-pose matrix directory. Default experimental_data/pose_dockq_matrices/<dataset>.
    --dead:      complexes to skip (pxm2025 only). Default experimental_data/pxm2025/DEAD_COMPLEXES.txt.
    --out:       experiment directory. Default experimental_data/seed_scatter_50 (pxm2025), else
                 <dataset data>/seed_scatter_50.
    --tag:       queue tag. Default scatter50.

OUTPUTS
    <out>/selection.csv: iface, pdb, seed, pick (decoy_top_iptm | decoy_best_dockq | committed |
                         scattered), agreement (blank for decoy picks), seeds_ranked; plus, for decoy
                         picks, the picked pose's ipTM and crystal DockQ
    <out>/queue.tsv:     one free fold per (complex, seed), folder.py's queue format, decoy folds first
    stdout:              counts and the agreement of each pick group

EXAMPLE
    analysis/seed_scatter_select.py

FUNCTIONS
    seed_agreement(): one matrix -> {seed: agreement} over seeds with 5 samples
    main():           pick, write, summarise
"""
import argparse
import collections
import csv
import glob
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
from score_experiment import classes_at, pool_by_seed  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experimental_data")


def seed_agreement(npz):
    """in: a pose-to-pose matrix file.  out: {seed: mean pairwise DockQ of its 5 samples}"""
    d = np.load(npz)
    rows = collections.defaultdict(list)
    for i, s in enumerate(d["seed"]):
        rows[str(s)].append(i)
    out = {}
    for s, ix in rows.items():
        if len(ix) == 5:
            m = d["dockq"][np.ix_(ix, ix)]
            out[s] = float(m[np.triu_indices(5, 1)].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--decoys-only", action="store_true",
                    help="only the decoy picks, no committed/scattered population")
    ap.add_argument("--matrices", default=None, help="default experimental_data/pose_dockq_matrices/<dataset>")
    ap.add_argument("--dead", default=f"{DATA}/pxm2025/DEAD_COMPLEXES.txt")
    ap.add_argument("--out", default=None,
                    help="default experimental_data/seed_scatter_50 (pxm2025), <dataset data>/seed_scatter_50 otherwise")
    ap.add_argument("--tag", default="scatter50")
    A = ap.parse_args()
    ds = paths.DATASETS[A.dataset]
    A.matrices = A.matrices or f"{DATA}/pose_dockq_matrices/{A.dataset}"
    A.out = A.out or (f"{DATA}/seed_scatter_50" if A.dataset == "pxm2025" else f"{ds.data}/seed_scatter_50")
    dead = ({l.split()[0] for l in open(A.dead) if l.strip() and not l.startswith("#")}
            if A.dataset == "pxm2025" else set())
    os.makedirs(A.out, exist_ok=True)

    picks, short = [], 0
    for f in ([] if A.decoys_only else sorted(glob.glob(f"{A.matrices}/*.npz"))):
        iface = os.path.basename(f)[:-4]
        pdb = iface.split("_")[0]
        if pdb in dead:
            continue
        ag = seed_agreement(f)
        if len(ag) < 2:
            short += 1
            continue
        ranked = sorted(ag, key=lambda s: (ag[s], int(s)))
        picks.append((iface, pdb, ranked[-1], "committed", ag[ranked[-1]], len(ranked)))
        picks.append((iface, pdb, ranked[0], "scattered", ag[ranked[0]], len(ranked)))

    pool, first = pool_by_seed(50, str(ds.pool_csv))
    cls = classes_at(pool, first)
    dpicks, cpicks = [], []
    for iface in sorted(i for i, c in cls.items() if c == "decoy"):
        pdb = iface.split("_")[0]
        if pdb in dead:
            continue
        keep = [r for r in pool[iface] if r[0] in first[pdb]]          # (seed, sample, dockq, iptm)
        top = max(keep, key=lambda r: r[3])
        best = max(keep, key=lambda r: r[2])
        dpicks.append((iface, pdb, top[0], "decoy_top_iptm", top[3], top[2]))
        dpicks.append((iface, pdb, best[0], "decoy_best_dockq", best[3], best[2]))
        # the seed MOST committed to a correct pose: most of its 5 poses acceptable, then mean DockQ
        per = collections.defaultdict(list)
        for r in keep:
            per[r[0]].append(r)
        cc = max(per, key=lambda sd: (sum(r[2] >= 0.23 for r in per[sd]),
                                      statistics.mean(r[2] for r in per[sd]), -int(sd)))
        cpicks.append((iface, pdb, cc, "decoy_committed_correct",
                       max(r[3] for r in per[cc]), max(r[2] for r in per[cc]),
                       sum(r[2] >= 0.23 for r in per[cc])))

    with open(f"{A.out}/selection.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iface", "pdb", "seed", "pick", "agreement", "seeds_ranked", "pose_iptm", "pose_dockq",
                    "acceptable_of_5"])
        for p in dpicks:
            w.writerow([*p[:4], "", "", f"{p[4]:.4f}", f"{p[5]:.4f}", ""])
        for p in cpicks:
            w.writerow([*p[:4], "", "", f"{p[4]:.4f}", f"{p[5]:.4f}", p[6]])
        for p in picks:
            w.writerow([*p[:4], f"{p[4]:.4f}", p[5], "", "", ""])
    front = list(dict.fromkeys((p[1], p[2]) for p in dpicks))
    second = [f for f in dict.fromkeys((p[1], p[2]) for p in cpicks) if f not in front]
    rest = sorted({(p[1], p[2]) for p in picks} - set(front) - set(second), key=lambda x: (x[0], int(x[1])))
    folds = front + second + rest
    with open(f"{A.out}/queue.tsv", "w") as fh:
        for pdb, seed in folds:
            fh.write(f"free\t{pdb}\t*\t{A.tag}\t{seed}\n")
    print(f"  committed-correct folds next: {len(second)} new (of {len(cpicks)} picks; "
          f"{sum(p[6] >= 3 for p in cpicks)} have >= 3 of 5 acceptable)")
    print(f"  decoy interfaces: {len(dpicks) // 2}   decoy folds at the front: {len(front)} "
          f"(of which also in the scatter picks: {len(set(front) & {(p[1], p[2]) for p in picks})})")

    print(f"  interfaces picked: {len(picks) // 2}   (skipped, under 2 seeds with 5 samples: {short})")
    print(f"  complexes: {len({p[1] for p in picks})}   folds (unique complex, seed): {len(folds)}")
    for k in ("committed", "scattered"):
        v = [p[4] for p in picks if p[3] == k]
        if v:
                print(f"  {k:9s} n={len(v)}  agreement median {statistics.median(v):.3f}  "
                  f"range {min(v):.3f}-{max(v):.3f}")
    print(f"  wrote {A.out}/selection.csv, {A.out}/queue.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
