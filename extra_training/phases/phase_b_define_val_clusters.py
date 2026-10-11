#!/usr/bin/env python
"""Pick the validation clusters for the confidence-head finetune: 50 clusters, ~10% of structures,
spread across release dates.

SUMMARY
    The training pool is 557 SAbDab2 ab_ag_clusters / 2,886 structures (see phase_a_define_train_set.py
    for how it is derived). Validation is held out BY CLUSTER, never by structure, so near-duplicates
    cannot straddle the split.

    CLUSTER SIZES ARE VERY SKEWED -- cluster 171 alone is 1,001 structures -- so drawing 50 clusters at
    random almost never lands near 10% of structures. Instead:

      1. each cluster gets its median deposition date (PDBdepo), and clusters are split into
         --n-bins date bins of equal cluster count, so every era is represented
      2. --draws random draws each take n_val / n_bins clusters from every bin
      3. draws in which any one cluster exceeds --max-share of the target are rejected -- otherwise a
       single 183-structure cluster hits 10% alone and validation is effectively one system
    4. the remaining draw whose structure count is closest to --frac of the pool is kept

    Deterministic given --seed, so the split reproduces.

INPUTS
    --n-val:   validation clusters. Default 50.
    --frac:    target share of structures. Default 0.10.
    --n-bins:  date bins. Default 5 (so 10 clusters per bin).
    --draws:   random draws searched. Default 20000.
    --seed:    RNG seed. Default 20261010.
    --out:     output csv. Default extra_training/results/val_clusters.csv.

OUTPUTS
    <out>                one row per validation cluster: cluster, n_structures, median_depo, date_bin
    stdout               structure share, per-bin counts, date range covered

FUNCTIONS
    cluster_table(): the training rows -> one row per cluster with size and median date
    main():          bin, search draws, write
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# This file lives at <repo>/extra_training/phases/, so the repo root is three levels up, not two.
# paths.py stays at the repo root: 20+ files across the project import it, and it is shared
# infrastructure rather than anything specific to this pipeline.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import phase_a_define_train_set as A  # noqa: E402



def cluster_table():
    """out: DataFrame, one row per surviving training cluster: cluster, n_structures, median_depo"""
    # Count only the polypeptide-antigen rows that make up the training pool. Counting every row of a
    # cluster (haptens, sugars, apo) inflated the pool to 3,765 structures instead of 2,886.
    _clusters, rows = A.training_clusters(A.SPLITS, A.EXCLUDED)
    s = pd.read_csv(A.SPLITS, low_memory=False, usecols=["PDB_ID", "PDBdepo"])
    s = s.drop_duplicates("PDB_ID").set_index("PDB_ID")
    s = rows[["PDB_ID", "ab_ag_cluster"]].join(s, on="PDB_ID")
    s["depo"] = pd.to_datetime(s.PDBdepo, errors="coerce")
    per_struct = s.drop_duplicates(["ab_ag_cluster", "PDB_ID"])
    return (per_struct.groupby("ab_ag_cluster")
            .agg(n_structures=("PDB_ID", "nunique"), median_depo=("depo", "median"))
            .reset_index().rename(columns={"ab_ag_cluster": "cluster"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--max-share", type=float, default=0.15,
                    help="no single cluster may exceed this share of the validation structures")
    ap.add_argument("--out", default=f"{REPO}/extra_training/results/val_clusters.csv")
    a = ap.parse_args()

    t = cluster_table().sort_values("median_depo").reset_index(drop=True)
    total = int(t.n_structures.sum())
    target = a.frac * total
    t["date_bin"] = pd.qcut(t.median_depo.rank(method="first"), a.n_bins, labels=False)
    per_bin = a.n_val // a.n_bins
    bins = [t.index[t.date_bin == b].to_numpy() for b in range(a.n_bins)]

    rng = np.random.default_rng(a.seed)
    best, best_err = None, np.inf
    for _ in range(a.draws):
        pick = np.concatenate([rng.choice(ix, per_bin, replace=False) for ix in bins])
        sizes = t.n_structures.to_numpy()[pick]
        # one giant cluster can hit the 10% target on its own; that is one system, not 50
        if sizes.max() > a.max_share * target:
            continue
        err = abs(sizes.sum() - target)
        if err < best_err:
            best, best_err = pick, err

    v = t.loc[np.sort(best)].copy()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    v.to_csv(a.out, index=False)

    n = int(v.n_structures.sum())
    print(f"pool: {len(t)} clusters, {total} structures; target {target:.0f} structures")
    print(f"validation: {len(v)} clusters, {n} structures = {100*n/total:.1f}% "
          f"(train keeps {len(t)-len(v)} clusters, {total-n} structures)")
    print(f"largest val cluster: {v.n_structures.max()} structures; "
          f"date range {v.median_depo.min().date()} .. {v.median_depo.max().date()}")
    print("| date bin | bin date range | val clusters | val structures |")
    print("|---|---|---|---|")
    for b in range(a.n_bins):
        tb, vb = t[t.date_bin == b], v[v.date_bin == b]
        print(f"| {b} | {tb.median_depo.min().date()} .. {tb.median_depo.max().date()} "
              f"| {len(vb)} | {int(vb.n_structures.sum())} |")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
