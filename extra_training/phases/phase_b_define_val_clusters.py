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
    --out:     output csv. Default extra_training/splits/val_split_proposal_clusters.csv,
               the LIVE 40-cluster split. The superseded 50-cluster draw is archived.

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



def cluster_table(splits=None, excluded=None):
    """in: optional splits csv and exclusion csv.  out: one row per surviving training cluster:
    cluster, n_structures, median_depo

    Both paths are arguments rather than module defaults so this phase can be pointed at an
    alternative splits release the same way phases A and C can -- otherwise B silently reads the
    default while A and C read whatever they were told to, and the three disagree about the pool.
    """
    splits = splits or A.SPLITS
    excluded = excluded or A.EXCLUDED
    # Count only the polypeptide-antigen rows that make up the training pool. Counting every row of a
    # cluster (haptens, sugars, apo) inflated the pool to 3,765 structures instead of 2,886.
    _clusters, rows = A.training_clusters(splits, excluded)
    s = pd.read_csv(splits, low_memory=False, usecols=["PDB_ID", "PDBdepo"])
    s = s.drop_duplicates("PDB_ID").set_index("PDB_ID")
    s = rows[["PDB_ID", "ab_ag_cluster"]].join(s, on="PDB_ID")
    s["depo"] = pd.to_datetime(s.PDBdepo, errors="coerce")
    per_struct = s.drop_duplicates(["ab_ag_cluster", "PDB_ID"])
    return (per_struct.groupby("ab_ag_cluster")
            .agg(n_structures=("PDB_ID", "nunique"), median_depo=("depo", "median"))
            .reset_index().rename(columns={"ab_ag_cluster": "cluster"}))


LIVE_VAL = f"{REPO}/extra_training/splits/val_split_proposal_clusters.csv"
LEGACY_VAL = f"{REPO}/extra_training/splits/archive/val_clusters_SUPERSEDED_50cluster.csv"
# THE LIVE VALIDATION SPLIT IS val_split_proposal_clusters.csv, 40 clusters, drawn 2026-10-13 by
# phases/phase_b2_propose_val_split.py. val_clusters.csv is the ORIGINAL 50-cluster phase-B draw and is
# LEGACY: 22 of the live 40 are not in it, so using it as the validation list leaves those 22 in the
# training pool. That is exactly what happened -- the FT3 cache selection picked up 28 structures
# that had been folded as validation. Anything that needs "the validation clusters" takes LIVE_VAL.
# Clusters held out for the old split and not in the live one are training data again (32 of them);
# the pool is pool - LIVE_VAL, nothing else.


def write_training(keep, krows, out, excluded):
    """write train_clusters.csv, after asserting it shares nothing with the exclusion list

    THE ASSERTION IS THE POINT. train_clusters.csv is the file the training stage reads, and it was
    committed holding 487 clusters of which four -- 947, 2238, 2483, 2557 -- the current leakage
    search excludes. Nothing compared the two, because the exclusion was applied when the pool was
    built and never re-checked against the artifact that outlived it. A list that is written from the
    pool is not automatically disjoint from a newer exclusion file, so it is checked here, where the
    file is produced, rather than trusted.
    """
    ex = set()
    if excluded and os.path.exists(excluded):
        import csv as _csv
        with open(excluded, newline="") as fh:
            ex = {str(r["cluster"]).strip() for r in _csv.DictReader(fh) if r.get("cluster")}
    bad = sorted(str(c) for c in keep if str(c) in ex)
    if bad:
        raise SystemExit(
            f"STOP: {len(bad)} cluster(s) are in BOTH the training list and the exclusion file "
            f"{excluded}: {bad[:10]}. Each one shares an antigen or a CDR-H3 with a complex we "
            "evaluate on, so training on it is leakage. Re-run phase A so the exclusion file is "
            "current, then re-run this.")
    tc = os.path.join(os.path.dirname(out), "train_clusters.csv")
    with open(tc, "w") as fh:
        fh.write("cluster\n")
        for c in sorted(keep):
            fh.write(f"{c}\n")
    # PROVENANCE, so a consumer can tell this file is stale instead of trusting it. train_clusters.csv
    # outlived the validation split it was subtracted from once already; a plain csv carries nothing
    # that would have shown it. The sidecar names the validation file and hashes its contents.
    import hashlib as _h, json as _j
    meta = {
        "written": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "val_file": os.path.abspath(out),
        "val_sha256": _h.sha256(open(out, "rb").read()).hexdigest(),
        "n_val": sum(1 for _ in open(out)) - 1,
        "n_train": len(keep),
    }
    with open(tc.replace(".csv", ".meta.json"), "w") as fh:
        _j.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"wrote {tc}: {len(keep)} training clusters ({krows.PDB_ID.nunique()} structures), "
          f"validation removed, disjoint from {len(ex)} excluded clusters")
    return tc


def train_minus_val(splits=None, excluded=None, val_csv=None):
    """in: optional splits, exclusion and validation csv paths.
    out: (training cluster set with validation removed, those rows of the splits table)

    THE ONLY PLACE THE TRAIN/VAL SPLIT IS ENFORCED, and it exists because nothing else enforced it.
    Phase A derives the 557-cluster pool, phase B holds 50 of them out, and phase C deliberately
    builds MSAs for all 557 -- correctly, since the validation complexes have to be folded. So every
    phase up to here legitimately carries the full pool, and the subtraction has to happen wherever
    the training indices are built. At the time of writing nothing did it: the failure mode is
    training on your own validation set with nothing raised, and then reporting the result.

    A MISSING VALIDATION CSV IS FATAL, never "nothing to hold out". That is the same reasoning as
    the exclusion-csv guard in phase A: an absent artifact means the split was never applied.

    Any training-index builder must take its cluster list from here, not from
    phase_a_define_train_set.training_clusters(), which returns the pool INCLUDING validation.
    """
    val_csv = val_csv or LIVE_VAL
    if not os.path.exists(val_csv):
        raise SystemExit(
            f"STOP: {val_csv} does not exist.\n"
            "It is written by this phase's main(). Without it the training set would silently "
            "include the validation clusters.\n"
            f"  python {os.path.basename(__file__)} --out {val_csv}")
    val = set(pd.read_csv(val_csv).cluster)
    if not val:
        raise SystemExit(f"STOP: {val_csv} lists no clusters; the split would hold nothing out.")
    clusters, rows = A.training_clusters(splits or A.SPLITS, excluded or A.EXCLUDED)
    missing = val - clusters
    if missing:
        raise SystemExit(
            f"STOP: {len(missing)} validation cluster(s) are not in the training pool: "
            f"{sorted(missing)[:10]}\n"
            "the validation csv and the pool have drifted apart -- re-run phases A and B in order.")
    keep = clusters - val
    return keep, rows[~rows.ab_ag_cluster.isin(val)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--max-share", type=float, default=0.15,
                    help="no single cluster may exceed this share of the validation structures")
    ap.add_argument("--prune", action="store_true",
                    help="drop pinned validation clusters that a later leakage check has excluded "
                         "from the pool, keeping the rest pinned. The minimal correct repair.")
    ap.add_argument("--force", action="store_true",
                    help="draw a NEW validation split even though one exists. Off by default: the "
                         "committed split is pinned, and redrawing after results have been seen is "
                         "what makes a validation set stop being one.")
    ap.add_argument("--splits", default=None, help="SAbDab2 abag_split.csv; default phase A's")
    ap.add_argument("--excluded", default=None, help="phase A's exclusion csv; default phase A's")
    ap.add_argument("--select", choices=["target", "band"], default="target",
                    help="target: the draw closest to --frac (reproduces results/val_clusters.csv). "
                         "band: uniform among draws inside --frac-lo/--frac-hi. See main().")
    ap.add_argument("--frac-lo", type=float, default=0.05,
                    help="accept any draw holding at least this share of the pool's structures")
    ap.add_argument("--frac-hi", type=float, default=0.12,
                    help="...and at most this share. See the band comment in main().")
    ap.add_argument("--out", default=LIVE_VAL,
                    help="the validation split to hold out; defaults to the LIVE 40-cluster file. "
                         "val_clusters.csv is the legacy 50-cluster draw -- see LIVE_VAL above.")
    a = ap.parse_args()

    # THE PINNED SPLIT IS NOT OVERWRITTEN. results/val_clusters.csv was drawn from a 557-cluster
    # pool; the pool is now 555, and redrawing changes 45 of its 50 clusters. Re-drawing a validation
    # split after the fact is what this phase's own docstring warns against, and it would happen on
    # any plain re-run. So an existing output is kept, and only train_clusters.csv is regenerated
    # from it -- which is also the check that the pinned split is still a subset of the pool.
    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} exists; NOT redrawing. Use --force to draw a new split.")
        # PRUNED, NOT REDRAWN. A later leakage check can exclude a cluster that is already in the
        # pinned split -- extending filter 2 to all 227 benchmark complexes excluded 4 of the 50
        # (125, 266, 278, 922). Those cannot stay: validating on a cluster that shares an antigen
        # with a benchmark complex is validating on data that leaks into the benchmark. Removing them
        # keeps the other 46 pinned, which is the minimal correct change; redrawing would replace 45
        # of 50 for no reason connected to the leak.
        v0 = pd.read_csv(a.out)
        pool0, _r0 = A.training_clusters(a.splits or A.SPLITS, a.excluded or A.EXCLUDED)
        poolk = {A._ckey(c) for c in pool0}
        gone = [c for c in v0.cluster if A._ckey(c) not in poolk]
        if gone:
            if not a.prune:
                raise SystemExit(
                    f"STOP: {len(gone)} pinned validation cluster(s) are no longer in the training "
                    f"pool: {sorted(gone)}\nThey were excluded by a later leakage check, so they "
                    "cannot be validated on. Re-run with --prune to drop exactly those and keep the "
                    "rest pinned, or --force to draw an entirely new split.")
            v0[~v0.cluster.isin(gone)].to_csv(a.out, index=False)
            print(f"  PRUNED {len(gone)} excluded cluster(s) from the pinned split: {sorted(gone)}")
            print(f"  validation is now {len(v0)-len(gone)} clusters; the rest stay pinned")
        keep, krows = train_minus_val(a.splits, a.excluded, a.out)
        tc = write_training(keep, krows, a.out, a.excluded or A.EXCLUDED)
        v = pd.read_csv(a.out)
        print(f"  pinned split: {len(v)} validation clusters, every one still in the pool")
        return

    t = cluster_table(a.splits, a.excluded).sort_values("median_depo").reset_index(drop=True)
    total = int(t.n_structures.sum())
    target = a.frac * total
    t["date_bin"] = pd.qcut(t.median_depo.rank(method="first"), a.n_bins, labels=False)
    # THE REMAINDER IS DISTRIBUTED, NOT DISCARDED. `n_val // n_bins` silently returned 50 clusters
    # for --n-val 51, so the file said 50 and the run reported success. The first (n_val % n_bins)
    # bins take one extra cluster each.
    base, extra = divmod(a.n_val, a.n_bins)
    per_bin = [base + (1 if b < extra else 0) for b in range(a.n_bins)]
    bins = [t.index[t.date_bin == b].to_numpy() for b in range(a.n_bins)]
    for b, (k, ix) in enumerate(zip(per_bin, bins)):
        if k > len(ix):
            raise SystemExit(f"STOP: bin {b} holds {len(ix)} clusters but {k} were asked for. "
                             f"Lower --n-val or --n-bins.")
    assert sum(per_bin) == a.n_val, (per_bin, a.n_val)

    # TWO SELECTION RULES. --select target (the default, and what results/val_clusters.csv was built
    # with) keeps the draw closest to --frac of the pool's structures. --select band accepts every
    # draw inside [--frac-lo, --frac-hi] and picks uniformly among them.
    #
    # WHAT IS KNOWN ABOUT THE DIFFERENCE, measured on this pool. Of 20,000 draws, 11,296 pass the
    # dominant-cluster rule; their held-out share runs 2.4% to 10.1%, median 4.5%. "target" with
    # --frac 0.10 therefore selects the single largest-holdout draw of the 11,296 -- the edge of the
    # distribution, not the middle. It gets there because holding out more structures requires
    # picking larger clusters: median cluster size 2 against 1, and 29 of its 50 clusters have two or
    # more distinct PDB entries against 22 for a typical draw. That matters because only clusters
    # with two entries can contribute two folded representatives, so "target" yields 79 folded
    # structures where a uniform draw yields a median of 72.
    #
    # WHAT IS NOT KNOWN: whether that makes validation easier or harder. Large clusters here are
    # NEWER (median deposition 2021-07-30 for clusters of 6+, against 2019-10-09 for singletons; only
    # 39% of their structures predate Protenix's 2021-09-30 cutoff, against 64%), so if anything they
    # are less likely to sit inside the model's training data. The rank correlation between cluster
    # size and date is 0.04. Post-cutoff coverage is identical either way -- 17 of 50 clusters -- as
    # the date binning guarantees. So "target" is an extreme draw on cluster size, in a direction
    # that is not established to flatter the result, and it buys 7 more folded structures.
    # It stays the default on that basis; "band" is kept because the bias is real and may matter once
    # per-cluster DockQ exists to measure it against.
    rng = np.random.default_rng(a.seed)
    sizes_all = t.n_structures.to_numpy()
    lo, hi = a.frac_lo * total, a.frac_hi * total
    ok, rejected_share, rejected_giant = [], 0, 0
    best_err = np.inf
    for _ in range(a.draws):
        pick = np.concatenate([rng.choice(ix, k, replace=False) for k, ix in zip(per_bin, bins)])
        sizes = sizes_all[pick]
        # one giant cluster can hit the target on its own; that is one system studied N times, not 50
        if sizes.max() > a.max_share * target:
            rejected_giant += 1
            continue
        if a.select == "target":
            err = abs(sizes.sum() - target)
            if err < best_err:
                ok, best_err = [pick], err
            continue
        if not (lo <= sizes.sum() <= hi):
            rejected_share += 1
            continue
        ok.append(pick)
    if a.select == "target":
        print(f"draws: {a.draws}   rejected {rejected_giant} for a dominant cluster   "
              f"kept the draw {best_err:.0f} structures from the {target:.0f} target")
    else:
        print(f"draws: {a.draws}   accepted {len(ok)}   "
              f"rejected {rejected_giant} for a dominant cluster, {rejected_share} outside "
              f"[{100*a.frac_lo:.0f}%, {100*a.frac_hi:.0f}%]")
    # NO SILENT None. The old code left `best` as None when every draw tripped --max-share and then
    # failed inside pandas with an error naming neither the cause nor the knob.
    if not ok:
        raise SystemExit(
            f"STOP: none of {a.draws} draws satisfied the constraints.\n"
            f"  {rejected_giant} had a cluster above --max-share ({a.max_share:.0%} of "
            f"{target:.0f} structures)\n"
            + (f"  {rejected_share} fell outside --frac-lo/--frac-hi "
               f"([{lo:.0f}, {hi:.0f}] structures)\n" if a.select == "band" else "")
            + "Raise --max-share or --draws"
            + (", or widen the band." if a.select == "band" else "."))
    # "target" leaves exactly one survivor; "band" leaves many and picks uniformly.
    best = ok[0] if a.select == "target" else ok[rng.integers(len(ok))]

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
    # THE TRAINING CLUSTER LIST, AS A FILE. train_minus_val() is the only enforcement of the split
    # and nothing calls it yet, because the training stage does not exist -- and a helper nobody is
    # obliged to call does not prevent the failure it was written for. Writing the list means the
    # training stage reads a file that already has validation removed, so using the full pool
    # requires deliberately going around the pipeline rather than merely forgetting a call.
    keep, krows = train_minus_val(a.splits, a.excluded, a.out)
    write_training(keep, krows, a.out, a.excluded or A.EXCLUDED)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
