#!/usr/bin/env python
"""Which (interface, seed) should the next REPEAT arm fold 50 times? Chosen without a crystal.

SUMMARY
    Folds do not reproduce. Re-folding one seed draws a new trunk outcome each time, and at 9n6e_GF
    seed 300046 those draws split 14 clean / 10 mixed over 24 repeats of one seed -- "clean" meaning
    all five diffusion samples acceptable. ipTM reads that fold-level commitment and nothing finer
    (within-fold pair_iptm spread, median 0.008), so a clean draw wins the argmax and a mixed one
    loses it. Repeating one seed is therefore a way to re-roll the ipTM lottery until the trunk
    commits, and it rescued 9n6e_GF from DockQ 0.022 to 0.644 at 46% of the free-50 budget.

    This file picks the next target. It must do so WITHOUT the crystal, or the method is circular.

THE RULE
    Within one fold, average the pose-to-pose DockQ over its five samples' ten pairs. That is the
    fold's AGREEMENT: high means the five poses are the same placement, low means the fold is
    internally split. It reads no reference structure -- it compares predictions to each other.
    The candidate seed for an interface is its MINIMUM-agreement fold: the one most spread apart,
    which is the signature of a trunk that failed to commit and so the one with room to be re-rolled.

    Interfaces are restricted to class `decoy` (an acceptable pose exists in the pool but the
    top-pair_iptm pose is not one). That restriction DOES use the crystal, which is legitimate: it
    selects the test cases to spend GPU hours on, exactly as picking a benchmark does. The seed
    within the interface must not, because that is the step a deployed method would have to perform.

WHAT IS REPORTED BUT NEVER SELECTED ON
    Oracle DockQ of the pool, the candidate fold's own best and worst DockQ, and the DockQ of the
    pose ipTM currently picks. These say whether the pick was lucky and are printed last for that
    reason. Selecting on any of them would make the result unusable.

CALIBRATION
    9n6e_GF is printed separately whatever its rank, with seed 300046 marked, so the threshold that
    worked once is visible beside the candidates rather than remembered.

INPUTS
    --dataset:  pxm2025 or foldbench, repeatable. Default: both.
    --seeds:    how many seeds define the pool and the classes. Default 50, as every reporter.
    --top:      how many candidates to print per dataset. Default 12.
    --min-poses: skip a fold with fewer than this many poses in the matrix. Default 5.
    --out:      per-interface csv. Optional.

OUTPUTS
    stdout: per dataset, the decoys ranked by candidate-fold agreement ascending, then 9n6e_GF
    --out:  dataset, iface, n_seeds, cand_seed, cand_agreement, cand_n_poses, iface_med_agreement,
            cand_mean_iptm, cand_best_dockq, cand_worst_dockq, pool_oracle, pool_sel_dockq

EXAMPLE
    analysis/pick_repeat_seed.py --dataset pxm2025 --top 15

FUNCTIONS
    fold_agreement(): matrix + row indices -> mean pose-to-pose DockQ over the fold's pairs
    interface_rows(): one dataset -> one candidate row per decoy interface
    main():           rank, print, calibrate against 9n6e_GF
"""
import argparse
import collections
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

DEAD = {"9r10"}
CALIBRATION = ("9n6e_GF", "300046")


def fold_agreement(dq, idx):
    """out: (mean, max) pose-to-pose DockQ over every pair within one fold. None if <2 poses.

    BOTH are needed and the calibration is why. Ranking decoys by MEAN alone (most spread apart)
    picks folds whose five poses are each wrong in a different direction -- on 9n6e_GF it picks
    seed 300038, whose best pose is DockQ 0.127, and ranks the interface 44/87. Those folds have
    nothing to re-roll: no placement in them is right.

    A MIXED fold is the target instead: a subset of poses that agree strongly with each other,
    embedded in disagreement. Its mean is low but its MAX pair is high, because the good poses
    agree with one another. High max inside a low mean is the label-free signature of a trunk that
    found a placement and then failed to commit to it, which is exactly what re-rolling fixes.
    """
    if len(idx) < 2:
        return None
    sub = dq[np.ix_(idx, idx)]
    pairs = sub[np.triu_indices(len(idx), 1)]
    return float(pairs.mean()), float(pairs.max())


def interface_rows(dsn, n_seeds, min_poses, rule):
    """out: list of candidate rows, one per decoy interface that has a matrix."""
    DS = paths.DATASETS[dsn]
    pool, first = pool_by_seed(n_seeds, str(DS.pool_csv))
    cls = classes_at(pool, first)
    mdir = str(DS.matrices)
    rows = []
    for iface, c in sorted(cls.items()):
        pdb = iface.split("_")[0]
        if pdb in DEAD or c != "decoy":
            continue
        mp = f"{mdir}/{iface}.npz"
        if not os.path.exists(mp):
            continue
        z = np.load(mp, allow_pickle=True)
        dq, seeds = z["dockq"], [str(s) for s in z["seed"]]
        ip = z["pair_iptm"] if "pair_iptm" in z.files else None

        by_seed = collections.defaultdict(list)
        for i, s in enumerate(seeds):
            by_seed[s].append(i)
        agr, mx = {}, {}
        for s, idx in by_seed.items():
            if len(idx) < min_poses:
                continue
            a = fold_agreement(dq, idx)
            if a is not None:
                agr[s], mx[s] = a
        if not agr:
            continue
        if rule == "minmean":
            cand = min(agr, key=agr.get)
        elif rule == "maxpair":
            cand = max(mx, key=mx.get)
        else:                                      # maxgap: strongest agreeing subset, least overall
            cand = max(agr, key=lambda s: mx[s] - agr[s])
        ci = by_seed[cand]

        # reported, never selected on
        keep = [r for r in pool[iface] if r[0] in first[pdb]]
        oracle = max(r[2] for r in keep) if keep else float("nan")
        sel_dq = max(keep, key=lambda r: r[3])[2] if keep else float("nan")
        # the candidate fold's own DockQ against the crystal comes from the pool csv, not the matrix
        cpool = [r for r in keep if r[0] == cand]
        cbest = max((r[2] for r in cpool), default=float("nan"))
        cworst = min((r[2] for r in cpool), default=float("nan"))
        cip = float(np.mean([ip[i] for i in ci])) if ip is not None else float("nan")

        rows.append(dict(
            dataset=dsn, iface=iface, n_seeds=len(by_seed), cand_seed=cand,
            cand_agreement=agr[cand], cand_maxpair=mx[cand], cand_gap=mx[cand] - agr[cand],
            cand_n_poses=len(ci),
            iface_med_agreement=st.median(agr.values()), cand_mean_iptm=cip,
            cand_best_dockq=cbest, cand_worst_dockq=cworst,
            pool_oracle=oracle, pool_sel_dockq=sel_dq,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", choices=paths.DATASET_NAMES, default=None)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-poses", type=int, default=5)
    ap.add_argument("--rule", choices=("minmean", "maxpair", "maxgap"), default="maxgap",
                    help="minmean = most spread apart (FAILS calibration); maxpair = strongest "
                         "agreeing subset; maxgap = strong subset inside a split fold (default)")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    allrows = []
    for dsn in (A.dataset or list(paths.DATASET_NAMES)):
        rows = interface_rows(dsn, A.seeds, A.min_poses, A.rule)
        allrows += rows
        rows.sort(key=lambda r: -r["cand_gap"] if A.rule != "minmean" else r["cand_agreement"])
        print(f"\n=== {dsn} ===  {len(rows)} decoy interfaces with a pose-to-pose matrix")
        print("  LABEL-FREE columns: cand seed, agreement (mean pose-to-pose DockQ in that fold),"
              " its poses, interface-median agreement, fold-mean pair_iptm")
        print("  CRYSTAL columns (reported, NOT selected on): the fold's best/worst DockQ,"
              " the pool's oracle and what ipTM currently picks")
        hdr = (f"  {'interface':14}{'cand':>8}{'mean':>7}{'maxpr':>7}{'gap':>7}{'medagr':>8}"
               f"{'iptm':>7}{'|':>3}{'fBest':>7}{'fWorst':>8}{'oracle':>8}{'selDQ':>7}")
        print(hdr)
        for r in rows[:A.top]:
            print(f"  {r['iface']:14}{r['cand_seed']:>8}{r['cand_agreement']:>7.3f}"
                  f"{r['cand_maxpair']:>7.3f}{r['cand_gap']:>7.3f}"
                  f"{r['iface_med_agreement']:>8.3f}{r['cand_mean_iptm']:>7.3f}"
                  f"{'|':>3}{r['cand_best_dockq']:>7.3f}{r['cand_worst_dockq']:>8.3f}"
                  f"{r['pool_oracle']:>8.3f}{r['pool_sel_dockq']:>7.3f}")

        cal = [r for r in rows if r["iface"] == CALIBRATION[0]]
        if cal:
            r = cal[0]
            rank = rows.index(r) + 1
            mark = "  <-- the seed that worked" if r["cand_seed"] == CALIBRATION[1] else \
                   f"  <-- NOTE: rule picks {r['cand_seed']}, the arm used {CALIBRATION[1]}"
            print(f"\n  CALIBRATION  {r['iface']} ranks {rank}/{len(rows)} by this rule")
            print(f"  {r['iface']:14}{r['cand_seed']:>8}{r['cand_agreement']:>7.3f}"
                  f"{r['cand_maxpair']:>7.3f}{r['cand_gap']:>7.3f}"
                  f"{r['iface_med_agreement']:>8.3f}{r['cand_mean_iptm']:>7.3f}"
                  f"{'|':>3}{r['cand_best_dockq']:>7.3f}{r['cand_worst_dockq']:>8.3f}"
                  f"{r['pool_oracle']:>8.3f}{r['pool_sel_dockq']:>7.3f}{mark}")

    if A.out:
        os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
        with open(A.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        print(f"\n  wrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
