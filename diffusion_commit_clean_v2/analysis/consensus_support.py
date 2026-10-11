#!/usr/bin/env python
"""Does STRUCTURAL CONSENSUS behind the top-ipTM pose tell solved interfaces from failed ones?

SUMMARY
    ipTM >= 0.8 is already a strong "trust this" rule: it keeps 84 FoldBench interfaces at 98%
    precision and 96 pxm2025 ones at 94%. What it misses is a third of the solved interfaces, which
    sit below 0.8 mixed in with the decoys. This asks whether one extra, theory-light number rescues
    them.

    SUPPORT of the selected pose = the fraction of the interface's other 249 poses whose pose-to-pose
    DockQ with it is at least `--tau`. The argument needs no literature: fifty independent folds
    converged on this placement, so it is not an artifact of one sampling run. On a solved interface
    the pool is coherent (all-pairs pose-to-pose 0.37 / 0.55) so the selected pose has company; on a
    decoy there is no dominant cluster at all (0.11 / 0.14).

    EVERYTHING IS MEASURED IN THE BAND THAT MATTERS. An AUC over all interfaces flatters any feature
    correlated with ipTM, because ipTM already separates most of them. The number reported first is
    therefore the AUC among interfaces with selected pair_iptm < 0.8, against the ipTM-alone baseline
    in that same band. A feature that only works where ipTM already works is worth nothing.

    DISCOVER ON ONE BENCHMARK, CONFIRM ON THE OTHER. Hand-picked thresholds overfit as readily as a
    trained classifier if enough are tried. Pick tau and the support cut looking only at FoldBench,
    then run pxm2025 once. The per-threshold sweep is printed for both so that discipline is the
    reader's to apply, not something this file silently does for them.

    GROUND TRUTH is used only to label solved / decoy / nocov and to report the result. The support
    statistic itself reads no crystal: it is computable at prediction time.

INPUTS
    --dataset:  pxm2025 or foldbench, repeatable. Default: both.
    --tau:      pose-to-pose DockQ at which two poses count as the same placement. Default 0.5.
    --iptm-cut: the published trust threshold. Default 0.8.
    --out:      per-interface csv. Optional.

OUTPUTS
    stdout: median support per group; AUC (solved vs decoy+nocov) overall and below the ipTM cut,
            for support and for ipTM alone; then a sweep of the combined two-threshold rule
    --out:  dataset, iface, class, sel_iptm, support, n_poses

EXAMPLE
    analysis/consensus_support.py --out ../experimental_results/consensus_support.csv

FUNCTIONS
    auc():     Mann-Whitney AUC, ties at half
    support(): one interface -> fraction of poses agreeing with the selected pose
    main():    per dataset, tabulate, sweep
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

ROOT = os.path.dirname(os.path.dirname(HERE))
DEAD = {"9r10"}


def auc(pos, neg):
    """P(a random positive scores above a random negative), ties at half. None if either is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", choices=paths.DATASET_NAMES, default=None)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--iptm-cut", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    datasets = A.dataset or list(paths.DATASET_NAMES)
    allrows = []

    for dsn in datasets:
        DS = paths.DATASETS[dsn]
        pool, first = pool_by_seed(50, str(DS.pool_csv))
        cls = classes_at(pool, first)
        mdir = str(DS.matrices)
        rows, missing = [], 0
        for iface, c in sorted(cls.items()):
            pdb = iface.split("_")[0]
            if pdb in DEAD or c not in ("solved", "decoy", "nocov"):
                continue
            keep = [r for r in pool[iface] if r[0] in first[pdb]]
            if not keep:
                continue
            sel = max(keep, key=lambda r: r[3])            # top pair_iptm pose: (seed, sample, dq, ip)
            mp = f"{mdir}/{iface}.npz"
            if not os.path.exists(mp):
                missing += 1
                continue
            z = np.load(mp, allow_pickle=True)
            key = [(str(s), int(x)) for s, x in zip(z["seed"], z["sample"])]
            try:
                i = key.index((str(sel[0]), int(sel[1])))
            except ValueError:
                missing += 1                               # selected pose was dropped from the matrix
                continue
            row = np.delete(z["dockq"][i], i)
            sup = float((row >= A.tau).mean())
            rows.append((iface, c, float(sel[3]), sup, len(row) + 1))
            allrows.append((dsn,) + rows[-1])

        lo = [r for r in rows if r[2] < A.iptm_cut]
        print(f"\n=== {dsn} ===  n={len(rows)} interfaces"
              + (f"  ({missing} skipped: no matrix or selected pose absent)" if missing else ""))
        print(f"  support = share of the other poses within pose-to-pose DockQ >= {A.tau} "
              f"of the selected pose")
        print(f"  {'group':30}{'n':>5}{'med support':>13}{'med sel iptm':>14}")
        for lbl, sel_fn in (("solved, iptm >= cut", lambda r: r[1] == "solved" and r[2] >= A.iptm_cut),
                            ("solved, iptm <  cut", lambda r: r[1] == "solved" and r[2] < A.iptm_cut),
                            ("decoy,  iptm <  cut", lambda r: r[1] == "decoy" and r[2] < A.iptm_cut),
                            ("nocov,  iptm <  cut", lambda r: r[1] == "nocov" and r[2] < A.iptm_cut)):
            v = [r for r in rows if sel_fn(r)]
            if v:
                print(f"  {lbl:30}{len(v):>5}{st.median(r[3] for r in v):>13.3f}"
                      f"{st.median(r[2] for r in v):>14.3f}")

        for band, lbl in ((rows, "ALL interfaces"), (lo, f"iptm < {A.iptm_cut} ONLY  <-- the band that matters")):
            pos = [r[3] for r in band if r[1] == "solved"]
            neg = [r[3] for r in band if r[1] != "solved"]
            pi = [r[2] for r in band if r[1] == "solved"]
            ni = [r[2] for r in band if r[1] != "solved"]
            a_s, a_i = auc(pos, neg), auc(pi, ni)
            if a_s is None:
                continue
            print(f"  {lbl:50} n={len(band):4} ({len(pos)} solved)  "
                  f"AUC support {a_s:.3f}   AUC ipTM alone {a_i:.3f}")

        # the two-threshold rule: trust if iptm >= cut OR support >= s
        print(f"\n  rule: trust if  iptm >= {A.iptm_cut}  OR  support >= s")
        print(f"  {'s':>6}{'kept':>7}{'solved kept':>13}{'precision':>11}"
              f"{'recall of solved':>18}{'vs iptm-only':>14}")
        base_keep = [r for r in rows if r[2] >= A.iptm_cut]
        n_solved = sum(1 for r in rows if r[1] == "solved")
        bp = sum(1 for r in base_keep if r[1] == "solved")
        print(f"  {'(none)':>6}{len(base_keep):>7}{bp:>13}{bp/max(len(base_keep),1):>11.2f}"
              f"{bp/max(n_solved,1):>18.2f}{'baseline':>14}")
        for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            k = [r for r in rows if r[2] >= A.iptm_cut or r[3] >= s]
            g = sum(1 for r in k if r[1] == "solved")
            print(f"  {s:>6.1f}{len(k):>7}{g:>13}{g/max(len(k),1):>11.2f}"
                  f"{g/max(n_solved,1):>18.2f}{g-bp:>+14}")

    if A.out:
        os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
        with open(A.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "iface", "class", "sel_iptm", "support", "n_poses"])
            w.writerows([r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}", r[5]] for r in allrows)
        print(f"\n  wrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
