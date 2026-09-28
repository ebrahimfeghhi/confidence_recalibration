#!/usr/bin/env python
"""Does templating raise the quality of the poses generated? Every pose, not the selected one.

SUMMARY
    Per interface, two pools of poses: the templated folds (the experiment's results.csv) and free +
    compute-matched (the first 50 free seeds plus the complex's matched extra seeds, chosen by
    compare_arms.matched_seeds as every reporter counts them). For each pool: the share of its poses that
    are acceptable (DockQ >= 0.23) and its mean pose DockQ. Pool sizes differ, so both measures are
    per-pose rates. Per class: the mean of each over interfaces, and on how many interfaces the templated
    pool's rate is higher / lower. EVALUATION ONLY: DockQ against the crystal.

INPUTS
    --dataset:  pxm2025 or foldbench. Default pxm2025.
    --exp:      the templated experiment. Default the dataset's self_template_fixed_50_seed.
    --out:      per-interface csv. Optional.

OUTPUTS
    stdout: one row per class; --out: iface, class, poses and rates for each pool

EXAMPLE
    analysis/pose_quality.py --out experimental_results/pose_quality_pxm2025.csv

FUNCTIONS
    rates(): DockQs -> (share acceptable, mean)
    main():  gather both pools per interface, tabulate
"""
import argparse
import collections
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import compare_arms as C  # noqa: E402
import paths  # noqa: E402
from fmt import table  # noqa: E402
from score_experiment import ACCEPTABLE, arm_defaults, classes_at, pool_by_seed  # noqa: E402


def rates(dq):
    """in: DockQs.  out: (share >= ACCEPTABLE, mean)"""
    return sum(x >= ACCEPTABLE for x in dq) / len(dq), statistics.mean(dq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    ds, exp, extra_csv = arm_defaults(A.dataset)
    A.exp = A.exp or exp

    pool, first = pool_by_seed(50, str(ds.pool_csv))
    cls = classes_at(pool, first)
    per, _ = C.donors_used(A.exp, first)
    extra = C.extra_free(extra_csv, first)
    tmpl = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{A.exp}/results.csv")):
        if r["dockq"]:
            tmpl[r["iface"]].append(float(r["dockq"]))
    by_pdb = collections.defaultdict(list)
    for i in pool:
        by_pdb[i.split("_")[0]].append(i)

    rows, unmatched = [], 0
    for iface in sorted(pool):
        pdb = iface.split("_")[0]
        if cls.get(iface) is None or not tmpl.get(iface):
            continue
        n = len(per.get(pdb, ()))
        seeds = C.matched_seeds(extra, by_pdb[pdb], n)
        if len(seeds) < n:
            unmatched += 1
            continue
        free = [dq for s, _, dq, _ in pool[iface] if s in first[pdb]]
        free += [dq for s in seeds for _, dq in extra[iface][s]]
        rows.append((iface, cls[iface], len(free), *rates(free), len(tmpl[iface]), *rates(tmpl[iface])))

    out = []
    for k in ("decoy", "solved", "nocov"):
        v = [r for r in rows if r[1] == k]
        if not v:
            continue
        m = lambda j: statistics.mean(r[j] for r in v)  # noqa: E731
        out.append([k, len(v), f"{100 * m(3):.1f}%", f"{100 * m(6):.1f}%", f"{m(4):.3f}", f"{m(7):.3f}",
                    sum(r[6] > r[3] for r in v), sum(r[6] < r[3] for r in v)])
    print(table(["class", "interfaces", "acceptable poses, free + matched", "acceptable poses, templated",
                 "mean pose DockQ, free + matched", "mean pose DockQ, templated",
                 "interfaces: templated share higher", "templated share lower"], out,
                align="l r r r r r r r"))
    print(f"\n  {A.dataset}  {A.exp}\n  interfaces excluded as not compute-matched: {unmatched}")
    if A.out:
        with open(A.out, "w") as fh:
            w = csv.writer(fh)
            w.writerow(["iface", "class", "free_poses", "free_acceptable_share", "free_mean_dockq",
                        "tmpl_poses", "tmpl_acceptable_share", "tmpl_mean_dockq"])
            w.writerows([r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}", r[5], f"{r[6]:.4f}", f"{r[7]:.4f}"]
                        for r in rows)
        print(f"  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
