#!/usr/bin/env python
"""How much does templating raise ipTM, for correct donors vs wrong ones? Decoy or solved interfaces.

SUMMARY
    Templating flipped the top-ipTM pick to an acceptable pose on 20 of 89 pxm2025 decoy interfaces
    without raising pose quality (pose_quality.py). The hypothesis: every donor's fold gets more
    confident, and correct donors get the bigger lift. This reads, per donor fold:
      free ipTM / DockQ   the donor pose itself, in the free pool, at this interface
      templated ipTM      the highest ipTM among the fold's 5 samples, and that sample's DockQ
      lift                templated ipTM - free ipTM
    A donor is CORRECT at an interface if its free DockQ >= 0.23. An interface is RESCUED if the
    highest-ipTM templated pose over all its folds is acceptable (the "method (all)" arm of
    compare_arms.py); with --cls solved the groups are BROKEN (that pose is not acceptable) and kept. Scores only -- no poses are read, so the 60% of folds without poses on disk count.
    EVALUATION ONLY: DockQ against the crystal.

INPUTS
    --dataset:  pxm2025 or foldbench. Default pxm2025.
    --exp:      the templated experiment. Default the dataset's self_template_fixed_50_seed.
    --cls:      decoy (default; groups rescued / not rescued) or solved (groups broken / kept).
    --out:      per-donor-fold csv. Optional.

OUTPUTS
    stdout: per group: medians over interfaces of the correct- and
            wrong-donor medians, and the ipTM gap between the best wrong and best correct donor
            before and after templating
    --out:  iface, rescued, arm, donor_correct, free_iptm, free_dockq, tmpl_iptm, tmpl_dockq, lift,
            tmpl_acceptable_share

EXAMPLE
    analysis/donor_lift.py --out experimental_results/donor_lift_pxm2025.csv
    analysis/donor_lift.py --dataset foldbench --cls solved --out experimental_results/foldbench/donor_lift_solved.csv

FUNCTIONS
    donor_key(): donor path -> (seed, sample)
    folds():     results.csv -> {interface: {arm: (donor, [(iptm, dockq)])}}
    main():      join donors to the free pool, summarise
"""
import argparse
import collections
import csv
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402
from fmt import table  # noqa: E402
from score_experiment import ACCEPTABLE, arm_defaults, classes_at, pool_by_seed  # noqa: E402

DONOR = re.compile(r"/free/[0-9a-z]{4}/(\d+)-[^/]*/predictions/[^/]*_sample_(\d+)\.cif$")


def donor_key(path):
    """in: a donor pose path.  out: (seed, sample), or None if the path does not parse"""
    m = DONOR.search(path)
    return (m.group(1), int(m.group(2))) if m else None


def folds(exp):
    """in: the experiment.  out: {interface: {arm: (donor path, [(iptm, dockq)])}}"""
    out = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{exp}/results.csv")):
        if not (r["dockq"] and r["iptm"]):
            continue
        d = out[r["iface"]].setdefault(r["arm"], (r["donor_pose"], []))
        d[1].append((float(r["iptm"]), float(r["dockq"])))
    return out


def med(v):
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--cls", choices=["decoy", "solved"], default="decoy")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    ds, exp, _ = arm_defaults(A.dataset)
    A.exp = A.exp or exp

    pool, first = pool_by_seed(50, str(ds.pool_csv))
    cls = classes_at(pool, first)
    F = folds(A.exp)

    recs, unjoined = [], 0
    for iface in sorted(F):
        if cls.get(iface) != A.cls:
            continue
        free = {(s, k): (dq, ip) for s, k, dq, ip in pool[iface]}
        allrows = [x for _, rows in F[iface].values() for x in rows]
        rescued = max(allrows)[1] >= ACCEPTABLE
        for arm, (donor, rows) in sorted(F[iface].items()):
            key = donor_key(donor)
            if key not in free:
                unjoined += 1
                continue
            fdq, fip = free[key]
            tip, tdq = max(rows)
            recs.append(dict(iface=iface, rescued=int(rescued), arm=arm,
                             donor_correct=int(fdq >= ACCEPTABLE),
                             free_iptm=fip, free_dockq=fdq, tmpl_iptm=tip, tmpl_dockq=tdq,
                             lift=tip - fip,
                             tmpl_acceptable_share=sum(d >= ACCEPTABLE for _, d in rows) / len(rows)))

    by = collections.defaultdict(list)
    for r in recs:
        by[r["iface"]].append(r)
    out_rows = []
    groups = ((("rescued", 1), ("not rescued", 0)) if A.cls == "decoy"
              else (("broken", 0), ("kept", 1)))
    for label, flag in groups:
        ifs = [i for i, rs in by.items() if rs[0]["rescued"] == flag]
        both = [i for i in ifs if {r["donor_correct"] for r in by[i]} == {0, 1}]
        stat = collections.defaultdict(list)
        bigger = 0
        for i in both:
            c = [r for r in by[i] if r["donor_correct"]]
            w = [r for r in by[i] if not r["donor_correct"]]
            for side, rs in (("c", c), ("w", w)):
                stat[side + "_free"].append(med([r["free_iptm"] for r in rs]))
                stat[side + "_tmpl"].append(med([r["tmpl_iptm"] for r in rs]))
                stat[side + "_lift"].append(med([r["lift"] for r in rs]))
                stat[side + "_stay"].append(med([r["tmpl_acceptable_share"] for r in rs]))
            bigger += stat["c_lift"][-1] > stat["w_lift"][-1]
            # gap: best wrong donor's ipTM minus best correct donor's, before and after templating
            stat["gap_free"].append(max(r["free_iptm"] for r in w) - max(r["free_iptm"] for r in c))
            stat["gap_tmpl"].append(max(r["tmpl_iptm"] for r in w) - max(r["tmpl_iptm"] for r in c))
        out_rows.append([label, len(ifs), len(both),
                         sum(r["donor_correct"] for i in both for r in by[i]),
                         sum(1 - r["donor_correct"] for i in both for r in by[i]),
                         med(stat["c_free"]), med(stat["w_free"]),
                         med(stat["c_tmpl"]), med(stat["w_tmpl"]),
                         med(stat["c_lift"]), med(stat["w_lift"]),
                         f"{bigger}/{len(both)}",
                         med(stat["gap_free"]), med(stat["gap_tmpl"]),
                         med(stat["c_stay"])])

    head = [f"{A.cls} interfaces", "n", "n with correct and wrong donors", "correct donor folds",
            "wrong donor folds", "free ipTM: correct", "free ipTM: wrong",
            "templated ipTM: correct", "templated ipTM: wrong", "lift: correct", "lift: wrong",
            "correct lift > wrong lift", "gap free (best wrong - best correct)",
            "gap templated", "correct donor: share of samples acceptable"]
    print(table(head, out_rows, dec={c: 3 for c in head[5:]}))
    print(f"\n  dataset: {A.dataset}   experiment: {A.exp}\n"
          f"  each ipTM/lift column: median over interfaces of the per-interface median over donor folds\n"
          f"  templated ipTM = highest of the fold's 5 samples; donor folds not joined to the pool: {unjoined}")
    if A.out:
        with open(A.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            for r in recs:
                w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
        print(f"  wrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
