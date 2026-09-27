#!/usr/bin/env python
"""Donors per complex and donor quality for one built experiment, on either dataset.

SUMMARY
    Answers "how many donors did selection give each complex, and how good are they" before any GPU
    time is spent folding them. The definitions are crystal_frame_selection.py's (its donors(),
    donor_quality() and summarise() are imported, not copied), so the numbers are comparable with
    crystal_frame_selection.md:

      donors per complex   distinct donor poses in queue.tsv, binned
      donors per class     per benchmark interface, its complex's donor count (a template serves the whole
                           complex), summarised per class: median, mean, how many at the cap, under 10
      donor quality        EVALUATION, against the crystal: for each benchmark interface, the DockQ of
                           every donor of its complex (a template is a whole complex and a fold is scored
                           on every interface), from the dataset's pool csv. Per class (solved / decoy /
                           nocov on the first 50 seeds): interfaces with at least one acceptable donor
                           (DockQ >= 0.23), the median best donor DockQ, the mean donor DockQ

INPUTS
    --dataset:  pxm2025 or foldbench. Default pxm2025.
    --exp:      the experiment whose queue.tsv is read. Default the dataset's self_template_fixed_50_seed.
    --out:      also write the tables here as markdown. Optional.

OUTPUTS
    stdout (and --out): the three tables

EXAMPLE
    analysis/donor_summary.py --dataset foldbench \\
        --out experimental_data/protenix/foldbench/self_template_fixed_50_seed/donor_summary.md

FUNCTIONS
    per_class(): donors per complex and the classes -> per-class donor counts, per interface
    main():      read the queue and the pool, tabulate
"""
import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402
from crystal_frame_selection import donor_quality, donors, summarise  # noqa: E402
from fmt import table  # noqa: E402
from score_experiment import arm_defaults, classes_at, pool_by_seed  # noqa: E402

BINS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 49), (50, 50)]
CAP = 50


def per_class(n, cls, classes):
    """in: {complex: donors}, {interface: class}, the classes.  out: table rows, one per class, each
    interface counted with its complex's donor count"""
    rows = []
    for k in classes:
        v = [n.get(i.split("_")[0], 0) for i, c in cls.items() if c == k]
        if v:
            rows.append([k, len(v), f"{statistics.median(v):g}", f"{statistics.mean(v):.1f}",
                         sum(x == CAP for x in v), sum(x < 10 for x in v)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    ds, exp, _ = arm_defaults(A.dataset)
    A.exp = A.exp or exp

    sel = donors(f"{A.exp}/queue.tsv")
    pool, first = pool_by_seed(50, str(ds.pool_csv))
    pool = {i: [r for r in rows if r[0] in first[i.split("_")[0]]] for i, rows in pool.items()}
    cls = classes_at(pool, first)
    missing = sorted({i.split("_")[0] for i in pool} - set(sel))

    v = sorted(len(x) for x in sel.values())
    counts = [[len(sel), sum(v), f"{statistics.median(v):g}", f"{statistics.mean(v):.1f}"]
              + [sum(1 for x in v if a <= x <= b) for a, b in BINS]]
    q, unscored = donor_quality(sel, pool)
    lines = [f"# Donors: {A.dataset}, `{os.path.basename(A.exp)}`", "",
             "Produced by `analysis/donor_summary.py`. Donor DockQ is against the crystal at each "
             "benchmark interface (evaluation only; selection never reads it).", "",
             "## Donors per complex", "",
             table(["complexes", "donors", "median", "mean"]
                   + [f"{a}-{b}" if a != b else f"{a} (cap)" for a, b in BINS], counts,
                   align=" ".join("r" * 10)), "",
             f"benchmark complexes with no donor: {len(missing)}" + (f" ({', '.join(missing)})" if missing else ""),
             "", "## Donors per class (benchmark interfaces)", "",
             "Each interface is counted with its complex's donor count: a template is a whole complex.", "",
             table(["class", "interfaces", "median donors", "mean donors", f"at the cap ({CAP})",
                    "under 10"], per_class({p: len(v) for p, v in sel.items()}, cls,
                                           ["decoy", "solved", "nocov"]), align="l r r r r r"), "",
             "## Donor quality per class (benchmark interfaces)", "",
             table(["class", "interfaces", "any acceptable donor", "median best donor DockQ",
                    "mean donor DockQ"], summarise(q, cls, ["decoy", "solved", "nocov"]),
                   align="l r r r r"), "",
             f"donor-interface pairs without a crystal score: {unscored}", ""]
    text = "\n".join(lines)
    print(text)
    if A.out:
        open(A.out, "w").write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
