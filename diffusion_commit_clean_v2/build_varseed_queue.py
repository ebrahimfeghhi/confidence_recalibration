#!/usr/bin/env python
"""Write the variable-seed arm's queue: the fixed arm's donors, each on its own compute-matched free seed.

SUMMARY
    The variable-seed arm folds the SAME donors with the SAME templates as the fixed-seed arm, but each
    donor on its own seed instead of all on 430000. Its seeds are exactly the free seeds that
    compare_arms.py counts as the complex's compute match (compare_arms.matched_seeds: extra free seeds
    scored on every interface, lowest-numbered first, as many as the complex has donors). So free +
    matched and the variable-seed arm spend the same folds -- the same seeds, once free and once
    templated.

    Per complex, the seeds are shuffled with random.Random(f"varseed-{pdb}") and assigned to its donors
    in (interface, tag) order, so which donor gets which seed is fixed and reproducible but carries no
    order. Seeds a free top-up has queued but not yet scored can be counted with --pending; rerun
    without it once they are scored, and the queue must come out identical.

    With --order-by-class, whole complexes are queued by the best class among their interfaces --
    any confident decoy first, then any solved, then no acceptable pose -- so the decoy results come in
    first. The classes come from the crystal (score_experiment.classes_at), but they only set the ORDER
    folds run in: every complex keeps the same donors, seeds and templates, so the results are the same
    whichever order is used. Complexes stay whole because an interface is only finished once all its
    complex's folds are scored.

INPUTS
    --fixed:    the fixed-seed arm's directory; its queue.tsv gives the donors. Required.
    --out:      the queue to write. Required.
    --pending:  free queues whose seeds count though not yet scored, colon-separated. Optional.
    --extra-csv: free results csvs, colon-separated. Default the free pool and the compute-matched run.
    --order-by-class: queue complexes decoy first, then solved, then nocov. Default sorted by pdb.

OUTPUTS
    <out>: one line per donor -- the fixed arm's line with the seed replaced

EXAMPLE
    build_varseed_queue.py --fixed experimental_data/self_template_fixed_50_seed \\
        --pending experimental_data/free_compute_matched/queue_free_topup_arm_a.tsv \\
        --order-by-class --out experimental_data/self_template_variable_50_seed/queue.tsv

FUNCTIONS
    class_rank(): the complex -> 0 any decoy interface, 1 any solved, 2 nocov only, 3 unclassed
    main(): read the donors and the free seeds, assign, write, and verify every complex is matched
"""
import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
from compare_arms import extra_free, matched_seeds  # noqa: E402
from score_experiment import classes_at, pool_by_seed  # noqa: E402

RANK = {"decoy": 0, "solved": 1, "nocov": 2}


def class_rank(cls, ifaces):
    """in: {interface: class}, the complex's interfaces.  out: 0 any decoy, 1 any solved, 2 nocov
    only, 3 none classed -- scheduling order only"""
    return min((RANK.get(cls.get(i), 3) for i in ifaces), default=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pending", default="")
    ap.add_argument("--extra-csv",
                    default=f"{paths.FREE_DIR}/results.csv:{paths.DATA_DIR}/free_compute_matched/results.csv")
    ap.add_argument("--order-by-class", action="store_true")
    A = ap.parse_args()

    pool, first = pool_by_seed(50)
    extra = extra_free(A.extra_csv, first)
    ifaces = collections.defaultdict(list)
    for i in pool:
        ifaces[i.split("_")[0]].append(i)
    pending = collections.defaultdict(set)
    for q in filter(None, A.pending.split(":")):
        for line in open(q):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 5:
                pending[f[1]].add(f[4])

    donors = collections.defaultdict(list)
    for line in open(f"{A.fixed}/queue.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            donors[f[1]].append(f)
    order = sorted(donors)
    if A.order_by_class:
        cls = classes_at(pool, first)
        order.sort(key=lambda p: class_rank(cls, ifaces[p]))
    out, short, n_pending = [], [], 0
    for pdb in order:
        rows = sorted(donors[pdb], key=lambda f: (f[2], f[3]))
        seeds = matched_seeds(extra, ifaces[pdb], len(rows), pending[pdb])
        if len(seeds) < len(rows):
            short.append(f"{pdb} ({len(seeds)}/{len(rows)})")
            continue
        n_pending += len(set(seeds) & pending[pdb])
        random.Random(f"varseed-{pdb}").shuffle(seeds)
        out += [f[:4] + [s] + f[5:] for f, s in zip(rows, seeds)]
    if short:
        raise SystemExit(f"{len(short)} complexes have fewer matched free seeds than donors: {short[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, "w") as fh:
        fh.writelines("\t".join(f) + "\n" for f in out)
    print(f"  {len(out)} folds over {len(donors)} complexes, each donor on its complex's matched free "
          f"seed ({n_pending} of them pending in a top-up)  -> {A.out}")
    if A.order_by_class:
        n = collections.Counter(class_rank(cls, ifaces[p]) for p in order)
        print(f"  order: {n[0]} complexes with a decoy, then {n[1]} with a solved, then {n[2]} nocov only"
              f"{f', then {n[3]} unclassed' if n[3] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
