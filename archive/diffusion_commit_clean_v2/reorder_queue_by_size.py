#!/usr/bin/env python
"""Reorder a queue so complexes run smallest-first, grouped by class.

    ./reorder_queue_by_size.py <queue.tsv> [<queue.tsv> ...] [--apply] [--pair]

Default is a DRY RUN that prints the new order; --apply rewrites the files in place after saving
`<queue>.bak`. Rows are only moved, never changed: the script asserts the new file is a permutation
of the old one (same multiset of lines, same count), so a reorder can never add, drop or edit a fold.

Order is class group first -- decoy-containing, then nocov, then solved -- and ATOM COUNT ASCENDING
within each group. Small complexes finish faster, so the rows of the table fill sooner at the same
total cost; the class grouping stays on top because which class fills first is a deliberate choice.

--pair applies ONE order to every queue named, built from the union of their complexes. The matched
baseline and the method arm are compared interface by interface, so they have to cover the same
complexes at the same time; letting each sort itself would drift them apart and leave the table with
one arm's complexes finished and the other's not.

A running worker does NOT pick this up: folder.py reads the queue once at startup. Restart the
workers afterwards -- claims and results.csv make that idempotent, so nothing is refolded.
"""
import argparse
import collections
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/tts_abag")

import compare_arms as C  # noqa: E402
from score_experiment import classes_at, pool_by_seed  # noqa: E402

SIZES = ("/home/ebrahim/tts_abag/experimental_data/protenix/foldbench/"
         "selftmpl_mask_on_nc10/sizes.json")
GROUP = {"decoy": 0, "nocov": 1, "solved": 2}


def complex_class(dataset):
    """out: {complex: class} -- a complex holding a decoy interface IS a decoy."""
    ds, _, _ = C.arm_defaults(dataset)
    pool, first = pool_by_seed(50, str(ds.pool_csv))
    per = collections.defaultdict(set)
    for iface, k in classes_at(pool, first).items():
        per[iface.split("_")[0]].add(k)
    return {p: "decoy" if "decoy" in s else ("nocov" if "nocov" in s and "solved" not in s
                                             else "solved")
            for p, s in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queues", nargs="+")
    ap.add_argument("--dataset", default="foldbench")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--pair", action="store_true", help="one shared order across every queue")
    A = ap.parse_args()

    klass = complex_class(A.dataset)
    sizes = json.load(open(SIZES))
    files = {q: [l.rstrip("\n") for l in open(q) if l.strip()] for q in A.queues}

    def key(p):
        # a complex with no cached size sorts last in its group rather than crashing the reorder
        return (GROUP[klass.get(p, "solved")], sizes.get(p, 10 ** 9), p)

    shared = sorted({l.split("\t")[1] for ls in files.values() for l in ls}, key=key) \
        if A.pair else None

    for q, lines in files.items():
        by = collections.defaultdict(list)
        for l in lines:
            by[l.split("\t")[1]].append(l)
        order = [p for p in shared if p in by] if shared else sorted(by, key=key)
        out = [l for p in order for l in by[p]]
        assert collections.Counter(out) == collections.Counter(lines), \
            f"{q}: reorder is not a permutation -- refusing to write"
        print(f"\n{os.path.basename(os.path.dirname(q))}: {len(order)} complexes, {len(out)} folds")
        print("  " + " ".join(f"{p}:{klass.get(p,'?')[0]}{sizes.get(p,0)//1000}k" for p in order))
        if A.apply:
            if not os.path.exists(q + ".bak"):
                shutil.copy2(q, q + ".bak")
            with open(q, "w") as fh:
                fh.write("\n".join(out) + "\n")
            print("  WRITTEN (backup at " + os.path.basename(q) + ".bak)")

    if not A.apply:
        print("\ndry run -- pass --apply to rewrite, then restart the workers")


if __name__ == "__main__":
    main()
