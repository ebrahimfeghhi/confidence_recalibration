#!/usr/bin/env python
"""Is a seed complete, PER INTERFACE, and which interfaces are still missing.

WHY NOT pool.csv's INTERFACE COUNT. pool.csv names 51 interfaces across the 22 degenerate
structures; only 48 have ever been scored, at any seed, by any route. 6gv4_BE, 8dv6_BD and 9bbn_FH
are the 1wej/4p3c class: OST groups the antibody H and L chains into one chemical class, leaves a
reference chain unassigned and drops the interface. Counting against pool.csv therefore demands rows
that nothing can produce, and a gate built on it can never be satisfied.

SO THE AUTHORITY IS A REFERENCE SEED. An interface counts as scorable when the reference seed holds
a full set of samples for it. That is measured, it names the interfaces rather than counting them,
and it degrades honestly: a structure the reference seed never scored is reported as such instead of
silently reading as complete (a Counter's 0 == 0).

ARM IS FILTERED. The seed is shared across arms for templated folds, so a row from another arm would
otherwise count toward a free fold's completeness.
"""
import argparse
import collections
import csv
import math
import sys


def load(csv_path, arm):
    per = collections.defaultdict(set)          # (pdb, seed, iface) -> {sample}
    for r in csv.DictReader(open(csv_path)):
        if r["arm"] != arm:
            continue
        try:
            if not (math.isfinite(float(r["dockq"])) and math.isfinite(float(r["iptm"]))):
                continue
            k = int(r["sample"])
        except (TypeError, ValueError, KeyError):
            continue
        per[(r["iface"].split("_")[0], r["seed"], r["iface"])].add(k)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training/fold/results.csv")
    ap.add_argument("--structures", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--ref-seed", default="500000")
    ap.add_argument("--arm", default="free")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--mode", choices=("complete", "targets"), default="complete")
    a = ap.parse_args()

    want = [l.strip() for l in open(a.structures) if l.strip() and l[0] != "#"]
    per = load(a.csv, a.arm)

    scorable = collections.defaultdict(set)
    for (pdb, seed, iface), ks in per.items():
        if seed == a.ref_seed and pdb in set(want) and len(ks) >= a.nsample:
            scorable[pdb].add(iface)
    no_ref = [p for p in want if not scorable[p]]

    missing = []
    for p in want:
        for iface in sorted(scorable[p]):
            if len(per.get((p, a.seed, iface), ())) < a.nsample:
                missing.append(iface)

    if a.mode == "targets":
        # phase_j_rescore_unmapped.py takes "<pdb>:<iface>", and ONLY the interfaces that are
        # actually missing -- handing it one that already has rows makes the merge refuse the
        # whole batch as "already in the canonical file".
        print(",".join(f"{i.split('_')[0]}:{i}" for i in missing))
        return 0

    n_if = sum(len(v) for v in scorable.values())
    print(f"  seed {a.seed}: {n_if - len(missing)} of {n_if} scorable interface(s) complete "
          f"over {len(want) - len(no_ref)} structure(s) x {a.nsample} sample(s)")
    if no_ref:
        print(f"  WARNING: {len(no_ref)} structure(s) have no full reference seed {a.ref_seed}: "
              f"{','.join(no_ref)}")
    if missing:
        print(f"  missing: {len(missing)} interface(s), e.g. {','.join(missing[:6])}")
    # A structure with no reference is not a pass: nothing establishes what complete would mean.
    return 1 if (missing or no_ref) else 0


if __name__ == "__main__":
    sys.exit(main())
