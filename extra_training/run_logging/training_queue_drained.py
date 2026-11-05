#!/usr/bin/env python
"""Is the training fold queue ACTUALLY drained? Exit 0 if yes, 1 if not, printing why.

WHY A LOG STRING IS NOT ENOUGH. Raised in review. folder.py prints "queue complete: N folds run,
M already scored, K failed" with K>0 included, and then runs a SECOND PASS over whatever is still
unscored -- a unit that became claimable only after the walk went past it is picked up there. So
the message appears while the worker is still folding, and appears even when folds failed. An
arming script that waits for that string launches onto cards the training fold still holds, which
is the exact failure it was written to prevent.

THE QUEUE IS DRAINED WHEN THE DATA SAYS SO: every queued unit has a trunk and scored rows.
"""
import argparse, csv, glob, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="training/fold/queue.tsv")
ap.add_argument("--trunks", default="training/fold/trunks")
ap.add_argument("--csv", default="training/fold/results.csv")
ap.add_argument("--pool", default="training/fold/pool.csv")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

units = []
for ln in open(a.queue):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 5:
        units.append((p[1], p[3], p[4]))          # pdb, target/tag, seed

# an interface count per structure, so "scored" means every interface of the unit
n_if = {}
for r in csv.DictReader(open(a.pool)):
    n_if[r["pdb"]] = n_if.get(r["pdb"], 0) + 1

scored = {}
for r in csv.reader(open(a.csv)):
    if len(r) >= 2 and r[0] and r[1]:
        scored.setdefault((r[0].split("_")[0], r[1]), set()).add(r[0])

no_trunk, unscored = [], []
for pdb, tag, seed in units:
    if not glob.glob(f"{a.trunks}/{pdb}/*_{seed}.npz"):
        no_trunk.append((pdb, seed)); continue
    got = scored.get((pdb, seed), set())
    if len(got) < n_if.get(pdb, 1):
        unscored.append((pdb, seed, len(got), n_if.get(pdb, 1)))

if not a.quiet:
    print(f"  queued {len(units)} unit(s): {len(no_trunk)} without a trunk, "
          f"{len(unscored)} folded but not fully scored")
    for pdb, seed in no_trunk[:5]:
        print(f"     no trunk: {pdb} seed {seed}")
    for pdb, seed, got, want in unscored[:5]:
        print(f"     partial:  {pdb} seed {seed}: {got} of {want} interface(s)")

sys.exit(0 if not no_trunk and not unscored else 1)
