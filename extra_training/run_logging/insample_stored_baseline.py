#!/usr/bin/env python
"""SHIPPED MAX-POOLED IN-SAMPLE SELECTION OVER THE CURRENT TRAINING SET, ON CPU, NO FORWARD PASS.

WHY THIS IS FREE AND THE MEAN-POOLED NUMBER IS NOT. training/fold/results.csv stores ONE ipTM
column per pose and it is the shipped head's MAX-pooled value, written when the poses were
scored. Selecting by it is therefore a pure argmax over rows that already exist: the shipped
max-pooled number on any pool is a CPU aggregation, not a GPU pass. The mean-pooled shipped
number is a different story -- there is no mean-pooled column anywhere, so it needs the stock
head actually forwarded with --pool-mean, which is what insample_fullset.py --head shipped does.
This script exists because that distinction was being paid for in GPU time it did not need.

IT IS THE SAME POPULATION AND THE SAME CANDIDATES AS insample_fullset.py, deliberately, so the
number it prints is the baseline for that script's trained-head passes rather than a near miss:

  * the pool is training/combined/pool.csv -- 1688 interfaces over 853 structures, which is what
    the live arms train on. Every earlier in-sample number in this repo is over
    training/full_train/pool_winnable.csv, 738 interfaces over 368 structures, and the two are
    not comparable.
  * the interface id is f"{pdb}_{chain1}{chain2}", which is how results.csv names interfaces.
  * each structure's (seed base, budget) is READ OFF ITS OWN TRUNK STORE -- base is its lowest
    trunk seed and budget the contiguous run from there -- because the 2026-11-04 prune left 437
    structures at 15 seeds, 415 at exactly one, and 9dzj at the single seed 500002 rather than
    500000. A hardcoded base would silently drop 9dzj's two interfaces or claim poses that do
    not exist.

A MISSING SCORE IS A STOP, NOT A SMALLER MEAN. Every intended (iface, seed, sample) must be in
results.csv. Dropping the absent ones would move the metric by an unknown amount in an unknown
direction and nothing in the output would say so.
"""
import argparse
import collections
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLD = "training/fold"
POOL = "training/combined/pool.csv"
NSAMPLE = 5
ACCEPTABLE = 0.23

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=POOL)
ap.add_argument("--out", default="")
a = ap.parse_args()

rows = list(csv.DictReader(open(os.path.join(ROOT, a.pool))))
ifaces = [f"{r['pdb']}_{r['chain1']}{r['chain2']}" for r in rows]
structs = sorted({r["pdb"] for r in rows})
if len(set(ifaces)) != len(ifaces):
    dup = [k for k, n in collections.Counter(ifaces).items() if n > 1]
    sys.exit(f"STOP: {a.pool} names {len(dup)} interface(s) twice, e.g. {dup[:4]}")
print(f"pool {a.pool}: {len(ifaces)} interfaces over {len(structs)} structures")


def window(pdb):
    """(base, budget) from the trunks this structure actually has."""
    seeds = sorted(int(f.split("_")[1].split(".")[0])
                   for f in os.listdir(os.path.join(ROOT, FOLD, "trunks", pdb))
                   if f.startswith("free_") and f.endswith(".npz"))
    if not seeds:
        sys.exit(f"STOP: {pdb} has no trunk")
    n = 1
    while n < len(seeds) and seeds[n] == seeds[0] + n:
        n += 1
    return seeds[0], n


win = {p: window(p) for p in structs}
subs = collections.Counter(win.values())
for (base, budget), n in sorted(subs.items()):
    print(f"  seed base {base} budget {budget:2d}: {n:4d} structures")

# (iface, seed, sample) -> (dockq, shipped max-pooled iptm)
res = {}
with open(os.path.join(ROOT, FOLD, "results.csv")) as fh:
    for r in csv.DictReader(fh):
        if r["arm"] != "free" or r["dockq"] in ("", None):
            continue
        res[(r["iface"], int(r["seed"]), int(r["sample"]))] = (float(r["dockq"]),
                                                               float(r["iptm"]))

sel, orc, missing = [], [], []
for r, iface in zip(rows, ifaces):
    base, budget = win[r["pdb"]]
    cands = []
    for seed in range(base, base + budget):
        for s in range(NSAMPLE):
            v = res.get((iface, seed, s))
            if v is None:
                missing.append((iface, seed, s))
            else:
                cands.append(v)
    if not cands:
        missing.append((iface, base, -1))
        continue
    sel.append(max(cands, key=lambda t: t[1])[0])
    orc.append(max(d for d, _ in cands))

if missing:
    sys.exit(f"STOP: {len(missing)} intended (iface, seed, sample) row(s) are absent from "
             f"results.csv, e.g. {missing[:4]}. The mean would silently cover a subset.")


def _m(v):
    return sum(v) / len(v)


out = {
    "pool": a.pool,
    "n_interfaces": len(sel),
    "n_structures": len(structs),
    "n_candidate_poses": sum(win[r["pdb"]][1] for r in rows) * NSAMPLE,
    "head": "shipped",
    "select_by": "max",
    "source": "stored iptm in training/fold/results.csv -- no forward pass",
    "mean_dockq_select_max": round(_m(sel), 4),
    "frac_acc_select_max": round(_m([1.0 if d >= ACCEPTABLE else 0.0 for d in sel]), 4),
    "oracle_mean_dockq": round(_m(orc), 4),
    "oracle_frac_acc": round(_m([1.0 if d >= ACCEPTABLE else 0.0 for d in orc]), 4),
}
out["headroom"] = round(out["oracle_mean_dockq"] - out["mean_dockq_select_max"], 4)
print()
for k, v in out.items():
    print(f"  {k:24s} {v}")
if a.out:
    with open(os.path.join(ROOT, a.out), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n-> {a.out}")
