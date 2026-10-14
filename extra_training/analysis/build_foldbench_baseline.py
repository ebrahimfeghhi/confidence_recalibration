#!/usr/bin/env python
"""One FoldBench table: the 50-seed pool first, then every top-up, one row per pose.

    ./build_foldbench_baseline.py --out .../foldbench_baseline_results.csv

WHY. The free seeds for FoldBench are spread over six directories written at different times in two
different schemas, and the pooled table built from them (`free_pool/results.csv`) was built before the
running top-off existed, so it is already behind. An analysis that reads one directory and not another
silently reports the wrong seed depth.

WHAT IS IN, AND WHY `other_seeds` IS NOT. Five blocks, at identical settings (bf16, N_cycle 10, same
template policy) on DISJOINT seed ranges. `other_seeds` is left out: it covers only 48 of the 160
interfaces, and only the complexes someone had experimented on before, so pooling it would weight the
table toward whatever was interesting at the time. Its per-interface agreement with the pool is
actually fine (+0.002 mean DockQ, 0 disagreements at p<0.001); the problem is WHICH interfaces it has,
not what it says about them.

WHY `matched` IS IN, reversing how it was tiered. It was marked suspect for disagreeing with the pool
on how often a seed finds an acceptable pose. But `top100` REFOLDED exactly those complexes -- that is
what `--credit-suspect-except-flagged` did -- and its fresh folds reproduce `matched`, not the pool:

    8up2_AC   pool 0.96   matched 0.46   top100 0.45
    7x8p_AB   pool 0.76   matched 0.38   top100 0.28
    8vyl_AF   pool 0.30   matched 0.00   top100 0.00

Two independent refolds on different seed ranges agree with each other and not with the pool, so the
pool is the outlier on those four complexes. Recorded rather than hidden: the `metadata` column names
every row's block, so anyone can drop it again with one filter.

NO SEED IS REFOLDED. Measured, not assumed: zero (interface, seed, sample) rows appear in more than one
block, and the ranges do not overlap -- 101-150, 151-200, 501-510, 600000-600039, 601000-601039. The
uniqueness check below is therefore an assertion, not a merge rule; if it ever fires, the inputs
changed and a human has to decide which fold wins.
"""
import argparse
import collections
import csv
import os

ROOT = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"

# ORDER IS THE OUTPUT ORDER: the published 50-seed baseline first, then the top-ups in the order they
# were folded. `metadata` is what each row's provenance reads as in the file.
BLOCKS = [
    ("pool",     "pool, seeds 101-150, the published 50-seed baseline"),
    ("matched",  "top-up, seeds 151-200, refold-confirmed (see docstring)"),
    ("extra10",  "top-up, seeds 501-510, all 160 interfaces"),
    ("top100",   "top-up, seeds 600000-600039"),
    ("top100b",  "top-up, seeds 601000-601039, the completion run"),
]
FIELDS = ["pdb", "chain1", "chain2", "iface", "seed", "sample", "dockq", "pair_iptm", "metadata"]


def from_pooled(path):
    """the five-block pooled table; `top100b` is absent from it and comes from its own file"""
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        out[r["block"]].append({
            "pdb": r["pdb"], "chain1": r["chain1"], "chain2": r["chain2"],
            "iface": f"{r['pdb']}_{r['chain1']}{r['chain2']}",
            "seed": r["seed"], "sample": r["sample"],
            "dockq": r["dockq"], "pair_iptm": r["pair_iptm"]})
    return out


def from_block(path):
    """a raw block results.csv: iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm

    FREE ROWS ONLY. A block directory can hold templated arms too, and those are a different
    experiment; `donor_pose` is -1 exactly when nothing was templated.
    """
    rows = []
    for r in csv.DictReader(open(path)):
        if r.get("donor_pose", "-1") != "-1":
            continue
        pdb, chains = r["iface"].rsplit("_", 1)
        rows.append({"pdb": pdb, "chain1": chains[0], "chain2": chains[1], "iface": r["iface"],
                     "seed": r["seed"], "sample": r["sample"],
                     "dockq": r["dockq"], "pair_iptm": r.get("iptm", "")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", default=f"{ROOT}/free_pool/results.csv")
    ap.add_argument("--top100b", default=f"{ROOT}/free_top100b/results.csv")
    ap.add_argument("--out", default=f"{ROOT}/foldbench_baseline_results.csv")
    a = ap.parse_args()

    have = from_pooled(a.pooled)
    if os.path.exists(a.top100b):
        have["top100b"] = from_block(a.top100b)
    rows = []
    for blk, note in BLOCKS:
        for r in sorted(have.get(blk, []), key=lambda t: (t["iface"], int(t["seed"]),
                                                          int(t["sample"]))):
            rows.append({**r, "metadata": note})
        print(f"  {blk:9s} {len(have.get(blk, [])):6d} rows   {note}")

    # ONE ROW PER (interface, seed, sample). This is an assertion about the inputs, not a dedupe:
    # the blocks hold disjoint seeds, so a collision means something upstream changed.
    seen = collections.Counter((r["iface"], r["seed"], r["sample"]) for r in rows)
    dup = [k for k, n in seen.items() if n > 1]
    if dup:
        raise SystemExit(f"STOP: {len(dup)} (interface, seed, sample) key(s) appear twice, e.g. "
                         f"{dup[:3]}. The blocks are supposed to hold disjoint seeds, so this is a "
                         "change upstream -- decide which fold wins before pooling.")
    # and no interface may carry the same seed twice
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["iface"]][r["seed"]] += 1
    nsample = collections.Counter(n for c in per.values() for n in c.values())

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    ifaces = {r["iface"] for r in rows}
    depth = sorted(len(c) for c in per.values())
    print(f"\nwrote {a.out}")
    print(f"  {len(rows)} rows, {len(ifaces)} interfaces, no duplicated (interface, seed, sample)")
    print(f"  seeds per interface: min {depth[0]}, median {depth[len(depth)//2]}, max {depth[-1]}")
    print(f"  samples per (interface, seed): {dict(nsample)}")


if __name__ == "__main__":
    main()
