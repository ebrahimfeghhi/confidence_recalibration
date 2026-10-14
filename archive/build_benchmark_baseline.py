#!/usr/bin/env python
"""One table per benchmark: the published pool first, then every top-up, one row per pose.

    ./build_benchmark_baseline.py --dataset foldbench
    ./build_benchmark_baseline.py --dataset pxm2025

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

PROTENIX = "/home/ebrahim/tts_abag/experimental_data/protenix"

# Per benchmark: the block order (which is the OUTPUT order), and what each row's `metadata` says.
# FoldBench's blocks hold disjoint seed ranges; PXM's OVERLAP -- pool 300000-300094, free100
# 300050-300129, free149 300100-300149 -- but no (interface, seed) pair appears in two blocks, so the
# overlap is between different interfaces and nothing collides across blocks.
BENCH = {
    "foldbench": dict(dirname="foldbench", extra=("top100b", "foldbench_baseline_remaining_folds"),
                      blocks=[
        ("pool",     "pool, seeds 101-150, the published 50-seed baseline"),
        ("matched",  "top-up, seeds 151-200, refold-confirmed"),
        ("extra10",  "top-up, seeds 501-510, all 160 interfaces"),
        ("top100",   "top-up, seeds 600000-600039"),
        ("top100b",  "top-up, seeds 601000-601039, the completion run"),
    ]),
    "pxm2025": dict(dirname="pxm2025", extra=None, blocks=[
        ("pool",     "pool, seeds 300000-300094, the published baseline"),
        ("free100",  "top-up, seeds 300050-300129"),
        ("free149",  "top-up, seeds 300100-300149"),
        ("extra10",  "top-up, seeds 300501-300520, all 292 interfaces"),
        ("top100",   "top-up, seeds 300601-300639"),
    ]),
}

# ORDER IS THE OUTPUT ORDER: the published 50-seed baseline first, then the top-ups in the order they
# were folded. `metadata` is what each row's provenance reads as in the file.
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
    ap.add_argument("--dataset", default="foldbench", choices=sorted(BENCH))
    ap.add_argument("--pooled", default=None)
    ap.add_argument("--extra", default=None, help="a block absent from the pooled table")
    ap.add_argument("--out", default=None)
    ap.add_argument("--queue", default=None,
                    help="also write the folds still missing to reach --target, as a folder.py queue")
    ap.add_argument("--target", type=int, default=100, help="seeds per structure to reach")
    ap.add_argument("--seed0", type=int, default=602000,
                    help="first seed to allocate; must be clear of every seed already in the table")
    a = ap.parse_args()
    spec = BENCH[a.dataset]
    ROOT = f"{PROTENIX}/{spec['dirname']}"
    BLOCKS = spec["blocks"]
    a.pooled = a.pooled or f"{ROOT}/free_pool/results.csv"
    a.out = a.out or f"{ROOT}/{a.dataset}_baseline_results.csv"
    if a.extra is None and spec["extra"]:
        a.extra = f"{ROOT}/{spec['extra'][1]}/results.csv"

    # THE BLOCK DIRECTORIES ARE GONE. They were deleted once this table replaced them, so a rebuild
    # from source is no longer possible and the table itself is the input. That is a one-way door and
    # it should look like one: if the sources are absent the script reads what it wrote last time,
    # and says so, rather than failing or quietly producing an empty table.
    if os.path.exists(a.pooled):
        have = from_pooled(a.pooled)
        if a.extra and os.path.exists(a.extra):
            have[spec["extra"][0]] = from_block(a.extra)
        rows = []
        for blk, note in BLOCKS:
            for r in sorted(have.get(blk, []), key=lambda t: (t["iface"], int(t["seed"]),
                                                              int(t["sample"]))):
                rows.append({**r, "metadata": note})
            print(f"  {blk:9s} {len(have.get(blk, [])):6d} rows   {note}")
        write = True
    elif os.path.exists(a.out):
        rows = list(csv.DictReader(open(a.out)))
        print(f"the per-block sources are gone; reading the existing table instead")
        print(f"  {a.out}: {len(rows)} rows")
        write = False
    else:
        raise SystemExit(f"STOP: neither the block sources ({a.pooled}) nor an existing table "
                         f"({a.out}) is present; there is nothing to build from.")

    # ONE ROW PER (interface, seed, sample). FoldBench has no collisions at all -- its blocks hold
    # disjoint seed ranges -- so for it this is purely an assertion. PXM has exactly one unit folded
    # twice, 9mnz_BD seed 300112, samples 0-4, BOTH inside the free149 block: two folds of the same
    # seed with different DockQ (0.0142 vs 0.0144 at sample 0), which is expected since folds are not
    # reproducible here.
    #
    # THE LATER FOLD WINS, and "later" is the source file's APPEND ORDER -- results.csv is appended to
    # as units finish, and the two folds sit consecutively at rows 64542-64546 and 64547-64551, so the
    # second block of five is the more recent. Nothing else in either benchmark is affected.
    seen = collections.defaultdict(list)
    for idx, r in enumerate(rows):
        seen[(r["iface"], r["seed"], r["sample"])].append(idx)
    drop, resolved = set(), []
    for k, idxs in seen.items():
        if len(idxs) > 1:
            drop.update(idxs[:-1])          # keep the last = the most recently appended
            resolved.append((k, len(idxs)))
    if resolved:
        print(f"\n  {len(resolved)} refolded (interface, seed, sample) key(s); kept the most recent:")
        for k, n in resolved[:6]:
            print(f"     {k[0]} seed {k[1]} sample {k[2]} -- {n} folds, kept the last")
        rows = [r for i, r in enumerate(rows) if i not in drop]
    # and no interface may carry the same seed twice
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["iface"]][r["seed"]] += 1
    nsample = collections.Counter(n for c in per.values() for n in c.values())

    if write:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.out}")
    ifaces = {r["iface"] for r in rows}
    depth = sorted(len(c) for c in per.values())
    print(f"  {len(rows)} rows, {len(ifaces)} interfaces, no duplicated (interface, seed, sample)")
    print(f"  seeds per interface: min {depth[0]}, median {depth[len(depth)//2]}, max {depth[-1]}")
    print(f"  samples per (interface, seed): {dict(nsample)}")

    # THE REMAINING FOLDS, derived from the table rather than from a directory listing. A fold is per
    # (STRUCTURE, SEED) -- every interface of a complex comes out of one fold -- so the deficit is
    # counted per structure, not per interface. Counting per interface double-counts a complex with
    # two of them and asks for folds that already exist.
    if a.queue:
        per_pdb = collections.defaultdict(set)
        for r in rows:
            per_pdb[r["pdb"]].add(int(r["seed"]))
        used = set().union(*per_pdb.values()) if per_pdb else set()
        if a.seed0 <= max(used, default=-1):
            raise SystemExit(f"STOP: --seed0 {a.seed0} is at or below the highest seed already in "
                             f"the table ({max(used)}). Allocating from there would reuse seeds.")
        lines, short = [], {}
        for pdb in sorted(per_pdb):
            need = a.target - len(per_pdb[pdb])
            if need <= 0:
                continue
            short[pdb] = need
            # SKIP ANY SEED THE STRUCTURE ALREADY HAS, even above seed0, so a re-run of this builder
            # can never issue a fold that exists.
            seed, issued = a.seed0, 0
            while issued < need:
                if seed not in per_pdb[pdb]:
                    lines.append(f"free\t{pdb}\t*\tfree\t{seed}")
                    issued += 1
                seed += 1
        with open(a.queue, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote {a.queue}")
        print(f"  {len(lines)} folds over {len(short)} structure(s) below {a.target} seeds")
        print(f"  seeds allocated from {a.seed0}; highest already in use {max(used)}")
        print(f"  deficit per structure: {sorted(set(short.values()))}")


if __name__ == "__main__":
    main()
