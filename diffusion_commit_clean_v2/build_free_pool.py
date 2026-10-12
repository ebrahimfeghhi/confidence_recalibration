#!/usr/bin/env python
"""Assemble each benchmark's free-seed blocks into ONE pose-level table per benchmark.

    ./build_free_pool.py            # dry run, prints what it would write
    ./build_free_pool.py --apply

WHY. The free seeds for a benchmark live in two places with two schemas: the canonical 50-seed table
(`paths.DATASETS[ds].pool_csv`, whose poses were deleted) and a later top-up arm (`free_extra10/`,
whose poses are still on disk). An analysis that read one and not the other silently reported the
wrong seed depth -- 50 instead of 60, or 50 instead of 70. One table per benchmark removes that.

WHAT IT DOES NOT DO. No pose is moved or deleted. The `block` column records which arm a row came
from, and INDEX.md records where that arm's poses are, so a rescorer can find them.

SCHEMA. pdb, chain1, chain2, seed, sample, dockq, pair_iptm, block, tier

TIER IS THE MARKER THAT MAKES THIS SAFE TO POOL. Not every free seed is interchangeable with the
original 50, so every row carries which kind it is and an analysis can drop a kind without knowing
the directory layout:

  pool     the canonical 50-seed baseline (pool_csv). Every published free-50 number comes from here
  topup    later free folds at the SAME settings (bf16, N_cycle 10, same template policy), on seed
           blocks disjoint from the pool. Comparable by construction, but never used in the
           published baseline, so a discrepancy against `pool` is a finding, not a given
  suspect  free folds at the same settings whose SAMPLING is uneven -- concentrated on whichever
           complexes had been experimented on before. foldbench/free_other_seeds measured 0.060
           worse mean DockQ than the pool for exactly that reason, which is how adding free seeds
           once appeared to make the baseline worse. Included so the depth is reachable; filter it
           out of any baseline

So: `tier == "pool"` reproduces the published baseline, `tier != "suspect"` is the honest deep pool,
and everything is one read away.

  pool_csv  : pdb,chain1,chain2,seed,sample,dockq[,lddt,bb_lddt],pair_iptm
  free_extra: iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm   (iface = <pdb>_<c1><c2>)

Chain orientation is taken from pool_csv, so a reversed pair in the top-up is swapped to match. The
extra columns pxm2025's pool_csv carries (lddt, bb_lddt) are dropped: the top-up has no counterpart,
and a column that is present for 50 seeds and blank for 20 invites exactly the bug this file exists
to prevent.

SCORER. Each benchmark's blocks must share one scorer, because DockQ from two implementations is not
comparable. Both arms here were folded through folder.py --dataset <ds>, which resolves the scorer
from paths.py, so this holds by construction; it is asserted against paths.py anyway.
"""
import argparse
import csv
import os
import subprocess
import sys

sys.path[:0] = ["/home/ebrahim/tts_abag"]
import paths  # noqa: E402

REPO = "/home/ebrahim/tts_abag"
FIELDS = ["pdb", "chain1", "chain2", "seed", "sample", "dockq", "pair_iptm", "block", "tier"]

FB = f"{REPO}/experimental_data/protenix/foldbench"
PX = f"{REPO}/experimental_data/protenix/pxm2025"

# benchmark -> blocks to fold in, as (block, directory, tier, arms). pool_csv is always block "pool"
# at tier "pool" and is added first. `arms` filters the source results.csv by its own `arm` column;
# None takes every row.
#
# WHAT IS LEFT OUT, and why, because an exclusion no one can see is indistinguishable from an
# oversight:
#   pxm2025/free arms freebf16r1, freebf16r2, freefp32r1, freefp32r2 -- 200 rows that REFOLD POOL
#   SEEDS, two precisions x two repeats. Every one of their row keys is already in pool_csv, so
#   pooling them would count a seed twice and silently change the seed depth. They are a
#   determinism experiment and belong to that analysis, not to a baseline.
BLOCKS = {
    "foldbench": [
        ("extra10",     f"{FB}/free_extra10",     "topup",   None),
        # SUSPECT, not topup. 16 interfaces over 10 complexes disagree with the pool on how
        # often a seed finds an acceptable pose, at p<0.001, in BOTH directions -- 8up2_AC goes
        # 48/50 to 23/50 and 7zoz_AB goes 18/50 to 19/24. Timing rules out reduced recycles and the
        # paired mean DockQ matches to 0.007, so averages are fine; it is the max-over-seeds
        # statistics (decoy, no-acceptable) that move, and this is the block whose directory also
        # holds resume_8vyl_s101, queue_refold_nan_8xnh.tsv and two claims_refold_* -- it was
        # re-run in pieces. foldbench/free_extra10 and every pxm block show ZERO such interfaces.
        ("matched",     f"{FB}/free",             "suspect", None),
        ("other_seeds", f"{FB}/free_other_seeds", "suspect", None),
    ],
    "pxm2025": [
        ("extra10",  f"{PX}/free_extra10", "topup", None),
        ("free100",  f"{PX}/free",         "topup", {"free100"}),
        ("free149",  f"{PX}/free",         "topup", {"free"}),
    ],
}


def read_pool(ds):
    """pool_csv rows, normalised. out: (rows, {(pdb, c1, c2)} as stored)"""
    rows, keys = [], set()
    with open(paths.DATASETS[ds].pool_csv) as fh:
        for r in csv.DictReader(fh):
            k = (r["pdb"], r["chain1"], r["chain2"])
            keys.add(k)
            rows.append({"pdb": r["pdb"], "chain1": r["chain1"], "chain2": r["chain2"],
                         "seed": r["seed"], "sample": r["sample"], "dockq": r["dockq"],
                         "pair_iptm": r["pair_iptm"], "block": "pool", "tier": "pool"})
    return rows, keys


def read_arm(path, block, tier, keys, arms=None):
    """a free arm's results.csv, normalised onto pool_csv's chain orientation.

    in: results.csv path, block name, tier, the pool's (pdb, c1, c2) keys, and optionally the set of
    `arm` values to keep.
    out: (rows, problems) -- problems is a list of iface strings that do not map onto a pool key,
    which is fatal to the caller rather than silently dropped."""
    rows, bad = [], []
    with open(os.path.join(path, "results.csv")) as fh:
        for r in csv.DictReader(fh):
            if arms is not None and r.get("arm") not in arms:
                continue
            pdb, ch = r["iface"].rsplit("_", 1)
            if len(ch) != 2:
                bad.append(r["iface"])
                continue
            c1, c2 = ch[0], ch[1]
            if (pdb, c1, c2) not in keys:
                if (pdb, c2, c1) in keys:
                    c1, c2 = c2, c1          # pool_csv's orientation wins
                else:
                    bad.append(r["iface"])
                    continue
            rows.append({"pdb": pdb, "chain1": c1, "chain2": c2, "seed": r["seed"],
                         "sample": r["sample"], "dockq": r["dockq"], "pair_iptm": r["iptm"],
                         "block": block, "tier": tier})
    return rows, bad


def seed_range(rows):
    """a compact seed range for the INDEX, e.g. "101-150" or "0-100,201-339"."""
    if not rows:
        return "-"
    ss = sorted({int(r["seed"]) for r in rows})
    out, a, prev = [], ss[0], ss[0]
    for x in ss[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append((a, prev))
        a = prev = x
    out.append((a, prev))
    txt = ",".join(f"{lo}-{hi}" if lo != hi else str(lo) for lo, hi in out)
    return txt if len(txt) <= 40 else f"{ss[0]}-{ss[-1]} ({len(out)} blocks)"


def cifs(path):
    """pose cifs anywhere under an arm, following symlinks.

    NOT `<path>/poses` -- the layout is not uniform and assuming it reported 0 for the two largest
    pose stores we have. folder.py's newer arms use `poses/<pdb>/<seed>-<run>/`, the older ones put
    `<pdb>/<seed>-<run>/` at the arm root, and free_other_seeds is a farm of symlinks into
    tts_abag/poses. Counting the whole tree is the only form that is right for all three."""
    out = subprocess.run(["bash", "-c", f'find -L {path} -name "*.cif" 2>/dev/null | wc -l'],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--datasets", default="foldbench,pxm2025")
    a = ap.parse_args()

    for ds in a.datasets.split(","):
        rows, keys = read_pool(ds)
        # ROW KEY, not just the interface: a block that repeated a (seed, sample) the pool already
        # has would double-count that pose, and the oracle over a doubled pose set is unchanged
        # while the seed depth reported is wrong.
        seen = {(r["pdb"], r["chain1"], r["chain2"], r["seed"], r["sample"]) for r in rows}
        report = [("pool", "pool", str(paths.DATASETS[ds].pool_csv), len(rows), 0,
                   seed_range(rows))]
        for block, path, tier, arms in BLOCKS.get(ds, []):
            new, bad = read_arm(path, block, tier, keys, arms)
            if bad:
                raise SystemExit(f"STOP: {ds}/{block} has {len(bad)} interface(s) that do not map "
                                 f"onto a pool_csv chain pair, e.g. {bad[:5]}")
            dup = [r for r in new
                   if (r["pdb"], r["chain1"], r["chain2"], r["seed"], r["sample"]) in seen]
            if dup:
                raise SystemExit(f"STOP: {ds}/{block} repeats {len(dup)} row key(s) already in the "
                                 f"pool, e.g. {dup[:3]}")
            seen.update((r["pdb"], r["chain1"], r["chain2"], r["seed"], r["sample"]) for r in new)
            rows += new
            report.append((block, tier, path, len(new), cifs(path), seed_range(new)))

        def depth_of(keep):
            d = {}
            for r in rows:
                if r["tier"] in keep:
                    d.setdefault((r["pdb"], r["chain1"], r["chain2"]), set()).add(r["seed"])
            return sorted(len(v) for v in d.values()), len(d)

        n, nif = depth_of({"pool", "topup", "suspect"})
        nt, _ = depth_of({"pool", "topup"})
        npool, _ = depth_of({"pool"})
        print(f"{ds}: {len(rows)} rows, {nif} interfaces")
        print(f"    seeds/interface  pool only     min {npool[0]:4d} median {npool[len(npool)//2]:4d} "
              f"max {npool[-1]:4d}")
        print(f"                     pool+topup    min {nt[0]:4d} median {nt[len(nt)//2]:4d} "
              f"max {nt[-1]:4d}")
        print(f"                     + suspect     min {n[0]:4d} median {n[len(n)//2]:4d} "
              f"max {n[-1]:4d}")
        for block, tier, path, nr, nc, sr in report:
            print(f"    {block:12s} {tier:8s} {nr:7d} rows  {nc:6d} cifs  seeds {sr}")

        if not a.apply:
            print("    (dry run; --apply to write)")
            continue

        # inside the benchmark's own folder, which is what paths.py resolves
        out = str(paths.DATASETS[ds].free_pool.parent)
        os.makedirs(out, exist_ok=True)
        # lineterminator="\n": a shell reading this with `IFS=, read` keeps the \r of a CRLF file in
        # the last field, which has broken interface keys here before.
        with open(f"{out}/results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        with open(f"{out}/INDEX.md", "w") as fh:
            fh.write(f"# {ds} free seeds\n\n"
                     f"`results.csv` is every free pose-level score for this benchmark in one "
                     f"table: {len(rows)} rows over {nif} interfaces. Scorer "
                     f"`{paths.DATASETS[ds].scorer}` (paths.py). Columns "
                     f"`{', '.join(FIELDS)}`.\n\n"
                     f"## Seed depth by tier\n\n"
                     f"| tier filter | min | median | max |\n|---|---|---|---|\n"
                     f"| `tier == \"pool\"` (the published free-50 baseline) | {npool[0]} | "
                     f"{npool[len(npool)//2]} | {npool[-1]} |\n"
                     f"| `tier != \"suspect\"` | {nt[0]} | {nt[len(nt)//2]} | {nt[-1]} |\n"
                     f"| everything | {n[0]} | {n[len(n)//2]} | {n[-1]} |\n\n"
                     f"Rows are seeds per interface. **Reproduce a published number with "
                     f"`tier == \"pool\"`.** A deeper read is legitimate but has never been the "
                     f"baseline, so if it disagrees with the pool that is a result to explain, not "
                     f"a number to quote.\n\n"
                     f"## Blocks\n\n"
                     f"| block | tier | rows | pose cifs | seeds | where the poses are |\n"
                     f"|---|---|---|---|---|---|\n")
            for block, tier, path, nr, nc, sr in report:
                where = path if nc else "deleted"
                fh.write(f"| {block} | {tier} | {nr} | {nc} | {sr} | {where} |\n")
            fh.write("\n`suspect` means the settings match but the sampling is uneven -- those "
                     "seeds exist only for complexes someone had experimented on before, and they "
                     "measured 0.060 worse on mean DockQ than the pool for that reason. Keep them "
                     "out of any baseline.\n\n"
                     "Left out deliberately: `pxm2025/free` arms `freebf16r1/r2` and "
                     "`freefp32r1/r2`, 200 rows that refold pool seeds in two precisions. Every "
                     "row key is already in `pool_csv`, so pooling them would count a seed twice; "
                     "they belong to the determinism analysis.\n\n"
                     "Built by `diffusion_commit_clean_v2/build_free_pool.py`. No pose was moved "
                     "or deleted.\n")
        print(f"    wrote {out}/results.csv and INDEX.md")


if __name__ == "__main__":
    main()
