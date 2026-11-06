#!/usr/bin/env python
"""Is a seed COMPLETE? The exact key set, not one row per structure.

WHY THIS EXISTS, AND WHAT IT CAUGHT. chain_extra_seeds.sh judged a seed complete by counting rows
per PDB and asking whether the count was zero -- so a structure whose first interface scored and
whose second did not was indistinguishable from one that scored everything. 83.1% of the frozen 437
have more than one scoreable interface, and folder.py appends per INTERFACE (run_model/folder.py:774
"flushed per interface, not at the end"), so the loose test is wrong by construction, not by race.

Measured against training/combined/pool.csv, the pool the trainer actually reads:
    seed 500000   437 of 437 structures exact
    seed 500010   422 of 437            23 interfaces and 1 sample missing, gate said 437 of 437
The seed-500010 recovery sweep then compounded it: recover_seed selected only structures with ZERO
rows, so it repaired 1wej/4p3c/9blx and left those 15 partial structures untouched. The seed was
declared complete, published as complete, and chained onto.

WHAT "COMPLETE" MEANS HERE. For every structure in --structures, every interface the pool names for
it, and every sample 0..nsample-1: exactly one row, arm=free, finite dockq and iptm. A duplicate key
is a failure, an unexpected key is a failure, and a structure with no rows is a failure. Nothing is
inferred from a count.

WHICH POOL. --pool defaults to training/combined/pool.csv because that is what training reads, and
it is the only one of the three that is exactly right: training/fold/pool.csv names 10 interfaces
that OST can never score (the chem-group drop -- 3rhw_DK, 4rdq_DF, 8xcd_AC and 7 more), so it
reports permanent failures as gaps; pool_scoreable.csv still carries 8xcd_AC. A gate that cries
every seed gets switched off, so the expected set has to be the achievable one.
"""
import argparse, collections, csv, math, os, sys

ARM = "free"


def pool_ifaces(path, structures=None):
    d = collections.defaultdict(set)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            p = r["pdb"].strip().lower()
            if structures is not None and p not in structures:
                continue
            d[p].add(f"{p}_{''.join(sorted((r['chain1'], r['chain2'])))}")
    return d


def coverage(results, pool, structures, seed, nsample):
    """out: (exact, partial, absent, extra) -- partial/absent name structures, extra names keys."""
    want_if = {i for p in structures for i in pool.get(p, ())}
    seen = collections.Counter()
    extra = []
    with open(results) as fh:
        for r in csv.DictReader(fh):
            if r["seed"] != str(seed):
                continue
            ifc = r["iface"]
            if ifc not in want_if:
                continue
            # A ROW THAT IS NOT FINITE IS NOT A SCORE. The builder treats a non-finite dockq as
            # unscored elsewhere; counting it here would let a seed pass on NaNs.
            try:
                if not (math.isfinite(float(r["dockq"])) and math.isfinite(float(r["iptm"]))):
                    continue
            except ValueError:
                continue
            if r["arm"] != ARM:
                extra.append((ifc, r["sample"], f"arm={r['arm']!r}"))
                continue
            seen[(ifc, str(r["sample"]))] += 1
    samples = [str(k) for k in range(nsample)]
    exact, partial, absent = [], [], []
    for p in structures:
        ifs = pool.get(p)
        if not ifs:
            absent.append((p, "no interface in --pool"))
            continue
        want = {(i, s) for i in sorted(ifs) for s in samples}
        got = {k for k in want if seen.get(k)}
        dup = sorted(k for k in want if seen.get(k, 0) > 1)
        if dup:
            partial.append((p, len(got), len(want), [f"DUPLICATE {k[0]} sample {k[1]}" for k in dup[:3]]))
        elif not got:
            absent.append((p, f"0 of {len(want)} key(s)"))
        elif got == want:
            exact.append(p)
        else:
            miss = sorted(want - got)
            byif = collections.Counter(k[0] for k in miss)
            det = [f"{i} ({n} sample(s))" for i, n in sorted(byif.items())]
            partial.append((p, len(got), len(want), det))
    return exact, partial, absent, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--pool", default="training/combined/pool.csv",
                    help="the pool whose interfaces define the expected set; see the docstring for "
                         "why this is combined/pool.csv and not fold/pool.csv")
    ap.add_argument("--structures", default="docs/extra_seed_structures.txt")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="exit status only, for a shell gate")
    ap.add_argument("--targets-out", help="write the pdb:iface target string a recovery sweep takes")
    ap.add_argument("--limit", type=int, default=12, help="how many partial structures to print")
    a = ap.parse_args()

    structures = [l.strip().lower() for l in open(a.structures)
                  if l.strip() and not l.startswith("#")]
    pool = pool_ifaces(a.pool, set(structures))
    exact, partial, absent, extra = coverage(a.results, pool, structures, a.seed, a.nsample)

    n = len(structures)
    if not a.quiet:
        print(f"  seed {a.seed}: {len(exact)} of {n} structure(s) EXACT "
              f"(every interface in {os.path.basename(a.pool)} x {a.nsample} sample(s))")
        if partial:
            print(f"    {len(partial)} PARTIAL -- the old row-count gate called these complete:")
            for p, g, w, det in sorted(partial)[:a.limit]:
                print(f"      {p}: {g} of {w} key(s); missing {', '.join(det[:4])}")
            if len(partial) > a.limit:
                print(f"      ... and {len(partial) - a.limit} more")
        if absent:
            print(f"    {len(absent)} with NO rows: {[p for p, _ in sorted(absent)[:8]]}")
        if extra:
            print(f"    {len(extra)} unexpected key(s) e.g. {extra[:3]}")

    # THE SWEEP TARGET STRING, so the recovery acts on the gap this found rather than on its own
    # independently derived idea of the gap. That divergence is the defect itself.
    if a.targets_out:
        tgt = sorted({f"{p}:{i}" for p, _g, _w, _d in partial for i in pool[p]}
                     | {f"{p}:{i}" for p, _why in absent for i in pool.get(p, ())})
        tgt = [t for t in tgt if len(t.split(":")[1].split("_")[-1]) == 2]
        with open(a.targets_out, "w") as fh:
            fh.write(",".join(tgt) + ("\n" if tgt else ""))
        if not a.quiet:
            print(f"    wrote {len(tgt)} sweep target(s) -> {a.targets_out}")

    return 0 if (len(exact) == n) else 1


if __name__ == "__main__":
    sys.exit(main())
