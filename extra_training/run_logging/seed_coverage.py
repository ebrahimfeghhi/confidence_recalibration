#!/usr/bin/env python
"""Is a seed COMPLETE? The exact key set, and FAIL CLOSED on anything unexpected.

WHY THIS EXISTS, AND WHAT IT CAUGHT. chain_extra_seeds.sh judged a seed complete by counting rows
per PDB and asking whether the count was zero -- so a structure whose first interface scored and
whose second did not was indistinguishable from one that scored everything. 83.1% of the frozen 437
have more than one scoreable interface, and folder.py appends per INTERFACE (run_model/folder.py:774
"flushed per interface, not at the end"), so the loose test is wrong by construction, not by race.

Measured against training/combined/pool.csv, the pool the trainer actually reads:
    seed 500000   437 of 437 structures exact
    seed 500010   422 of 437            seed 500012   423 of 437
    seed 500011   423 of 437            seed 500013   423 of 437
and the chain reported 437 of 437 for every one of them.

FAIL CLOSED, WHICH THE FIRST VERSION DID NOT DO AND ITS DOCSTRING CLAIMED IT DID. Raised in review
and correct: rows whose interface was not in the expected set were silently skipped, and wrong-arm
rows were collected into a list that the exit status never consulted -- so an unexpected scored pair
or a stray arm could coexist with complete expected coverage and the tool returned 0 while promising
it refused them. Every row belonging to a requested (pdb, seed) is now classified, and the exit
status depends on exact expected coverage AND the absence of forbidden extras.

WHAT COUNTS AS FORBIDDEN: an interface the pool does not name for that structure, a sample index
outside 0..nsample-1, an arm other than `free`, or any key appearing twice. These fail.
WHAT COUNTS AS MERELY UNSCORED: a row whose dockq or iptm is not finite, or is outside [0, 1].
folder.open_csv would skip such a row, so treating it as a score would be worse than treating it as
absent -- it is reported and makes the interface short, which is the same outcome as no row at all.

WHICH POOL. --pool defaults to training/combined/pool.csv because that is what training reads, and
it is the only one of the three that is exactly right: training/fold/pool.csv names 10 interfaces
OST can never score (the chem-group drop -- 3rhw_DK, 4rdq_DF, 8xcd_AC and 7 more), so it reports
permanent failures as gaps; pool_scoreable.csv still carries 8xcd_AC. A gate that cries on every
seed gets switched off, so the expected set has to be the achievable one.

THE PLAN IS THE POINT OF --plan-out. A recovery whose idea of the gap is derived independently of
the gate that judges it is the defect this whole file exists to fix, so the exact missing
(iface, seed, sample) keys are written for phase_l_merge_recovered.py --plan to require equality
against. That is also what makes a PARTIALLY scored interface recoverable: the merge's old contract
demanded exactly seeds x nsample new rows per interface, which a 4-of-5 fill can never satisfy.
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


def classify(results, pool, structures, seed, nsample):
    """out: (good, forbidden, unscored) -- good is {(iface, sample)} that count as scored.

    EVERY ROW OF A REQUESTED (pdb, seed) IS LOOKED AT, including ones whose interface the pool does
    not name. Skipping those is what let an unexpected pair pass.
    """
    sset = set(structures)
    samples = {str(k) for k in range(nsample)}
    seen = collections.Counter()
    forbidden, unscored = [], []
    with open(results) as fh:
        for r in csv.DictReader(fh):
            if r["seed"] != str(seed):
                continue
            ifc = r["iface"]
            pdb = ifc.split("_")[0]
            if pdb not in sset:
                continue
            smp = str(r["sample"])
            if ifc not in pool.get(pdb, ()):
                forbidden.append((ifc, smp, f"interface not named by the pool for {pdb}"))
                continue
            if smp not in samples:
                forbidden.append((ifc, smp, f"sample index outside 0..{nsample - 1}"))
                continue
            if r["arm"] != ARM:
                forbidden.append((ifc, smp, f"arm={r['arm']!r}, expected {ARM!r}"))
                continue
            try:
                dq, ip = float(r["dockq"]), float(r["iptm"])
                if not (math.isfinite(dq) and math.isfinite(ip)):
                    unscored.append((ifc, smp, "dockq or iptm not finite")); continue
                if not (0.0 <= dq <= 1.0 and 0.0 <= ip <= 1.0):
                    unscored.append((ifc, smp, f"dockq {dq} or iptm {ip} outside [0, 1]")); continue
            except (ValueError, KeyError):
                unscored.append((ifc, smp, "dockq or iptm unparseable")); continue
            seen[(ifc, smp)] += 1
    for k, n in sorted(seen.items()):
        if n > 1:
            forbidden.append((k[0], k[1], f"appears {n} times"))
    return {k for k in seen}, forbidden, unscored


def coverage(results, pool, structures, seed, nsample):
    """out: (exact, partial, absent, forbidden, unscored). partial/absent name structures."""
    good, forbidden, unscored = classify(results, pool, structures, seed, nsample)
    samples = [str(k) for k in range(nsample)]
    exact, partial, absent = [], [], []
    for p in structures:
        ifs = pool.get(p)
        if not ifs:
            absent.append((p, "no interface in --pool", []))
            continue
        want = {(i, s) for i in sorted(ifs) for s in samples}
        got = want & good
        if not got:
            absent.append((p, f"0 of {len(want)} key(s)", sorted(want)))
        elif got == want:
            exact.append(p)
        else:
            partial.append((p, len(got), len(want), sorted(want - got)))
    return exact, partial, absent, forbidden, unscored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--pool", default="training/combined/pool.csv",
                    help="the pool whose interfaces define the expected set; see the docstring")
    ap.add_argument("--structures", default="docs/extra_seed_structures.txt")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="exit status only, for a shell gate")
    ap.add_argument("--targets-out", help="the pdb:iface target string a recovery sweep takes")
    ap.add_argument("--plan-out",
                    help="csv of the EXACT missing (iface, seed, sample) keys, for the merge to "
                         "require equality against")
    ap.add_argument("--limit", type=int, default=12, help="how many partial structures to print")
    a = ap.parse_args()

    structures = [l.strip().lower() for l in open(a.structures)
                  if l.strip() and not l.startswith("#")]
    pool = pool_ifaces(a.pool, set(structures))
    exact, partial, absent, forbidden, unscored = coverage(
        a.results, pool, structures, a.seed, a.nsample)

    n = len(structures)
    if not a.quiet:
        print(f"  seed {a.seed}: {len(exact)} of {n} structure(s) EXACT "
              f"(every interface in {os.path.basename(a.pool)} x {a.nsample} sample(s))")
        if partial:
            print(f"    {len(partial)} PARTIAL -- the old row-count gate called these complete:")
            for p, g, w, miss in sorted(partial)[:a.limit]:
                byif = collections.Counter(k[0] for k in miss)
                det = ", ".join(f"{i} ({c} sample(s))" for i, c in sorted(byif.items()))
                print(f"      {p}: {g} of {w} key(s); missing {det}")
            if len(partial) > a.limit:
                print(f"      ... and {len(partial) - a.limit} more")
        if absent:
            print(f"    {len(absent)} with NO rows: {[p for p, _w, _m in sorted(absent)[:8]]}")
        if unscored:
            print(f"    {len(unscored)} row(s) present but NOT A SCORE (treated as missing): "
                  f"{unscored[:3]}")
        if forbidden:
            print(f"    {len(forbidden)} FORBIDDEN key(s) -- these fail the gate on their own: "
                  f"{forbidden[:4]}")

    miss_keys = sorted({k for _p, _g, _w, m in partial for k in m}
                       | {k for _p, _w, m in absent for k in m})
    # ONLY SORTED SINGLE-LETTER PAIRS ARE ADDRESSABLE by the sweep's pdb:pdb_C1C2 form.
    def addressable(ifc):
        return len(ifc.split("_")[-1]) == 2

    if a.targets_out:
        # ONLY THE INTERFACES ACTUALLY SHORT, never every interface of a partial structure. The
        # merge runs with --replace, so naming a complete interface would overwrite good rows with
        # rows from a DIFFERENT scoring route -- a repair that silently changes correct data. At
        # seed 500010 the loose derivation named 46 interfaces where 20 are short.
        tgt = sorted({f"{i.split('_')[0]}:{i}" for i, _s in miss_keys if addressable(i)})
        with open(a.targets_out, "w") as fh:
            fh.write(",".join(tgt) + ("\n" if tgt else ""))
        if not a.quiet:
            print(f"    wrote {len(tgt)} sweep target(s) -> {a.targets_out}")

    if a.plan_out:
        rows = [(i, a.seed, s) for i, s in miss_keys if addressable(i)]
        with open(a.plan_out, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["iface", "seed", "sample"])
            w.writerows(rows)
        if not a.quiet:
            skipped = len(miss_keys) - len(rows)
            print(f"    wrote a {len(rows)}-key plan -> {a.plan_out}"
                  + (f" ({skipped} key(s) not addressable by the sweep)" if skipped else ""))

    # THE EXIT STATUS DEPENDS ON BOTH. Exact coverage of what is expected, and nothing forbidden.
    return 0 if (len(exact) == n and not forbidden) else 1


if __name__ == "__main__":
    sys.exit(main())
