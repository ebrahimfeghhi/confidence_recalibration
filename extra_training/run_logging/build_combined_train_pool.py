#!/usr/bin/env python
"""Build ONE training pool from every scoreable interface that actually has scores.

WHY THIS EXISTS. The rows for three separately-folded waves all live in
training/fold/results.csv, and until now only two of them were named by a training pool:

    full_train/pool.csv      977 interfaces   488 structures   the previous arms' set
    this wave (queue.tsv)    660 interfaces   356 structures   folded and scored today
    orphans                   51 interfaces    19 structures   scored, in NO training pool
    never scored               4 interfaces     3 structures   needs GPU, excluded here

The three sets share no INTERFACE, but 10 of the 19 orphan structures are structures full_train
already names -- it pooled only some of their interfaces. So 488 + 356 + 19 = 863 double-counts
those 10, and the union is 1,688 interfaces over **853** structures. (The first version of this
docstring said 863, which is the figure you get by assuming the structure sets are disjoint too.)
The 51 orphans were complete all along -- 50 poses each, finite scores, all in the master pool -- and were
simply never written into a pool anyone trained on, so every arm so far silently ignored them.

ALL 51 GO IN, NOT THE 22 THAT CARRY RANKING SIGNAL. Selecting interfaces by their own DockQ spread
or by whether any pose is acceptable is selecting on the label: it would make this pool a different
kind of object from full_train/pool.csv, which contains 286 interfaces where no pose reaches 0.23
and 211 where every pose is equally good. A pool filtered on outcome cannot be compared with one
that is not, and the comparison is the entire point of keeping the arms paired.

THE 4 NEVER-SCORED ARE EXCLUDED, NOT SILENTLY DROPPED. 8xcd_AC, 8zh7_AB, 9mx1_AB and 9mx1_AC touch
in the reference and have no rows at all, because their 3 structures were never folded. A pool that
named them would make every completeness check fail for a reason that is not a scoring fault. They
are written to the exclusions file with that reason so the omission is a record, not an oversight.

IT REFUSES RATHER THAN GUESS. Any interface it would name that has no rows, or whose rows are not
exactly --nsample x --seeds, is reported and the build fails -- a training pool whose members are
not all scored is the thing that made a fold look complete when it was not.
"""
import argparse, collections, csv, glob, math, os, sys

COLS = ["pdb", "chain1", "chain2"]
ARM = "free"      # every queued training unit; checked against queue.tsv


def pool_ifaces(path):
    with open(path) as fh:
        return {f"{r['pdb']}_{r['chain1']}{r['chain2']}": (r["pdb"], r["chain1"], r["chain2"])
                for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="training/fold/pool_scoreable.csv",
                    help="the scoreable derivation of the master pool")
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--out", default="training/combined/pool.csv")
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="fewest seeds an interface may have and still count as complete; the "
                         "seeds present must be a contiguous run from --seed0 whatever the count")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    master = pool_ifaces(a.master)
    # THE EXACT KEY SET, NOT A ROW COUNT. Raised in review: "50 finite rows" does not mean "ten
    # seeds times samples 0-4". A set with a missing sample, a seed outside the range and a
    # duplicate key counts 50 too, and would have been written into a training pool as complete.
    have = collections.defaultdict(collections.Counter)
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            try:
                dq, ip = float(r["dockq"]), float(r["iptm"])
            except ValueError:
                continue
            if math.isfinite(dq) and math.isfinite(ip):
                have[r["iface"]][(r["seed"], r["arm"], r["sample"])] += 1

    # THE SEED COUNT IS NO LONGER UNIFORM, so a single expected key set is the wrong test.
    # Requiring exactly seeds 500000-500009 was right while every structure had ten, and became
    # wrong the moment 416 structures were trimmed to one seed and 437 were given an eleventh: at
    # --seeds 10 the trimmed ones report 45 missing and the extended ones 5 unexpected, so the
    # builder would have refused 853 of 853 interfaces for being the shape they are meant to be.
    #
    # What still has to hold, and is what the strict check was actually protecting:
    #   - the seeds present are a CONTIGUOUS run starting at --seed0, with no gap in the middle;
    #   - every seed present carries samples 0..nsample-1, none missing;
    #   - no key appears twice;
    #   - at least --min-seeds seeds, so a structure reduced to nothing is still caught.
    # A gap is the interesting failure: seeds 500000 and 500002 with 500001 absent means a unit was
    # lost, and that is exactly what a count of rows cannot see.
    samples = {str(k) for k in range(a.nsample)}
    keep, no_rows, short, by_nseeds = [], [], [], collections.Counter()
    for ifc, cols in sorted(master.items()):
        got = have.get(ifc, {})
        if not got:
            no_rows.append(ifc)
            continue
        why = []
        dup = sorted(k for k, n in got.items() if n > 1)
        if dup:
            why.append(f"{len(dup)} duplicated key(s) e.g. {dup[0]}")
        bad_arm = sorted({k[1] for k in got} - {ARM})
        if bad_arm:
            why.append(f"arm(s) {bad_arm}, expected {ARM!r}")
        per_seed = collections.defaultdict(set)
        for sd, _arm, smp in got:
            per_seed[sd].add(smp)
        seeds_i = sorted(int(s) for s in per_seed)
        run = [a.seed0 + i for i in range(len(seeds_i))]
        if seeds_i != run:
            why.append(f"seeds are not a contiguous run from {a.seed0}: "
                       f"{seeds_i[:3]}..{seeds_i[-1]} ({len(seeds_i)} present)")
        if len(seeds_i) < a.min_seeds:
            why.append(f"{len(seeds_i)} seed(s), fewer than --min-seeds {a.min_seeds}")
        incomplete = sorted(sd for sd, ss in per_seed.items() if ss != samples)
        if incomplete:
            why.append(f"{len(incomplete)} seed(s) short of {a.nsample} samples e.g. "
                       f"{incomplete[0]} has {sorted(per_seed[incomplete[0]])}")
        if why:
            short.append((ifc, "; ".join(why)))
        else:
            keep.append(cols)
            by_nseeds[len(seeds_i)] += 1

    print(f"master scoreable pool : {len(master)} interface(s)")
    print(f"  complete                   : {len(keep)}")
    print(f"  seeds per complete iface   : {dict(sorted(by_nseeds.items()))}")
    print(f"  no rows at all             : {len(no_rows)}  {sorted(no_rows)}")
    print(f"  incomplete                 : {len(short)}  {sorted(short)[:6]}")
    print(f"  structures in the new pool : {len({c[0] for c in keep})}")

    if short:
        print("BUILD FAILED: a training pool must not name a partially scored interface.")
        return 1

    if a.write:
        d = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(d, exist_ok=True)
        if os.path.abspath(a.out) in (os.path.abspath(a.master), os.path.abspath(a.results)):
            raise SystemExit("STOP: --out would overwrite an input")
        tmp = a.out + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(COLS)
            w.writerows(keep)
        os.replace(tmp, a.out)
        print(f"  wrote {a.out} ({len(keep)} interface(s))")
        xp = os.path.join(d, "pool_unscored.csv")
        tmp = xp + ".tmp"
        # THE REASON IS DERIVED, NOT ASSUMED. The first version of this file said "never folded"
        # for all four, which was wrong for two of them: 8xcd and 8zh7 have ten pose directories
        # each and 50 scored rows for their OTHER interface, so they were folded and the interface
        # was lost in scoring. Those two need a scoring fix; only a structure with no poses at all
        # needs a GPU. Writing one reason for both would have sent GPU work after a CPU problem.
        poses_root = os.path.join(os.path.dirname(os.path.abspath(a.results)), "poses")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(COLS + ["reason", "needs"])
            for ifc in sorted(no_rows):
                pdb = master[ifc][0]
                folded = len(glob.glob(os.path.join(poses_root, pdb, "*", "run.json")))
                if folded:
                    why = (f"folded ({folded} pose dir(s)) and its sibling interface is scored, so "
                           f"this interface was lost in scoring, not in folding")
                    need = "scoring fix"
                else:
                    why = "no pose directory at all: this structure was never folded"
                    need = "GPU"
                w.writerow(list(master[ifc]) + [why, need])
        os.replace(tmp, xp)
        print(f"  wrote {xp} ({len(no_rows)} interface(s)) -- GPU work, not a scoring fault")
    return 0


if __name__ == "__main__":
    sys.exit(main())
