#!/usr/bin/env python
"""THE FULL-TRAINING-POOL IN-SAMPLE PASS, SPLIT BY FOLDED DEPTH, BOTH POOLINGS.

    trainfull_strata.py <rid> <epoch> [<rid2> <epoch2> ...] [--md]

WHY SPLIT IT. training/fulltrain_mixed holds 1,688 interfaces, but 741 of them sit on structures
with ONE folded seed: 5 candidates to choose among, and an oracle only 0.014 above their own
shipped baseline. Pooled with the 947 interfaces that have 15 seeds and 0.167 of headroom, they
halve every delta by construction -- the combined number is honest about the denominator and
useless for telling arms apart. The strata are reported separately for that reason.

BOTH POOLINGS, FROM THE ROWS. The metric json's `strata` block carries the max-pooled column
only; `iptm` and `iptm_mean` are both in the shard rows, so the mean-pooled column is derived
the same way rather than re-run. Shipped comes from `stored_iptm` in those same rows, so every
column of a stratum is the same poses scored three ways.

THE ORACLE DOES NOT DEPEND ON POOLING -- it is the best DockQ available among the candidates,
which is what makes `headroom` the right denominator for reading a delta.
"""
import argparse, collections, csv, glob, json, os, statistics, sys

ACC = 0.23
MANIFEST = "training/fulltrain_mixed/seed_manifest.json"


def load(rid, ep):
    by = collections.defaultdict(list)
    for f in glob.glob(f"training/runs/{rid}/trainfull_eval_shards_ep{ep}/rows*_s*.csv"):
        if "paeloss" in f:
            continue
        for r in csv.DictReader(open(f)):
            try:
                by[r["iface"]].append((float(r["stored_iptm"]), float(r["iptm"]),
                                       float(r["iptm_mean"]), float(r["dockq"])))
            except (KeyError, ValueError):
                continue
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="rid epoch rid epoch ...")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--depth", type=int, action="append",
                    help="only this folded depth; repeatable. Default: every stratum present.")
    a = ap.parse_args()
    if len(a.pairs) % 2:
        sys.exit("pass rid epoch pairs")
    jobs = [(a.pairs[i], a.pairs[i + 1]) for i in range(0, len(a.pairs), 2)]

    man = json.load(open(MANIFEST))["structures"]
    depth = {p: (len(v["seeds"]) if isinstance(v, dict) and "seeds" in v else len(v))
             for p, v in man.items()}

    out = collections.defaultdict(dict)
    strata_n = {}
    for rid, ep in jobs:
        by = load(rid, ep)
        if not by:
            print(f"  {rid} ep{ep}: no rows", file=sys.stderr)
            continue
        groups = collections.defaultdict(list)
        for iface in by:
            groups[depth.get(iface.split("_")[0], 0)].append(iface)
        for d, ifaces in groups.items():
            strata_n[d] = len(ifaces)
            # RANK OF THE FIRST ACCEPTABLE POSE. The candidates are sorted by the pooled ipTM,
            # descending, and this is the 1-based position of the first one with DockQ >= ACC.
            # It is the quantity the selector actually controls and it is GRADED, where top-1
            # frac acceptable only moves when the pick crosses the threshold -- an arm that
            # lifts an acceptable pose from rank 30 to rank 4 shows nothing in frac and a clear
            # move here.
            #
            # RESTRICTED TO WINNABLE INTERFACES, and that choice is the one thing to know about
            # it: 6.5% of depth-15 interfaces have NO acceptable pose among their candidates, so
            # "the rank of the first acceptable pose" is undefined for them. Including them would
            # need a censoring convention (rank = n+1, say), which would make the mean a function
            # of the candidate count and so incomparable between the two strata -- 75 candidates
            # against 5. They are dropped instead and their count is reported, so nothing is
            # hidden and the two strata stay comparable.
            row = {}
            win = [i for i in ifaces if any(t[3] >= ACC for t in by[i])]
            row["_n_win"] = len(win)
            for name, idx in (("shipped", 0), ("max", 1), ("mean", 2)):
                v = [max(by[i], key=lambda t: t[idx])[3] for i in ifaces]
                rk = []
                for i in win:
                    order = sorted(by[i], key=lambda t: -t[idx])
                    rk.append(next(j for j, t in enumerate(order, 1) if t[3] >= ACC))
                row[name] = (statistics.fmean(v),
                             100.0 * sum(1 for x in v if x >= ACC) / len(v),
                             statistics.median(rk) if rk else float("nan"),
                             statistics.fmean(rk) if rk else float("nan"))
            orc = [max(t[3] for t in by[i]) for i in ifaces]
            row["oracle"] = (statistics.fmean(orc),
                             100.0 * sum(1 for x in orc if x >= ACC) / len(orc),
                             1.0, 1.0)
            out[d][f"{rid} ep{ep}"] = row

    # ONE TABLE PER POOLING, each carrying its own shipped and oracle rows. SHIPPED AND ORACLE
    # ARE THE SAME IN BOTH, by construction: shipped selects on `stored_iptm`, a single stored
    # value with no pooling to choose, and the oracle is the best DockQ available among the
    # candidates regardless of how they were ranked. Repeating them in each table is the point --
    # a delta read against the wrong baseline is the mistake this layout exists to prevent.
    for pool in ("max", "mean"):
        print(("\n**" + f"{pool.upper()}-POOLED ipTM" + "**") if a.md
              else ("\n=== " + f"{pool.upper()}-POOLED ipTM" + " ==="))
        cols = ["stratum", "n", "arm", "DockQ", "frac acc",
                "median rank of 1st acceptable", "mean rank"]
        if a.md:
            print("\n| " + " | ".join(cols) + " |")
            print("|" + "---|" * len(cols))
        # SHIPPED FIRST, THEN THE ARMS, THEN THE ORACLE -- the order you read the stratum in.
        # The baseline has to be above the deltas it is the baseline for; putting it underneath
        # invites reading an arm's row against the stratum above it.
        def emit(lab, n, nm, t):
            c = [lab, str(n), nm, f"{t[0]:.4f}", f"{t[1]:.1f}%",
                 f"{t[2]:.1f}", f"{t[3]:.2f}"]
            print(("| " + " | ".join(c) + " |") if a.md
                  else "  ".join(x.ljust(15) for x in c))

        for d in sorted(out, reverse=True):
            if a.depth and d not in a.depth:
                continue
            lab = f"depth {d} ({5*d} cand.)"
            r0 = next(iter(out[d].values()))
            emit(lab, strata_n[d], "shipped", r0["shipped"])
            for arm, r in out[d].items():
                emit(lab, strata_n[d], arm, r[pool])
            emit(lab, strata_n[d], "oracle", r0["oracle"])
            print((f"\n_{r0['_n_win']} of {strata_n[d]} interfaces have an acceptable pose "
                   f"among their candidates; the rank columns cover those._") if a.md
                  else f"  ({r0['_n_win']} of {strata_n[d]} winnable)")


if __name__ == "__main__":
    main()
