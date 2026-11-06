#!/usr/bin/env python
"""What the training fold actually offers a reranker: spread, acceptability and the oracle gap.

THE METRICS ARE THE PROJECT'S. Selection is max(score, -seed, -sample) and acceptability is
dockq >= 0.23, exactly as phase_k_eval and abag_rank/eval.py do. The tie-break is not cosmetic:
row order differs between a sharded run and a single-process one, so two poses with equal score
would otherwise select different DockQ depending on how the work was split. The shipped baseline is
recomputed from these same rows using their own stored ipTM, never quoted from elsewhere, so every
figure below is paired on identical poses.

WHY SPREAD AND ACCEPTABILITY ARE REPORTED AS FAILURES OF THE EXAMPLE, not of the model. A reranker
learns to choose among poses, so an interface whose 50 poses all score the same teaches it nothing
whichever one it picks -- there is no better choice to find. An interface where no pose reaches
0.23 teaches it nothing either, because every available choice is a failure. Both are real
training examples that cannot contribute ranking signal, and counting them is the difference
between "the method gained nothing" and "the method had nothing to gain here".

  THE TWO ZERO-SPREAD CASES ARE NOT THE SAME THING and are reported apart. All 50 poses good is a
  solved complex; all 50 bad is one the folder cannot do. They look identical to a ranking loss and
  completely different to anyone deciding what to fold next.

THE ORACLE GAP IS THE HEADROOM, and which gap depends on what it is compared against:
  oracle - shipped   what a perfect selector would add over the ipTM the model already emits.
                     This is the number a reranking method is trying to close.
  oracle - mean      what it would add over picking a pose at random. Always larger, and not the
                     target, because ipTM is not random.
"""
import argparse, collections, csv, math, os, statistics, sys

ACCEPTABLE = 0.23
FLAT = 0.05


def load(results_csv, queue_tsv, seeds, nsample, pool_csv=None, seed0=500000,
         only=None, manifest=None):
    """iface -> [(seed, sample, iptm, dockq)] for the scope only.

    SCOPE IS A QUEUE OR A POOL, never "every row in the file". training/fold/results.csv holds rows
    for several pools folded in different waves -- full_train's 488 structures and this fold's 356
    among them, with no structure in both -- so a report that read the whole file would mix two
    training sets and compare neither. A queue.tsv names the units actually folded in one wave; a
    pool plus a seed range is how to scope a wave whose queue is no longer around.
    """
    scope = set()
    want_ifaces = None
    if manifest:
        # A PER-STRUCTURE SEED SET, BECAUSE THE KEPT TRUNKS ARE NOT A CONTIGUOUS RANGE. The wave-1
        # validation trim kept 30 anchored-coverage seeds per structure, and 29 of the 30
        # structures got a DIFFERENT set -- 7shz keeps 400037 and 400070, which no --seeds N range
        # starting at 400000 contains. Scoring rows exist for all 100 seeds, so a contiguous
        # budget still reports a number; it is just not the budget this pool can be EVALUATED at,
        # because an eval needs the trunk. Reading the manifest is the only way to measure the
        # headroom that is actually reachable.
        import json
        m = json.load(open(manifest))
        for pdb, v in m["structures"].items():
            for sd in v["seeds"]:
                scope.add((pdb, str(sd)))
    elif queue_tsv:
        with open(queue_tsv) as fh:
            for ln in fh:
                f = ln.rstrip("\n").split("\t")
                if len(f) >= 5:
                    scope.add((f[1], f[4]))
    else:
        # SCOPING BY POOL MEANS ITS INTERFACES, NOT ITS STRUCTURES. Scoping on structure alone
        # collected 992 interfaces for full_train's 488 structures when its pool names 977: the
        # master training/fold pool lists more interfaces per structure than full_train's own, and
        # the extra 15 were never part of that training set. Comparing a 992-interface set against
        # a 977-interface one as though they were the same thing is exactly the mistake this
        # report exists to avoid.
        want_ifaces = {f"{r['pdb']}_{r['chain1']}{r['chain2']}"
                       for r in csv.DictReader(open(pool_csv))}
        pdbs = {i.split("_")[0] for i in want_ifaces}
        scope = {(p, str(seed0 + i)) for p in pdbs for i in range(seeds)}
    # RESTRICTING TO A WAVE IS A SCOPE NARROWING, NOT A FILTER ON OUTCOME. The two validation
    # waves were folded months apart against the same 100-seed budget, and their headroom is the
    # thing being compared, so each has to be measured over its own structures alone. This clips
    # the scope BEFORE any row is read, so pooled/scoreable counts and the gap are all computed
    # over the same restricted set rather than one being the whole pool and another a subset.
    if only:
        missing = only - {p for p, _ in scope}
        if missing:
            raise SystemExit(f"STOP: {len(missing)} --only structure(s) are not in scope, so the "
                             f"report would silently cover fewer than asked: {sorted(missing)[:8]}")
        scope = {(p, sd) for p, sd in scope if p in only}
    if want_ifaces is None and not queue_tsv and pool_csv:
        want_ifaces = {f"{r['pdb']}_{r['chain1']}{r['chain2']}"
                       for r in csv.DictReader(open(pool_csv))}
    out = collections.defaultdict(list)
    with open(results_csv) as fh:
        for r in csv.DictReader(fh):
            pdb = r["iface"].split("_")[0]
            if (pdb, r["seed"]) not in scope:
                continue
            try:
                dq, ip = float(r["dockq"]), float(r["iptm"])
            except ValueError:
                continue
            if not (math.isfinite(dq) and math.isfinite(ip)):
                continue
            if want_ifaces is not None and r["iface"] not in want_ifaces:
                continue
            out[r["iface"]].append((int(r["seed"]), int(r["sample"]), ip, dq))
    return out, scope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training/fold/results.csv")
    ap.add_argument("--queue", default="training/fold/queue.tsv")
    ap.add_argument("--pool", default="training/fold/pool.csv")
    ap.add_argument("--scoreable", default="training/fold/pool_scoreable.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--flat", type=float, default=FLAT)
    ap.add_argument("--seed0", type=int, default=500000,
                    help="first seed, used only when --queue is empty and scope comes from --pool")
    ap.add_argument("--seed-manifest", default="",
                    help="json with structures->seeds; scope each structure to its OWN kept seeds "
                         "rather than a contiguous --seeds range")
    ap.add_argument("--only", default="",
                    help="file listing structures, one per line; restrict the whole report to them")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    only = None
    if a.only:
        only = {l.split()[0] for l in open(a.only) if l.strip() and l[0] != "#"}
        if not only:
            raise SystemExit(f"STOP: --only {a.only} names no structure")
    per, scope = load(a.csv, a.queue, a.seeds, a.nsample, a.pool, a.seed0, only,
                      a.seed_manifest)
    pdbs = {p for p, _ in scope}
    want = a.seeds * a.nsample

    pooled = {f"{r['pdb']}_{r['chain1']}{r['chain2']}"
              for r in csv.DictReader(open(a.pool))} if os.path.exists(a.pool) else set()
    scoreable = {f"{r['pdb']}_{r['chain1']}{r['chain2']}"
                 for r in csv.DictReader(open(a.scoreable))} \
        if os.path.exists(a.scoreable) else set()
    pooled_in_scope = {i for i in pooled if i.split("_")[0] in pdbs}
    scoreable_in_scope = {i for i in scoreable if i.split("_")[0] in pdbs}

    print("=" * 78)
    print("STRUCTURES AND INTERFACES IN THE TRAINING FOLD")
    print("=" * 78)
    print(f"  structures queued                 {len(pdbs)}")
    print(f"  interfaces named by pool.csv      {len(pooled_in_scope)}")
    print(f"  of those, scoreable (chains touch){len(scoreable_in_scope):>4}")
    print(f"  interfaces with >=1 scored pose   {len(per)}")
    print(f"  poses per interface               {a.seeds} seeds x {a.nsample} samples = {want}")
    short = {i: len(v) for i, v in per.items() if len(v) != want}
    print(f"  interfaces NOT at the full {want}      {len(short)}"
          + (f"  {sorted(short.items())[:5]}" if short else ""))
    by_pdb = collections.Counter(i.split("_")[0] for i in per)
    print(f"  interfaces per structure          "
          f"{dict(sorted(collections.Counter(by_pdb.values()).items()))}")

    rows = []
    for ifc, v in sorted(per.items()):
        dq = [t[3] for t in v]
        oracle = max(dq)
        mean = statistics.mean(dq)
        sd = statistics.stdev(dq) if len(dq) > 1 else 0.0
        shipped = max(v, key=lambda t: (t[2], -t[0], -t[1]))[3]
        rows.append(dict(iface=ifc, pdb=ifc.split("_")[0], n=len(dq), oracle=oracle, mean=mean,
                         std=sd, shipped=shipped, worst=min(dq),
                         flat=sd < a.flat, no_acc=oracle < ACCEPTABLE))

    n = len(rows)
    flat = [r for r in rows if r["flat"]]
    noacc = [r for r in rows if r["no_acc"]]
    either = [r for r in rows if r["flat"] or r["no_acc"]]
    both = [r for r in rows if r["flat"] and r["no_acc"]]
    usable = [r for r in rows if not (r["flat"] or r["no_acc"])]
    flat_good = [r for r in flat if not r["no_acc"]]

    print()
    print("=" * 78)
    print(f"EXAMPLES THAT CANNOT CONTRIBUTE RANKING SIGNAL  (of {n} interfaces)")
    print("=" * 78)
    print(f"  DockQ std < {a.flat:<5} (no spread: every pose scores alike)  "
          f"{len(flat):>4}  {100*len(flat)/n:5.1f}%")
    print(f"      of those, all poses ACCEPTABLE (solved complex)   {len(flat_good):>4}")
    print(f"      of those, all poses BAD (folder cannot do it)     {len(both):>4}")
    print(f"  no acceptable pose (oracle DockQ < {ACCEPTABLE})            "
          f"{len(noacc):>4}  {100*len(noacc)/n:5.1f}%")
    print(f"  FAILS EITHER TEST                                   "
          f"{len(either):>4}  {100*len(either)/n:5.1f}%")
    print(f"  fails both                                          {len(both):>4}")
    print(f"  USABLE for ranking                                  "
          f"{len(usable):>4}  {100*len(usable)/n:5.1f}%")
    # NO ACCEPTABLE POSE IS NOT THE SAME AS NO RANKING INFORMATION. Raised in review: an interface
    # whose oracle is 0.20 with a spread of 0.15 offers nothing to a SELECTION metric -- every
    # choice is a failure by the 0.23 criterion -- but it still carries graded supervision, because
    # 0.20 really is better than 0.02 and a relative loss can learn that ordering. These are
    # counted apart so that "usable" is read as "usable for the acceptability metric", which is
    # what it means, rather than as "contains no signal".
    graded = [r for r in noacc if not r["flat"]]
    print(f"  of the no-acceptable-pose ones, still GRADED        "
          f"{len(graded):>4}  (spread >= {a.flat}, so a relative loss can order them)")
    print(f"  genuinely inert (flat AND no acceptable pose)       {len(both):>4}")
    us = {r["pdb"] for r in usable}
    print(f"  structures with >=1 usable interface                 "
          f"{len(us):>4}  of {len(pdbs)}")

    def gap(rs, lbl):
        if not rs:
            print(f"  {lbl:<34} n=0")
            return
        orc = statistics.mean(r["oracle"] for r in rs)
        shp = statistics.mean(r["shipped"] for r in rs)
        mn = statistics.mean(r["mean"] for r in rs)
        d = [r["oracle"] - r["shipped"] for r in rs]
        se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
        print(f"  {lbl:<34} n={len(rs):<4} oracle {orc:.4f}  shipped {shp:.4f}  "
              f"mean {mn:.4f}")
        print(f"      oracle - shipped  {orc - shp:+.4f} +- {se:.4f}"
              f"      oracle - mean  {orc - mn:+.4f}")
        print(f"      frac acceptable: oracle {sum(1 for r in rs if r['oracle'] >= ACCEPTABLE)/len(rs):.4f}"
              f"   shipped {sum(1 for r in rs if r['shipped'] >= ACCEPTABLE)/len(rs):.4f}")

    print()
    print("=" * 78)
    print(f"ORACLE GAP  (budget {want} poses; selection max(score, -seed, -sample))")
    print("=" * 78)
    gap(rows, "every scored interface")
    gap(usable, "usable interfaces only")
    gap(noacc, "no-acceptable-pose interfaces")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out} ({len(rows)} interface(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
