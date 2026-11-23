#!/usr/bin/env python
"""WHICH INTERFACES CHANGED, not how the mean moved.

A net +2.2 points of frac acceptable@5 on 141 interfaces is three interfaces. The aggregate cannot
say whether an arm rescued three and broke none, or rescued eight and broke five, and it cannot say
whether two arms rescued the SAME three. This reports gains and losses separately, per interface,
and the rank of the best acceptable pose on every interface whose status changed.

RANK IS 1-BASED OVER THE CANDIDATES THE RANKER ORDERS, and "rank of the best acceptable pose" means
the position of the FIRST candidate at or above the acceptance threshold once sorted by score. An
interface with no acceptable candidate anywhere is UNWINNABLE: no ranker can score on it, so it is
excluded from the gain/loss counts and reported separately, exactly as --contestable does for the
aggregate tables.

TIES take the row order the pass wrote, the same convention the evaluator's own argmax uses.
"""
import argparse, collections, csv, glob, os, sys
ACC = 0.23
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load(dirs, keep=None):
    by = collections.defaultdict(list)
    for d in dirs:
        for f in glob.glob(os.path.join(d, "rows_*.csv")):
            if f.endswith(".pae.csv"):
                continue
            for r in csv.DictReader(open(f)):
                try:
                    m = r.get("iptm_mean")
                    by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]),
                                           float(r["dockq"]),
                                           float(m) if m not in (None, "") else None))
                except (ValueError, KeyError, TypeError):
                    continue
    if keep is not None:
        by = {i: c for i, c in by.items() if i in keep}
    return by


def rank_of_first_acceptable(cands, key):
    """1-based rank of the best-ranked acceptable pose, or None if the ranker finds none."""
    for i, t in enumerate(sorted(cands, key=key, reverse=True), start=1):
        if t[2] >= ACC:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--pooling", choices=["max", "mean"], default="max")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", default="")
    ap.add_argument("--list", action="store_true", help="name every interface that changed")
    a = ap.parse_args()
    keep = None
    if a.pool and os.path.exists(a.pool):
        keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}' for r in csv.DictReader(open(a.pool))}
    by = load([x for x in a.dirs.split(",") if x], keep)
    ship = lambda t: t[1]
    ours = (lambda t: t[0]) if a.pooling == "max" else (lambda t: t[3] if t[3] is not None else t[0])

    unwinnable, rows = [], []
    for iface, c in by.items():
        if not any(t[2] >= ACC for t in c):
            unwinnable.append(iface)
            continue
        rows.append((iface, rank_of_first_acceptable(c, ship), rank_of_first_acceptable(c, ours)))

    def bucket(k):
        g = [r for r in rows if (r[1] is None or r[1] > k) and r[2] is not None and r[2] <= k]
        l = [r for r in rows if (r[2] is None or r[2] > k) and r[1] is not None and r[1] <= k]
        return g, l

    print(f"\n=== {a.label or a.dirs}  [{a.pooling}-pooled] ===")
    print(f"  {len(by)} interface(s); {len(unwinnable)} unwinnable (no acceptable pose exists), "
          f"{len(rows)} contestable")
    for k in (1, a.k):
        g, l = bucket(k)
        print(f"  TOP-{k}: gained {len(g)}, lost {len(l)}, net {len(g)-len(l):+d}"
              f"   ({100.0*(len(g)-len(l))/max(1,len(by)):+.1f} pp over all {len(by)})")
        if a.list:
            for iface, sr, orr in sorted(g):
                print(f"      GAIN {iface:16s} shipped rank {sr if sr else '-':>4} -> ours {orr}")
            for iface, sr, orr in sorted(l):
                print(f"      LOSS {iface:16s} shipped rank {sr:>4} -> ours {orr if orr else '-'}")
    # WHERE THE RESCUED ONES LANDED: a gain at 5 that is also a gain at 1 is a different thing
    # from a gain that stops at rank 4, and the aggregate cannot tell them apart.
    g5, _ = bucket(a.k)
    at1 = sum(1 for r in g5 if r[2] == 1)
    print(f"  of the {len(g5)} gained at top-{a.k}, {at1} reached RANK 1; "
          f"rank histogram {dict(sorted(collections.Counter(r[2] for r in g5).items()))}")
    print(f"  INTERFACES_GAINED_{a.k}: {','.join(sorted(r[0] for r in g5))}")


if __name__ == "__main__":
    main()
