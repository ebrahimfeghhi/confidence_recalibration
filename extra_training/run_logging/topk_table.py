#!/usr/bin/env python
"""TOP-1 AND TOP-5 on one eval pass, for shipped and for the arm, both poolings. Plus PAE loss.

NO EXTRA GPU. Every number here comes from the rows_*.csv an eval pass already wrote -- each row
is (iface, seed, sample, iptm, dockq, stored_iptm[, iptm_mean]) -- so Top-5 is a reader-side
reduction over candidates that were already scored. Adding it costs no forward pass.

WHAT TOP-5 MEANS HERE, stated because there are two defensible readings and they answer different
questions:
  frac_acc@5  the fraction of interfaces with AT LEAST ONE acceptable pose among the 5 the
              ranker put first. This is the number that matters if a human or a downstream
              filter looks at five candidates.
  best@5      the mean over interfaces of the BEST DockQ within those 5.
Both are a ranker's-eye view: the 5 are chosen by the SCORE, and only then is DockQ read. Taking
the best 5 by DockQ would be the oracle, which is reported separately.

TIES ARE BROKEN BY THE ROW ORDER THE PASS WROTE, which is the same convention the evaluator's own
argmax uses. Rounding is what manufactures ties -- the rows carry full repr() precision for
exactly this reason -- so a tie here is a genuine one.

SHIPPED IS MAX-POOLED AND HAS NO MEAN-POOLED TWIN: every row carries one `stored_iptm`, because
mean pooling needs the per-anchor-row PAE distribution and that is computed only for the head
under test. So the mean-pooled column is OURS only.
"""
import argparse
import collections
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
ACC = 0.23


def load(dirs):
    by = collections.defaultdict(list)
    for d in dirs:
        for f in glob.glob(os.path.join(d, "rows_*.csv")):
            # THE PAE SIDECAR IS NOT A ROWS FILE. It is named paeloss_*.csv precisely so this
            # glob cannot pick it up, but the guard stays: six readers share this pattern.
            if f.endswith(".pae.csv"):
                continue
            for r in csv.DictReader(f_ := open(f)):
                try:
                    m = r.get("iptm_mean")
                    by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]),
                                           float(r["dockq"]),
                                           float(m) if m not in (None, "") else None))
                except (ValueError, KeyError, TypeError):
                    continue
            f_.close()
    return by


def topk(by, key, k):
    """(frac_acc@k, best@k) -- at least one acceptable in the top k, and the best DockQ there."""
    n = len(by)
    if not n:
        return float("nan"), float("nan")
    hit, best = 0, []
    for c in by.values():
        top = sorted(c, key=key, reverse=True)[:k]
        dqs = [t[2] for t in top]
        if any(d >= ACC for d in dqs):
            hit += 1
        best.append(max(dqs))
    return hit / n, sum(best) / n


def pae_split(dirs, results, pool=None):
    """Good/bad/all PAE loss, grouped ONLY BY THE POOL'S INTERFACES.

    WHY THE POOL FILTER IS NOT OPTIONAL. results.csv carries a DockQ row for every interface the
    FOLD scored, which is a superset of the pool an arm trains on: 10fj has rows for both AB and
    AC while the hard 280 lists only AB. Without the filter, 6,375 of 13,500 poses were classified
    partly by interfaces the arm never trains on and the metric never reads -- which put the
    shipped baseline's groups at 2,731/9,450/1,319 against the trainer's 3,040/9,985/475 over the
    same poses, and made the baseline lines on the loss figure incomparable to the arm curves they
    were drawn against. With the filter the two agree exactly. The trainer restricts to its
    --pae-scored-pairs mask and is the reference behaviour here.
    """
    keep = None
    if pool and os.path.exists(pool):
        keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}' for r in csv.DictReader(open(pool))}
    dq = collections.defaultdict(list)
    if os.path.exists(results):
        for r in csv.DictReader(open(results)):
            if keep is not None and r["iface"] not in keep:
                continue
            try:
                dq[(r["iface"].split("_")[0], int(r["seed"]), int(r["sample"]))].append(
                    float(r["dockq"]))
            except (ValueError, KeyError):
                continue
    good, bad, allv = [], [], []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "paeloss_*.csv")):
            for r in csv.DictReader(open(f)):
                try:
                    v = float(r["pae_loss"])
                    k = (r["pdb"], int(r["seed"]), int(r["sample"]))
                except (ValueError, KeyError):
                    continue
                allv.append(v)
                g = dq.get(k)
                if not g:
                    continue
                if all(x >= ACC for x in g):
                    good.append(v)
                elif all(x < ACC for x in g):
                    bad.append(v)
    if not allv:
        return None
    f = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return f(good), f(bad), f(allv), len(good), len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="comma-separated eval shard dirs")
    ap.add_argument("--label", default="")
    ap.add_argument("--results", default="training/fold/results_with_trunks.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", default="",
                    help="pool csv whose interfaces define the good/bad grouping; "
                         "without it the grouping uses every interface results.csv "
                         "has, which is a SUPERSET and does not match the trainer")
    ap.add_argument("--restrict", default="",
                    help="csv of interfaces (pdb,chain1,chain2) that the DOCKQ TABLE is limited "
                         "to, not merely grouped by. --pool only chooses which interfaces define "
                         "the good/bad PAE split; the top-k table still aggregated every interface "
                         "in the eval dirs, so passing --pool a subset returned the SAME DockQ "
                         "numbers for every subset and the difference the subsets exist to show "
                         "was invisible. Use this to score one bin of the slice (the thin 1/5-4/5 "
                         "items against the 5/5 ones, say): shipped and ours are then both means "
                         "over the same restricted interface set, which is the only way the two "
                         "columns are comparable.")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--contestable", action="store_true",
                    help="keep only interfaces where the choice can matter: at least one "
                         "acceptable candidate AND at least one unacceptable one. An interface "
                         "whose every pose clears 0.23 scores 100%% for any ranker, and one where "
                         "none does scores 0 for any ranker, so both are constants added to every "
                         "arm's mean -- they dilute a real difference without being able to show "
                         "one. On val83 they are 42.5%% of the 141 interfaces (32.6%% all-good, "
                         "9.9%% all-bad), so the undiluted comparison is over the remaining 57.5%%.")
    a = ap.parse_args()
    dirs = [x for x in a.dirs.split(",") if x]
    by = load(dirs)
    if a.restrict:
        keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'
                for r in csv.DictReader(open(a.restrict))}
        _n0 = len(by)
        by = {i: c for i, c in by.items() if i in keep}
        print(f"  RESTRICTED to {len(by)} of {_n0} interface(s) "
              f"({len(keep)} listed in {os.path.basename(a.restrict)})")
    if a.contestable:
        _n0 = len(by)
        by = {i: c for i, c in by.items()
              if any(t[2] >= ACC for t in c) and any(t[2] < ACC for t in c)}
        print(f"  CONTESTABLE ONLY: {len(by)} of {_n0} interface(s) have both an acceptable and "
              f"an unacceptable candidate")
    if not by:
        print(f"  {a.label or dirs[0]}: no rows"); return
    ship = lambda t: t[1]
    mx = lambda t: t[0]
    mn = lambda t: t[3] if t[3] is not None else t[0]
    rows = []
    for nm, key in (("shipped", ship), ("ours max", mx), ("ours mean", mn)):
        a1, b1 = topk(by, key, 1)
        ak, bk = topk(by, key, a.k)
        rows.append((nm, b1, a1, bk, ak))
    hdr = f"{a.label or ''} (n={len(by)} interface(s))"
    if a.md:
        print(f"**{hdr}**\n")
        print(f"| | top-1 mean DockQ | top-1 frac acc | best@{a.k} | frac acc@{a.k} |")
        print("|---|---|---|---|---|")
        for nm, b1, a1, bk, ak in rows:
            print(f"| {nm} | {b1:.4f} | {a1*100:.1f}% | {bk:.4f} | {ak*100:.1f}% |")
    else:
        print(f"  {hdr}")
        print(f"    {'':11s} {'top1 DockQ':>11s} {'top1 acc':>9s} {'best@'+str(a.k):>9s} "
              f"{'acc@'+str(a.k):>9s}")
        for nm, b1, a1, bk, ak in rows:
            print(f"    {nm:11s} {b1:>11.4f} {a1*100:>8.1f}% {bk:>9.4f} {ak*100:>8.1f}%")
    pl = pae_split(dirs, a.results, a.pool)
    if pl:
        if a.md:
            print(f"\n| PAE loss | good ({pl[3]}) | bad ({pl[4]}) | all |")
            print("|---|---|---|---|")
            print(f"| ours | {pl[0]:.4f} | {pl[1]:.4f} | {pl[2]:.4f} |")
        else:
            print(f"    PAE loss  good {pl[0]:.4f} ({pl[3]})  bad {pl[1]:.4f} ({pl[4]})  "
                  f"all {pl[2]:.4f}")
    print()


if __name__ == "__main__":
    main()
