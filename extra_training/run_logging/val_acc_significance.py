#!/usr/bin/env python
"""IS THE frac-ACCEPTABLE MOVE SIGNIFICANT? McNemar on the accept indicator, per epoch.

    val_acc_significance.py <rid> [<rid2> ...]

WHY A DIFFERENT TEST. The DockQ columns are continuous and paired, so a paired t is right for
them: at n=139, df=138, two-sided p<0.05 needs |t| >= 1.977, which is why `val_two_tables.py`
prints the t. frac acceptable is NOT continuous -- it is the mean of a binary indicator
(top-1 DockQ >= 0.23) over the same 139 interfaces, and 73.4% -> 74.8% is TWO interfaces
changing. A paired t on a 0/1 variable is the wrong instrument; the paired test for a binary
outcome is McNemar, which looks ONLY at the interfaces where the arm and shipped disagree and
asks whether the gains outnumber the losses by more than a coin flip would.

THE DISCORDANT COUNTS ARE THE WHOLE RESULT. b = shipped unacceptable -> arm acceptable,
c = the reverse. Concordant interfaces carry no information about the difference, so an arm can
move frac by a point while only 3 or 4 interfaces are in play -- and with b+c that small, the
exact binomial p cannot go below 0.125 no matter which way they fall (b=3, c=0 gives 0.25
two-sided; b=4, c=0 gives 0.125). That is a property of the pool size, not of the arm, and it
is why the frac column must never be quoted on its own.
"""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_two_tables import load, epochs_of, ACC
import csv, glob

RESTRICT = "training/valfull83/pool_noleak.csv"


def exact_two_sided(b, c):
    """Exact binomial on the b+c discordant pairs at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def main():
    rids = sys.argv[1:]
    if not rids:
        sys.exit(__doc__)
    keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'
            for r in csv.DictReader(open(RESTRICT))} if os.path.exists(RESTRICT) else None

    eps = {}
    for rid in rids:
        for e in epochs_of(rid):
            d = load(rid, e)
            if d:
                eps[e] = d
    if not eps:
        sys.exit("no val row csvs found")
    keys = sorted(set.intersection(*[set(d) for d in eps.values()]))
    if keep is not None:
        keys = [k for k in keys if k in keep]
    n = len(keys)

    # THE t THRESHOLD IS A PROPERTY OF n, NOT OF THE ARM -- print it so the DockQ deltas can be
    # read against it instead of against an eyeballed "looks big".
    df = n - 1
    print(f"n = {n} interfaces, df = {df}")
    print(f"paired-t threshold for the DockQ columns: |t| >= 1.977 (two-sided p<0.05), "
          f"|t| >= 1.656 (p<0.10)\n")

    def acc_vec(d, idx):
        return [1 if max(d[k], key=lambda t: t[idx])[3] >= ACC else 0 for k in keys]

    base = eps[min(eps)]
    for label, idx in (("MAX-POOLED", 1), ("MEAN-POOLED", 2)):
        print(f"{label} -- accept indicator, top-1 DockQ >= {ACC}")
        print("| epoch | acc | shipped acc | gained | lost | discordant | McNemar p |")
        print("|---|---|---|---|---|---|---|")
        s = acc_vec(base, 0)
        for e in sorted(eps):
            a = acc_vec(eps[e], idx)
            b = sum(1 for x, y in zip(a, s) if x == 1 and y == 0)
            c = sum(1 for x, y in zip(a, s) if x == 0 and y == 1)
            p = exact_two_sided(b, c)
            print(f"| {e} | {100*sum(a)/n:.1f}% | {100*sum(s)/n:.1f}% | {b} | {c} | {b+c} | "
                  f"{p:.3f} |")
        print()


if __name__ == "__main__":
    main()
