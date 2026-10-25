#!/usr/bin/env python
"""SCALE-INVARIANT ordering accuracy: does the head put the better pose higher, sign only.

    python -I run_logging/order_accuracy.py <rid> [--budget 30]

WHY THIS EXISTS, AND WHAT IT CORRECTS. The ranking loss is scale-DEPENDENT: it weights each pair
by softplus(-m/tau), so a head can lower it two ways -- by fixing the ORDER, or by shrinking every
margin m toward 0, which drives softplus toward ln2 and the loss toward 2*tau*ln2 from above no
matter what the order is. The loss alone cannot tell those apart. This can: it counts the FRACTION
of pairs whose sign is right, and the |dDockQ|-weighted version, both invariant to any rescaling of
the scores.

MEASURED ON x5frz6 (within-seed objective, tau 0.0036), held-out val10:

    epoch    within correct / weighted     cross correct / weighted     within-seed spread
      1        0.5295 / 0.5014               0.5811 / 0.7284              0.00593
     11        0.5400 / 0.5574               0.5755 / 0.7239              0.00186
  SHIPPED      0.5476 / 0.6028               0.5765 / 0.7469              0.01294

So its ordering did NOT improve past the shipped head on either family -- it is slightly WORSE
than shipped on all four figures -- while its held-out loss fell from 0.01196 to 0.00563 against
shipped's 0.02145. That loss gap is a MARGIN gap, not an ordering gap: shipped is better ordered
but states it with 7x the margin, and a scale-dependent loss punishes a confident error far more
than a timid one. Quoting the loss as evidence of better ordering is wrong, and this module exists
so that mistake is one command away from being caught.

IT ALSO CORRECTS THE READING OF 2*tau*ln2. "Above the line" does NOT mean the head is
anti-correlated with DockQ: shipped sits at 0.0803, nearly double the 0.04228 line, while ordering
57.65% of cross-seed pairs correctly and 74.69% by DockQ weight. The line is crossed from above by
a head whose WEIGHTED PENALTY beats silence, and softplus is asymmetric -- a correct pair saturates
toward 0 while a wrong one costs ~m/tau without bound -- so a minority of confident errors can hold
the loss above the line even when most pairs are right.
"""
import argparse, csv, glob
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("rid")
ap.add_argument("--budget", type=int, default=30)
a = ap.parse_args()


def load(ep):
    rows = []
    for f in glob.glob(f"training/runs/{a.rid}/train_eval_shards_ep{ep}/rows_b{a.budget}_*.csv"):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def order(rows, col, family):
    byif = defaultdict(list)
    for r in rows:
        byif[r["iface"]].append((r["seed"], float(r[col]), float(r["dockq"])))
    ok = n = 0
    wok = wn = 0.0
    for v in byif.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                p, q = v[i], v[j]
                same = p[0] == q[0]
                if (family == "within") != same:
                    continue
                if abs(p[2] - q[2]) <= 1e-6:
                    continue
                n += 1
                w = abs(p[2] - q[2])
                wn += w
                if p[1] == q[1]:
                    continue
                g = (p[1] > q[1]) == (p[2] > q[2])
                ok += g
                wok += w * g
    return (ok / n if n else float("nan"), wok / wn if wn else float("nan"), n)


eps = sorted(int(d.rsplit("ep", 1)[1])
             for d in glob.glob(f"training/runs/{a.rid}/train_eval_shards_ep*"))
print(f"{a.rid}: scale-invariant ordering accuracy, held out (0.5 = coin flip)")
print(f"{'ep':>4s} | {'within':>8s} {'within wtd':>11s} | {'cross':>8s} {'cross wtd':>10s}")
for ep in eps:
    rows = load(ep)
    if not rows:
        continue
    w = order(rows, "iptm", "within")
    c = order(rows, "iptm", "cross")
    print(f"{ep:4d} | {w[0]:8.4f} {w[1]:11.4f} | {c[0]:8.4f} {c[1]:10.4f}")
rows = load(eps[-1])
if rows:
    w = order(rows, "stored_iptm", "within")
    c = order(rows, "stored_iptm", "cross")
    print(f"{'SHIP':>4s} | {w[0]:8.4f} {w[1]:11.4f} | {c[0]:8.4f} {c[1]:10.4f}")
    print(f"\npairs: within {w[2]:,}  cross {c[2]:,}")
