#!/usr/bin/env python
"""Where the ranking loss stalls, and why that number is 0.042.

    python -I run_logging/zero_margin.py <rid> [<rid> ...]

THE LANDMARK. The scaled softplus is L = (2 tau / W) * sum_ij w_ij softplus(-m_ij / tau) with
W = sum_ij w_ij. Set every margin m_ij to zero and softplus(0) = ln 2 factors out of the sum while
the weights normalise away, leaving

    L(no ordering) = 2 * tau * ln 2 = 2 * 0.0305 * 0.6931 = 0.04228

independent of the pair set, the DockQ weights and the number of interfaces. So 0.04228 is not a
floor -- the floor is 0 -- it is the value the loss reads when the head assigns every pose of an
interface the SAME score, i.e. when it carries no ordering information at all. Below it the head
orders better than a constant; above it, worse.

THAT IS THE "0.042 WALL" the run notes describe. Arms do not stall there because the objective
runs out of gradient; they stall there because that is the zero-information point, and getting
under it requires real positive margin, which is exactly what these arms cannot produce much of.

Measured, recomputed through rank_loss.group_loss so it agrees with rank_rho's xscanon to 5dp:

    shipped head   0.0803   ABOVE  -- its ipTM is anti-correlated with DockQ in the weighted sense
    ykhvzs ep12    0.0497   ABOVE
    8fkrsu ep8     0.0493   ABOVE
    4y3ve6 ep1     0.0489   ABOVE
    4y3ve6 ep10    0.0399   BELOW  <- the only arm that breaks through
    4y3ve6 ep20    0.0392   BELOW
    nzqhdf ep2     0.0362   BELOW  (full head; selection collapsed anyway)

4y3ve6 is also the only run in docs/overfit30_val10_runs.md that finishes above shipped on BOTH
DockQ and frac. Those two facts are the same fact.
"""
import csv, glob, math, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import rank_loss as RKL

ZERO = 2 * RKL.DEFAULT_TAU * math.log(2)


def epoch_loss(rid, ep, budget=30):
    rows = []
    for f in glob.glob(f"training/runs/{rid}/train_eval_shards_ep{ep}/rows_b{budget}_*.csv"):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        return None
    s, d = defaultdict(dict), defaultdict(dict)
    ss = defaultdict(dict)
    for r in rows:
        k = (r["seed"], r["sample"])
        s[r["iface"]][k] = float(r["iptm"])
        ss[r["iface"]][k] = float(r["stored_iptm"])
        d[r["iface"]][k] = float(r["dockq"])
    out = []
    for src in (s, ss):
        Ls = [RKL.group_loss(src[i], d[i], form="softplus", tau=RKL.DEFAULT_TAU, pairs="cross")
              for i in src]
        Ls = [L for L, n in Ls if n]
        out.append(sum(Ls) / len(Ls) if Ls else float("nan"))
    return out


print(f"tau = {RKL.DEFAULT_TAU};  ZERO-INFORMATION VALUE 2*tau*ln2 = {ZERO:.5f}")
print("below it the head orders better than a constant score; above it, worse\n")
print(f"{'run':9s} {'ep':>4s} {'trained':>9s} {'shipped':>9s}   verdict")
for rid in sys.argv[1:]:
    eps = sorted(int(x.rsplit("ep", 1)[1])
                 for x in glob.glob(f"training/runs/{rid}/train_eval_shards_ep*"))
    for ep in eps:
        r = epoch_loss(rid, ep)
        if not r:
            continue
        t, sh = r
        print(f"{rid:9s} {ep:4d} {t:9.5f} {sh:9.5f}   "
              f"{'BELOW (real ordering)' if t < ZERO else 'above (worse than constant)'}")
