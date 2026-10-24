"""How large a within-seed multiplier buys how much of the gradient, at tau = 0.0305.

Sets the knob BY MEASUREMENT rather than by guess. The quantity is the pairwise score-space
gradient mass, the same one measure_tau.py reports:

    per-pair contribution to dL/d(s_i - s_j)  =  2 |dDockQ| sigmoid(-m/tau) / W

normalised per interface by that interface's own W, so every interface counts equally. A
multiplier L on within-seed pairs scales both their numerator and their share of W.

WHY A MULTIPLIER IS A BETTER KNOB THAN A SECOND TAU: it is monotone and its units are the thing
we actually want to control (share of gradient mass), so it can be set to hit a target. A second
tau changes the SHAPE of the weighting, and its effect on mass share is not monotone.
"""
import argparse, csv, math, os, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_tau as MT

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", default=["training/fold/results.csv"])
ap.add_argument("--restrict-to", default=None)
ap.add_argument("--tau", type=float, default=0.0305)
a = ap.parse_args()

keep = None
if a.restrict_to:
    keep = {(r.get("pdb") or next(iter(r.values()))).lower()
            for r in csv.DictReader(open(a.restrict_to))}
data = MT.load(a.results, keep)

# (gap, margin, is_within) per interface
per_iface = []
for ifc, poses in data.items():
    pr = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            si, ipi, dqi = poses[i]
            sj, ipj, dqj = poses[j]
            dd = dqi - dqj
            if abs(dd) < MT.DQ_TOL:
                continue
            pr.append((abs(dqi - dqj), (1.0 if dd > 0 else -1.0) * (ipi - ipj), si == sj))
    if len(pr) >= 10:
        per_iface.append(pr)

print(f"{len(per_iface)} interface(s), tau {a.tau}"
      + (f", restricted to {a.restrict_to}" if keep else ", FULL TRAINING SET"))
nw = sum(1 for pr in per_iface for g, m, w in pr if w)
nt = sum(len(pr) for pr in per_iface)
print(f"within-seed pairs: {nw:,} of {nt:,} = {100*nw/nt:.1f}% BY COUNT\n")
print(f"  {'multiplier L':>12} {'within share of gradient mass':>31} {'mass on inversions':>20}")
for L in (1, 2, 3, 5, 8, 12, 20, 40):
    wi = tot = inv = 0.0
    for pr in per_iface:
        W = sum(g * (L if w else 1.0) for g, m, w in pr)
        for g, m, w in pr:
            c = 2.0 * g * (L if w else 1.0) * MT.sigmoid(-m / a.tau) / W
            tot += c
            if w:
                wi += c
            if m <= 0:
                inv += c
    print(f"  {L:>12} {100*wi/tot:>30.1f}% {100*inv/tot:>19.1f}%")
