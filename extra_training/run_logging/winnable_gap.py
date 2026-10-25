#!/usr/bin/env python
"""How far the fraction-above-0.23 metric actually needs the scores to move, per epoch.

    python -I run_logging/winnable_gap.py <rid> [--thresh 0.23] [--budget 30]

WHY. frac can only change on interfaces that are WINNABLE BUT LOST: shipped's argmax is below the
threshold while some acceptable pose exists. On valfull that is 11 of 92. For each one this prints
the ipTM gap between the current argmax and the best-scoring acceptable pose -- the distance the
score function has to close for that interface to flip. The target set is fixed once from SHIPPED
scores so the rows are comparable across epochs.

WHAT IT SETTLED. The gap is the quantity the metric depends on, and it is ~200x the top1-top2
margin: median 0.1745 ipTM on valfull, against 0.00087 between the top two poses. So the deciding
comparisons for THIS metric are wide pairs, not near-ties, and tau 0.1429 is matched to them. The
near-tie argument in e808f846/0b224d80 explains why the argmax churns harmlessly on ~55 interfaces
an epoch; it does NOT explain a flat frac, and should not be quoted as doing so.

It also showed 8fkrsu is not failing to learn. It closed the median gap 0.1745 -> 0.053 in two
epochs, 70% of the way, on 9-11 of 11 interfaces -- and then stalled there for five epochs without
crossing on any. A stall at 59% gradient weight (m/tau = -0.35 for a still-inverted pair) is not
the loss being satisfied, so the binding constraint is elsewhere: the L2-SP anchor, or the capacity
of the 8 trainable tensors.
"""
import argparse, csv, glob, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("rid")
ap.add_argument("--thresh", type=float, default=0.23)
ap.add_argument("--budget", type=int, default=30)
a = ap.parse_args()
T = a.thresh


def load(ep):
    rows = []
    for f in glob.glob(f"training/runs/{a.rid}/train_eval_shards_ep{ep}/rows_b{a.budget}_*.csv"):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        return None
    by = defaultdict(list)
    for r in rows:
        by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"])))
    return by


base = load(1)
if base is None:
    raise SystemExit(f"no shards for {a.rid}")
targets = {}
for ifc, ps in base.items():
    s = sorted(ps, key=lambda p: -p[1])
    if s[0][2] >= T or not any(p[2] >= T for p in s):
        continue
    targets[ifc] = max((p for p in s if p[2] >= T), key=lambda p: p[1])[1]
    targets[ifc] = s[0][1] - targets[ifc]

print(f"{a.rid}: {len(targets)} winnable-but-lost interface(s) of {len(base)}")
print(f"shipped median gap to close: {st.median(list(targets.values())):.4f} ipTM")
print("epoch  median gap  mean gap  closed vs shipped  crossed")
for ep in range(1, 200):
    by = load(ep)
    if not by:
        continue
    gaps = []
    for ifc, g0 in targets.items():
        ps = by.get(ifc)
        if not ps:
            continue
        cand = [p for p in ps if p[2] >= T]
        if not cand:
            continue
        gaps.append(max(ps, key=lambda p: p[0])[0] - max(cand, key=lambda p: p[0])[0])
    cl = sum(1 for g, g0 in zip(gaps, targets.values()) if g < g0)
    print(f"  {ep:<5d}  {st.median(gaps):.4f}      {sum(gaps)/len(gaps):.4f}    "
          f"{cl}/{len(gaps)}              {sum(1 for g in gaps if g < 0)}")
