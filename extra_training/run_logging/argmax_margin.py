#!/usr/bin/env python
"""The ipTM gap that actually decides selection, against the tau the loss was given.

    python -I run_logging/argmax_margin.py <rid> <epoch> [--tau 0.1429] [--budget 30]

WHY. tau is measured as a median |ipTM gap| over the pairs the loss is trained on. With
--rank-straddle those are pairs that straddle DockQ 0.23, which are WIDE -- 0.1429 on the full
fold. But selection is an argmax, and an argmax is decided by the gap between the top pose and the
runner-up, which is a different and far smaller quantity. This prints both, plus the softplus
gradient weight sigmoid(-m/tau) at the deciding margin, where 0.5 means the loss assigns a
correctly-ordered pair and its inversion the same pull and so expresses no preference between them.

This is the same units error rank_loss.py's own docstring records for the first softplus attempt
(median gap 0.0081 against tau=1 gave 0.495 vs 0.505, "a straight line through ln2"), arrived at
from the opposite direction: there tau was too large because it was never set, here because it was
measured on a deliberately wide subpopulation.
"""
import argparse, csv, glob, math, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("rid")
ap.add_argument("epoch", type=int)
ap.add_argument("--tau", type=float, default=0.1429)
ap.add_argument("--budget", type=int, default=30)
a = ap.parse_args()

rows = []
for f in glob.glob(f"training/runs/{a.rid}/train_eval_shards_ep{a.epoch}/rows_b{a.budget}_*.csv"):
    with open(f) as fh:
        rows += list(csv.DictReader(fh))
if not rows:
    raise SystemExit(f"no shards for {a.rid} epoch {a.epoch}")

by = defaultdict(list)
for r in rows:
    by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"])))

print(f"{a.rid} epoch {a.epoch}: {len(by)} interface(s), tau = {a.tau}")
for name, idx in (("trained", 0), ("shipped", 1)):
    g12, g15, bad, n = [], [], 0, 0
    for ps in by.values():
        s = sorted(ps, key=lambda p: -p[idx])
        if len(s) < 5:
            continue
        n += 1
        g12.append(s[0][idx] - s[1][idx])
        g15.append(s[0][idx] - s[4][idx])
        if max(s[:5], key=lambda p: p[2])[2] - s[0][2] > 1e-9:
            bad += 1
    m12, m15 = st.median(g12), st.median(g15)
    w = 1.0 / (1.0 + math.exp(m12 / a.tau))
    print(f"  {name}: median top1-top2 gap {m12:.5f}  top1-top5 {m15:.5f}")
    print(f"           m/tau at the deciding margin {m12/a.tau:.4f}, "
          f"softplus gradient weight {w:.4f}  (0.5 = no preference)")
    print(f"           top-1 is not the best DockQ of its own top-5 in {bad}/{n} "
          f"({100*bad/n:.0f}%)")
