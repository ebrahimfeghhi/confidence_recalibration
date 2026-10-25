#!/usr/bin/env python
"""Why a falling ranking loss can buy no fraction-above-0.23, measured from a run's own shards.

    python -I run_logging/crossing_audit.py <rid> [--thresh 0.23] [--budget 30]

WHAT IT SEPARATES. The reported criterion is, per interface, 1.0 if the SELECTED pose clears the
threshold. Selection is argmax of chain_pair_iptm over that interface's poses. So the criterion
moves only when the argmax crosses the threshold -- not when the argmax merely changes. This
counts those two events apart:

  contested   interfaces holding poses on BOTH sides of the threshold. Only these can ever move;
              the rest are decided whatever the head does.
  MOVED       trained argmax is a different pose than shipped argmax.
  crossed     of those, the ones where the move also changed which side of the threshold the
              selected pose sits on, split by direction.

WHY IT WAS WRITTEN. 8fkrsu drove held-out ranking loss on the canonical cross-seed objective 28%
below the shipped head over four epochs while frac-above-0.23 sat at or under shipped. This says
where that work went: the argmax moves on roughly half of all interfaces every epoch and almost
none of those moves cross the threshold, so the churn is same-side reordering the criterion cannot
see. A straddle-filtered training objective does not prevent this, because the filter constrains
which PAIRS carry gradient, not which pose ends up winning a ~30-way argmax.

The crossing count is a check on itself: shipped_frac*n + net_crossings must reproduce the frac
this run reports in rank_rho_b30_v*.csv, and it does for every epoch of 8fkrsu.
"""
import argparse, csv, glob
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("rid")
ap.add_argument("--thresh", type=float, default=0.23)
ap.add_argument("--budget", type=int, default=30)
a = ap.parse_args()
T = a.thresh

for ep in range(1, 200):
    rows = []
    for f in glob.glob(f"training/runs/{a.rid}/train_eval_shards_ep{ep}/rows_b{a.budget}_*.csv"):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        continue
    by = defaultdict(list)
    for r in rows:
        by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"]),
                               r.get("pose") or r.get("unit") or ""))
    contested = moved = up = dn = 0
    for ps in by.values():
        if any(p[2] >= T for p in ps) and any(p[2] < T for p in ps):
            contested += 1
        t = max(ps, key=lambda p: p[0])
        s = max(ps, key=lambda p: p[1])
        if t[3] != s[3] or abs(t[2] - s[2]) > 1e-9:
            moved += 1
            up += (t[2] >= T > s[2])
            dn += (t[2] < T <= s[2])
    print(f"ep{ep}: ifaces {len(by)}  contested {contested}  selection MOVED {moved}  "
          f"crossed UP {up}  crossed DOWN {dn}  net {up-dn:+d}")
