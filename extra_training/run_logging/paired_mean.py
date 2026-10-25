#!/usr/bin/env python
"""Mean paired (trained - shipped) DockQ over an epoch range, from a run's own shards.

    python -I run_logging/paired_mean.py <rid> [first_epoch] [last_epoch]

Prints one line: MEAN <value> POSITIVE <k>/<n>. Exists so a shell script can act on the number
without re-deriving it, and so every place that quotes it computes it the same way.
"""
import csv, glob, math, statistics as st, sys
from collections import defaultdict

rid = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 8
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 15
ms = []
for ep in range(lo, hi + 1):
    rows = []
    for f in glob.glob(f"training/runs/{rid}/train_eval_shards_ep{ep}/rows_b30_*.csv"):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        continue
    by = defaultdict(list)
    for r in rows:
        by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"])))
    dt = [max(ps, key=lambda t: t[0])[2] for ps in by.values()]
    ds = [max(ps, key=lambda t: t[1])[2] for ps in by.values()]
    d = [a - b for a, b in zip(dt, ds)]
    ms.append(sum(d) / len(d))
if not ms:
    print("MEAN nan POSITIVE 0/0")
    sys.exit(1)
print(f"MEAN {sum(ms)/len(ms):.6f} POSITIVE {sum(1 for m in ms if m > 0)}/{len(ms)}")
