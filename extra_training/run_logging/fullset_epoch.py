#!/usr/bin/env python
"""The newest epoch the FULL-SET arm has finished, across its segments. Prints a number.

Written because a grep over training/runs/*/run.log finds every run on the box -- the first
version of the plot watcher reported epoch 200 from an unrelated single-item probe and then never
fired again, because no full-set epoch would exceed it for months.
"""
import glob
import json
import os
import re

best = 0
for a in glob.glob("training/runs/*/args.json"):
    try:
        d = json.load(open(a))
    except (ValueError, OSError):
        continue
    d = d.get("args", d)
    if not (d.get("manifest_selection") or "").endswith("fullset/structures.csv"):
        continue
    p = os.path.join(os.path.dirname(a), "run.log")
    if not os.path.exists(p):
        continue
    for line in open(p, errors="ignore"):
        m = re.search(r"epoch (\d+) PAE BY POSE GROUP", line)
        if m:
            best = max(best, int(m.group(1)))
print(best)
