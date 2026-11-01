#!/usr/bin/env python
"""Was a FULL-STRUCTURE fallback pose ever the one selection picked?

WHY A COUNT IS NOT AN ANSWER. --crop-predicted-margin scores a pose on the whole structure when
its crop is undefinable (no inter-chain contact). "1 pose of 7,950" sounds negligible, but
selection is an argmax per interface, so that one pose can be the winner -- and it is scored with
a different token count, hence a different ipTM d0, than everything it is being compared against.
This reads the sidecars phase_k_eval writes beside its rows and reports fallback WINNERS, for both
selectors, rather than fallback counts.

The sidecar is keyed (pdb, seed, SAMPLE id) -- the joined id that rows use. An earlier version
stored the cached pose index, which cannot be matched to a row without redoing the coordinate
join; that was the review finding this closes.

    python -I run_logging/fallback_selected.py <rows_dir> [<rows_dir> ...]
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

dirs = sys.argv[1:]
if not dirs:
    sys.exit(__doc__)

for d in dirs:
    fb = set()
    for f in glob.glob(os.path.join(d, "*.fallback.json")):
        for pdb, sd, k in json.load(open(f)):
            fb.add((str(pdb), str(sd), str(k)))
    rows = []
    for f in glob.glob(os.path.join(d, "rows_b*_s*.csv")):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        print(f"  {d}: no rows")
        continue
    by = defaultdict(list)
    for r in rows:
        by[r["iface"]].append(r)
    print(f"  {d}")
    print(f"    {len(fb)} fallback pose(s) recorded, {len(by)} interface(s) scored")
    if not fb:
        print("    no fallback poses, so none can have been selected")
        continue
    for pool, col in (("MEAN", "iptm_mean"), ("MAX", "iptm")):
        won = []
        for iface, ps in by.items():
            best = max(ps, key=lambda r: float(r[col]))
            # the sidecar is per STRUCTURE; an interface id carries the pdb as its prefix
            key = (iface.split("_")[0], best["seed"], best["sample"])
            if key in fb:
                won.append((iface, best["seed"], best["sample"], float(best["dockq"])))
        print(f"    {pool}-pooled: {len(won)} interface(s) selected a FULL-STRUCTURE pose"
              + (f" -> {won[:5]}" if won else ""))
