#!/usr/bin/env python
"""Gate for the training trim: every trimmed structure must keep exactly one usable seed.

The validation trim has verify_val_manifest.py. This is the training equivalent, and it checks the
three things that would make the trim silently wrong:

  1. EXACTLY ONE unit remains per trimmed structure. Zero means the structure is gone and its
     cluster coverage with it -- the whole reason for trimming to 1 rather than deleting. More
     than one means the trim did not complete and the disk figure is wrong.
  2. THE SURVIVING UNIT OPENS and carries the keys the trunk readers use. A truncated npz passes
     an existence test and fails everything downstream.
  3. ITS DockQ ROWS STILL EXIST in results.csv, for all --nsample poses. The retained seed is only
     useful if its poses can still be joined to their labels, and results.csv is the join.

Exit 0 only when all three hold for every structure.
"""
import argparse, collections, csv, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--trimmed", required=True, help="one pdb per line: the structures trimmed to 1 seed")
ap.add_argument("--trunks", default="training/fold/trunks")
ap.add_argument("--results", default="training/fold/results.csv")
ap.add_argument("--nsample", type=int, default=5)
ap.add_argument("--show", type=int, default=6)
a = ap.parse_args()

pdbs = [l.strip() for l in open(a.trimmed) if l.strip()]
prob = collections.defaultdict(list)
kept = {}
for p in pdbs:
    d = os.path.join(ROOT, a.trunks, p)
    if not os.path.isdir(d):
        prob["one_unit"].append((p, "trunk dir gone")); continue
    us = sorted(f for f in os.listdir(d) if re.search(r"_(\d+)\.npz$", f))
    if len(us) != 1:
        prob["one_unit"].append((p, f"{len(us)} unit(s), expected 1")); continue
    kept[p] = int(re.search(r"_(\d+)\.npz$", us[0]).group(1))
    try:
        import numpy as np
        with np.load(os.path.join(d, us[0]), allow_pickle=True) as z:
            need = ("ci_coordinate", "ci_feat_asym_id", "ci_feat_atom_to_token_idx")
            miss = [k for k in need if k not in z.files]
            if miss:
                prob["unit_opens"].append((p, f"missing {miss}"))
    except Exception as e:
        prob["unit_opens"].append((p, f"{type(e).__name__}"))

rows = collections.defaultdict(lambda: collections.defaultdict(int))
for r in csv.DictReader(open(os.path.join(ROOT, a.results))):
    i = (r.get("iface") or "").strip()
    if not i:
        continue
    try:
        rows[i.split("_")[0]][int(r["seed"])] += 1
    except (ValueError, TypeError, KeyError):
        continue
for p, s in kept.items():
    n = rows.get(p, {}).get(s, 0)
    if n < a.nsample:
        prob["dockq_rows"].append((p, f"seed {s} has {n} row(s), expected >= {a.nsample}"))

print(f"training trim gate over {len(pdbs)} structure(s)")
for c in ("one_unit", "unit_opens", "dockq_rows"):
    n = len(prob[c])
    print(f"  {'PASS' if not n else 'FAIL'}  {c:<12} {n} problem(s)")
    for k, why in prob[c][:a.show]:
        print(f"          {k}: {why}")
    if n > a.show:
        print(f"          ... and {n-a.show} more")
bad = sum(len(v) for v in prob.values())
print(f"\n{'GATE PASSED' if not bad else 'GATE FAILED'}: {bad} problem(s)")
sys.exit(0 if not bad else 1)
