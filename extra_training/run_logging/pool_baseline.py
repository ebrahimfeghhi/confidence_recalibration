#!/usr/bin/env python
"""Shipped-head frac-above-0.23 and mean DockQ for any pool, from stored labels. No GPU.

    python -I run_logging/pool_baseline.py [--labels training/fold/results.csv] [--thresh 0.23]
                                           [--structures <csv>] [--pool <csv>] [--manifest-skip a,b]

Selection is argmax of the STORED ipTM over an interface's poses, which is what the shipped head
does, so the whole baseline is a groupby over the label file. The trained head's version is not
free -- it needs a forward pass per pose -- but the shipped one costs nothing and is the reference
every arm is scored against.

--structures takes the interfaces of every pdb in that manifest (what --manifest-selection trains
on). --pool takes exactly the pdb,chain1,chain2 rows listed (what --pae-scored-pairs masks). They
are NOT the same set and mixing them up silently changes the answer.

WHAT IT FOUND, 2026-10-25. The two pools this project trains and scores on are not comparable
populations, and the gap is in the metric itself:

    whole training fold   1,028 interfaces   shipped DockQ 0.4074  frac 0.5175
    overfit30 training       78 interfaces   shipped DockQ 0.0417  frac 0.0000
    (overfit30 oracle)                       DockQ 0.5049  frac 0.8718

overfit30 is a pool on which the shipped head selects an unacceptable pose for EVERY ONE of its 78
interfaces, while 87% of them hold an acceptable pose. 68 of 78 are winnable-but-lost. val10, the
pool those arms are scored on, has shipped frac 0.5556. So an overfit30 arm is trained where the
criterion is 0 and judged where it is 0.56.
"""
import argparse, csv
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="training/fold/results.csv")
ap.add_argument("--thresh", type=float, default=0.23)
ap.add_argument("--structures", default="")
ap.add_argument("--pool", default="")
ap.add_argument("--manifest-skip", default="")
a = ap.parse_args()
T = a.thresh
skips = {s for s in a.manifest_skip.split(",") if s}

by = defaultdict(list)
with open(a.labels) as fh:
    for r in csv.DictReader(fh):
        by[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))

if a.structures:
    with open(a.structures) as fh:
        pdbs = {r["pdb"].lower() for r in csv.DictReader(fh)} - skips
    keys = [k for k in by if k.split("_")[0].lower() in pdbs]
    label = f"{a.structures} ({len(pdbs)} pdbs)"
elif a.pool:
    with open(a.pool) as fh:
        keys = [f"{r['pdb']}_{r['chain1']}{r['chain2']}" for r in csv.DictReader(fh)
                if r["pdb"] not in skips]
    label = a.pool
else:
    keys, label = list(by), f"ALL of {a.labels}"

sub = {k: by[k] for k in keys if k in by}
if not sub:
    raise SystemExit(f"no interfaces matched for {label}")
sd = [max(v, key=lambda p: p[0])[1] for v in sub.values()]
od = [max(v, key=lambda p: p[1])[1] for v in sub.values()]
cont = sum(1 for v in sub.values()
           if any(p[1] >= T for p in v) and any(p[1] < T for p in v))
win = sum(1 for v in sub.values()
          if max(v, key=lambda p: p[0])[1] < T and any(p[1] >= T for p in v))
poses = [len(v) for v in sub.values()]
print(f"{label}: {len(sub)} interface(s), {min(poses)}-{max(poses)} poses each")
print(f"  SHIPPED  mean DockQ {sum(sd)/len(sd):.4f}   frac >= {T} {sum(1 for d in sd if d>=T)/len(sd):.4f}")
print(f"  ORACLE   mean DockQ {sum(od)/len(od):.4f}   frac >= {T} {sum(1 for d in od if d>=T)/len(od):.4f}")
print(f"  headroom {sum(od)/len(od)-sum(sd)/len(sd):.4f} DockQ")
print(f"  contested {cont}/{len(sub)}   winnable-but-lost {win}/{len(sub)}")
