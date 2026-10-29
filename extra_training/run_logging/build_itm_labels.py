#!/usr/bin/env python
"""Turn itm_ceiling.py's CSVs into a per-structure label cache the trainer can read.

WHY A SEPARATE STEP AND NOT A RECOMPUTE. itm_ceiling.py already computed the true iTM for every
one of the 36,900 training poses (738 winnable interfaces x 10 seeds x 5 samples) at a cost of
10.8 min on 9 CPU workers. Recomputing it inside the trainer would pay that every epoch for a
quantity that cannot change -- the true error of a fixed pose against a fixed structure. So it is
computed once, cached, and read.

THE KEY IS (seed, sample, IFACE), NOT the asym pair -- and that is a correction. The first version
keyed by (seed, sample, asym_x, asym_y) using interface_chain_pairs' STATIC mapping, while the
value itself had been computed on that pose's OST-resolved pair. On the 52 structures where the
two disagree that silently paired predicted ipTM for one chain pair with true iTM for another:
7stf's cache said (3,1) while its value came from (3,0). The coverage check could not see it
because the keys matched. Caught in review.

The interface id (e.g. 9gvn_BF, in REFERENCE chain letters) is the one identifier that is stable
across poses -- the asym pair is not, because OST resolves it per pose. So the label is keyed by
it, and the trainer resolves the asym pair per pose for BOTH its prediction and its lookup.

BOTH POOLINGS ARE STORED. The arm being built trains on the MEAN, but the max is the shipped
readout and the mean-vs-max question is not settled on a trained head (measured: mean wins at the
oracle by +0.0140 paired t=4.06, and LOSES on the current head by -0.0130 t=-2.68). Storing both
costs nothing and lets the target be switched without rebuilding.

OUTPUT: <fold>/itm_labels/<pdb>.npz with parallel int/float arrays, plus a manifest.json carrying
the row count and a digest. The trainer refuses to start if a unit's label file is missing rather
than training on a silently empty target -- that is the row-weight bool-cast failure in another
costume.

Usage:  python -I run_logging/build_itm_labels.py [--out <fold>/itm_labels]
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import provenance as PROV                                   # noqa: E402
from phase_k_eval import interface_chain_pairs              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ceiling", default="training/runs/gt_iptm_ceiling")
ap.add_argument("--pool", default="training/full_train/pool_winnable.csv")
ap.add_argument("--roles", default="training/fold/chain_roles.csv")
ap.add_argument("--out", default="training/fold/itm_labels")
a = ap.parse_args()

# IFACE -> PDB ONLY. The asym pair deliberately does NOT come from here: it varies per pose and
# the trainer resolves it itself. All this needs is which structure's file a row belongs in, and
# that the interface is one the pool actually names.
pairs_by_pdb, unres = interface_chain_pairs(os.path.join(ROOT, a.pool),
                                            os.path.join(ROOT, a.roles))
where = {}
for pdb, lst in pairs_by_pdb.items():
    for iface, _ax, _ay in lst:
        where[iface] = pdb

# REFUSE A STATICALLY-MAPPED CEILING. The labels must come from the per-pose OST join, the one
# phase_k_eval scores with. The first attempt at this used the static chain_roles join and was
# wrong on 3,154 of 36,900 rows (8.55%) across 52 structures -- and the "fix" run silently used
# the static join anyway, because the parent never forwarded --ost-mapping to its workers. That
# was caught only because the rebuilt labels came out bit-identical. The ceiling's own provenance
# now records the mapping, so this can be checked rather than remembered.
# AN UNRECORDED MAPPING IS ALSO REFUSED, not just a recorded static one. The first version of
# this guard only fired when chain_mapping said something non-OST -- so the original static
# ceiling, written before that field existed, sailed straight through it. A mapping that cannot be
# confirmed is not a mapping that can be trusted, and the cost of being wrong here is 8.55% of
# labels pointing at the wrong chain pair.
_cp = os.path.join(ROOT, a.ceiling, "provenance.json")
_cm = ((json.load(open(_cp)) or {}).get("chain_mapping", "")
       if os.path.exists(_cp) else "")
if "OST" not in _cm:
    sys.exit(f"STOP: {a.ceiling} reports chain_mapping "
             f"{_cm!r} -- training labels must come from the PER-POSE OST join the evaluator "
             f"scores with, and an unrecorded mapping cannot be confirmed to be one. Rerun "
             f"run_logging/itm_ceiling.py with --ost-mapping (it records the mapping in its own "
             f"provenance.json).")
print(f"  ceiling mapping: {_cm}")

rows = collections.defaultdict(list)
n_in = n_skip = 0
for f in sorted(glob.glob(os.path.join(ROOT, a.ceiling, "gt_iptm_s*.csv"))):
    for r in csv.DictReader(open(f)):
        n_in += 1
        pdb = where.get(r["iface"])
        if pdb is None:
            n_skip += 1
            continue
        rows[pdb].append((r["iface"], int(r["seed"]), int(r["sample"]),
                          float(r["gt_iptm_mean"]), float(r["gt_iptm_max"])))
if not rows:
    sys.exit(f"STOP: no rows joined from {a.ceiling}; is the ceiling pass done?")

OUT = os.path.join(ROOT, a.out)
os.makedirs(OUT, exist_ok=True)
agg = hashlib.sha256()
n_out = 0
for pdb in sorted(rows):
    v = sorted(rows[pdb])
    p = os.path.join(OUT, f"{pdb}.npz")
    np.savez_compressed(
        p + ".tmp",
        iface=np.array([x[0] for x in v]),
        seed=np.array([x[1] for x in v], dtype=np.int64),
        sample=np.array([x[2] for x in v], dtype=np.int64),
        g_mean=np.array([x[3] for x in v], dtype=np.float32),
        g_max=np.array([x[4] for x in v], dtype=np.float32))
    os.replace(p + ".tmp.npz", p)
    agg.update(pdb.encode()); agg.update(str(len(v)).encode())
    n_out += len(v)

man = {"structures": len(rows), "rows": n_out, "rows_read": n_in, "rows_unmapped": n_skip,
       "digest": agg.hexdigest(), "pool": a.pool, "roles": a.roles, "ceiling": a.ceiling,
       "keys": "(seed, sample, iface) -> g_mean, g_max",
       "key_note": "keyed by the REFERENCE interface id, not an asym pair: the asym pair is "
                   "resolved per pose by OST and is not stable across poses",
       "unresolved_pool_pairs": unres[:20]}
with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(man, fh, indent=1)
print(f"  {len(rows)} structure(s), {n_out} label row(s) -> {a.out}")
print(f"  read {n_in}, unmapped {n_skip}" + ("  <-- CHECK THIS" if n_skip else ""))
if unres:
    print(f"  NOTE {len(unres)} pool pair(s) unresolved in chain_roles: {unres[:5]}")
per = collections.Counter(len(v) for v in rows.values())
print(f"  rows per structure: {dict(sorted(per.items()))}")
PROV.write(OUT, ROOT, inputs=[a.pool, a.roles, os.path.join(a.ceiling, "gt_iptm_s*.csv")],
           outputs_glob="*.npz", extra={"manifest": man})
