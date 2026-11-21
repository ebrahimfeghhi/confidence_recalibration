#!/usr/bin/env python
"""Build a --manifest-selection csv for the 83 validation structures.

WHY ONE HAS TO BE BUILT. Every existing val manifest stops at 53 structures
(train_val_splits/val_all_structures.csv, label_audit_validation.csv): they predate the valfull83
pool and the later wave that took it to 83. Nothing in the repo lists all 83 with their seed
depths.

WHY THAT IS CHEAP RATHER THAN A BLOCKER. phase_k_train reads only `pdb` from this file, plus
OPTIONAL `tokens` (for --epoch-order tokens) and OPTIONAL `seeds` (the per-structure depth). The
`instance` column that the training manifests carry is never read -- confirmed at
phase_k_train.py:3457-3466 -- so the missing instance mapping for the 30 newer structures does
not matter.

DECLARED, NOT DISCOVERED, which is the contract the trainer's own comment insists on: the seeds
come from training/valfull83/seed_manifest.json, the same file every held-out eval uses, so a
structure whose trunk is missing is an ERROR here rather than a quietly smaller run. The trunk
directory is cross-checked against it and any disagreement is a hard stop.
"""
import csv
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUNKS = os.path.join(ROOT, "validation/fold/trunks")
SEEDMAN = os.path.join(ROOT, "training/valfull83/seed_manifest.json")
POOL = os.path.join(ROOT, "training/valfull83/pool.csv")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "train_val_splits/val83_manifest.csv")

# THE MAPPING IS man["structures"][pdb]["seeds"], not a top-level dict. The manifest's top level
# is the trim's own record -- algo, k, seed_range, weights, digest -- and the per-structure entry
# holds both `seeds` and the per-interface coverage the trim was chosen for. Reading it as a flat
# pdb -> seeds dict reported all 83 structures as "absent from the seed manifest", which is the
# shape of a guard firing correctly on a wrong assumption rather than a missing file.
man = json.load(open(SEEDMAN))
_st = man.get("structures")
if not isinstance(_st, dict):
    sys.exit(f"STOP: {SEEDMAN} has no 'structures' map (top level: {sorted(man)[:8]})")
seeds_by = {p: v["seeds"] for p, v in _st.items() if isinstance(v, dict) and "seeds" in v}
if len(seeds_by) != len(_st):
    sys.exit(f"STOP: {len(_st) - len(seeds_by)} structure(s) in {SEEDMAN} carry no 'seeds' list")
_k = man.get("k")
_bad_k = {p: len(v) for p, v in seeds_by.items() if _k and len(v) != int(_k)}
if _bad_k:
    sys.exit(f"STOP: the manifest declares k={_k} but {len(_bad_k)} structure(s) list a different "
             f"number of seeds, e.g. {list(_bad_k.items())[:4]}. The depth is the contract.")

pool_pdbs = sorted({r["pdb"].strip().lower() for r in csv.DictReader(open(POOL))})
trunk_pdbs = sorted(d for d in os.listdir(TRUNKS) if os.path.isdir(os.path.join(TRUNKS, d)))

missing_trunk = [p for p in pool_pdbs if p not in set(trunk_pdbs)]
missing_seed = [p for p in pool_pdbs if p not in seeds_by]
if missing_trunk:
    sys.exit(f"STOP: {len(missing_trunk)} pool structure(s) have no trunk dir: {missing_trunk[:5]}")
if missing_seed:
    sys.exit(f"STOP: {len(missing_seed)} pool structure(s) absent from the seed manifest: "
             f"{missing_seed[:5]}")

def _tokens(pdb, seed):
    """Token count from the trunk's s_trunk HEADER, without reading the array.

    THE COLUMN IS OPTIONAL TO THE TRAINER AND MANDATORY TO THE GATE, which is the whole reason
    it is here. phase_k_train reads `tokens` through .get() and silently falls back to pdb order
    for --epoch-order tokens; phase_k_fingerprint_scan indexes r["tokens"] directly and dies with
    KeyError on a manifest without it. Emitting it satisfies both and restores size ordering.

    A zip member header is ~100 bytes, so this is 83 small reads rather than 83 trunk loads --
    s_trunk is (n_tok, 384) uint16, a few hundred MB across the set if actually read.
    """
    f = os.path.join(TRUNKS, pdb, f"free_{seed}.npz")
    with zipfile.ZipFile(f) as z, z.open("s_trunk.npy") as fh:
        np.lib.format.read_magic(fh)
        shape = np.lib.format.read_array_header_1_0(fh)[0]
    if len(shape) != 2:
        raise SystemExit(f"STOP: {f} s_trunk has shape {shape}, expected (n_tok, c)")
    return int(shape[0])


rows, short = [], []
for p in pool_pdbs:
    sd = sorted(int(s) for s in seeds_by[p])
    have = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(os.path.join(TRUNKS, p))
            if f.startswith("free_") and f.endswith(".npz")}
    absent = [s for s in sd if s not in have]
    if absent:
        short.append((p, len(absent)))
        continue
    # COMMA, NOT SPACE. _parse_seed_field (phase_k_train.py:3482-3505) splits on "," and ";"
    # only, never whitespace, then int()s each part -- so a space-joined field STOPS at startup
    # with "seed field '400000 400001 ...' is not an integer". A lo-hi range is not an option
    # either: the trim keeps a SPARSE set (29tj is 400000 400002 400004 ... 400092).
    rows.append({"pdb": p, "tokens": _tokens(p, sd[0]),
                 "seeds": ",".join(str(s) for s in sd)})
if short:
    sys.exit(f"STOP: {len(short)} structure(s) name a seed with no trunk, e.g. {short[:4]}. "
             f"The manifest is a contract; a missing trunk must be an error, not a smaller run.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pdb", "tokens", "seeds"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"{len(rows)} structure(s) -> {OUT}")
print(f"  seed depths: {sorted({len(r['seeds'].split(',')) for r in rows})}")
print(f"  total units: {sum(len(r['seeds'].split(',')) for r in rows)}")
_tk = sorted(r["tokens"] for r in rows)
print(f"  tokens: min {_tk[0]}, median {_tk[len(_tk) // 2]}, max {_tk[-1]}")
