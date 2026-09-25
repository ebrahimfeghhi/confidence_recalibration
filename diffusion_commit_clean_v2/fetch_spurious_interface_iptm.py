#!/usr/bin/env python
"""Get pair_iptm for the interfaces the predictions declare but the reference does not contain.

SUMMARY
    Runs after pose_interfaces.py. That step declares an interface for any chain pair in contact within
    5.0 A in at least one pose, which recovers all 292 reference interfaces and adds 53 more. Those 53
    have no rows in the pool csv -- it is keyed by reference interface -- so the next step cannot see
    them, and selection has to treat them as interfaces or the reference-free interface list is pointless.

    Where the ipTM comes from: each pose's summary_confidence json carries a full chain_pair_iptm matrix,
    one entry per ordered chain pair, whether or not the pair touches. Its rows are not labelled, so the
    mapping to chain names is the cif's own chain order. That was verified rather than assumed -- for 800
    reference interfaces the json value at the cif index matched the pool csv's pair_iptm to within 1e-3,
    800 of 800, and the pool csv comes from a different code path so agreement cannot be a shared bug.

    dockq is written EMPTY. These pairs have no counterpart in the reference, so no ground truth exists.
    Writing 0.0 would read as "a very bad interface" in every mean and every acceptable-rate denominator
    rather than "not evaluable". Empty becomes NaN downstream, which compares false against 0.23.

INPUTS
    --ifaces: pose_interfaces.csv from pose_interfaces.py. Required. Rows with have_gt=0 are the ones
              handled here.
    --out:    where the two csvs go. Required.
    --jobs:   worker processes reading confidence jsons in parallel. Default 8. CPU only.

    Files read:
    anchor maps: chain_maps_anchor.csv, to translate an anchor-frame pair into each pose's own chains
    pool csv:    which poses exist for each complex
    per-pose confidence jsons and cifs: the ipTM matrix, and the chain order it is indexed by

OUTPUTS
    <out>/spurious_pool.csv:  pool-shaped rows the next step can read alongside the real pool --
                              pdb, chain1, chain2, seed, sample, dockq (empty), pair_iptm
    <out>/spurious_roles.csv: which chains are antibody and antigen for each of those pairs

EXAMPLE
    fetch_spurious_interface_iptm.py --ifaces $ANCHOR_DIR/pose_interfaces.csv --out $ANCHOR_DIR --jobs 8

FUNCTIONS
    pair_iptm_rows(): one complex -> (pdb, ab, ag, seed, sample, pair_iptm) for every requested pair
    main():           collect the predicted-only pairs, read their ipTM in parallel, write both csvs
"""
import argparse
import collections
import csv
import json
import os
import sys

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemmi  # noqa: E402
import paths  # noqa: E402
from fmt import table  # noqa: E402

import pool_index  # noqa: E402

ANCHOR_DIR = str(paths.ANCHOR_DIR)


def pair_iptm_rows(task):
    """in: (pdb, [(seed, sample, pose path, chain map)], pairs).
    out: (pdb, ab, ag, seed, sample, pair_iptm) per pose per pair
    """
    pdb, items, pairs = task
    out = []
    for seed, samp, pose, amap in items:
        j = pose.replace(f"{pdb}_sample_{samp}.cif",
                         f"{pdb}_summary_confidence_sample_{samp}.json")
        if not os.path.exists(j) or not os.path.exists(pose):
            continue
        try:
            M = json.load(open(j))["chain_pair_iptm"]
            st = gemmi.read_structure(pose)
            st.setup_entities()
            idx = {c.name: i for i, c in enumerate(st[0])}
        except Exception:
            continue
        if len(M) != len(idx):
            continue        # chain counts disagree, so the positional order cannot be trusted
        for a, g in pairs:
            ma, mg = amap.get(a), amap.get(g)
            if ma in idx and mg in idx:
                out.append((pdb, a, g, seed, samp, float(M[idx[ma]][idx[mg]])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifaces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--pool", default=None,
                    help="pool csv naming which poses exist. Default the full pxm2025 pool; pass a "
                         "filtered one to restrict which seeds are read")
    A = ap.parse_args()

    spur = collections.defaultdict(list)
    for r in csv.DictReader(open(A.ifaces)):
        if r["have_gt"] == "0":
            spur[r["pdb"]].append((r["ab_chain"], r["ag_chain"]))
    amaps = pool_index.load_chain_maps(f"{ANCHOR_DIR}/chain_maps_anchor.csv")
    xmaps = pool_index.load_chain_maps()

    poses = collections.defaultdict(set)
    with open(A.pool or str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            if r["pdb"] in spur:
                poses[r["pdb"]].add((r["seed"], int(r["sample"])))

    tasks = []
    for pdb, pairs in sorted(spur.items()):
        items = []
        for s, x in sorted(poses.get(pdb, ())):
            m = amaps.get((pdb, s, x)) or xmaps.get((pdb, s, x))
            if m:
                items.append((s, x, paths.pool_pose(pdb, s, x), m))
        if items:
            tasks.append((pdb, items, sorted(pairs)))
    print(f"  complexes with spurious pairs: {len(tasks)}   "
          f"pairs: {sum(len(t[2]) for t in tasks)}", flush=True)

    import multiprocessing as mp
    rows = []
    with mp.Pool(A.jobs) as p:
        for i, out in enumerate(p.imap_unordered(pair_iptm_rows, tasks), 1):
            rows += out
            if i % 5 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  rows {len(rows)}", flush=True)

    os.makedirs(A.out, exist_ok=True)
    with open(f"{A.out}/spurious_pool.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain1", "chain2", "seed", "sample", "dockq", "pair_iptm"])
        for pdb, a, g, s, x, v in rows:
            w.writerow([pdb, a, g, s, x, "", f"{v:.6f}"])
    with open(f"{A.out}/spurious_roles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain_1", "chain_2", "ab_chain", "ag_chain"])
        for pdb, pairs in sorted(spur.items()):
            for a, g in sorted(pairs):
                w.writerow([pdb, a, g, a, g])
    n = sum(len(v) for v in spur.values())
    print()
    print(table(["", "count"], [["spurious interfaces", n], ["pose rows written", len(rows)]],
                align="l r"))
    print(f"  wrote {A.out}/spurious_pool.csv and {A.out}/spurious_roles.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
