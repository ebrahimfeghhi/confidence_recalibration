#!/usr/bin/env python
"""Pool rows and roles for the interfaces the PREDICTIONS declare but the reference does not contain.

WHY THIS IS NEEDED. pose_interfaces.py defines an interface as a chain pair in contact within 5.0 A in
at least one pose. That recovers all 292 reference interfaces and declares 53 more. The 53 have no rows
in the pool CSV -- that file is keyed by reference interface -- so build_matrices cannot see them, yet
selection must treat them as interfaces or the whole point of a reference-free interface list is lost.

WHERE pair_iptm COMES FROM. The per-pose summary_confidence json carries a FULL chain_pair_iptm matrix,
one entry per ordered chain pair, whether or not the pair touches. Its indices are not labelled, so the
mapping to chain names is the cif's own chain order. That assumption was verified, not assumed: for 800
reference interfaces the json value at the cif index equals the pool CSV's pair_iptm to within 1e-3,
800/800. The pool CSV is produced by a different code path, so agreement cannot come from a shared bug.

dockq_gt IS LEFT EMPTY, deliberately. These pairs have no counterpart in the reference, so no ground
truth exists. Writing 0.0 would be a lie that silently reads as "a very bad interface" in every average
and every acceptable-rate denominator. Empty becomes NaN downstream, which compares false against the
0.23 threshold and is excluded from means.

    spurious_pool.py --ifaces anchor_merged/pose_interfaces.csv --out anchor_merged [--jobs 8]
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemmi  # noqa: E402
import metrics  # noqa: E402
import paths  # noqa: E402
from fmt import table  # noqa: E402

ANCHOR_DIR = str(paths.ANCHOR_DIR)


def one(task):
    """pair_iptm for each requested anchor-frame pair, for every pose of one complex."""
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
            continue                     # cannot trust positional order if the counts disagree
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
    A = ap.parse_args()

    spur = collections.defaultdict(list)
    for r in csv.DictReader(open(A.ifaces)):
        if r["have_gt"] == "0":
            spur[r["pdb"]].append((r["ab_chain"], r["ag_chain"]))
    amaps = metrics.load_chain_maps(f"{ANCHOR_DIR}/chain_maps_anchor.csv")
    xmaps = metrics.load_chain_maps()

    poses = collections.defaultdict(set)
    with open(str(paths.PXM2025_POOL_CSV)) as fh:
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
        for i, out in enumerate(p.imap_unordered(one, tasks), 1):
            rows += out
            if i % 5 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  rows {len(rows)}", flush=True)

    # pool rows in the same schema load_pool reads, with dockq left EMPTY (-> NaN)
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
