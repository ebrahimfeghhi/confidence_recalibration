#!/usr/bin/env python
"""Name every pose's chains after one chosen pose, so "interface A-B" means the same thing in all of them.

SUMMARY
    Protenix names duplicate chains arbitrarily: the antigen copy it calls B in one pose can be C in
    the next. Later steps sort poses per interface, so they need one consistent naming across the pool.
    This picks one pose per complex as the reference and expresses every other pose in its terms. No
    native structure is involved.

    WHEN IT DOES ANYTHING -- only when a complex has more than one copy of an entity. With one antibody
    and one antigen there is a single possible assignment (the anchor's A can only match the pose's A,
    the only chain with that sequence), so the mapping is the identity. Of 131 benchmark complexes, 52
    have no duplication, 44 duplicate the antibody, 35 duplicate both; 9jo7 has 5 of each. Over the 73
    multi-interface complexes, 24 never produce a non-identity mapping and 42% of poses get a chain
    renamed.

    WHY THE LABEL MATTERS when the copies are identical -- they share a sequence but not a position, and
    in a given pose the antibody touches one and not the other:

        pose 1: antibody A binds the copy labelled B  ->  (A,B) is a real interface, (A,C) is nothing
        pose 2: antibody A binds the copy labelled C  ->  (A,C) is a real interface, (A,B) is nothing

    Ranking by "ipTM at A-B" then compares pose 1's docking site against a pair of chains in pose 2 that
    never touch, and pose-pose DockQ at A-B makes two identically-docked poses look unrelated.

    ONE ANCHOR PER COMPLEX, not per interface. The mapping pairs each anchor chain with exactly one pose
    chain and no two share a partner, so the anchor's (A,B) and (A,C) necessarily land on different chain
    pairs of the pose. Two independent per-interface mappings could both pick the pose's (A,B) pair,
    putting one physical interface into two candidate lists.

INPUTS
    --out-dir: where to write the csvs and the manifest. Default paths.ANCHOR_DIR.
    --pool:    pool csv listing the poses and their pair_iptm. Default paths.PXM2025_POOL_CSV. This is
               what identifies the dataset: there is no separate foldbench pool index today, so the
               pool file is the only honest record of what was run.
    --pdbs:    comma-separated subset of complexes. Default: every complex in the pool csv.
    --jobs:    how many worker processes map poses in parallel. Default 8. CPU only; each process reads
               one pose cif at a time, so this is bounded by disk and cores, not by GPU.

    Files read, none of them arguments:
    pool csv:      paths.PXM2025_POOL_CSV -- pair_iptm per pose per interface, used to pick the anchor
    pose cifs:     via paths.pool_pose(), read by the chain mapper
    crystal files: the reference roles and mappings, used ONLY to decide which of the anchor's chain
                   pairs is called "8w5d_AB", so the output lines up against the crystal-framed run

OUTPUTS
    <out-dir>/chain_maps_anchor.csv:  per pose, {anchor_chain: pose_chain}
    <out-dir>/chain_roles_anchor.csv: per interface, which ANCHOR chains are the antibody and antigen
    <out-dir>/anchor_manifest.json:   what this run covered -- the pool csv, the complexes, the anchor
                                      pose chosen for each, and how many poses were mapped or failed.
                                      Without it, a later step reading these csvs cannot tell which
                                      complexes they describe.

    The two csvs are drop-in for build_matrices.py --maps/--roles.

EXAMPLE
    anchor_relabel.py --out-dir anchor_merged --jobs 8
    anchor_relabel.py --out-dir /tmp/try --pdbs 8w5d,8w5f --jobs 4

FUNCTIONS
    pool_rows():     pool csv -> {pdb: {(seed, sample): {interface: pair_iptm}}}
    pick_anchor():   the poses of one complex -> the (seed, sample) to use as reference
    map_to_anchor(): one pose -> its chain mapping onto the anchor, as a csv row
    anchor_roles():  the anchor's chain pair per interface, for the roles csv
    main():          pick an anchor per complex, map every pose, write both csvs
"""
import argparse
import collections
import csv
import datetime
import json
import logging
import os
import sys
import time

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402

import pool_index  # noqa: E402

logging.disable(logging.WARNING)   # pxmeter logs a warning per unmapped ligand, thousands of poses

MAP_FIELDS = ["pdb", "seed", "sample", "chain_mapping", "anchor_seed", "anchor_sample", "seconds"]
ROLE_FIELDS = ["pdb", "chain_1", "chain_2", "ab_chain", "ag_chain"]


def pool_rows(pool_csv, pdbs=None):
    """in: pool csv, optional subset of complexes.
    out: {pdb: {(seed, sample): {interface: pair_iptm}}}
    """
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(pool_csv) as fh:
        for r in csv.DictReader(fh):
            if pdbs and r["pdb"] not in pdbs:
                continue
            # pool csv spells these chain1/chain2; the roles csv spells them chain_1/chain_2
            key = f"{r['chain1']}{r['chain2']}"
            out[r["pdb"]][(r["seed"], int(r["sample"]))][key] = float(r["pair_iptm"])
    return out


def pick_anchor(poses):
    """in: {(seed, sample): {interface: pair_iptm}}.  out: the (seed, sample) with the highest mean.

    Averaged over the interfaces a pose appears in, not over all of them: requiring presence
    everywhere would exclude the poses that bind an unusual copy, which are the ones worth surfacing.
    """
    return max(poses, key=lambda k: sum(poses[k].values()) / max(len(poses[k]), 1))


def map_to_anchor(task):
    """in: (pdb, seed, sample, pose path, anchor seed, anchor sample, anchor path)
    out: a csv row; chain_mapping is "" and anchor_sample carries the reason when mapping fails
    """
    pdb, seed, sample, path, aseed, asamp, apath = task
    from pxmeter.mapping import MappingResult
    t = time.time()
    base = {"pdb": pdb, "seed": seed, "sample": sample,
            "anchor_seed": aseed, "anchor_sample": asamp}
    if (seed, sample) == (aseed, asamp):
        import gemmi                              # anchor against itself is the identity mapping
        st = gemmi.read_structure(apath)
        st.setup_entities()
        cm = {c.name: c.name for c in st[0]}
    else:
        try:
            m = MappingResult.from_cifs(apath, path)      # (reference, model) -> ref_to_model
        except Exception as e:
            return {**base, "chain_mapping": "", "seconds": round(time.time() - t, 2),
                    "anchor_sample": f"FAILED: {type(e).__name__}"}
        cm = {str(k): str(v) for k, v in m.chain_mapping.items()}
    return {**base, "chain_mapping": json.dumps(cm, sort_keys=True),
            "seconds": round(time.time() - t, 2)}


def anchor_roles(pdb, interfaces, anchor_xtal_map, xtal_roles):
    """in: the anchor's crystal mapping and the reference roles.
    out: rows of [pdb, c1, c2, anchor ab chain, anchor ag chain]
    """
    rows = []
    for ik in sorted(interfaces):
        iface = f"{pdb}_{ik}"
        if iface not in xtal_roles:
            continue
        ab_ref, ag_ref = xtal_roles[iface]
        if ab_ref in anchor_xtal_map and ag_ref in anchor_xtal_map:
            rows.append([pdb, ik[0], ik[1], anchor_xtal_map[ab_ref], anchor_xtal_map[ag_ref]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(paths.ANCHOR_DIR))
    ap.add_argument("--pool", default=str(paths.PXM2025_POOL_CSV))
    ap.add_argument("--pdbs", default=None, help="comma-separated subset; default is every complex")
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()

    pool = pool_rows(A.pool, set(A.pdbs.split(",")) if A.pdbs else None)
    os.makedirs(A.out_dir, exist_ok=True)
    fmap = f"{A.out_dir}/chain_maps_anchor.csv"
    frole = f"{A.out_dir}/chain_roles_anchor.csv"
    xtal_roles, xtal_maps = pool_index.load_roles(), pool_index.load_chain_maps()

    tasks, roles, skipped, anchors = [], [], [], {}
    for pdb, poses in sorted(pool.items()):
        aseed, asamp = pick_anchor(poses)
        anchors[pdb] = {"seed": aseed, "sample": asamp}
        apath = paths.pool_pose(pdb, aseed, asamp)
        anchor_xtal_map = xtal_maps.get((pdb, aseed, asamp))
        if not os.path.exists(apath) or not anchor_xtal_map:
            skipped.append(f"{pdb} (anchor pose has no cif or no crystal mapping)")
            continue
        for seed, sample in sorted(poses):
            p = paths.pool_pose(pdb, seed, sample)
            if os.path.exists(p):
                tasks.append((pdb, seed, sample, p, aseed, asamp, apath))
        roles += anchor_roles(pdb, poses.get((aseed, asamp), {}), anchor_xtal_map, xtal_roles)
    for s in skipped:
        print(f"  SKIPPED {s}")

    with open(frole, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ROLE_FIELDS)
        w.writerows(roles)
    print(f"  interfaces with anchor-frame roles: {len(roles)}  ->  {frole}")

    print(f"  poses to map: {len(tasks)}  jobs: {A.jobs}", flush=True)
    t0, nfail = time.time(), 0
    with open(fmap, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MAP_FIELDS)
        w.writeheader()
        import multiprocessing as mp
        with mp.Pool(A.jobs) as workers:
            for i, row in enumerate(workers.imap_unordered(map_to_anchor, tasks, chunksize=8), 1):
                w.writerow(row)
                fh.flush()          # a kill loses the pose in flight, not the run
                nfail += not row["chain_mapping"]
                if i % 500 == 0 or i == len(tasks):
                    el = time.time() - t0
                    print(f"  {i}/{len(tasks)}  {el/i:.3f}s/pose  "
                          f"eta {(len(tasks)-i)*el/i/60:.0f}m  failed {nfail}", flush=True)
    print(f"  wrote {fmap}  ({len(tasks)} poses, {nfail} failed, {time.time()-t0:.0f}s)")

    manifest = {
        "pool_csv": A.pool,
        "complexes": sorted(anchors),
        "n_complexes": len(anchors),
        "anchors": anchors,
        "poses_mapped": len(tasks),
        "poses_failed": nfail,
        "skipped_complexes": skipped,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    fman = f"{A.out_dir}/anchor_manifest.json"
    json.dump(manifest, open(fman, "w"), indent=1)
    print(f"  wrote {fman}  ({len(anchors)} complexes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
