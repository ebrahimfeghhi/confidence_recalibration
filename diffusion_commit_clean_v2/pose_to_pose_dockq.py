#!/usr/bin/env python
"""Compute the pose-to-pose DockQ matrix for each interface -- how similar every pose is to every other.

SUMMARY
    Runs after pose_interfaces.py. The next step keeps a pose only if it differs from everything already
    kept, which needs all N x N pairs precomputed. This is the GPU-heavy part of the pipeline: roughly
    27 s per interface at 250 poses, across ~290 interfaces.

    SYMMETRISED. Raw DockQ is directional -- Fnat divides by the reference's contact count, so
    DockQ(i as reference, j as model) does not equal the reverse. The stored value is the average of the
    two directions, because a similarity used for filtering has to be symmetric.

    POSES WITH NO CONTACTS ARE DROPPED before anything is computed. On a complex with several antigen
    copies a pose can bind a different copy than this interface asks about, leaving zero antibody-antigen
    contacts for the pair being measured. Its similarity to everything would be 0 by construction, which
    is not diversity, it is the absence of an interface. The drop count is recorded per interface rather
    than left to be inferred from a row count.

    The matrix is stored WITH its labels -- which pose each row is, which chains were compared, the
    pose's ipTM and its ground-truth DockQ. A matrix whose row order has drifted from its labels is a
    silent and unfalsifiable bug, and row order here is nothing more than a list index.

    dockq_gt is stored for EVALUATION ONLY. Nothing in the selection path may read it.

INPUTS
    --out:      directory for the .npz files. Required. An interface whose file already exists is
                skipped, so a killed run resumes.
    --ifaces:   comma-separated subset. Default: every interface in the pool.
    --maps:     chain-map csv. Default the crystal frame; pass anchor_relabel.py's output to work in
                the anchor frame.
    --roles:    roles csv, in the same frame as --maps. The two must move together -- a crystal roles
                file against anchor-frame maps misses every lookup instead of failing.
    --pool:     pool csv. Default paths.PXM2025_POOL_CSV.
    --extra-pool: a second pool csv for predicted-only interfaces, from
                fetch_spurious_interface_iptm.py. They have no rows in the main pool.
    --min-poses: fewest contacting poses an interface needs to get a matrix. Default 5. Below that a
                similarity matrix says nothing. Set to 1 when the interface SET is what is being
                measured, so this threshold does not silently act as an interface filter.
    --device:   cuda or cpu. Default cuda.
    --shard:    "i/n" to split the work across processes, taking every n'th interface.

OUTPUTS
    <out>/<interface>.npz, one per interface:
        dockq      (N, N) float32, symmetric, the similarity the next step filters on
        seed       (N,)   which seed each row is
        sample     (N,)   which diffusion sample each row is
        model_ab   (N,)   the antibody chain compared, per pose
        model_ag   (N,)   the antigen chain compared, per pose
        pair_iptm  (N,)   the model's own confidence, what the next step sorts by
        dockq_gt   (N,)   ground truth against the reference, for evaluation only

    <out>/build_log[_<shard>].csv: per interface, poses in the pool, rows kept, poses dropped for a
        missing mapping, poses dropped for no contacts, seconds

EXAMPLE
    pose_to_pose_dockq.py --out matrices_anchor --maps $ANCHOR_DIR/chain_maps_anchor.csv \\
                          --roles $ANCHOR_DIR/chain_roles_anchor.csv

    pose_to_pose_dockq.py --out matrices_spur --maps $ANCHOR_DIR/chain_maps_anchor.csv \\
                          --roles $ANCHOR_DIR/chain_roles_all.csv \\
                          --extra-pool $ANCHOR_DIR/spurious_pool.csv --min-poses 1

FUNCTIONS
    dockq_matrix_for(): one interface -> the symmetric matrix and its labels, or None if too few poses
    main():             build every requested interface, skipping those already on disk
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
import poses as P  # noqa: E402


def dockq_matrix_for(iface, rows, maps, roles, device, min_poses):
    """in: an interface, its pool rows, a chain map, roles, device, minimum poses.
    out: (arrays dict or None, poses with no mapping, poses with no contacts)
    """
    items, nomap = P.items_for_interface(iface, [(s, x) for s, x, _, _ in rows], maps, roles)
    # re-pair the labels to what survived rather than assuming the two lists still line up
    by_key = {(s, x): (d, i) for s, x, d, i in rows}
    dropped = set(nomap)
    keys = [(s, x) for s, x, _, _ in rows if (s, x) not in dropped]
    keep = [(it, k) for it, k in zip(items, keys)
            if len(P.epitope_set(it[0], it[1], it[2], cutoff=P.CONTACT_CUTOFF)) > 0]
    nocontact = len(items) - len(keep)
    if len(keep) < min_poses:
        return None, len(nomap), nocontact
    items = [it for it, _ in keep]
    keys = [k for _, k in keep]

    sys.path.insert(0, str(paths.CLUSTERING_DIR))
    from dockq_torch import dockq_matrix
    M = np.asarray(dockq_matrix(items, device=device, verbose=False), dtype=np.float32)
    out = dict(
        dockq=(M + M.T) / 2.0,
        seed=np.array([k[0] for k in keys]),
        sample=np.array([k[1] for k in keys], dtype=np.int16),
        model_ab=np.array([it[1] for it in items]),
        model_ag=np.array([it[2] for it in items]),
        dockq_gt=np.array([by_key[k][0] for k in keys], dtype=np.float32),
        pair_iptm=np.array([by_key[k][1] for k in keys], dtype=np.float32),
    )
    return out, len(nomap), nocontact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ifaces", default=None)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--roles", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--extra-pool", default=None)
    ap.add_argument("--min-poses", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/n -- take every n'th interface, offset i")
    A = ap.parse_args()
    si, sn = (int(v) for v in A.shard.split("/"))

    os.makedirs(A.out, exist_ok=True)
    pool = P.load_pool(A.pool, A.extra_pool)
    maps, roles = P.load_chain_maps(A.maps), P.load_roles(A.roles)
    want = set(A.ifaces.split(",")) if A.ifaces else None
    todo = sorted(k for k in pool if (not want or k in want) and k in roles)
    todo = [k for k in todo if not os.path.exists(f"{A.out}/{k}.npz")][si::sn]
    print(f"  interfaces to build: {len(todo)}  (shard {si}/{sn})", flush=True)

    # one log per shard: several processes appending to one csv interleaves partial lines
    log = open(f"{A.out}/build_log{'' if sn == 1 else f'_{si}'}.csv", "a", newline="")
    w = csv.writer(log)
    if log.tell() == 0:
        w.writerow(["interface", "poses_in_pool", "rows", "dropped_unmapped",
                    "dropped_no_contact", "seconds"])
    t0 = time.time()
    for n, iface in enumerate(todo, 1):
        t = time.time()
        try:
            out, nomap, nocon = dockq_matrix_for(iface, pool[iface], maps, roles,
                                                 A.device, A.min_poses)
        except Exception as e:
            print(f"  {iface}: FAILED {type(e).__name__}: {e}", flush=True)
            w.writerow([iface, len(pool[iface]), 0, "", "", round(time.time() - t, 1)])
            log.flush()
            continue
        nrow = 0 if out is None else len(out["seed"])
        if out is not None:
            np.savez_compressed(f"{A.out}/{iface}.npz", **out)
        w.writerow([iface, len(pool[iface]), nrow, nomap, nocon, round(time.time() - t, 1)])
        log.flush()
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el/n:.1f}s/iface  eta {(len(todo)-n)*el/n/60:.0f}m", flush=True)
    log.close()
    print(f"  done in {(time.time()-t0)/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
