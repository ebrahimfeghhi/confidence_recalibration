#!/usr/bin/env python
"""Compute the pose-to-pose DockQ matrix for each interface -- how similar every pose is to every other.

SUMMARY
    Runs after pose_interfaces.py. The next step keeps a pose only if it differs from everything already
    kept, which needs all N x N pairs precomputed. This is the GPU-heavy part of the pipeline.

    SYMMETRISED. Raw DockQ is directional -- Fnat divides by the reference's contact count, so
    DockQ(i as reference, j as model) does not equal the reverse. The stored value is the average of the
    two directions, because a similarity used for filtering has to be symmetric.

    POSES WITH NO CONTACTS ARE DROPPED before anything is computed. On a complex with several antigen
    copies a pose can bind a different copy than this interface asks about, leaving zero antibody-antigen
    contacts for the pair being measured. Its similarity to everything would be 0 by construction, which
    is not diversity, it is the absence of an interface. The drop count is recorded per interface.

    THE SORT KEY IS READ AT THE COMPARED CHAINS: pair_iptm comes from each pose's own confidence json
    at (model_ab, model_ag), the chains this matrix compares and the template will pin. It used to be
    copied from the crystal-scored pool csv, whose chains a crystal mapping chose; on duplicate-copy
    complexes that was a different copy for ~25% of poses.

    NOTHING HERE READS THE CRYSTAL. The matrix is stored WITH its labels -- which pose each row is,
    which chains were compared, the pose's ipTM -- because a row order drifted from its labels is a
    silent and unfalsifiable bug.

INPUTS
    --poses:     csv of pdb, seed, sample -- the pool. Required.
    --maps:      anchor_relabel.py's chain_maps_anchor.csv. Required.
    --roles:     pose_interfaces.py's interfaces csv. Required. Every interface in it gets a matrix.
    --out:       directory for the .npz files. Required. An interface whose file exists is skipped, so
                 a killed run resumes.
    --ifaces:    comma-separated subset. Default every interface in --roles.
    --min-poses: fewest contacting poses an interface needs to get a matrix. Default 5. Below that a
                 similarity matrix says nothing.
    --device:    cuda or cpu. Default cuda.
    --shard:     "i/n" to split the work across processes, taking every n'th interface.

OUTPUTS
    <out>/<interface>.npz, one per interface:
        dockq      (N, N) float32, symmetric, the similarity the next step filters on
        seed       (N,)   which seed each row is
        sample     (N,)   which diffusion sample each row is
        model_ab   (N,)   the antibody chain compared, per pose
        model_ag   (N,)   the antigen chain compared, per pose
        pair_iptm  (N,)   the model's own confidence at (model_ab, model_ag), what the next step sorts by

    <out>/build_log[_<shard>].csv: per interface, poses of the complex, rows kept, poses dropped for a
        missing mapping, poses dropped for no contacts, seconds

EXAMPLE
    pose_to_pose_dockq.py --poses exp/poses.csv --maps exp/chain_frame/chain_maps_anchor.csv \\
                          --roles exp/interfaces.csv --out exp/pose_dockq_matrices

FUNCTIONS
    dockq_matrix_for(): one interface -> the symmetric matrix and its labels, or None if too few poses
    main():             build every requested interface, skipping those already on disk
"""
import argparse
import collections
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contacts  # noqa: E402
import pool_index  # noqa: E402


def dockq_matrix_for(iface, keys, maps, roles, device, min_poses):
    """in: an interface, its complex's (seed, sample) list, the anchor maps and interface roles.
    out: (arrays dict or None, poses with no mapping, poses with no contacts)
    """
    items, nomap = pool_index.items_for_interface(iface, keys, maps, roles)
    dropped = set(nomap)
    keys = [k for k in keys if k not in dropped]           # items_for_interface keeps input order
    keep = [(it, k) for it, k in zip(items, keys) if contacts.has_contact(*it)]
    nocontact = len(items) - len(keep)
    if len(keep) < min_poses:
        return None, len(nomap), nocontact
    items = [it for it, _ in keep]
    keys = [k for _, k in keep]

    # dockq_torch sits beside this file; the import stays local so the GPU code travels with it
    from dockq_torch import dockq_matrix
    M = np.asarray(dockq_matrix(items, device=device, verbose=False), dtype=np.float32)
    out = dict(
        dockq=(M + M.T) / 2.0,
        seed=np.array([k[0] for k in keys]),
        sample=np.array([k[1] for k in keys], dtype=np.int16),
        model_ab=np.array([it[1] for it in items]),
        model_ag=np.array([it[2] for it in items]),
        pair_iptm=np.array([pool_index.pair_iptm_at(*it) for it in items], dtype=np.float32),
    )
    return out, len(nomap), nocontact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ifaces", default=None)
    ap.add_argument("--min-poses", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/n -- take every n'th interface, offset i")
    A = ap.parse_args()
    si, sn = (int(v) for v in A.shard.split("/"))

    os.makedirs(A.out, exist_ok=True)
    keys = collections.defaultdict(set)
    for r in csv.DictReader(open(A.poses)):
        keys[r["pdb"]].add((r["seed"], int(r["sample"])))
    maps, roles = pool_index.load_chain_maps(A.maps), pool_index.load_roles(A.roles)
    want = set(A.ifaces.split(",")) if A.ifaces else None
    todo = sorted(k for k in roles if not want or k in want)
    todo = [k for k in todo if not os.path.exists(f"{A.out}/{k}.npz")][si::sn]
    print(f"  interfaces to build: {len(todo)}  (shard {si}/{sn})", flush=True)

    # one log per shard: several processes appending to one csv interleaves partial lines
    log = open(f"{A.out}/build_log{'' if sn == 1 else f'_{si}'}.csv", "a", newline="")
    w = csv.writer(log)
    if log.tell() == 0:
        w.writerow(["interface", "poses", "rows", "dropped_unmapped", "dropped_no_contact", "seconds"])
    t0 = time.time()
    for n, iface in enumerate(todo, 1):
        t = time.time()
        pdb_keys = sorted(keys[iface.split("_")[0]])
        try:
            out, nomap, nocon = dockq_matrix_for(iface, pdb_keys, maps, roles, A.device, A.min_poses)
        except Exception as e:
            print(f"  {iface}: FAILED {type(e).__name__}: {e}", flush=True)
            w.writerow([iface, len(pdb_keys), 0, "", "", round(time.time() - t, 1)])
            log.flush()
            continue
        nrow = 0 if out is None else len(out["seed"])
        if out is not None:
            # write-then-rename: a kill mid-write must not leave a truncated npz the resume would skip
            tmp = f"{A.out}/.{iface}.tmp.npz"
            np.savez_compressed(tmp, **out)
            os.replace(tmp, f"{A.out}/{iface}.npz")
        w.writerow([iface, len(pdb_keys), nrow, nomap, nocon, round(time.time() - t, 1)])
        log.flush()
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el/n:.1f}s/iface  eta {(len(todo)-n)*el/n/60:.0f}m", flush=True)
    log.close()
    print(f"  done in {(time.time()-t0)/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
