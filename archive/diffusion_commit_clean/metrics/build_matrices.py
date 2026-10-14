#!/usr/bin/env python
"""Compute the three pose-similarity matrices for every interface, so cluster purity can decide which
metric defines a "basin".

ONE FILE PER INTERFACE, an .npz holding everything needed to evaluate it later:

    epitope, fcc, dockq   (N, N) float32 similarity, symmetric, diagonal 1
    seed, sample          (N,)   which pose each row is
    model_ab, model_ag    (N,)   the chains actually compared, from chain_map.py's mapping
    dockq_gt, pair_iptm   (N,)   ground truth and confidence, the labels purity is scored against

Keeping the labels beside the matrices is deliberate: a matrix whose row order has drifted from its
labels is a silent, unfalsifiable bug, and row order here is nothing more than a list index.

RECOMPUTED FROM SCRATCH, not reused. 55 cached matrices exist from before the chain mapping, and 46%
of all poses turn out to have a permuted mapping, so some unknown subset of those was built on the
wrong chains. Identifying which would cost as much as redoing them.

POSES WITH NO CONTACTS ARE DROPPED. On a multi-copy assembly a pose can bind a different antigen copy
than the interface asks about, leaving zero antibody-antigen contacts for the pair we resolved (1 of
12 on 8w5d_AB). Its similarity to everything is 0 by construction, which is not a basin -- it is an
absence of an interface. Dropping is done BEFORE any matrix is built so all three share a row order,
and the count is recorded per interface rather than left to be inferred from a row count.

    build_matrices.py --out DIR [--ifaces 8w5d_AB,...] [--device cuda]

Resumable: an interface whose .npz exists is skipped.
"""
import argparse
import collections
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import paths  # noqa: E402
from pool import load_pool  # noqa: E402  one definition, shared with validate.py


def build(iface, rows, maps, roles, device, min_poses=5):
    """Three matrices plus aligned labels for one interface, or None if too few usable poses."""
    items, nomap = metrics.items_for_interface(iface, [(s, x) for s, x, _, _ in rows], maps, roles)
    # items_for_interface drops unmapped poses, so re-pair labels to what survived rather than
    # assuming the two lists still line up
    by_key = {(s, x): (d, i) for s, x, d, i in rows}
    keys = [(s, x) for (s, x) in [(s, x) for s, x, _, _ in rows]
            if (s, x) not in set(nomap)]
    # Filter at the CONTACT cutoff (5.0 A), not the epitope cutoff (4.5 A). Using 4.5 dropped ~1,110
    # poses that DockQ scores as real interfaces: on ring assemblies like 9kmp (5 antigens, 5
    # antibodies, each antigen bound by two) the second interface of the ring is a glancing 4.70 A
    # contact -- outside 4.5, inside DockQ's 5.0 -- and 162 of those poses had GT DockQ above 0.80.
    keep = [(it, k) for it, k in zip(items, keys)
            if len(metrics.epitope_set(it[0], it[1], it[2], cutoff=metrics.CONTACT_CUTOFF)) > 0]
    nocontact = len(items) - len(keep)
    # min_poses was a hard 5, written when these matrices were only for basin clustering, where a
    # matrix over <5 poses says nothing. It silently became an interface FILTER once the interface set
    # started coming from pose contacts: 11 of 53 predicted-only pairs were dropped by it, making 5 a
    # tuned parameter for a job it was never chosen for. It is now explicit and caller-set.
    if len(keep) < min_poses:
        return None, len(nomap), nocontact
    items = [it for it, _ in keep]
    keys = [k for _, k in keep]
    M, _ = metrics.set_matrix(items, "epitope")
    F, _ = metrics.set_matrix(items, "fcc")
    D = metrics.dockq_pose_matrix(items, device=device)
    out = dict(
        epitope=M, fcc=F, dockq=D,
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
    ap.add_argument("--ifaces", default=None, help="comma-separated subset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/n -- take every n'th interface, offset i")
    ap.add_argument("--min-poses", type=int, default=5,
                    help="minimum contacting poses for an interface to get a matrix")
    ap.add_argument("--pool", default=None,
                    help="extra pool rows for interfaces absent from the reference")
    ap.add_argument("--maps", default=None,
                    help="chain_maps csv; default is the crystal-anchored one from chain_map.py")
    ap.add_argument("--roles", default=None,
                    help="chain_roles csv; must be in the same frame as --maps")
    A = ap.parse_args()
    si, sn = (int(v) for v in A.shard.split("/"))

    os.makedirs(A.out, exist_ok=True)
    # --maps/--roles must move together: roles name the REFERENCE chains that maps is keyed on, so a
    # crystal roles file against an anchor-frame maps file misses every lookup instead of erroring
    pool = load_pool(A.pool)
    maps, roles = metrics.load_chain_maps(A.maps), metrics.load_roles(A.roles)
    want = set(A.ifaces.split(",")) if A.ifaces else None
    todo = sorted(k for k in pool if (not want or k in want) and k in roles)
    todo = [k for k in todo if not os.path.exists(f"{A.out}/{k}.npz")][si::sn]
    print(f"  interfaces to build: {len(todo)}  (shard {si}/{sn})", flush=True)

    # one log per shard: three processes appending to one csv interleaves partial lines
    log = open(f"{A.out}/build_log{'' if sn == 1 else f'_{si}'}.csv", "a", newline="")
    w = csv.writer(log)
    if log.tell() == 0:
        w.writerow(["interface", "poses_in_pool", "rows", "dropped_unmapped",
                    "dropped_no_contact", "seconds"])
    t0 = time.time()
    for n, iface in enumerate(todo, 1):
        t = time.time()
        try:
            out, nomap, nocon = build(iface, pool[iface], maps, roles, A.device, A.min_poses)
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
