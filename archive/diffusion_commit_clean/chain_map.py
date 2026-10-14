#!/usr/bin/env python
"""Recover, for every pose, which model chain plays which reference chain.

WHY THIS EXISTS. On a multi-copy assembly the model builds several copies of the antigen and
different poses bind different copies, so a fixed chain pair is NOT the same interface in every pose.
Measured on 8w5d seed 300000: sample 0 maps reference B->model C and C->model B, while samples 1 and 2
are identity. Comparing those three poses on a fixed pair would silently compare different molecules.

The mapping was already solved correctly, once per pose, by the scorer -- PXMeter runs
ChainPermutation (anchor superposition + linear_sum_assignment over same-entity chains, every anchor
tried, best kept) BEFORE computing DockQ. It is a prerequisite of DockQ, not a by-product, which is
why our GPU pose-vs-pose DockQ cannot produce it: dockq_torch takes the chain pair as an ARGUMENT.

Then we threw it away. pxmeter writes it to pxm_json/<sample>.json, but only 4 of 6,919 legacy folds
retain that file, and eval/pxm_dockq.py computes `model_chains` per interface and drops it on the way
to the pool CSV. dockq_torch was written expecting a `chain_mapping` column that the PXMeter scoring
path never wrote. This script backfills it.

ANCHORED TO THE GROUND TRUTH, NOT POSE TO POSE. Permuting per pair of poses would make similarity
non-transitive: A could match B through copy C while B matches D through copy D, so A~B and B~D would
be measured in different frames and agglomerative clustering would merge on inconsistent evidence --
silently, since every individual number looks fine. Mapping each pose to the GT once puts every pose
in one canonical frame, and all pairwise comparisons become consistent by construction.

Calls MappingResult.from_cifs, the same entry point the scorer uses, rather than driving
ChainPermutation directly: that guarantees the recovered mapping is what scoring actually used. A
leaner call exists (get_heurisitic_chain_mapping alone) but at 0.18 s/pose the saving is not worth
risking a divergence.

    chain_map.py --out chain_maps.csv [--pdbs 8w5d,9s36] [--jobs 8]

Resumable: poses already in --out are skipped, and rows are flushed one at a time, so a kill loses at
most the pose in flight.
"""
import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402

logging.disable(logging.WARNING)   # pxmeter logs a warning per unmapped ligand; thousands of poses

FIELDS = ["pdb", "seed", "sample", "chain_mapping", "anchor_ref", "anchor_model", "seconds"]


def poses(pdbs=None):
    """Every legacy pool pose as (pdb, seed, sample, path), sorted so runs are reproducible."""
    out = []
    for d in sorted(glob.glob(f"{paths.FREE_DIR}/*/*-{paths.LEGACY_RUN}/predictions")):
        parts = d.split("/")
        pdb, seed = parts[-3], parts[-2].split("-")[0]
        if pdbs and pdb not in pdbs:
            continue
        for p in sorted(glob.glob(f"{d}/{pdb}_sample_*.cif")):
            m = re.search(r"_sample_(\d+)\.cif$", p)
            if m:
                out.append((pdb, seed, int(m.group(1)), p))
    return out


def one(task):
    """Map a single pose onto its ground truth. Returns a row dict, or None on failure.

    Failures are expected and must not kill the sweep: a pose too far from the reference raises
    ValueError from get_heurisitic_chain_mapping by design.
    """
    pdb, seed, sample, path = task
    from pxmeter.mapping import MappingResult
    t = time.time()
    try:
        m = MappingResult.from_cifs(f"{paths.PXM2025_CIF}/{pdb}.cif", path)
    except Exception as e:
        return {"pdb": pdb, "seed": seed, "sample": sample, "chain_mapping": "",
                "anchor_ref": "", "anchor_model": f"FAILED: {type(e).__name__}",
                "seconds": round(time.time() - t, 2)}
    # keys and values come back as numpy str_, which json cannot serialise
    cm = {str(k): str(v) for k, v in m.chain_mapping.items()}
    a = {str(k): str(v) for k, v in (m.chain_mapping_anchors or {}).items()}
    return {"pdb": pdb, "seed": seed, "sample": sample, "chain_mapping": json.dumps(cm, sort_keys=True),
            "anchor_ref": a.get("ref", ""), "anchor_model": a.get("model", ""),
            "seconds": round(time.time() - t, 2)}


def done_keys(out):
    if not os.path.exists(out):
        return set()
    with open(out) as fh:
        return {(r["pdb"], r["seed"], int(r["sample"])) for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdbs", default=None, help="comma-separated subset")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after N poses (for timing)")
    A = ap.parse_args()

    want = set(A.pdbs.split(",")) if A.pdbs else None
    todo = poses(want)
    have = done_keys(A.out)
    todo = [t for t in todo if (t[0], t[1], t[2]) not in have]
    if A.limit:
        todo = todo[:A.limit]
    print(f"  poses to map: {len(todo)}  (already done: {len(have)})  jobs: {A.jobs}", flush=True)
    if not todo:
        return 0

    fresh = not os.path.exists(A.out)
    t0 = time.time()
    nfail = 0
    with open(A.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        import multiprocessing as mp
        with mp.Pool(A.jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(one, todo, chunksize=8), 1):
                w.writerow(row)
                fh.flush()          # a kill loses the pose in flight, not the run
                if row["anchor_model"].startswith("FAILED"):
                    nfail += 1
                if i % 500 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {el/i:.3f}s/pose  eta {(len(todo)-i)*el/i/60:.0f}m  "
                          f"failed {nfail}", flush=True)
    print(f"  wrote {A.out}  ({len(todo)} poses, {nfail} failed, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
