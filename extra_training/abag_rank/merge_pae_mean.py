#!/usr/bin/env python
"""Merge each complex's per-shard PAE partial sums into the exact pae_col_mean. CPU only.

WHY A MERGE STEP EXISTS AT ALL. ABAG-Rank centres every PAE feature by `pae_col_mean`: the mean,
over ALL of a complex's poses, of that residue pair's PAE -- including the poses in which the pair
is far apart and which our sparse per-pose dump deliberately omits. Measured on 22gi, computing
that mean from the sparse pairs alone gives 1.532 A against the true 6.858 A, a 4.5x error in the
baseline every centred feature is measured against. So `phase_k_eval --dump-pae` accumulates the
DENSE sum on the GPU, where the tensor already exists, and writes one partial per shard; a
complex's poses are split across shards, so the partials must be summed before the mean is right.

THE POPULATION IS CHECKED, NOT ASSUMED. A partial set that is merely *present* proves nothing: if
one shard died, or two shards collided on a filename, the surviving sums produce a perfectly
plausible mean over fewer poses than the complex has. So the merge requires that the pose files on
disk and the accumulated counts agree exactly, per complex, and refuses otherwise.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="the --dump-pae directory")
    ap.add_argument("--out-name", default="_pae_mean.npz")
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge complexes whose pose files and accumulated counts disagree. Off "
                         "by default: such a mean is over a different population than the dataset.")
    a = ap.parse_args()

    prov = os.path.join(a.dump, "_provenance.json")
    if not os.path.exists(prov):
        raise SystemExit(f"STOP: no {prov}. That file records which checkpoint produced these PAE "
                         f"values; without it this dump's identity is unknown.")
    nshards = int(json.load(open(prov)).get("nshards") or 1)

    pdbs = sorted(d for d in os.listdir(a.dump)
                  if os.path.isdir(os.path.join(a.dump, d)))
    ok = bad = 0
    report = []
    for pdb in pdbs:
        d = os.path.join(a.dump, pdb)
        parts = sorted(glob.glob(os.path.join(d, "_paesum_s*.npz")))
        poses = sorted(f for f in glob.glob(os.path.join(d, "*.npz"))
                       if "_paesum" not in os.path.basename(f)
                       and os.path.basename(f) != a.out_name)
        if not parts:
            report.append(f"{pdb}: no partial sums"); bad += 1; continue
        tot = None
        n = 0
        shapes = set()
        for f in parts:
            z = np.load(f)
            shapes.add(z["pae_sum"].shape)
            tot = z["pae_sum"].astype(np.float64) if tot is None else tot + z["pae_sum"]
            n += int(z["n_poses"])
        if len(shapes) != 1:
            report.append(f"{pdb}: partials disagree on shape {shapes}"); bad += 1; continue
        # THE TWO COUNTS MUST AGREE. n is how many poses the accumulator saw; len(poses) is how
        # many pose files the dump wrote. A mean over a different population than the dataset uses
        # is exactly the error this whole mechanism exists to prevent.
        if n != len(poses):
            report.append(f"{pdb}: accumulated {n} pose(s) but {len(poses)} pose file(s) on disk")
            bad += 1
            if not a.allow_partial:
                continue
        if len(parts) > nshards:
            report.append(f"{pdb}: {len(parts)} partials for {nshards} shard(s)"); bad += 1
            if not a.allow_partial:
                continue
        mean = (tot / max(n, 1)).astype(np.float32)
        tmp = os.path.join(d, a.out_name + ".tmp")
        # np.savez_compressed appends .npz to a name lacking it.
        np.savez_compressed(tmp, pae_col_mean=mean, n_poses=np.int64(n),
                            n_shards=np.int32(len(parts)), pdb=pdb)
        os.replace(tmp + ".npz", os.path.join(d, a.out_name))
        ok += 1

    print(f"  merged {ok} complex(es); {bad} refused")
    for line in report[:12]:
        print(f"    {line}")
    if bad and not a.allow_partial:
        print(f"\n  STOP: {bad} complex(es) did not merge. Their pae_col_mean would be over a "
              f"different population than the poses in the dataset, which silently biases every "
              f"centred feature. Re-export them, or pass --allow-partial deliberately.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
