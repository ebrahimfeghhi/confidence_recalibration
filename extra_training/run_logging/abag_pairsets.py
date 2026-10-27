#!/usr/bin/env python
"""Pass 1 of the ABAG-Rank export: fix each complex's inter-chain pair set. CPU only, no GPU.

WHY A FIXED SET PER COMPLEX. ABAG-Rank's 03_prepare_dataset.py computes per-complex normalisation
stats with np.stack(pae_list, axis=0) over the complex's samples, which requires every pose of a
complex to have the SAME number of pairs. A per-pose distance cutoff does not give that -- on 10fj
the 12 A inter-chain counts range 1,784 to 2,018 across its 50 poses, so the stack would raise.
Their own stage 1 avoids it by writing ALL inter-chain pairs (259,600 for 10fj, ~76 GB over our
38,150 poses) and letting the dataloader cut.

So this takes the UNION of each complex's per-pose cutoff sets: measured on 10fj that is 4,096
pairs, 2.2x the per-pose mean and 1.05% of T^2, about 1.3 GB over the whole dump. Lengths are then
constant within a complex, stage 3 stacks unchanged, and the dataloader's own per-sample
valid_pair_mask still applies the real cutoff -- so the model sees exactly what it would have seen.

THE REPRESENTATIVE ATOM IS THE DISTOGRAM ONE, not "the first atom of the token". The trunk npz
carries ci_feat_distogram_rep_atom_mask for exactly this, and using anything else would put the
distances on a different convention from everything that reads them.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rep_coords(z, pose):
    """(T, 3) representative-atom coordinates for one pose, by the distogram mask."""
    a2t = z["ci_feat_atom_to_token_idx"]
    mask = z["ci_feat_distogram_rep_atom_mask"].astype(bool)
    T = len(z["ci_feat_asym_id"])
    out = np.full((T, 3), np.nan, dtype=np.float32)
    seen = np.zeros(T, dtype=bool)
    xyz = z["ci_coordinate"][pose]
    for ai in np.where(mask)[0]:
        t = int(a2t[ai])
        if not seen[t]:
            out[t] = xyz[ai]
            seen[t] = True
    return out, seen


def pair_set(pdb, fold, cutoff):
    """Union over every (seed, pose) of this complex of its inter-chain pairs under `cutoff`."""
    files = sorted(glob.glob(os.path.join(fold, "trunks", pdb, "free_*.npz")))
    if not files:
        return None
    union = None
    asym = None
    npose = 0
    for f in files:
        z = np.load(f)
        if asym is None:
            asym = z["ci_feat_asym_id"]
            inter = asym[:, None] != asym[None, :]
        for p in range(z["ci_coordinate"].shape[0]):
            rep, seen = rep_coords(z, p)
            d = np.linalg.norm(rep[:, None, :] - rep[None, :, :], axis=-1)
            ok = seen[:, None] & seen[None, :]
            m = inter & ok & (d < cutoff)
            union = m if union is None else (union | m)
            npose += 1
    if union is None or not union.any():
        return None
    i, j = np.where(union)
    return {"pdb": pdb, "n_tokens": int(len(asym)), "n_poses": npose,
            "n_pairs": int(len(i)), "inter_idx": i.astype(np.int32),
            "inter_jdx": j.astype(np.int32), "asym_id": asym.astype(np.int32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="training/full_train/structures_no9dzj.csv")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--cutoff", type=float, default=12.0,
                    help="matches config_ABAG_rank_no_esm.yaml's distance_cutoff")
    ap.add_argument("--out", default="training/abag/pairsets")
    ap.add_argument("--limit", type=int, default=0, help="first N structures, for a smoke test")
    a = ap.parse_args()

    pdbs = [r["pdb"] for r in csv.DictReader(open(os.path.join(ROOT, a.selection)))]
    if a.limit:
        pdbs = pdbs[:a.limit]
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    tot_pairs = tot_poses = done = skipped = tot_bytes = 0
    for n, pdb in enumerate(pdbs, 1):
        dst = os.path.join(out, f"{pdb}.npz")
        if os.path.exists(dst):
            done += 1
            continue
        ps = pair_set(pdb, os.path.join(ROOT, a.fold), a.cutoff)
        if ps is None:
            skipped += 1
            print(f"  SKIP {pdb}: no trunks or no inter-chain pair under {a.cutoff} A", flush=True)
            continue
        np.savez_compressed(dst, inter_idx=ps["inter_idx"], inter_jdx=ps["inter_jdx"],
                            asym_id=ps["asym_id"], n_tokens=ps["n_tokens"],
                            n_poses=ps["n_poses"], cutoff=a.cutoff)
        tot_pairs += ps["n_pairs"]
        tot_poses += ps["n_poses"]
        # bytes this complex will cost in the dump: one float32 PAE and one float32 CA distance
        # per pair per pose. Accumulated per complex because pairs/complex and poses/complex both
        # vary, so a product of the two means is not the sum of the products.
        tot_bytes += ps["n_pairs"] * ps["n_poses"] * 4 * 2
        done += 1
        if n % 25 == 0 or n == len(pdbs):
            print(f"  {n}/{len(pdbs)}  last {pdb}: T={ps['n_tokens']} poses={ps['n_poses']} "
                  f"pairs={ps['n_pairs']:,}", flush=True)
    print(f"\n  {done} complex(es) written, {skipped} skipped")
    if tot_pairs:
        print(f"  {tot_pairs:,} pairs over {tot_poses:,} poses; mean {tot_pairs/max(done,1):,.0f} "
              f"pairs/complex")
        print(f"  dump size for these complexes: {tot_bytes / 1e9:.3f} GB; "
              f"projected for {len(pdbs)} complexes: "
              f"{tot_bytes / max(done, 1) * len(pdbs) / 1e9:.2f} GB")
    json.dump({"cutoff": a.cutoff, "selection": a.selection, "n_complexes": done},
              open(os.path.join(out, "_meta.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
