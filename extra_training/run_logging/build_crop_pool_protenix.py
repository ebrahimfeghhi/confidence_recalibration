"""Protenix's OWN SpatialInterfaceCropping, enumerated per structure. No scoring, no filtering.

WHAT THIS IS. protenix/utils/cropping.py's SpatialInterfaceCropping picks a reference token that
lies within `interface_minimal_distance` (15 A) of ANY other chain, then takes the `crop_size`
nearest tokens of the whole assembly. configs_base.py ships train_crop_size = 256. That is all
this reproduces.

WHAT IT DELIBERATELY DOES NOT DO, and why that is the point. build_crop_pool.py additionally
(a) restricts the anchor to SCORED interfaces and (b) keeps only crops whose DockQ regret is
within a threshold. Both exist to protect a RANKING loss that reads ipTM on a specific chain
pair. A PAE-only arm has no scored pair and no ranking objective -- the PAE target is a per
token-pair cross-entropy computed from the cropped unit's own coordinates, so it is self
consistent under any crop. Those filters would therefore be unmotivated here, and the failure
they caused (a pool validated against the static chain_roles join while training scored the
per-pose OST join, which killed 3pdv7i 21 minutes into epoch 1) cannot arise without them.

The anchor set is every eligible token, so a structure sees a different Protenix crop each epoch
via `epoch % n_crop` rather than a fresh random draw. The distribution is the same; the rotation
is reproducible.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phases"))

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import crop_itm_rank as C                                           # noqa: E402


def all_chain_pairs(asym, resolved):
    """Every distinct ordered chain pair present, so the anchor rule is Protenix's own:
    `chain_id[None, :] != chain_id[:, None]` -- a contact with ANY other chain qualifies."""
    ch = sorted({int(c) for c in asym[resolved].tolist()}) if bool(resolved.any()) else []
    return [(f"{i}_{j}", i, j) for n, i in enumerate(ch) for j in ch[n + 1:]]


def one_structure(pdb, a):
    got = None
    for seed in C.SEEDS:
        got = C.unit_tensors(a.fold, pdb, seed)
        if got is not None:
            break
    if got is None:
        return None, "no tensors"
    asym, hf, fai, rep, tr, lb = got
    xyz, resolved = C.token_geometry(lb, rep)
    asym = asym.reshape(-1).long()
    N = int(asym.numel())
    pairs = all_chain_pairs(asym, resolved)
    if not pairs:
        return None, "fewer than two chains with resolved coordinates"
    crops, n_elig = C.sample_crops_pair_anchor(
        xyz, resolved, asym, pairs, a.crop_size, a.crops, C.crop_seed(a.seed, pdb))
    if crops is None or len(crops) == 0:
        return None, "no eligible anchor"
    arr = np.asarray(crops, dtype=np.uint16)
    return {"c0": arr,
            "sizes": np.full(arr.shape[0], arr.shape[1], np.int16),
            "n_crop": np.int32(arr.shape[0]),
            "n_token": np.int32(N),
            "n_eligible_anchors": np.int32(n_elig)}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--manifest", default="train_val_splits/cache_train_structures_extended.csv")
    ap.add_argument("--skip", default="4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m")
    ap.add_argument("--crop-size", type=int, default=256, help="Protenix train_crop_size")
    ap.add_argument("--crops", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="training/crop_pool_protenix")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    skip = set(x for x in a.skip.split(",") if x)
    want = [r["pdb"] for r in csv.DictReader(open(a.manifest)) if r["pdb"] not in skip]
    todo = want[a.shard::a.workers] if a.workers > 1 else want
    if a.limit:
        todo = todo[:a.limit]

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:                                               # noqa: BLE001
        commit = ""
    meta = json.dumps({"anchor": "protenix_any_chain_15A", "crop_size": a.crop_size,
                       "n_crops": a.crops, "seed": a.seed, "commit": commit,
                       "builder": "build_crop_pool_protenix.py", "filtering": "none",
                       "source": "protenix SpatialInterfaceCropping, train_crop_size 256"})

    wrote, skipped = 0, {}
    for pdb in todo:
        f = os.path.join(a.out, f"{pdb}.npz")
        if a.skip_existing and os.path.exists(f):
            continue
        try:
            out, why = one_structure(pdb, a)
        except Exception as e:                                      # noqa: BLE001
            skipped[pdb] = f"{type(e).__name__}: {e}"
            continue
        if out is None:
            skipped[pdb] = why
            continue
        out["__meta"] = np.array(meta)
        np.savez_compressed(f, **out)
        wrote += 1
    print(f"[shard {a.shard}] wrote {wrote} structure(s) to {a.out}   skipped {skipped}",
          flush=True)


if __name__ == "__main__":
    main()
