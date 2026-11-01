#!/usr/bin/env python
"""Does any crop drop a WHOLE chain? If so, every ipTM read under that crop is mis-indexed.

WHY THIS IS NOT A STYLE QUESTION. Protenix compacts asym ids before building chain_pair_iptm:
sample_confidence.py:508-512 remaps them to 0..N-1 whenever `len(unique) != max+1`, i.e. exactly
when a chain is missing. phase_k_eval indexes that matrix with the ORIGINAL asym ids, so a crop
that removes chain 1 makes the evaluator read pair (0,3) where it meant (0,2), and its
`max(ai,aj) >= cpi.shape[0]` guard silently drops the last pair instead of flagging anything.

So the question "is this bug live on our data" is exactly "does any crop ever drop a whole chain",
and that is CPU-only and cheap. It is asked of EVERY crop mode, not just the new per-pose one,
because the published section 9b and section 13 numbers were produced under the other two.

    python -I run_logging/audit_chain_survival.py --pool training/valfull/pool.csv \
        --trunks validation/fold/trunks --labels validation/fold/labels \
        --masks training/runs/union_crop/valfull_m5.npz,training/runs/union_crop/valfull_m2.5.npz
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_j_build_cache as BC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--trunks", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--masks", default="", help="comma-separated union-crop npz files to audit too")
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margin", type=float, default=10.0)
ap.add_argument("--seed-base", type=int, default=400000)
ap.add_argument("--budget", type=int, default=30)
ap.add_argument("--gt-all-poses", action="store_true",
                help="audit the GT crop with all_poses=True. OFF BY DEFAULT because "
                     "phase_k_eval's --crop-all-poses is a store_true that defaults to FALSE, so "
                     "the historical runs cropped from POSE 0 ONLY. Auditing the union instead "
                     "is a weaker test in the wrong direction: a union can keep a chain that "
                     "pose 0 drops, so it can return a clean verdict for a crop that was not "
                     "clean. Raised in review.")
a = ap.parse_args()

pdbs = []
for r in csv.DictReader(open(a.pool)):
    if r["pdb"] not in pdbs:
        pdbs.append(r["pdb"])

masks = {}
for mp in [m for m in a.masks.split(",") if m]:
    z = np.load(mp, allow_pickle=True)
    masks[os.path.basename(mp)] = {k: z[k] for k in z.files
                                   if not k.endswith("__n") and k != "__meta"}

lost = {k: [] for k in ["GT crop (label coords)"] + list(masks)}
lost["per-pose predicted crop"] = []
n_struct = n_pose = n_unit = n_gt_undef = 0
missing = []
expect_units = 0

for pdb in pdbs:
    tdir, ldir = os.path.join(a.trunks, pdb), os.path.join(a.labels, pdb)
    t0 = os.path.join(tdir, f"free_{a.seed_base}.npz")
    l0 = os.path.join(ldir, f"free_{a.seed_base}.npz")
    expect_units += a.budget
    if not (os.path.exists(t0) and os.path.exists(l0)):
        missing.append(f"structure {pdb}: no trunk/label at seed {a.seed_base}")
        continue
    n_struct += 1
    z0 = np.load(t0, allow_pickle=True)
    lab0 = np.load(l0, allow_pickle=True)
    asym = np.asarray(z0["ci_feat_asym_id"]).reshape(-1)
    allc = set(np.unique(asym).tolist())

    for nm, mk in masks.items():
        if pdb not in mk:
            # COVERAGE IS PART OF THE AUDIT. A missing mask entry used to be skipped silently,
            # so an audit over zero structures printed the same clean verdict as a real one.
            missing.append(f"{nm}: no mask entry for {pdb}")
            continue
        m = np.asarray(mk[pdb]).astype(bool)
        miss = allc - set(np.unique(asym[m]).tolist())
        if miss:
            lost[nm].append((pdb, sorted(miss), len(allc)))

    for sd in range(a.seed_base, a.seed_base + a.budget):
        tp, lp = os.path.join(tdir, f"free_{sd}.npz"), os.path.join(ldir, f"free_{sd}.npz")
        if not (os.path.exists(tp) and os.path.exists(lp)):
            missing.append(f"unit {pdb}/free_{sd}.npz absent")
            continue
        n_unit += 1
        zc = np.load(tp, allow_pickle=True)
        lb = np.load(lp, allow_pickle=True)

        # THE GT CROP, PER UNIT, AT THE HISTORICAL SETTING. Every unit, not just the first seed:
        # the label npz is written per unit and its coordinates are permuted per pose, so one
        # seed does not stand for thirty.
        kt, _ = BC.interface_crop_tokens(zc, lb, a.contact, a.margin, all_poses=a.gt_all_poses)
        if kt is None:
            n_gt_undef += 1
        else:
            miss = allc - set(np.unique(asym[kt]).tolist())
            if miss:
                lost["GT crop (label coords)"].append(
                    (f"{pdb}/{sd}", sorted(miss), len(allc)))
        px = np.asarray(zc["ci_coordinate"])
        pm = np.isfinite(px).all(axis=-1)
        for pi in range(px.shape[0]):
            n_pose += 1
            k, _ = BC.crop_tokens_from_coords(zc, lb, px, pm, a.contact, a.margin, poses=[pi])
            if k is None:
                continue
            miss = allc - set(np.unique(asym[k]).tolist())
            if miss:
                lost["per-pose predicted crop"].append((f"{pdb}/{sd}/{pi}", sorted(miss), len(allc)))

print(f"  chain survival audit -- {n_struct} structure(s), {n_unit} unit(s), {n_pose} pose(s), "
      f"contact {a.contact} A, margin {a.margin} A, GT all_poses={a.gt_all_poses}")
print(f"  a crop that drops a whole chain makes Protenix compact the asym ids, and the "
      f"evaluator's\n  original-id indexing then reads the wrong chain pair.\n")
bad = 0
for nm, v in lost.items():
    print(f"  {nm:<28} crops that lost a chain: {len(v)}")
    for row in v[:5]:
        print(f"      {row[0]} lost chain(s) {row[1]} of {row[2]}")
    bad += len(v)
print()
# A VERDICT ON AN EMPTY AUDIT IS WORSE THAN NO VERDICT. Nothing may be skipped silently.
if missing:
    print(f"  INCOMPLETE: {len(missing)} expected item(s) were not examined, e.g.")
    for m in missing[:8]:
        print(f"      {m}")
    raise SystemExit("STOP: the audit did not cover everything it was asked to, so it cannot "
                     "certify anything. Fix the inputs or narrow the request explicitly.")
if n_unit != expect_units or n_struct == 0:
    raise SystemExit(f"STOP: examined {n_unit} unit(s) across {n_struct} structure(s), expected "
                     f"{expect_units}. An audit that silently examined nothing prints the same "
                     f"clean verdict as one that examined everything.")
print(f"  coverage: {n_unit}/{expect_units} unit(s), {n_gt_undef} GT crop(s) undefinable")
print("  VERDICT: " + ("no crop in any mode drops a whole chain on this pool, so the "
                       "mis-indexing is NOT live here" if bad == 0 else
                       f"{bad} crop(s) drop a chain -- the mis-indexing IS live"))
