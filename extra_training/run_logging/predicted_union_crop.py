#!/usr/bin/env python
"""A label-free crop: the UNION, over every candidate pose, of each pose's own predicted interface.

THE IDEA. Section 9's crop is defined from `label_coordinate`, so it cannot be built at inference
(section 9c). This one is built from `ci_coordinate` -- what the fold PREDICTED for each pose --
so it needs no experimental structure. Taking the union over all poses means the crop does not
encode where the right epitope is: a pose docking on the wrong face contributes its own residues
too. And because it is ONE token set per structure, every candidate is scored over the same
residues, with the same n, hence the SAME d0 -- which is what per-pose cropping cannot offer
(run_logging/predcrop_size_control.py measured d0 swinging 3.00 -> 4.81 A within one interface).

THE CROP RULE IS COPIED, NOT REINVENTED. `_crop_from_coords` is interface_crop_tokens' body with
the coordinate array as an argument. `--verify` feeds it the label arrays and asserts the result is
BIT-IDENTICAL to interface_crop_tokens on every structure measured; if that assertion fails, every
comparison below is between a crop and a different crop's reimplementation and means nothing.

MEASURED, FULL POOL (53 structures, 150 poses each, 92 interfaces):

    full structure            median  481 tokens
    GROUND-TRUTH crop         median  236 tokens  (49.1% of full)   d0 5.70 A
    PREDICTED union crop      median  426 tokens  (88.6% of full)   d0 7.42 A
    predicted, ONE pose only  median  323 tokens  (67.2% of full)

    recall    (GT tokens kept by the predicted union)   1.0000  (p10 0.9849)
    precision (predicted-union tokens that are GT)      0.6119
    Jaccard                                             0.6119

THE UNION FINDS THE INTERFACE AND THEN KEEPS ALMOST EVERYTHING ELSE TOO. Recall
is 1.0000: the decoys' own interfaces cover the true one on every structure
measured, worst case 0.930. But it discards 11.4% of the structure where the
ground-truth crop discards 50.9%, so as a CONCENTRATOR it does about a fifth of
the work. A single pose's crop (67.2%) is nearer the ground-truth crop in size
than the union is -- the union's breadth is the price of not knowing which pose
is right.

WHAT THAT PREDICTS. Section 12 measured the available gain as concentrating
supervision on near pairs; removing an eighth of the tokens concentrates little,
so this crop should recover materially less than section 9b's +0.0376 (MAX) /
+0.0226 (MEAN). It is still worth scoring, because it costs one eval and it
settles whether section 9b was concentration or leakage -- but it should not be
launched expecting 9b's number.

ANY-CHAIN CONTACTS, as the ground-truth crop uses. interface_crop_tokens seeds its interface from
heavy-atom pairs on DIFFERENT chains, which for an H/L/antigen complex includes the heavy-light
pair. Restricting the predicted version to antibody-antigen only would compare two different
definitions and attribute the difference to the coordinates.
"""
import argparse
import collections
import csv
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_j_build_cache as BC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="training/valfull/pool.csv")
ap.add_argument("--trunks", default="validation/fold/trunks")
ap.add_argument("--labels", default="validation/fold/labels")
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margin", type=float, default=10.0)
ap.add_argument("--structs", type=int, default=60)
ap.add_argument("--seeds", type=int, default=30)
ap.add_argument("--verify", action="store_true", default=True)
a = ap.parse_args()


def _crop_from_coords(z, coords, cmask_all, rep, contact, margin):
    """interface_crop_tokens' rule, over an arbitrary (n_pose, n_atom, 3) coordinate array."""
    asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
    a2t = np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)
    n_tok = asym.shape[0]
    in_range = (a2t >= 0) & (a2t < n_tok)
    keep_tok = np.zeros(n_tok, dtype=bool)
    any_pose = False
    for pi in range(coords.shape[0]):
        cmask = cmask_all[pi].astype(bool) & in_range
        if cmask.sum() < 4:
            continue
        xyz, tok_of_atom = coords[pi][cmask], a2t[cmask]
        chain_of_atom = asym[tok_of_atom]
        if np.unique(chain_of_atom).size < 2:
            continue
        iface = np.zeros(n_tok, dtype=bool)
        for i, j in cKDTree(xyz).query_pairs(contact):
            if chain_of_atom[i] != chain_of_atom[j]:
                iface[tok_of_atom[i]] = True
                iface[tok_of_atom[j]] = True
        if not iface.any():
            continue
        tok_xyz = np.full((n_tok, 3), np.nan, dtype=np.float64)
        for ai in np.nonzero(rep & cmask)[0]:
            t = int(a2t[ai])
            if np.isnan(tok_xyz[t, 0]):
                tok_xyz[t] = coords[pi][ai]
        good = ~np.isnan(tok_xyz[:, 0])
        centres = np.nonzero(iface & good)[0]
        if centres.size == 0:
            keep_tok |= iface
            any_pose = True
            continue
        gi = np.nonzero(good)[0]
        d = np.linalg.norm(tok_xyz[gi][:, None, :] - tok_xyz[centres][None, :, :], axis=-1)
        near = np.zeros(n_tok, dtype=bool)
        near[gi[(d < margin).any(axis=1)]] = True
        keep_tok |= iface | near
        any_pose = True
    return keep_tok if any_pose and keep_tok.any() else None


pdbs = []
for r in csv.DictReader(open(a.pool)):
    if r["pdb"] not in pdbs:
        pdbs.append(r["pdb"])

rows, verified, vfail = [], 0, []
for pdb in pdbs:
    if len(rows) >= a.structs:
        break
    tdir, ldir = os.path.join(a.trunks, pdb), os.path.join(a.labels, pdb)
    if not (os.path.isdir(tdir) and os.path.isdir(ldir)):
        continue
    units = sorted(f for f in os.listdir(tdir) if f.endswith(".npz"))[:a.seeds]
    if not units:
        continue
    gt = None
    pred_union = None
    pred_single = None
    n_pose = 0
    for k, u in enumerate(units):
        try:
            z = np.load(os.path.join(tdir, u), allow_pickle=True)
            lab = np.load(os.path.join(ldir, u), allow_pickle=True)
        except (OSError, ValueError):
            continue
        rep = np.asarray(lab["lf_pae_rep_atom_mask"]).reshape(-1).astype(bool)
        if gt is None:
            gt, _ = BC.interface_crop_tokens(z, lab, a.contact, a.margin, all_poses=True)
            if gt is None:
                break
            if a.verify:
                mine = _crop_from_coords(z, np.asarray(lab["label_coordinate"]),
                                         np.asarray(lab["label_coordinate_mask"]), rep,
                                         a.contact, a.margin)
                if mine is None or not np.array_equal(mine, gt):
                    vfail.append(pdb)
                else:
                    verified += 1
        pc = np.asarray(z["ci_coordinate"])
        pm = np.isfinite(pc).all(axis=-1)
        pu = _crop_from_coords(z, pc, pm, rep, a.contact, a.margin)
        if pu is not None:
            pred_union = pu if pred_union is None else (pred_union | pu)
            n_pose += pc.shape[0]
        if k == 0:
            ps = _crop_from_coords(z, pc[:1], pm[:1], rep, a.contact, a.margin)
            pred_single = ps
    if gt is None or pred_union is None:
        continue
    inter = int((gt & pred_union).sum())
    rows.append(dict(
        pdb=pdb, n_tok=int(gt.size), n_pose=n_pose,
        gt=int(gt.sum()), pu=int(pred_union.sum()),
        ps=int(pred_single.sum()) if pred_single is not None else -1,
        recall=inter / max(int(gt.sum()), 1),
        precision=inter / max(int(pred_union.sum()), 1),
        jacc=inter / max(int((gt | pred_union).sum()), 1),
    ))

if not rows:
    raise SystemExit("no structure measured")


def d0(n):
    return 0.5 if n <= 15 else 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8


def med(k):
    return float(np.median([r[k] for r in rows]))


print(f"{len(rows)} structure(s), {int(np.median([r['n_pose'] for r in rows]))} poses each "
      f"(median), contact {a.contact} A, margin {a.margin} A")
if a.verify:
    print(f"VERIFY: _crop_from_coords reproduces interface_crop_tokens bit-identically on "
          f"{verified}/{verified + len(vfail)} structure(s)"
          + (f"  FAILED: {vfail[:5]}" if vfail else ""))
print()
print(f"  full structure            median {med('n_tok'):6.0f} tokens")
print(f"  GROUND-TRUTH crop         median {med('gt'):6.0f} tokens  "
      f"({100*med('gt')/med('n_tok'):4.1f}% of full)   d0 {d0(med('gt')):.2f} A")
print(f"  PREDICTED union crop      median {med('pu'):6.0f} tokens  "
      f"({100*med('pu')/med('n_tok'):4.1f}% of full)   d0 {d0(med('pu')):.2f} A")
print(f"  predicted, ONE pose only  median {med('ps'):6.0f} tokens  "
      f"({100*med('ps')/med('n_tok'):4.1f}% of full)")
print()
print(f"  recall    (GT tokens kept by the predicted union)   {med('recall'):.4f}")
print(f"  precision (predicted-union tokens that are GT)      {med('precision'):.4f}")
print(f"  Jaccard                                            {med('jacc'):.4f}")
print()
r_lo = float(np.percentile([r["recall"] for r in rows], 10))
worst = sorted(rows, key=lambda r: r["recall"])[:3]
print(f"  recall p10 {r_lo:.4f};  worst: " +
      ", ".join(f"{r['pdb']} {r['recall']:.3f}" for r in worst))
