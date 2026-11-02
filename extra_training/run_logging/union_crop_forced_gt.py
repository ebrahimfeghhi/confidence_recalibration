#!/usr/bin/env python
"""What does FORCING the ground-truth interface into the pose-union crop cost in tokens?

THE QUESTION. union_crop_margin_sweep.py measured the union over poses at margin 2.5 on TRAIN at
41.4% of tokens with interface coverage 1.0000 AT THE MEDIAN but p10 0.9500 -- so the bottom
decile of structures loses 5%+ of the contacts the loss is asking the head to score. Median
coverage is not a guarantee; this objective needs one, because the DockQ label is a global number
computed on the full structure and cannot be cropped to match a partially-kept interface.

The fix is to union the bare ground-truth contact tokens in regardless of margin, which makes
coverage 1.0 BY CONSTRUCTION. This measures what that costs.

ALSO MEASURED: the PER-INTERFACE union. The random-anchor design needs containment of ONE
interface's union over poses, not of all interfaces at once, so its size is the number that
decides whether a random anchor can ever contain one.

CPU only -- cKDTree over cached coordinates, no GPU, safe beside a training run.

  python -I run_logging/union_crop_forced_gt.py --structs 80
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_j_build_cache as BC                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="training/full_train/pool_winnable.csv")
ap.add_argument("--roles", default="training/fold/chain_roles.csv")
ap.add_argument("--trunks", default="training/fold/trunks")
ap.add_argument("--labels", default="training/fold/labels")
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margins", default="1,2.5,5")
ap.add_argument("--ref-margin", type=float, default=10.0)
ap.add_argument("--structs", type=int, default=80)
ap.add_argument("--seeds", type=int, default=10, help="npz files per structure; the arm has 10, "
                                                      "each holding 5 poses = all 50")
a = ap.parse_args()
MARGINS = [float(x) for x in a.margins.split(",")]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def pose_ifaces(z, coords, cmask_all, rep, contact, want_pair=None):
    """Per pose: (interface token mask, token rep coords). want_pair restricts to ONE chain pair."""
    asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
    a2t = np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)
    n_tok = asym.shape[0]
    in_range = (a2t >= 0) & (a2t < n_tok)
    out = []
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
            ci, cj = chain_of_atom[i], chain_of_atom[j]
            if ci == cj:
                continue
            if want_pair is not None and tuple(sorted((int(ci), int(cj)))) != want_pair:
                continue
            iface[tok_of_atom[i]] = True
            iface[tok_of_atom[j]] = True
        if not iface.any():
            continue
        tok_xyz = np.full((n_tok, 3), np.nan)
        for ai in np.nonzero(rep & cmask)[0]:
            t = int(a2t[ai])
            if np.isnan(tok_xyz[t, 0]):
                tok_xyz[t] = coords[pi][ai]
        out.append((iface, tok_xyz))
    return out


def expand(iface, tok_xyz, margin, n_tok):
    good = ~np.isnan(tok_xyz[:, 0])
    centres = np.nonzero(iface & good)[0]
    if centres.size == 0:
        return iface.copy()
    gi = np.nonzero(good)[0]
    d = np.linalg.norm(tok_xyz[gi][:, None, :] - tok_xyz[centres][None, :, :], axis=-1)
    near = np.zeros(n_tok, dtype=bool)
    near[gi[(d < margin).any(axis=1)]] = True
    return iface | near


roles = {}
for r in csv.DictReader(open(a.roles)):
    roles.setdefault(r["pdb"], {})[r["ref_chain"]] = r["model_chain"]
pool = {}
for r in csv.DictReader(open(a.pool)):
    m1 = roles.get(r["pdb"], {}).get(r["chain1"])
    m2 = roles.get(r["pdb"], {}).get(r["chain2"])
    if m1 and m2:
        pool.setdefault(r["pdb"], set()).add((min(ord(m1), ord(m2)) - ord("A"),
                                              max(ord(m1), ord(m2)) - ord("A")))

rows = []
for pdb in sorted(pool):
    if len(rows) >= a.structs:
        break
    tdir, ldir = os.path.join(a.trunks, pdb), os.path.join(a.labels, pdb)
    if not (os.path.isdir(tdir) and os.path.isdir(ldir)):
        continue
    units = sorted(f for f in os.listdir(tdir) if f.endswith(".npz"))[:a.seeds]
    if not units:
        continue
    gt, true_iface, n_tok = None, None, None
    uni = {m: None for m in MARGINS}
    per_if = {}                                   # (pair, margin) -> union mask
    n_pose = 0
    for u in units:
        try:
            z = np.load(os.path.join(tdir, u), allow_pickle=True)
            lab = np.load(os.path.join(ldir, u), allow_pickle=True)
        except (OSError, ValueError):
            continue
        rep = np.asarray(lab["lf_pae_rep_atom_mask"]).reshape(-1).astype(bool)
        asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
        a2t = np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)
        n_tok = asym.shape[0]
        if gt is None:
            gt, _ = BC.interface_crop_tokens(z, lab, a.contact, a.ref_margin, all_poses=True)
            if gt is None:
                break
            # TRUE contact tokens, from the LABEL structure, scored pairs only.
            lc = np.asarray(lab["label_coordinate"])[0]
            lm = np.asarray(lab["label_coordinate_mask"])[0].astype(bool)
            ok = lm & (a2t >= 0) & (a2t < n_tok)
            ch = asym[a2t]
            ti = np.zeros(n_tok, dtype=bool)
            for (x, y) in pool[pdb]:
                m1, m2 = ok & (ch == x), ok & (ch == y)
                if not m1.any() or not m2.any():
                    continue
                i1, i2 = np.where(m1)[0], np.where(m2)[0]
                for q, h in enumerate(cKDTree(lc[m2]).query_ball_point(lc[m1], a.contact)):
                    if h:
                        ti[a2t[i1[q]]] = True
                        for j in h:
                            ti[a2t[i2[j]]] = True
            true_iface = ti
        pc = np.asarray(z["ci_coordinate"])
        pm = np.isfinite(pc).all(axis=-1)
        # ALL interfaces at once
        for iface, txyz in pose_ifaces(z, pc, pm, rep, a.contact):
            n_pose += 1
            for m in MARGINS:
                e = expand(iface, txyz, m, n_tok)
                uni[m] = e if uni[m] is None else (uni[m] | e)
        # ONE interface at a time
        for pr in sorted(pool[pdb]):
            for iface, txyz in pose_ifaces(z, pc, pm, rep, a.contact, want_pair=pr):
                for m in MARGINS:
                    e = expand(iface, txyz, m, n_tok)
                    k = (pr, m)
                    per_if[k] = e if k not in per_if else (per_if[k] | e)
    if gt is None or true_iface is None or not true_iface.any() or uni[MARGINS[-1]] is None:
        continue
    rec = {}
    for m in MARGINS:
        pu = uni[m]
        forced = pu | true_iface
        pf = [per_if[(pr, m)].mean() for pr in sorted(pool[pdb]) if (pr, m) in per_if]
        rec[m] = dict(frac=float(pu.mean()), icov=float((true_iface & pu).sum() / true_iface.sum()),
                      f_frac=float(forced.mean()), added=int(forced.sum() - pu.sum()),
                      per_if=pf)
    rows.append(dict(pdb=pdb, n_tok=n_tok, ti=int(true_iface.sum()), n_if=len(pool[pdb]),
                     gt_frac=float(gt.mean()), n_pose=n_pose, per=rec))

if not rows:
    raise SystemExit("no structure measured")

print(f"\n{len(rows)} structure(s), contact {a.contact} A, {int(np.median([r['n_pose'] for r in rows]))} "
      f"pose(s) unioned (median)")
print(f"  full structure            median {np.median([r['n_tok'] for r in rows]):.0f} tokens")
print(f"  GT interface crop (m10)   median {100*np.median([r['gt_frac'] for r in rows]):.1f}% of full")
print(f"  true contact tokens       median {np.median([r['ti'] for r in rows]):.0f}")
print(f"  scored interfaces         median {np.median([r['n_if'] for r in rows]):.0f}\n")

h = (f"  {'margin':>6}  {'union %':>8}  {'cov':>6}  {'cov p10':>8}  {'cov MIN':>8}  "
     f"{'+GT %':>7}  {'added tok':>10}  {'worst add':>10}  {'1 iface %':>10}")
print(h); print("  " + "-" * (len(h) - 2))
for m in MARGINS:
    fr = np.array([r["per"][m]["frac"] for r in rows])
    cov = np.array([r["per"][m]["icov"] for r in rows])
    ff = np.array([r["per"][m]["f_frac"] for r in rows])
    ad = np.array([r["per"][m]["added"] for r in rows])
    pif = np.concatenate([r["per"][m]["per_if"] for r in rows if r["per"][m]["per_if"]]) \
        if any(r["per"][m]["per_if"] for r in rows) else np.array([np.nan])
    print(f"  {m:>6.1f}  {100*np.median(fr):>7.1f}%  {np.median(cov):>6.4f}  "
          f"{np.percentile(cov,10):>8.4f}  {cov.min():>8.4f}  {100*np.median(ff):>6.1f}%  "
          f"{np.median(ad):>10.0f}  {ad.max():>10.0f}  {100*np.median(pif):>9.1f}%")
print(f"\n  'added tok' = tokens the forced GT contact set adds on top of the pose union.")
print(f"  '1 iface %' = median size of a SINGLE interface's pose-union, as % of full.")
n_bad = {m: int((np.array([r['per'][m]['icov'] for r in rows]) < 1.0).sum()) for m in MARGINS}
for m in MARGINS:
    print(f"  margin {m:g}: {n_bad[m]} of {len(rows)} structure(s) had coverage < 1.0 "
          f"BEFORE forcing (= silently truncated interfaces)")
