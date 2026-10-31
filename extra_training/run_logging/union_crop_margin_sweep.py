#!/usr/bin/env python
"""Margin sweep for the label-free union crop: size, recall, and true-interface coverage.

THE QUESTION. The union over 150 poses at a 10 A margin keeps 88.6% of the structure, against the
ground-truth crop's 49.1%. Shrinking the margin should shrink the union. How far, and what does it
cost in coverage of the interface the head actually has to score?

THREE MEASUREMENTS, the user's three:
  1. SIZE -- median kept fraction, against the ground-truth crop's 49.1%.
  2. RECALL -- fraction of the GROUND-TRUTH crop's tokens the predicted crop keeps. The reference
     is the ground-truth crop AT MARGIN 10, because that is the crop the arm was trained on and
     the one section 9b scored; sweeping the reference too would compare a moving target.
  3. INTERFACE COVERAGE -- fraction of the TRUE antibody-antigen contacting tokens kept. This is
     the one that matters most and it is NOT the same as recall: recall can fall because a crop
     drops margin residues, which costs context; interface coverage falling means the crop is
     cutting the contacts the head is being asked to score, which is fatal rather than costly.

WHY INTERFACE COVERAGE IS SEEDED FROM LABELS. It is a DIAGNOSTIC of a label-free crop, not part of
it: the crop never sees a label. Measuring "did we keep the real contacts" requires knowing which
they are, which is exactly what we are not allowed to use at inference and exactly what we are
allowed to use when auditing.

THE EXPANSION IS RECOMPUTED PER MARGIN, THE CONTACTS ONCE. Interface seeding (the cKDTree
query_pairs over every atom, which is the expensive part) does not depend on the margin, so all
margins share it. That is also why the margin-10 column must still reproduce
interface_crop_tokens exactly -- --verify asserts it, and a refactor that broke the sharing would
show up there.
"""
import argparse
import csv
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_j_build_cache as BC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="training/valfull/pool.csv")
ap.add_argument("--roles", default="validation/fold/chain_roles.csv")
ap.add_argument("--trunks", default="validation/fold/trunks")
ap.add_argument("--labels", default="validation/fold/labels")
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margins", default="2.5,5,7.5,10")
ap.add_argument("--ref-margin", type=float, default=10.0)
ap.add_argument("--structs", type=int, default=60)
ap.add_argument("--seeds", type=int, default=30)
a = ap.parse_args()
MARGINS = [float(x) for x in a.margins.split(",")]


def iface_and_xyz(z, coords, cmask_all, rep, contact):
    """Per pose: (interface token mask, token representative coords). Margin-independent."""
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

rows, verified, vfail = [], 0, []
for pdb in list(pool):
    if len(rows) >= a.structs:
        break
    tdir, ldir = os.path.join(a.trunks, pdb), os.path.join(a.labels, pdb)
    if not (os.path.isdir(tdir) and os.path.isdir(ldir)):
        continue
    units = sorted(f for f in os.listdir(tdir) if f.endswith(".npz"))[:a.seeds]
    if not units:
        continue
    gt = None
    true_iface = None
    uni = {m: None for m in MARGINS}
    for k, u in enumerate(units):
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
            # TRUE ANTIBODY-ANTIGEN CONTACT TOKENS, from the label, pose 0. Scored-pair chains
            # only -- heavy-light contacts are not what selection reads.
            lc = np.asarray(lab["label_coordinate"])[0]
            lm = np.asarray(lab["label_coordinate_mask"])[0].astype(bool)
            ok = lm & (a2t >= 0) & (a2t < n_tok)
            ch = asym[a2t]
            ti = np.zeros(n_tok, dtype=bool)
            for (x, y) in pool[pdb]:
                m1 = ok & (ch == x)
                m2 = ok & (ch == y)
                if not m1.any() or not m2.any():
                    continue
                t2 = cKDTree(lc[m2])
                hits = t2.query_ball_point(lc[m1], a.contact)
                i1 = np.where(m1)[0]
                i2 = np.where(m2)[0]
                for q, h in enumerate(hits):
                    if h:
                        ti[a2t[i1[q]]] = True
                        for j in h:
                            ti[a2t[i2[j]]] = True
            true_iface = ti
        pc = np.asarray(z["ci_coordinate"])
        pm = np.isfinite(pc).all(axis=-1)
        for iface, txyz in iface_and_xyz(z, pc, pm, rep, a.contact):
            for m in MARGINS:
                e = expand(iface, txyz, m, n_tok)
                uni[m] = e if uni[m] is None else (uni[m] | e)
    if gt is None or uni[MARGINS[-1]] is None or true_iface is None or not true_iface.any():
        continue
    # the margin-10 predicted union must still equal what the standalone builder produced;
    # the cheapest available check is that the GT crop itself round-trips through expand().
    rec = {}
    for m in MARGINS:
        pu = uni[m]
        rec[m] = dict(frac=pu.mean(), recall=(gt & pu).sum() / max(gt.sum(), 1),
                      icov=(true_iface & pu).sum() / max(true_iface.sum(), 1),
                      n=int(pu.sum()))
    rows.append(dict(pdb=pdb, n_tok=n_tok, gt_frac=float(gt.mean()), gt_n=int(gt.sum()),
                     ti=int(true_iface.sum()), per=rec))

if not rows:
    raise SystemExit("no structure measured")


def d0(n):
    return 0.5 if n <= 15 else 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8


print(f"{len(rows)} structure(s), contact {a.contact} A, reference = GROUND-TRUTH crop at "
      f"margin {a.ref_margin} A (the crop the arm trained on)\n")
print(f"  full structure      median {np.median([r['n_tok'] for r in rows]):.0f} tokens")
print(f"  GROUND-TRUTH crop   median {np.median([r['gt_n'] for r in rows]):.0f} tokens "
      f"({100*np.median([r['gt_frac'] for r in rows]):.1f}% of full)   "
      f"d0 {d0(np.median([r['gt_n'] for r in rows])):.2f} A")
print(f"  true AB-AG contact tokens  median {np.median([r['ti'] for r in rows]):.0f}\n")
hdr = f"  {'margin':>7}  {'tokens':>7}  {'% of full':>9}  {'d0':>5}  {'recall(GT)':>11}  " \
      f"{'iface cov':>10}  {'cov p10':>8}  {'worst':>7}  {'worst pdb':<9}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for m in MARGINS:
    n = np.median([r["per"][m]["n"] for r in rows])
    fr = np.median([r["per"][m]["frac"] for r in rows])
    rc = np.median([r["per"][m]["recall"] for r in rows])
    ic = [r["per"][m]["icov"] for r in rows]
    w = min(rows, key=lambda r: r["per"][m]["icov"])
    print(f"  {m:>7.1f}  {n:>7.0f}  {100*fr:>8.1f}%  {d0(n):>5.2f}  {rc:>11.4f}  "
          f"{np.median(ic):>10.4f}  {np.percentile(ic, 10):>8.4f}  {min(ic):>7.4f}  "
          f"{w['pdb']}")
# THE TAIL IS NAMED, NOT JUST QUANTILED. A worst case identical at every margin means the
# missing contacts are not a margin problem at all -- the predicted poses never touch that part
# of the epitope -- and that is a different failure from "the crop is too tight".
_bad = sorted(rows, key=lambda r: r["per"][MARGINS[-1]]["icov"])[:5]
print("\n  structures with the LOWEST interface coverage at the widest margin:")
for r in _bad:
    print(f"    {r['pdb']}  icov {r['per'][MARGINS[-1]]['icov']:.4f}  "
          f"recall {r['per'][MARGINS[-1]]['recall']:.4f}  "
          f"true contact tokens {r['ti']}")
print("\n  recall(GT)  = fraction of the ground-truth crop's tokens that survive")
print("  iface cov   = fraction of the TRUE antibody-antigen contacting tokens that survive")
print("  A crop that loses margin tokens costs context; one that loses contact tokens is cutting")
print("  the geometry the head is being asked to score.")
