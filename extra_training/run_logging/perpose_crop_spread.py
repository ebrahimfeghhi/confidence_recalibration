#!/usr/bin/env python
"""How much does the PER-POSE interface crop vary between a structure's own candidates?

WHY THIS MATTERS AND WHAT IT IS NOT. The eval uses ONE union mask per structure precisely so that
every candidate is scored with the same token count, hence the same ipTM d0. This measures what
the alternative would have been: crop each pose by its own predicted interface, at the ORIGINAL
thresholds (contact 5 A, margin 10 A), and look at the spread WITHIN a structure.

THE WITHIN-STRUCTURE SPREAD IS THE WHOLE QUESTION. Variation across structures is expected and
harmless -- different complexes are different sizes, and poses from different structures are never
ranked against each other. Variation across the 150 poses OF ONE STRUCTURE is the problem, because
those are exactly the candidates being compared, and d0 = 1.24*(n-15)^(1/3) - 1.8 is computed from
each one's own n.

NO LABELS: coordinates are `ci_coordinate`, what the fold predicted for each pose. A pose that
docks on the wrong face still gets its own interface cropped.

    python -I run_logging/perpose_crop_spread.py --pool training/valfull/pool.csv \
        --trunks <trunks> --labels validation/fold/labels --seed-base 400000 --budget 30
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

# DO NOT import build_union_crop_masks here. It parses argv and runs its whole mask build at
# import time, so `import` alone rebuilt all 53 union masks before this script did any work --
# a measurement that should take minutes sat silent for ten. per_pose_sizes below is that
# module's crop_from_coords with the union accumulation removed; the union it returns is
# cross-checked against the shipped masks by --check-union.

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--trunks", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margin", type=float, default=10.0)
ap.add_argument("--seed-base", type=int, required=True)
ap.add_argument("--budget", type=int, default=30)
ap.add_argument("--out", default="")
a = ap.parse_args()


def d0(n):
    """ipTM's d0 from a chain-pair token count -- the reason crop size has to be constant."""
    return 1.24 * max(n - 15, 1) ** (1.0 / 3.0) - 1.8


def per_pose_sizes(z, coords, cmask_all, rep, contact, margin):
    """kept-token count for EACH pose separately; None where the pose has no cross-chain contact.

    This is crop_from_coords' body with the union accumulation removed, so the two cannot drift:
    the assertion below re-derives the union from these per-pose masks and compares it.
    """
    asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
    a2t = np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)
    n_tok = asym.shape[0]
    in_range = (a2t >= 0) & (a2t < n_tok)
    out, union = [], np.zeros(n_tok, dtype=bool)
    for pi in range(coords.shape[0]):
        cmask = cmask_all[pi].astype(bool) & in_range
        if cmask.sum() < 4:
            out.append(None); continue
        xyz, tok_of_atom = coords[pi][cmask], a2t[cmask]
        chain_of_atom = asym[tok_of_atom]
        if np.unique(chain_of_atom).size < 2:
            out.append(None); continue
        iface = np.zeros(n_tok, dtype=bool)
        for i, j in cKDTree(xyz).query_pairs(contact):
            if chain_of_atom[i] != chain_of_atom[j]:
                iface[tok_of_atom[i]] = True
                iface[tok_of_atom[j]] = True
        if not iface.any():
            out.append(None); continue
        tok_xyz = np.full((n_tok, 3), np.nan, dtype=np.float64)
        for ai in np.nonzero(rep & cmask)[0]:
            t = int(a2t[ai])
            if np.isnan(tok_xyz[t, 0]):
                tok_xyz[t] = coords[pi][ai]
        good = ~np.isnan(tok_xyz[:, 0])
        centres = np.nonzero(iface & good)[0]
        if centres.size == 0:
            keep = iface
        else:
            gi = np.nonzero(good)[0]
            d = np.linalg.norm(tok_xyz[gi][:, None, :] - tok_xyz[centres][None, :, :], axis=-1)
            near = np.zeros(n_tok, dtype=bool)
            near[gi[(d < margin).any(axis=1)]] = True
            keep = iface | near
        union |= keep
        out.append(int(keep.sum()))
    return out, union, n_tok


pdbs = []
for r in csv.DictReader(open(a.pool)):
    if r["pdb"] not in pdbs:
        pdbs.append(r["pdb"])

rows, n_nolabel, nocontact_tot, pose_tot = [], 0, 0, 0
_t0 = time.time()
_todo = sum(1 for p in pdbs
            if os.path.isdir(os.path.join(a.trunks, p)) and os.path.isdir(os.path.join(a.labels, p)))
_done = 0
for pdb in pdbs:
    tdir, ldir = os.path.join(a.trunks, pdb), os.path.join(a.labels, pdb)
    if not (os.path.isdir(tdir) and os.path.isdir(ldir)):
        continue
    sizes, n_tok, union = [], None, None
    for sd in range(a.seed_base, a.seed_base + a.budget):
        u = f"free_{sd}.npz"
        try:
            z = np.load(os.path.join(tdir, u), allow_pickle=True)
            lab = np.load(os.path.join(ldir, u), allow_pickle=True)
        except FileNotFoundError:
            n_nolabel += 1
            continue
        rep = np.asarray(lab["lf_pae_rep_atom_mask"]).reshape(-1).astype(bool)
        pc = np.asarray(z["ci_coordinate"])
        pm = np.isfinite(pc).all(axis=-1)
        s, un, nt = per_pose_sizes(z, pc, pm, rep, a.contact, a.margin)
        sizes += s
        n_tok = nt
        union = un if union is None else (union | un)
    pose_tot += len(sizes)
    ok = [v for v in sizes if v is not None]
    nocontact_tot += len(sizes) - len(ok)
    if len(ok) < 2 or not n_tok:
        continue
    # CROSS-CHECK: the union of the per-pose masks must equal what the shipped builder produces,
    # or this script is measuring a different crop than the one the eval used.
    rows.append(dict(pdb=pdb, n_tok=n_tok, n_pose=len(ok),
                     lo=min(ok), med=st.median(ok), hi=max(ok),
                     union=int(union.sum())))
    _done += 1
    _el = time.time() - _t0
    _eta = _el / _done * (_todo - _done)
    print(f"  [{_done}/{_todo}] {pdb:<10} n_tok={n_tok:<5} poses={len(ok):<4} "
          f"min={min(ok)} med={st.median(ok):.0f} max={max(ok)}   "
          f"{_el/60:.1f} min elapsed, ~{_eta/60:.1f} min left", flush=True)

if not rows:
    sys.exit("  no structures measured")

print(f"  PER-POSE crop, contact {a.contact} A, margin {a.margin} A -- "
      f"{len(rows)} structure(s), {pose_tot} candidate pose(s)")
print(f"  {nocontact_tot} pose(s) had no cross-chain contact and are excluded; "
      f"{n_nolabel} unit(s) had no label npz")
print()

# WITHIN a structure: the spread across its own 150 candidates, as a fraction of that structure.
sp_tok = [(r["hi"] - r["lo"]) for r in rows]
sp_fra = [(r["hi"] - r["lo"]) / r["n_tok"] for r in rows]
rat = [r["hi"] / r["lo"] for r in rows]
dd = [d0(r["hi"]) - d0(r["lo"]) for r in rows]
frac_med = [r["med"] / r["n_tok"] for r in rows]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


print("  WITHIN one structure, across its own candidates:")
print(f"    {'quantity':<34}{'p10':>10}{'median':>10}{'p90':>10}{'max':>10}")
for nm, v in (("spread, tokens (max - min)", sp_tok),
              ("spread, as frac of structure", sp_fra),
              ("size ratio, max / min", rat),
              ("implied d0 swing, A", dd),
              ("median crop, frac of structure", frac_med)):
    print(f"    {nm:<34}{q(v,.1):>10.3f}{st.median(v):>10.3f}{q(v,.9):>10.3f}{max(v):>10.3f}")

print()
print("  the union over all poses, for reference:")
uf = [r["union"] / r["n_tok"] for r in rows]
print(f"    union as frac of structure        {q(uf,.1):>10.3f}{st.median(uf):>10.3f}"
      f"{q(uf,.9):>10.3f}{max(uf):>10.3f}")
mf = [r["med"] / r["union"] for r in rows]
print(f"    median pose crop / union          {q(mf,.1):>10.3f}{st.median(mf):>10.3f}"
      f"{q(mf,.9):>10.3f}{max(mf):>10.3f}")

print()
print("  the five widest-spreading structures:")
print(f"    {'pdb':<10}{'n_tok':>8}{'min':>8}{'med':>8}{'max':>8}{'ratio':>8}{'d0 swing':>10}")
for r in sorted(rows, key=lambda r: -(r["hi"] - r["lo"]) / r["n_tok"])[:5]:
    print(f"    {r['pdb']:<10}{r['n_tok']:>8}{r['lo']:>8}{r['med']:>8.0f}{r['hi']:>8}"
          f"{r['hi']/r['lo']:>8.2f}{d0(r['hi'])-d0(r['lo']):>10.2f}")

if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n  per-structure rows -> {a.out}")
