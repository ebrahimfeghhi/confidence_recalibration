#!/usr/bin/env python
"""WHERE the masked PAE term's supervision actually sits, relative to the binding interface.

WHY THIS EXISTS. `--pae-scored-pairs` restricts the PAE term to the scored chain pairs' CROSS-CHAIN
entries, which is what section 12 assumed when it blamed the uniform relative-iTM term for the
top-of-ranking dissociation. But "cross-chain" is not "interface": for a 300+300 residue complex
the mask admits ~90,000 ordered pairs, and the overwhelming majority are tens of angstroms apart.
Section 9's crop removed ~47% of tokens -- the ones FAR from the interface -- so it did not only
change the input distribution, it silently reweighted the training objective toward interface
geometry. This script measures that reweighting instead of assuming it.

THE MASK IS REBUILT WITH THE TRAINER'S OWN EXPRESSION, not an approximation of it: the block
    (fa[:, None] == x) & (aa[None, :] == y) | (fa[:, None] == y) & (aa[None, :] == x)
copied from phase_k_train._set_pae_mask, over the same chain_roles -> asym join and the same
pool file. If this script and the trainer disagree about which pairs are supervised, the numbers
here describe nothing.

GROUND TRUTH, AND SAID SO. Distances come from `label_coordinate`, the same array
interface_crop_tokens reads, so every number here is an ORACLE diagnostic. That is legitimate for
deciding how to WEIGHT A TRAINING LOSS -- training-time supervision may use labels -- but it is
not available at inference, so nothing here can become part of a selector.

THREE THINGS IT REPORTS, because they answer three different questions:
  1. SUPERVISION SHARE -- what fraction of the mask's pairs the crop would have kept. This is the
     size of the objective change cropping made.
  2. ipTM WEIGHT SHARE -- each pair's TM weight 1/(1+(d/d0)^2) at its TRUE distance, normalised
     per anchor row. ipTM is a mean of these over the row, so this says how much of the quantity
     SELECTION READS sits near the interface. The claim "even distant pairs contribute to ipTM" is
     true; this measures by how much.
  3. The gap between 1 and 2 IS the mismatch. A uniform PAE term spends its capacity in proportion
     to (1) while selection reads (2).
"""
import argparse
import csv
import os
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="training/full_train/pool_winnable.csv")
ap.add_argument("--roles", default="training/fold/chain_roles.csv")
ap.add_argument("--trunks", default="training/fold/trunks")
ap.add_argument("--labels", default="training/fold/labels")
ap.add_argument("--contact", type=float, default=5.0, help="heavy-atom contact, as the crop uses")
ap.add_argument("--margin", type=float, default=10.0, help="the crop's margin, --crop-margin")
ap.add_argument("--n", type=int, default=40, help="structures to measure")
ap.add_argument("--pose", type=int, default=0)
a = ap.parse_args()

roles = {}
with open(a.roles) as fh:
    for r in csv.DictReader(fh):
        roles.setdefault(r["pdb"], {})[r["ref_chain"]] = r["model_chain"]

pairs_by_pdb = {}
with open(a.pool) as fh:
    for r in csv.DictReader(fh):
        m1 = roles.get(r["pdb"], {}).get(r["chain1"])
        m2 = roles.get(r["pdb"], {}).get(r["chain2"])
        if m1 is None or m2 is None:
            continue
        x, y = ord(m1) - ord("A"), ord(m2) - ord("A")
        pairs_by_pdb.setdefault(r["pdb"], set()).add((min(x, y), max(x, y)))

def d0_of(n):
    # Protenix's ipTM normalisation, guarded the same way.
    if n <= 15:
        return 0.5
    return 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8

rows = []
done = 0
for pdb in sorted(pairs_by_pdb):
    if done >= a.n:
        break
    tdir = os.path.join(a.trunks, pdb)
    ldir = os.path.join(a.labels, pdb)
    if not os.path.isdir(tdir) or not os.path.isdir(ldir):
        continue
    units = sorted(f for f in os.listdir(tdir) if f.endswith(".npz"))
    if not units:
        continue
    u = units[0]
    try:
        zc = np.load(os.path.join(tdir, u), allow_pickle=True)
        lab = np.load(os.path.join(ldir, u), allow_pickle=True)
    except (OSError, ValueError):
        continue
    asym = np.asarray(zc["ci_feat_asym_id"]).reshape(-1)
    hasf = np.asarray(zc["ci_feat_has_frame"]).reshape(-1).astype(bool)
    a2t = np.asarray(zc["ci_feat_atom_to_token_idx"]).reshape(-1)
    rep = np.asarray(zc["ci_feat_distogram_rep_atom_mask"]).reshape(-1).astype(bool)
    xyz = np.asarray(lab["label_coordinate"])[a.pose]
    amask = np.asarray(lab["label_coordinate_mask"])[a.pose].astype(bool)
    if xyz.shape[0] != a2t.shape[0]:
        continue

    # token representative coordinates, for the pair distances
    tok_xyz = np.full((asym.shape[0], 3), np.nan, dtype=np.float64)
    sel = rep & amask
    tok_xyz[a2t[sel]] = xyz[sel]

    # INTERFACE TOKENS BY HEAVY-ATOM CONTACT, the crop's own definition.
    atom_ch = asym[a2t]
    iface_tok = np.zeros(asym.shape[0], dtype=bool)
    chains = np.unique(atom_ch[amask])
    for c in chains:
        m1 = amask & (atom_ch == c)
        m2 = amask & (atom_ch != c)
        if not m1.any() or not m2.any():
            continue
        t = cKDTree(xyz[m2])
        hit = t.query_ball_point(xyz[m1], a.contact)
        idx = np.where(m1)[0]
        for k, h in enumerate(hit):
            if h:
                iface_tok[a2t[idx[k]]] = True
    if not iface_tok.any():
        continue
    # tokens the CROP would keep: representative atom within margin of an interface token
    ok = ~np.isnan(tok_xyz[:, 0])
    kept_tok = np.zeros(asym.shape[0], dtype=bool)
    if (iface_tok & ok).any():
        t = cKDTree(tok_xyz[iface_tok & ok])
        near = t.query_ball_point(tok_xyz[ok], a.margin)
        kept_tok[np.where(ok)[0]] = np.array([len(h) > 0 for h in near])

    fa = asym[hasf]
    for (x, y) in sorted(pairs_by_pdb[pdb]):
        blk = (((fa[:, None] == x) & (asym[None, :] == y))
               | ((fa[:, None] == y) & (asym[None, :] == x)))
        if not blk.any():
            continue
        ri, cj = np.where(blk)
        gi = np.where(hasf)[0][ri]
        d = np.linalg.norm(tok_xyz[gi] - tok_xyz[cj], axis=1)
        fin = np.isfinite(d)
        if fin.sum() < 10:
            continue
        ri, cj, gi, d = ri[fin], cj[fin], gi[fin], d[fin]
        # 1. supervision share the crop kept
        in_crop = kept_tok[gi] & kept_tok[cj]
        # 2. ipTM weight share, per anchor row then averaged over rows
        n_pair_tok = int(((asym == x) | (asym == y)).sum())
        w = 1.0 / (1.0 + (d / d0_of(n_pair_tok)) ** 2)
        share_near, share_all = 0.0, 0.0
        for r_ in np.unique(ri):
            s = ri == r_
            tot = w[s].sum()
            if tot > 0:
                share_near += w[s][in_crop[s]].sum() / tot
                share_all += 1.0
        # THE BAND TABLE IS THE DECISIVE COMPARISON. "45% of pairs are in the crop" and "52% of
        # the TM mass is in the crop" look close, but the crop's 10 A margin is a coarse cut. The
        # mismatch is better read as: what share of SUPERVISED PAIRS, and what share of the TM
        # MASS selection reads, sit in each distance band. A uniform term spends capacity in
        # proportion to the first column and selection reads the second.
        bands = [(0, 10), (10, 20), (20, 30), (30, 1e9)]
        bshare, bmass = [], []
        for lo, hi in bands:
            s_ = (d >= lo) & (d < hi)
            bshare.append(float(s_.mean()))
            bmass.append(float(w[s_].sum() / max(w.sum(), 1e-12)))
        rows.append(dict(
            bshare=bshare, bmass=bmass,
            pdb=pdb, pair=f"{x}{y}", n_pairs=int(fin.sum()),
            frac_pairs_in_crop=float(in_crop.mean()),
            med_d=float(np.median(d)), frac_d_gt_30=float((d > 30).mean()),
            iptm_weight_in_crop=float(share_near / max(share_all, 1)),
        ))
    done += 1

if not rows:
    raise SystemExit("no structure measured -- check --trunks/--labels/--roles paths")

def med(k):
    return float(np.median([r[k] for r in rows]))

print(f"{len(rows)} scored chain pair(s) over {done} structure(s), pose {a.pose}")
print(f"crop definition: contact {a.contact} A, margin {a.margin} A (the arm's own values)\n")
print(f"  median supervised cross-chain pairs per interface   {med('n_pairs'):,.0f}")
print(f"  median inter-token distance of a supervised pair    {med('med_d'):.1f} A")
print(f"  median share of supervised pairs beyond 30 A        {100*med('frac_d_gt_30'):.1f}%")
print()
print(f"  1. SUPERVISION the crop kept   {100*med('frac_pairs_in_crop'):.1f}% of mask pairs")
print(f"  2. ipTM WEIGHT in those pairs  {100*med('iptm_weight_in_crop'):.1f}% of the row's TM mass")
print()
print("  (1) is what a uniform PAE term spends on interface-proximal geometry.")
print("  (2) is what selection actually reads there. The gap is the mismatch.")
print()
print("  BY DISTANCE BAND (median over interfaces):")
print("    band        share of supervised pairs   share of ipTM TM mass   ratio")
labels = ["0-10 A ", "10-20 A", "20-30 A", ">30 A  "]
import numpy as _np
for i, lb in enumerate(labels):
    sp = float(_np.median([r["bshare"][i] for r in rows]))
    mm = float(_np.median([r["bmass"][i] for r in rows]))
    rt = (mm / sp) if sp > 1e-9 else float("nan")
    print(f"    {lb}       {100*sp:21.1f}%   {100*mm:19.1f}%   {rt:5.2f}x")
print()
print("  A ratio below 1 means the uniform PAE term spends MORE capacity there than")
print("  selection reads; above 1 means it spends LESS than selection reads.")
