#!/usr/bin/env python
"""THE CONTROL THAT HAS TO RUN BEFORE PREDICTED-POSE CROPPING IS SCORED ON A GPU.

THE PROPOSAL. Crop each candidate pose to its own PREDICTED antibody-antigen interface (5 A
contacts, 10 A margin) and score the fine-tuned head on that crop. Unlike section 9's crop this
uses no labels, so it could run at inference. The question is whether it recovers the +0.0376
held-out gain the ground-truth crop produced.

THE CONFOUND, which is why this script exists. ipTM is a mean of TM weights 1/(1+(d/d0)^2) over a
chain pair's tokens, and PROTENIX SETS d0 FROM THE TOKEN COUNT: d0 = 1.24*(n-15)^(1/3) - 1.8. A
per-pose crop changes n per pose, so it changes d0 per pose, so it changes the scoring function
itself between the candidates being compared:

    n=250 -> d0=5.85 A, TM weight at 8 A = 0.349
    n=350 -> d0=6.81 A, TM weight at 8 A = 0.420      a 20% relative shift

And the shift is NOT random with respect to pose quality: a near-native pose has more contacts, so
a bigger crop, so a bigger d0, so a MORE GENEROUS weighting -- higher ipTM for geometric reasons
that have nothing to do with the trained head. A gain measured this way could be entirely an
artifact of crop size, with the head contributing nothing.

WHAT THIS MEASURES, on CPU, with no GPU and no model:
  1. within-interface Spearman(predicted crop size, DockQ)
  2. within-interface Spearman(predicted contact count, DockQ)
  3. mean selected DockQ if you select by crop size ALONE
Compare (3) against shipped (0.5834) and the ground-truth-crop arm (+0.0376). If selecting by crop
size alone already moves DockQ, then any gain from predicted-pose cropping must be reported net of
this baseline, and the honest finding would be "predicted contact count is a pose selector",
not "the confidence head transfers".

NO LABELS ARE READ FOR THE CROP. Coordinates come from `ci_coordinate` in the trunk cache, which
is what the fold PREDICTED for that pose. DockQ comes from results.csv only to score the result.
"""
import argparse
import csv
import collections
import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="training/valfull/pool.csv")
ap.add_argument("--roles", default="validation/fold/chain_roles.csv")
ap.add_argument("--trunks", default="validation/fold/trunks")
ap.add_argument("--results", default="validation/fold/results.csv")
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margin", type=float, default=10.0)
ap.add_argument("--acc", type=float, default=0.23)
ap.add_argument("--structs", type=int, default=12)
ap.add_argument("--seeds", type=int, default=30)
a = ap.parse_args()

roles = {}
for r in csv.DictReader(open(a.roles)):
    roles.setdefault(r["pdb"], {})[r["ref_chain"]] = r["model_chain"]

pool = collections.defaultdict(list)
for r in csv.DictReader(open(a.pool)):
    m1 = roles.get(r["pdb"], {}).get(r["chain1"])
    m2 = roles.get(r["pdb"], {}).get(r["chain2"])
    if m1 is None or m2 is None:
        continue
    x, y = ord(m1) - ord("A"), ord(m2) - ord("A")
    pool[r["pdb"]].append((f"{r['pdb']}_{r['chain1']}{r['chain2']}", min(x, y), max(x, y)))

dockq = {}
for r in csv.DictReader(open(a.results)):
    try:
        dockq[(r["iface"], int(r["seed"]), int(r["sample"]))] = float(r["dockq"])
    except (TypeError, ValueError):
        pass

rows = collections.defaultdict(list)   # iface -> [(crop_tokens, n_contacts, dockq)]
done = 0
for pdb in sorted(pool):
    if done >= a.structs:
        break
    tdir = os.path.join(a.trunks, pdb)
    if not os.path.isdir(tdir):
        continue
    units = sorted(f for f in os.listdir(tdir) if f.endswith(".npz"))[:a.seeds]
    if not units:
        continue
    got = False
    for u in units:
        seed = int(u.split("_")[-1].split(".")[0])
        try:
            z = np.load(os.path.join(tdir, u), allow_pickle=True)
            asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
            a2t = np.asarray(z["ci_feat_atom_to_token_idx"]).reshape(-1)
            rep = np.asarray(z["ci_feat_distogram_rep_atom_mask"]).reshape(-1).astype(bool)
            xyz_all = np.asarray(z["ci_coordinate"])        # PREDICTED, per pose
        except (OSError, ValueError, KeyError):
            continue
        atom_ch = asym[a2t]
        for pi in range(xyz_all.shape[0]):
            xyz = xyz_all[pi]
            if not np.isfinite(xyz).all():
                continue
            tok_xyz = np.full((asym.shape[0], 3), np.nan)
            tok_xyz[a2t[rep]] = xyz[rep]
            ok = ~np.isnan(tok_xyz[:, 0])
            for iface, x, y in pool[pdb]:
                dq = dockq.get((iface, seed, pi))
                if dq is None:
                    continue
                m1 = atom_ch == x
                m2 = atom_ch == y
                if not m1.any() or not m2.any():
                    continue
                t = cKDTree(xyz[m2])
                hit = t.query_ball_point(xyz[m1], a.contact)
                idx = np.where(m1)[0]
                ctok = set()
                ncon = 0
                for k, h in enumerate(hit):
                    if h:
                        ctok.add(int(a2t[idx[k]]))
                        ncon += len(h)
                        for j in h:
                            ctok.add(int(a2t[np.where(m2)[0][j]]))
                if not ctok:
                    rows[iface].append((0, 0, dq)); got = True; continue
                cl = sorted(c for c in ctok if ok[c])
                if not cl:
                    rows[iface].append((0, ncon, dq)); got = True; continue
                tt = cKDTree(tok_xyz[cl])
                near = tt.query_ball_point(tok_xyz[ok], a.margin)
                kept = int(sum(1 for h in near if h))
                rows[iface].append((kept, ncon, dq))
                got = True
    if got:
        done += 1

rows = {k: v for k, v in rows.items() if len(v) >= 10}
if not rows:
    raise SystemExit("no interface measured -- check --trunks/--results paths")

def d0(n):
    return 0.5 if n <= 15 else 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8

rho_size, rho_con, sel_size, sel_con, best, spread = [], [], [], [], [], []
for k, v in rows.items():
    sz = np.array([p[0] for p in v], float)
    nc = np.array([p[1] for p in v], float)
    dq = np.array([p[2] for p in v], float)
    if sz.std() > 0:
        rho_size.append(spearmanr(sz, dq).correlation)
    if nc.std() > 0:
        rho_con.append(spearmanr(nc, dq).correlation)
    sel_size.append(dq[int(np.argmax(sz))])
    sel_con.append(dq[int(np.argmax(nc))])
    best.append(dq.max())
    spread.append((d0(sz.min()), d0(sz.max())))

rho_size = [x for x in rho_size if x == x]
rho_con = [x for x in rho_con if x == x]
print(f"{len(rows)} interface(s), {sum(len(v) for v in rows.values())} pose(s), "
      f"{done} structure(s)")
print(f"predicted crop: {a.contact} A contacts, {a.margin} A margin -- NO LABELS\n")
print(f"  median within-interface Spearman(crop size,     DockQ)  {np.median(rho_size):+.4f}")
print(f"  median within-interface Spearman(contact count, DockQ)  {np.median(rho_con):+.4f}")
print()
print(f"  mean DockQ selecting by CROP SIZE alone       {np.mean(sel_size):.4f}")
print(f"  mean DockQ selecting by CONTACT COUNT alone   {np.mean(sel_con):.4f}")
print(f"  frac acceptable, crop size alone              {np.mean([d >= a.acc for d in sel_size]):.4f}")
print(f"  oracle (best pose per interface)              {np.mean(best):.4f}")
print()
allsz = np.array([p[0] for v in rows.values() for p in v], float)
print(f"  CROP SIZE DISTRIBUTION over all poses: min {allsz.min():.0f}  "
      f"p10 {np.percentile(allsz,10):.0f}  median {np.median(allsz):.0f}  "
      f"p90 {np.percentile(allsz,90):.0f}  max {allsz.max():.0f} tokens")
print(f"  poses with NO predicted contact (degenerate crop): "
      f"{int((allsz==0).sum())} of {allsz.size} ({100*(allsz==0).mean():.1f}%)")
print()
lo = np.median([s[0] for s in spread]); hi = np.median([s[1] for s in spread])
print(f"  d0 ACROSS POSES OF ONE INTERFACE: median range {lo:.2f} to {hi:.2f} A")
print(f"  TM weight at 8 A therefore moves {1/(1+(8/lo)**2):.3f} -> {1/(1+(8/hi)**2):.3f}")
print("  -- that is the scoring function changing BETWEEN the candidates being ranked.")
