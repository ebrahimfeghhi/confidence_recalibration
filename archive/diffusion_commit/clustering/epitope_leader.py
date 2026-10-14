#!/usr/bin/env python
"""Can a DIVERSITY shortlist put a good seed in the top 10, where ipTM manages only 44%?

THE PROBLEM. On 91 confident decoys that do contain an acceptable pose somewhere in their 50 seeds,
ranking seeds by ipTM puts a good seed in the top 10 just 44% of the time, and in the top 1 only 5%.
That is because the most confident seeds are the same dominant wrong basin over and over -- ranking by
confidence selects for agreement with the majority answer, which is exactly the failure mode.

THE IDEA. Shortlist for COVERAGE instead of confidence: cluster the seeds by which antigen surface
they bind, and take one representative per cluster. If 250 poses occupy only a handful of basins, ten
leaders should cover nearly all of them.

WHY EPITOPE SETS AND NOT POSE-POSE DockQ. DockQ between every pair of 250 poses is 31k OST calls per
interface, hours each. Two earlier results make the epitope a good stand-in: across 250 poses the
folds and interface patches are essentially identical (lDDT SD 0.007, patch RMSD 0.10 A) so only the
docking SITE moves; and the epitope explains 82% of the DockQ gain from seed scaling. So clustering on
"which antigen residues are contacted" should recover the same basins at O(n) cost instead of O(n^2).

LABEL-FREE. The epitope of a POSE is computed from the pose itself -- antigen residues within 5 A of
antibody. Nothing here reads the native structure or any DockQ. That matters: epitope_overlap_pxm.csv
holds overlap WITH THE NATIVE, and using it would leak the answer into the shortlist.

Usage: epitope_leader.py --n 15 [--thresh 0.5] [--k 10]
"""
import argparse
import collections
import csv
import glob
import os
import sys

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")
sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402  (needs the path insert above)
from fmt import table

REPO = str(paths.DATA_DIR)
POOL = f"{REPO}/pxm2025/dockq_pxm2025.csv"
ROLES = f"{REPO}/pxm2025/chain_roles.csv"


def pose_chains(path):
    """Polymer chains of a PREDICTED pose keyed by label_asym_id -> (coords, residue index).

    Mirrors scripts/decoy_set.py:native_chains, which reads label_asym_id explicitly rather than
    trusting gemmi's chain naming -- the two disagree for several complexes and a hand-rolled lookup
    silently compares the wrong chains.
    """
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB")]
    D = collections.defaultdict(lambda: ([], []))
    for a, s, x, y, z, g in zip(*cols):
        if g != "ATOM":
            continue
        D[a][0].append([float(x), float(y), float(z)])
        D[a][1].append(s)
    if not D:
        raise ValueError(f"{path}: no polymer atoms parsed")
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in D.items()}


def epitope(path, ab, ag, cutoff=5.0):
    """Antigen residues within cutoff of the antibody, as a set. Raises if a chain is absent.

    KD-TREE, NOT A BROADCAST DISTANCE MATRIX. The obvious
        np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=-1) < cutoff
    materialises an (n_ab, n_ag, 3) intermediate -- ~150 MB for a 600-residue complex -- allocated and
    discarded once per pose, thousands of times. A tree query with a fixed cutoff returns the same
    contacts in milliseconds and allocates almost nothing.
    """
    D = pose_chains(path)
    for c in (ab, ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, _ = D[ab]
    XB, IB = D[ag]
    pairs = cKDTree(XA).query_ball_tree(cKDTree(XB), r=cutoff)
    hit = {j for js in pairs for j in js}
    return {IB[j] for j in hit}


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def leaders(items, thresh):
    """Leader clustering: first item starts a cluster, each next joins the first leader it resembles.

    Deliberately order-dependent and single-pass -- the point is a cheap diverse shortlist, not an
    optimal clustering.
    """
    out = []
    for key, s in items:
        for L in out:
            if jac(s, L["set"]) >= thresh:
                L["members"].append(key)
                break
        else:
            out.append({"leader": key, "set": s, "members": [key]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="how many interfaces to test")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 10],
                    help="shortlist sizes, all evaluated from ONE clustering pass")
    A = ap.parse_args()

    # chain_roles.csv keys interfaces as "9jo7_A_B" (chains underscore-separated) while the pool csv
    # gives "9jo7_AB". Normalise to the pool form; the ab/ag assignment is what we actually need and it
    # must come from this file rather than being guessed -- 45% of interfaces were once measured on the
    # wrong chains by assuming an order.
    roles = {}
    for r in csv.DictReader(open(ROLES)):
        if r["interface"] == "interface":
            continue
        roles[f'{r["pdb"]}_{r["chain_1"]}{r["chain_2"]}'] = (r["ab_chain"], r["ag_chain"])

    P = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(POOL)):
        P[f"{r['pdb']}_{r['chain1']}{r['chain2']}"][r["seed"]].append(
            (int(r["sample"]), float(r["pair_iptm"]), float(r["dockq"])))

    # the confident decoys that DO contain an acceptable pose -- the set the shortlist has to find
    cands = []
    for iface, seeds in P.items():
        if len(seeds) < 50 or iface not in roles:
            continue
        allp = [p for v in seeds.values() for p in v]
        if max(allp, key=lambda x: x[1])[2] >= 0.23:
            continue
        if not any(d >= 0.23 for _, _, d in allp):
            continue
        cands.append(iface)
    cands.sort()
    print(f"  confident decoys available: {len(cands)}; testing {min(A.n, len(cands))}")

    rows = []
    for iface in cands[:A.n]:
        pdb = iface.split("_")[0]
        ab, ag = roles[iface]
        seeds = P[iface]
        # one representative pose per seed (its best-ipTM pose) keeps this O(seeds) not O(poses)
        reps = {}
        for s, v in seeds.items():
            samp = max(v, key=lambda x: x[1])[0]
            f = paths.pool_pose(pdb, s, samp)
            if os.path.exists(f):
                reps[s] = f
        if len(reps) < 40:
            print(f"    {iface}: only {len(reps)} pose files, skipping")
            continue
        try:
            epis = [(s, epitope(f, ab, ag)) for s, f in sorted(reps.items())]
        except (KeyError, ValueError) as e:
            print(f"    {iface}: {str(e)[:90]}")
            continue

        cl = leaders(epis, A.thresh)
        good = {s for s, v in seeds.items() if any(d >= 0.23 for _, _, d in v)}
        # ipTM shortlist for comparison
        by_iptm = sorted(seeds, key=lambda s: -max(x[1] for x in seeds[s]))
        # diversity shortlist: cluster leaders, biggest clusters first
        by_div = [L["leader"] for L in sorted(cl, key=lambda L: -len(L["members"]))]
        row = [iface, len(reps), len(cl), len(good)]
        for k in A.k:
            row.append("yes" if any(s in good for s in by_iptm[:k]) else "no")
            row.append("yes" if any(s in good for s in by_div[:k]) else "no")
        rows.append(row)
        print(f"    {iface}: {len(cl)} clusters, {len(good)}/{len(reps)} good seeds", flush=True)

    if not rows:
        print("  nothing measured")
        return 1
    print()
    hdr = ["interface", "seeds", "clusters", "good seeds"]
    for k in A.k:
        hdr += [f"ipTM@{k}", f"div@{k}"]
    print(table(hdr, rows, align="l r r r" + " l l" * len(A.k)))
    n = len(rows)
    print(f"\n  n = {n} interfaces, Jaccard threshold {A.thresh}")
    for i, k in enumerate(A.k):
        ip = sum(1 for r in rows if r[4 + 2 * i] == "yes")
        dv = sum(1 for r in rows if r[5 + 2 * i] == "yes")
        print(f"    shortlist {k:2d}:  ipTM {ip:3d}/{n} ({100*ip/n:3.0f}%)   "
              f"diversity {dv:3d}/{n} ({100*dv/n:3.0f}%)")
    cc = [r[2] for r in rows]
    print(f"    epitope clusters per interface: median {sorted(cc)[len(cc)//2]}, "
          f"range {min(cc)}-{max(cc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
