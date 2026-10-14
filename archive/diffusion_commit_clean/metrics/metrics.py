#!/usr/bin/env python
"""Three ways to measure how similar two poses of the same complex are.

WHY THREE. We want to cluster 250 poses into basins and self-template each basin, so the clustering
metric decides what a "basin" is. Three candidates, and the right one is an empirical question:

    epitope   which ANTIGEN RESIDUES the antibody touches            (a patch)
    fcc       which ANTIBODY-ANTIGEN RESIDUE PAIRS are in contact    (a patch AND a register)
    dockq     the CAPRI measure, computed pose against pose          (the field standard)

They differ in a way we have already been bitten by: two poses can touch the same antigen patch from
opposite directions. Epitope overlap calls those identical; FCC does not, because the pairs differ.
Measured earlier on 250 poses, epitope-Jaccard-to-truth in 0.4-0.6 was LESS often acceptable (9/65)
than 0.2-0.3 (15/18), and it traced to one interface whose poses shared the surface with the wrong
register -- exactly the failure epitope-only similarity cannot see. A 5-pose smoke test on 9s36_AB
showed the same split: one pose shared 11-27% of antigen residues with the other four but had FCC
0.000 against all of them, and pose-pose DockQ agreed with FCC (0.07-0.12), not with epitope Jaccard.

CUTOFFS follow the conventions we would be compared against:
    epitope   4.5 A heavy atom, the SAbDab/AsEP convention
    fcc       5.0 A heavy atom residue-residue pairs, the CAPRI/HADDOCK convention

FCC IS SYMMETRISED HERE. Rodrigues et al. define it as |A and B| / |A|, normalised by the reference,
so FCC(a,b) != FCC(b,a). Clustering needs a symmetric distance, so this returns the Jaccard form
|A and B| / |A or B| over contact pairs. Noted because the asymmetric number is what the FCC papers
report, and the two are not interchangeable.

EVERY FUNCTION TAKES ONE CHAIN PAIR PER POSE, never a single pair for the whole set. On a multi-copy
assembly the model builds several antigen copies and different poses bind different ones -- 23 of 40
8w5d poses are permuted relative to the reference -- so a fixed pair silently compares different
molecules. chain_map.py backfills the per-pose mapping (see its docstring for why it must be anchored
to the ground truth rather than pose-to-pose); items_for_interface() below turns that into the
per-pose chain pairs these functions expect. This mirrors dockq_torch, which has always taken the
pair as an argument for the same reason.

TWO VENVS. The metric functions need torch (Protenix venv); chain_map.py needs pxmeter. Imports here
are lazy so this module loads under either.
"""
import csv
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit/clustering")

import paths  # noqa: E402
from epitope_leader import pose_chains  # noqa: E402  (validated label_asym_id parser)

# 5.0 A for BOTH, deliberately, though the SAbDab/AsEP convention for an epitope is 4.5. The question
# here is which metric defines a basin, not which cutoff is conventional, and a tighter cutoff on
# epitope alone would confound the comparison: if epitope-Jaccard lost, we could not tell whether a
# patch carries less information than a register or whether I had simply been stingier with it. With
# both at 5.0 the two differ only in WHAT the set contains -- residues against residue pairs.
# Concretely it also matters: on ring assemblies like 9kmp the true interface is a 1-2 residue glance
# at ~4.7 A, so at 4.5 those poses would have an EMPTY epitope and score 0 against everything.
EPITOPE_CUTOFF = 5.0
CONTACT_CUTOFF = 5.0


# --- resolving which model chains to compare ---------------------------------------------------

def load_chain_maps(path=None):
    """{(pdb, seed, sample): {ref_chain: model_chain}} from chain_map.py's backfill."""
    path = path or f"{paths.PXM2025_DIR}/chain_maps.csv"
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["chain_mapping"]:            # blank means the mapping failed for that pose
                out[(r["pdb"], r["seed"], int(r["sample"]))] = json.loads(r["chain_mapping"])
    return out


def load_roles(path=None):
    """{"<pdb>_<c1><c2>": (ab_ref_chain, ag_ref_chain)} -- roles are in REFERENCE chain labels.

    Keyed off chain_1/chain_2 to match the pool CSV's interface key. NOT off the `interface` column,
    which already contains the pdb ("8w5d_A_B"), so using it yields "8w5d_8w5d_A_B" and every lookup
    misses.
    """
    path = path or str(paths.PXM2025_CHAIN_ROLES)
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[f"{r['pdb']}_{r['chain_1']}{r['chain_2']}"] = (r["ab_chain"], r["ag_chain"])
    return out


def items_for_interface(iface, seeds_samples, maps, roles):
    """[(pose_path, model_ab, model_ag), ...] for one interface.

    seeds_samples is [(seed, sample), ...]. Poses whose mapping is missing or does not cover both
    reference chains are DROPPED and reported, not silently defaulted to the reference labels -- that
    default is exactly the bug this plumbing exists to prevent.
    """
    pdb = iface.split("_")[0]
    ab_ref, ag_ref = roles[iface]
    items, dropped = [], []
    for seed, sample in seeds_samples:
        m = maps.get((pdb, str(seed), int(sample)))
        if not m or ab_ref not in m or ag_ref not in m:
            dropped.append((seed, sample))
            continue
        items.append((paths.pool_pose(pdb, seed, sample), m[ab_ref], m[ag_ref]))
    return items, dropped


# --- the three metrics -------------------------------------------------------------------------

def epitope_set(path, ab, ag, cutoff=EPITOPE_CUTOFF):
    """Antigen residues with a heavy atom within cutoff of the antibody."""
    D = pose_chains(path)
    for c in (ab, ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, _ = D[ab]
    XB, IB = D[ag]
    hit = {j for js in cKDTree(XA).query_ball_tree(cKDTree(XB), r=cutoff) for j in js}
    return {IB[j] for j in hit}


def contact_pairs(path, ab, ag, cutoff=CONTACT_CUTOFF):
    """(antibody residue, antigen residue) pairs in contact -- the patch AND the register."""
    D = pose_chains(path)
    for c in (ab, ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, IA = D[ab]
    XB, IB = D[ag]
    # query_ball_tree gives, for each antibody atom, the antigen atoms within cutoff; keep the
    # RESIDUE pair rather than the atom pair, which is what FCC counts
    near = cKDTree(XA).query_ball_tree(cKDTree(XB), r=cutoff)
    return {(IA[i], IB[j]) for i, js in enumerate(near) for j in js}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def set_matrix(items, kind):
    """Symmetric similarity matrix from a per-pose SET -- epitope residues or contact pairs.

    Sets are computed once per pose (N parses), then compared pairwise (N^2/2 cheap set ops). Doing
    it the other way round would re-parse the same file N times.
    """
    fn = epitope_set if kind == "epitope" else contact_pairs
    sets = [fn(p, ab, ag) for p, ab, ag in items]
    n = len(sets)
    M = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = jaccard(sets[i], sets[j])
    return M, [len(s) for s in sets]


def dockq_pose_matrix(items, device="cuda"):
    """Pose-against-pose DockQ, symmetrised.

    dockq_matrix returns [i, j] = DockQ(reference=i, model=j), which is not symmetric because Fnat
    divides by the reference's contact count. The average of the two directions is what we cluster on.
    """
    from dockq_torch import dockq_matrix
    M = np.asarray(dockq_matrix(items, device=device, verbose=False), dtype=np.float32)
    return (M + M.T) / 2.0


def all_metrics(items, device="cuda"):
    """{name: NxN similarity} for the three candidates, plus the per-pose set sizes."""
    out, sizes = {}, {}
    for kind in ("epitope", "fcc"):
        out[kind], sizes[kind] = set_matrix(items, kind)
    out["dockq"] = dockq_pose_matrix(items, device=device)
    return out, sizes
