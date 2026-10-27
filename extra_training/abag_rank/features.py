#!/usr/bin/env python
"""ABAG-Rank's per-pair features, ported from preprocess_shards.py. CPU only.

PORTED STEP FOR STEP from upstream's process_single_sample (preprocess_shards.py:133-275), in the
same ORDER, because the order changes the result: the X filter runs BEFORE the distance cutoff and
before the zero-contact fallback, and `pae_col_mean` is indexed through every mask the pair array
is, so a reordering silently misaligns the centring.

    1. drop pairs whose residue is 'X', and remap indices
    2. CA-CA <= 12 A; if nothing qualifies, keep the single closest pair
    3. sort by CA distance ascending, truncate to max_n (upstream default 1500)
    4. pae_centered = pae - pae_col_mean, the per-complex mean over ALL poses
    5. triangular positional encoding of (i, j)
    6. one-hot of both residues, 21 wide (20 + X)
    7. the label is log(p/(1-p)) of DockQ, not DockQ

WHAT IS NOT PORTED: their HDF5 schema, shard packer, manifest and train/test split. None of those
changes a tensor. See abag_rank/README.md.

pae_col_mean COMES FROM THE DENSE MEAN, not from the sparse pairs. Upstream's stage 1 writes every
inter-chain pair with no distance cutoff and stage 3 means over those, so the baseline includes
poses in which a pair is far apart. Measured on 22gi, taking the mean over only the sparse pairs
gives 1.532 A against the true 6.858 A -- a 4.5x error in the quantity every centred feature is
measured against. merge_pae_mean.py produces the dense version.
"""
import numpy as np

RESIDUE_TYPES = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
RES_ID = {r: i for i, r in enumerate(RESIDUE_TYPES)}
ONE_HOT_DIM = len(RESIDUE_TYPES)          # 21
CUTOFF = 12.0
MAX_N = 1500
EPS = 1e-6


def cantor(i, j):
    """Upstream's pairing function: a unique integer per (i, j)."""
    s = i + j
    return (s * (s + 1)) // 2 + j


def logit_label(dockq):
    """DockQ -> log(p/(1-p)). Upstream step 10, clipped away from 0 and 1."""
    p = np.clip(float(dockq), EPS, 1.0 - EPS)
    return float(np.log(p / (1.0 - p)))


def build(pose, pae_col_mean, letters, keep_chains=None, cutoff=CUTOFF, max_n=MAX_N):
    """[N, F] features for one pose, plus the pair count.

    `pose` is one npz from phase_k_eval --dump-pae. `pae_col_mean` is the dense [T, T] from
    merge_pae_mean.py. `letters` is [T] one-letter codes from restypes.py. `keep_chains`, when
    given, is the (asym_i, asym_j) of the interface being built -- an interface is a CHAIN PAIR,
    and 382 of our 487 training structures carry more than one, so a pose's pairs must be
    restricted to the chains of the interface this example is about.
    """
    ii = np.asarray(pose["inter_idx"], dtype=np.int64)
    jj = np.asarray(pose["inter_jdx"], dtype=np.int64)
    pae = np.asarray(pose["interchain_pae_vals"], dtype=np.float64)
    ca = np.asarray(pose["interchain_ca_distances"], dtype=np.float64)
    ai = np.asarray(pose["asym_i"], dtype=np.int64)
    aj = np.asarray(pose["asym_j"], dtype=np.int64)

    if keep_chains is not None:
        c1, c2 = keep_chains
        m = ((ai == c1) & (aj == c2)) | ((ai == c2) & (aj == c1))
        ii, jj, pae, ca = ii[m], jj[m], pae[m], ca[m]

    # 1. X filter. The dump already excluded X when choosing its pairs, but an interface subset
    #    is taken here, so it is applied again rather than assumed -- cheap, and a silent X would
    #    put a 21st one-hot column into a feature the model reads as a real residue.
    ok = np.array([letters[i] != "X" and letters[j] != "X" for i, j in zip(ii, jj)], dtype=bool)
    ii, jj, pae, ca = ii[ok], jj[ok], pae[ok], ca[ok]
    if len(ca) == 0:
        return np.zeros((0, 5 + 2 * ONE_HOT_DIM), dtype=np.float32), 0

    # 2. cutoff, with upstream's fallback to the single closest pair
    m = ca <= cutoff
    if not m.any():
        m = np.zeros_like(ca, dtype=bool)
        m[int(np.argmin(ca))] = True
    ii, jj, pae, ca = ii[m], jj[m], pae[m], ca[m]

    # 3. sort by distance, truncate
    order = np.argsort(ca, kind="stable")[:max_n]
    ii, jj, pae, ca = ii[order], jj[order], pae[order], ca[order]

    # 4. centring, indexed through the SAME pairs
    colmean = pae_col_mean[ii, jj].astype(np.float64)
    centered = pae - colmean

    # 5. positional encoding -- upstream's apply_triangular_encoding, literally.
    #
    #    TWO THINGS THAT ARE NOT T-1. Upstream normalises the Cantor code by the largest index
    #    PRESENT AMONG THE SELECTED PAIRS, and the index distance by that selection's OWN maximum
    #    distance -- not by the sequence length. Using T-1 for either gives systematically
    #    different numbers on every complex, and more different the more of the sequence sits
    #    outside the interface. Caught in review.
    #
    #    AND THE INDICES ARE REMAPPED AFTER THE X FILTER. Upstream drops X residues and
    #    renumbers what remains contiguously, so a pair's position is its position among KNOWN
    #    residues. ii/jj here are original token indices; rii/rjj are the remapped ones. The
    #    positional features use the remapped pair, pae_col_mean uses the ORIGINAL pair, because
    #    that matrix is in original token coordinates -- mixing them would be an alignment bug of
    #    exactly the kind this file exists to avoid.
    keep_tok = np.array([str(c) != "X" for c in letters], dtype=bool)
    remap = np.full(len(letters), -1, dtype=np.int64)
    remap[keep_tok] = np.arange(int(keep_tok.sum()))
    rii, rjj = remap[ii], remap[jj]
    mx = int(max(rii.max(), rjj.max())) if len(rii) else 0
    k = cantor(rii, rjj).astype(np.float64)
    if mx > 1:
        kmax = ((2 * (mx - 1)) * (2 * (mx - 1) + 1)) / 2 + (mx - 1)
        cantor_norm = k / kmax
    else:
        cantor_norm = np.zeros_like(k)
    dist = np.abs(rii - rjj).astype(np.float64)
    dmax = dist.max() if len(dist) else 0.0
    dist_norm = dist / dmax if dmax > 0 else np.zeros_like(dist)

    # 6. one-hots
    rid = np.array([RES_ID.get(str(c), RES_ID["X"]) for c in letters], dtype=np.int64)
    eye = np.eye(ONE_HOT_DIM, dtype=np.float64)
    oh_i, oh_j = eye[rid[ii]], eye[rid[jj]]

    feats = np.concatenate([
        cantor_norm[:, None], dist_norm[:, None], colmean[:, None],
        centered[:, None], ca[:, None], oh_i, oh_j], axis=1)
    return feats.astype(np.float32), len(ca)


FEATURE_DIM = 5 + 2 * ONE_HOT_DIM         # 47
FEATURE_NAMES = (["cantor_norm", "dist_norm", "pae_col_mean", "pae_centered", "ca_distance"]
                 + [f"res_i_{r}" for r in RESIDUE_TYPES] + [f"res_j_{r}" for r in RESIDUE_TYPES])
