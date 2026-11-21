#!/usr/bin/env python
"""PER-PAIR CENTRED TM ERROR. No cross-entropy, no other term.

WHAT IT IS, per pose and per scored cross-chain pair (i, j), on the SAME mask the PAE loss uses:

    s_pred(i,j) = sum_bins p_bin(i,j) * f(bin_center)      f(e) = 1 / (1 + (e/d0)^2)
    s_true(i,j) = f(true aligned error(i,j))
    e(i,j)      = s_pred - s_true                          (signed)
    c(i,j)      = e - e_mean(i,j)                          e_mean over the STRUCTURE's poses
    loss        = mean(c^2) over pairs and poses

WHY THE TM TRANSFORM IS REUSED, NOT REIMPLEMENTED. f and d0 are exactly what ipTM applies:
sample_confidence.calculate_chain_based_ptm builds
`per_bin_weight = 1 / (1 + (bin_center / ptm_norm) ** 2)` and reduces
`(pae_prob * per_bin_weight).sum(-1)`, with `ptm_norm = calculate_normalization(N_d)`. So
s_pred here IS that token_token_ptm, and both get_bin_centers and calculate_normalization are
imported from that module. A local copy of either would be a second definition of the metric
the whole project is trying to move.

d0 IS PER CHAIN PAIR, NOT PER STRUCTURE. calculate_normalization takes the number of tokens in
the pair of chains being scored, which is why phase_k_train's _set_pae_mask already stores
`int(((aa == x) | (aa == y)).sum())` beside each block: selection reads chain_pair_iptm[x, y],
one value per INTERFACE, each with its own d0. Using a structure-wide token count instead would
give every interface of a multi-interface structure the same d0, which is not the quantity ipTM
reads -- and 298 of 368 training structures score two or more pairs.

THE TRUE SIDE NEEDS NO SQRT. PAELoss.calculate_label returns the pairwise alignment error
SQUARED, so f(sqrt(sq)) = 1 / (1 + sq/d0^2) directly.

WHY e_mean IS A CONSTANT AND THE GRADIENT IS STILL EXACT. d/dtheta mean((e - e_mean)^2)
= 2/N sum (e - e_mean) * (de/dtheta - de_mean/dtheta), and sum(e - e_mean) = 0 by construction,
so the term carrying de_mean/dtheta vanishes. Detaching e_mean therefore costs nothing and needs
no correction -- which is what lets it be computed in a separate no-grad sweep.

ONLY PAIRS VALID IN EVERY POSE. A pair masked out for one pose has an e_mean over a different
population than the poses it is then subtracted from, so the mask used here is the AND across the
structure's poses, carried alongside e_mean.
"""
import torch

from protenix.model.sample_confidence import (  # noqa: E402
    calculate_normalization,
    get_bin_centers,
)


def per_bin_weight(d0, min_bin, max_bin, no_bins, device, dtype=torch.float32):
    """f(bin_center) for every bin, exactly as ipTM weights them."""
    bc = get_bin_centers(min_bin, max_bin, no_bins).to(device=device, dtype=dtype)
    return 1.0 / (1.0 + (bc / float(d0)) ** 2)


def d0_for_pair(n_pair_tokens):
    """ipTM's own normalisation for a chain pair of this many tokens."""
    return calculate_normalization(int(n_pair_tokens))


def pair_errors(logits, squared_pae, blocks, min_bin, max_bin, no_bins):
    """Signed per-pair TM error e = s_pred - s_true, and the pairs it is defined on.

    logits       [N_sample, N_frame, N_token, no_bins]  the head's PAE logits
    squared_pae  [N_sample, N_frame, N_token]           true aligned error SQUARED
    blocks       [(block_mask [N_frame, N_token] bool, n_pair_tokens), ...]

    Returns (e, m) both [N_sample, N_frame, N_token]; m is float 0/1. Each block contributes its
    own d0.

    m IS 0/1, NOT A COUNT, AND THAT IS A PROPERTY OF THE BLOCKS. A token pair's two chains
    determine its block uniquely -- block (x, y) holds exactly the pairs whose row is in x and
    column in y or the reverse -- so distinct scored chain pairs give disjoint masks and no pair is
    ever summed twice. If that stopped holding, the sum above would stay a SUM while centred_loss's
    denominator counted the pair once, which is not a mean over anything; phases/test_tm_centered.py
    asserts m.max() == 1 on the worst case this pool has (3 scored pairs over 3 chains).
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be [N_sample, N_frame, N_token, bins], got "
                         f"{tuple(logits.shape)}")
    # THE ROW AXIS MUST ALREADY BE THE FRAME AXIS, AND THAT IS NOT CHECKED BY BROADCASTING ALONE.
    # PAELoss.forward scores `logits[..., has_frame, :, :]` while the caller here hands the FULL
    # [N_sample, N_token, N_token, bins] head output, so the two agree only because every token in
    # this pool has a frame (measured: 856 of 856 training structures, has_frame all True). The
    # first structure with a frameless token -- a ligand, an UNK -- would otherwise either raise an
    # opaque broadcast error from inside the block loop or, under cropping, line up by accident
    # against the wrong rows. Named here, where the premise is used.
    if squared_pae.shape[-2] != logits.shape[-3] or squared_pae.shape[-1] != logits.shape[-2]:
        raise ValueError(
            f"squared_pae {tuple(squared_pae.shape)} does not match the PAE grid "
            f"{tuple(logits.shape[:-1])}: the label is [.., N_frame, N_token] and the logits are "
            f"[.., N_token, N_token], so these differ as soon as some token has no frame (or the "
            f"prediction is cropped and the label is not). Slice the logits by has_frame before "
            f"calling, as PAELoss.forward does.")
    prob = torch.softmax(logits.float(), dim=-1)
    e = torch.zeros(prob.shape[:-1], dtype=torch.float32, device=prob.device)
    m = torch.zeros_like(e)
    sq = squared_pae.float()
    for blk, npt in blocks:
        if not bool(blk.any()):
            continue
        if tuple(blk.shape) != tuple(prob.shape[-3:-1]):
            raise ValueError(f"block mask {tuple(blk.shape)} does not match the PAE grid "
                             f"{tuple(prob.shape[-3:-1])}")
        d0 = d0_for_pair(npt)
        w = per_bin_weight(d0, min_bin, max_bin, no_bins, prob.device)
        s_pred = (prob * w).sum(dim=-1)                       # == ipTM's token_token_ptm
        s_true = 1.0 / (1.0 + sq / (float(d0) ** 2))          # f(sqrt(sq)), no sqrt needed
        bm = blk.to(prob.device).to(torch.float32)
        e = e + (s_pred - s_true) * bm
        m = m + bm
    return e, m


def mean_pred_score(logits, blocks, min_bin, max_bin, no_bins):
    """The average predicted pair score over the scored pairs -- the squashing diagnostic.

    REPORTED BECAUSE THE LOSS CANNOT SEE IT. Centring removes any per-pair constant, so a head
    that drives every s_pred to 0 or to 1 can still have a small centred loss; this number is how
    that shows up.
    """
    with torch.no_grad():
        prob = torch.softmax(logits.float(), dim=-1)
        tot = torch.zeros((), dtype=torch.float32, device=prob.device)
        n = torch.zeros((), dtype=torch.float32, device=prob.device)
        for blk, npt in blocks:
            if not bool(blk.any()):
                continue
            w = per_bin_weight(d0_for_pair(npt), min_bin, max_bin, no_bins, prob.device)
            bm = blk.to(prob.device).to(torch.float32)
            tot = tot + ((prob * w).sum(dim=-1) * bm).sum()
            n = n + bm.sum() * prob.shape[0]
        return float(tot / n.clamp_min(1.0))


def centred_loss(e, m, e_mean, m_all):
    """mean((e - e_mean)^2) over pairs valid in EVERY pose of the structure.

    e_mean and m_all are constants from the no-grad sweep; m_all is the AND across poses.
    """
    keep = (m > 0).to(torch.float32) * m_all.to(e.device).to(torch.float32)
    if float(keep.sum()) == 0.0:
        return None
    c = (e - e_mean.to(e.device).detach()) * keep
    return (c * c).sum() / keep.sum()
