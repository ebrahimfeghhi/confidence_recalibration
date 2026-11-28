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


def pair_tm_weight(sq, blocks, out_dtype=None, device=None):
    """w = 1/(1 + e^2/d0^2) per scored pair, d0 from each block's own token count.

    THE WEIGHT --pae-tm-weight APPLIES, LIVING HERE SO A TEST CAN IMPORT IT. The first version of
    this lived as a closure inside phase_k_train.main(), which made it unimportable, so the CPU
    gate could only re-type the formula and assert against its own copy -- a test that passes for
    any implementation, including the bool-cast no-op --pae-row-frac actually shipped. Review
    caught exactly that. The trainer and run_logging/test_pae_tm_weight.py now call this.

    sq is the alignment error SQUARED, [N_frame, N_token] (no sample dim: the weight is per pose).
    blocks is the mask builder's list of (block_mask, n_pair_tokens).

    PAIRS IN NO BLOCK GET ZERO, which is correct because the union of the blocks IS the scored-pair
    mask -- they are built from the same loop -- so a masked pair always falls in exactly one
    block. A pose with no blocks at all is the caller's business, not this function's.
    """
    import torch
    dt = out_dtype or torch.float32
    dev = device or sq.device
    sqf = sq.to(dtype=dt, device=dev)
    if sqf.dim() != 2:
        raise ValueError(f"pair_tm_weight wants [N_frame, N_token], got {tuple(sqf.shape)}. The "
                         f"weight is per pose; a leading sample dim means --pose-batch > 1, which "
                         f"the flag refuses at startup.")
    w = torch.zeros(sqf.shape, dtype=dt, device=dev)
    for blk, npt in blocks:
        d0 = float(d0_for_pair(npt))
        w = torch.where(blk.to(dev), 1.0 / (1.0 + sqf / (d0 ** 2)), w)
    return w


def pair_tm_weight_max(true_bins, logits, blocks, out_dtype=None, device=None,
                       min_bin=0.0, max_bin=32.0, no_bins=64):
    """w = max(s_true, s_pred) per scored pair -- Arm N's weight.

    THE TWO NUMBERS, both in [0, 1] and both read off the SAME per-bin TM values so they are
    comparable:
      s_true  the TM value of the bin the pair's TRUE error falls in. Taken from the one-hot label
              rather than from the raw error, so it is the bin's value, not the exact error's.
      s_pred  the head's EXPECTED TM value: each bin's TM value times the probability the head puts
              on that bin, summed over all 64.

    WHY THE MAX. Arm M weighted by s_true alone, which spends the signal on pairs that are truly
    close -- often the easy ones the head already gets -- and gives almost nothing to a pair the
    head is confidently wrong about. Taking the max keeps a pair at full weight in either of the
    two cases that move ipTM: it really is a contact, OR the head believes it is one and it is not
    (a confident decoy). Only a pair that is truly wrong AND already predicted wrong falls away,
    and that is the mass the unweighted objective was being swallowed by.

    NO GRADIENT THROUGH THE WEIGHT. s_pred is a function of the logits, so without detaching, the
    head could lower the loss by lowering its own weights -- predicting uncertainty everywhere.
    The caller must be inside no_grad, and this returns a detached tensor either way.

    d0 IS PER BLOCK, from that interface's token count, so the bin values match the ipTM the arm is
    judged on rather than a lookalike at a different scale.
    """
    import torch
    dt = out_dtype or torch.float32
    dev = device or logits.device
    if true_bins.shape[-1] != no_bins:
        raise ValueError(f"true_bins last dim is {true_bins.shape[-1]}, expected {no_bins}")
    tb = true_bins.to(dtype=dt, device=dev)
    lg = logits.to(dtype=dt, device=dev)
    def _one_pose(x, name):
        """Collapse leading dims ONLY when they multiply to exactly one pose.

        THE OLD TEST WAS `x.shape[0] == 1`, which asks about the FIRST dim rather than about how
        many poses are present: [1, S, F, T, 64] with S > 1 passed it and returned pose 0's weight
        for every pose in the batch. Unreachable with today's call shape -- the head is driven one
        pose at a time -- but it would be a silent wrong answer rather than a crash the day that
        changed, which is the worst kind. Raised in review.
        """
        if x.dim() == 3:
            return x
        lead = 1
        for d in x.shape[:-3]:
            lead *= int(d)
        if lead != 1:
            raise ValueError(f"{name} holds {lead} poses {tuple(x.shape)}; the weight is per pose. "
                             f"--pose-batch > 1 is refused at startup for this reason.")
        return x.reshape(*x.shape[-3:])

    tb = _one_pose(tb, "true_bins")
    lg = _one_pose(lg, "logits")
    if tb.dim() != 3 or lg.dim() != 3:
        raise ValueError(f"wanted [N_frame, N_token, {no_bins}] per pose, got "
                         f"{tuple(tb.shape)} and {tuple(lg.shape)}. A leading sample dim means "
                         f"--pose-batch > 1, which the flag refuses at startup.")
    with torch.no_grad():
        p = torch.softmax(lg, dim=-1)
        w = torch.zeros(tb.shape[:2], dtype=dt, device=dev)
        for blk, npt in blocks:
            if tuple(blk.shape) != tuple(tb.shape[:2]):
                raise ValueError(f"block mask {tuple(blk.shape)} does not match the pair grid "
                                 f"{tuple(tb.shape[:2])}; torch.where would broadcast it silently")
            d0 = float(d0_for_pair(npt))
            tmv = per_bin_weight(d0, min_bin, max_bin, no_bins, dev, dt)   # [no_bins]
            s_true = (tb * tmv).sum(dim=-1)
            s_pred = (p * tmv).sum(dim=-1)
            w = torch.where(blk.to(dev), torch.maximum(s_true, s_pred), w)
    return w.detach()
