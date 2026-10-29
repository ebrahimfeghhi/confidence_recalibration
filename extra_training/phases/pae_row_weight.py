#!/usr/bin/env python
"""Row weighting for the PAE term: concentrate the loss on the anchor rows ipTM can read.

WHY THIS EXISTS. ipTM is max_i t_i over a chain pair's anchor rows -- our structures carry 29 to
801 of them, median 332 -- so selection reads ONE row while the PAE term grades every row equally.
These functions build a real-valued pair mask that tilts the term toward the rows selection reads.

WHAT IT DELIBERATELY DOES NOT DO. It never touches a LABEL. Every pair is still trained toward its
own true error bin, so the term cannot be lowered by distorting ipTM away from calibration, and it
stays sensitive to the absolute confidence level a softmax ranking loss is blind to (softmax is
invariant to adding a constant to every score). It only moves where capacity goes.

PER CHAIN PAIR, NOT PER STRUCTURE. Selection reads chain_pair_iptm[x, y], which Protenix computes
with token_mask = (asym == x) | (asym == y) -- one max per interface, each with its own d0 over
just those two chains. 298 of our 368 training structures have two or more scored pairs, so a
single ranking pooled over the union of their blocks ranks rows by a quantity no ipTM reads: a row
in the antigen would average its heavy-chain and light-chain targets together. Each scored pair
therefore gets its own t_i, its own d0 and its own top-k, applied to its own block of the mask.
The blocks are disjoint in (row, target) space, so the weighted mask is their sum.

THREE PROPERTIES THE TESTS PIN DOWN (run_logging/test_pae_row_weight.py):
  - max_i t_i over a pair's block equals Protenix's own chain_pair_iptm for that pair.
  - frac 1.0 reproduces the unweighted bool mask exactly, so any arm re-runs as its own control.
  - a weighted mask is NOT equal to the unweighted one at frac < 1. That assertion is here because
    its absence let a version ship where the weights were cast to bool and the whole term was a
    silent no-op.
"""
import math

import torch

from protenix.model.sample_confidence import calculate_normalization, get_bin_centers

# ROWS ARE PROCESSED IN CHUNKS so the softmax is never materialised for the whole crop. At the
# largest crop in our logs (1,473 tokens) a full fp32 [N, N, 64] softmax is ~555 MB, and row_t runs
# inside calculate_losses while the confidence head's autograd graph is live -- the same hazard
# phase_k_train refuses outright for --emit-iptm-rows outside --loss-only. 64 rows is ~24 MB.
ROW_CHUNK = 64


def _tm_weights(n_bins, d0, device, dtype, min_bin, max_bin):
    c = get_bin_centers(min_bin, max_bin, n_bins).to(device=device, dtype=dtype)
    return 1.0 / (1.0 + (c / d0) ** 2)


def row_t(logits, has_frame, target_mask, n_pair_tokens=None, min_bin=0.0, max_bin=32.0):
    """t_i for every anchor row: ipTM before the max. Returns ([N_frame], [N_frame] target counts).

    Replicates sample_confidence.calculate_iptm -- same bin centres, same
    d0 = 1.24*(max(N,19)-15)^(1/3) - 1.8, same per-bin weight 1/(1+(c/d0)^2) -- so that the max
    over a pair's rows equals chain_pair_iptm for that pair.

    Args:
        logits: [..., N_sample, N_token, N_token, n_bins] raw PAE head output.
        has_frame: [N_token] bool. Rows come back in has_frame order, which is the order
            PAELoss.calculate_label returns its [N_frame, N_token] pair mask in.
        target_mask: [N_frame, N_token] bool or float -- which targets each row averages over.
        n_pair_tokens: N_d for d0. Pass the token count of the TWO CHAINS for a chain pair, which
            is what chain_pair_iptm uses; None falls back to the whole crop, which is what
            calculate_ptm does and is wrong for an interface.
    """
    hf = has_frame.reshape(-1).bool()
    n = int(hf.sum())
    if n == 0:
        return None, None
    lg = logits
    while lg.dim() > 4:                 # drop batch dims, keep [N_sample, N_tok, N_tok, bins]
        lg = lg[0]
    if lg.dim() == 4:
        # t_i is LINEAR in the probabilities, so averaging prob over samples and then weighting is
        # identical to averaging per-sample t_i. One pair mask is shared across samples, so one
        # weight vector is what is needed.
        lg_rows = lg[:, hf]             # [N_sample, N_frame, N_token, bins]
    else:
        lg_rows = lg[hf].unsqueeze(0)
    nb = lg_rows.shape[-1]
    nd = int(n_pair_tokens) if n_pair_tokens is not None else int(hf.numel())
    d0 = calculate_normalization(nd)
    tm = target_mask.to(torch.float32)
    cnt = tm.sum(-1)
    out = torch.zeros(n, device=lg_rows.device, dtype=torch.float32)
    wb = None
    for lo in range(0, n, ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, n)
        chunk = lg_rows[:, lo:hi].float()                  # [S, C, N_token, bins]
        if wb is None:
            wb = _tm_weights(nb, d0, chunk.device, chunk.dtype, min_bin, max_bin)
        p = torch.softmax(chunk, dim=-1)
        tt = (p * wb).sum(-1).mean(0)                      # [C, N_token]
        out[lo:hi] = (tt * tm[lo:hi]).sum(-1) / cnt[lo:hi].clamp_min(1.0)
        del chunk, p, tt
    return out, cnt


def row_weights(t, valid, frac, mass, floor):
    """Per-row multiplier over the VALID rows, mean 1 across them. Returns (w, k); None if uniform.

    The top k = max(floor, ceil(frac * n_valid)) valid rows by t share `mass`; the rest share
    1 - mass. Rows that are not valid -- no targets in this block, or no true coordinates for their
    frame -- get 0, because weight spent on them is weight the loss cannot see, which would make
    the realised top-set share and the logged effective sample size both overstate the focus.

    NONE MEANS UNIFORM, NOT DISABLED. When k >= n_valid every valid row is in the top set and the
    weights are all 1, so returning None lets the caller keep the unweighted bool mask and makes
    --pae-row-frac 1.0 an exact identity rather than a float-error approximation.

    MASS IS NOT 1.0 BY DESIGN. At mass 1.0 the rows outside the top set are unsupervised and free
    to drift over-confident -- and since ipTM is a max, an over-confident row is exactly the one
    selected. A set rather than the single argmax row for the same reason: correcting the current
    max promotes the next row, so the set the argmax can move within must already be supervised.

    TIES ARE BROKEN BY INDEX, deterministically. torch.topk's order is arbitrary and unspecified on
    CUDA, so with ties the mass would land on a different row set in each of the three ranks. Rows
    with non-finite t sort LAST rather than first, which is where topk would have put them.
    """
    n = int(t.shape[-1])
    v = valid.reshape(-1).bool() if valid is not None else torch.ones_like(t, dtype=torch.bool)
    nv = int(v.sum())
    if n == 0 or nv == 0:
        return None, 0
    k = min(nv, max(int(floor), int(math.ceil(float(frac) * nv))))
    if k >= nv:
        # UNIFORM OVER THE SCORED ROWS, so the caller keeps the block's own mask unchanged. The
        # unscored rows need no zeroing here: a row with no targets in this block already has an
        # all-false row in the block mask, so it contributes nothing either way. Requiring nv == n
        # as well was a bug -- a block with any unscored row then took the weighted path at
        # frac 1.0 and came out uniformly rescaled instead of identical.
        return None, k
    key = torch.where(torch.isfinite(t), t, torch.full_like(t, float("-inf")))
    key = torch.where(v, key, torch.full_like(key, float("-inf")))
    order = torch.argsort(-key, stable=True)               # stable -> lowest index wins a tie
    top = order[:k]
    w = torch.zeros(n, device=t.device, dtype=torch.float32)
    n_bot = nv - k
    if n_bot > 0:
        w[v] = (1.0 - mass) * nv / n_bot
    w[top] = mass * nv / k
    return w, k


def weighted_mask(logits, has_frame, blocks, frac, mass, floor, min_bin=0.0, max_bin=32.0):
    """[N_frame, N_token] float mask, or None to keep the caller's unweighted bool mask.

    `blocks` is one (block_mask, n_pair_tokens) per scored chain pair, from the same pool the PAE
    term is masked to. Each block is ranked on its own t_i with its own d0 and weighted on its own
    top-k, then summed -- the blocks are disjoint in (row, target) space, so at uniform weights the
    sum is exactly the union mask the unweighted term uses.
    """
    if not blocks:
        return None
    acc, any_w = None, False
    for blk, nd in blocks:
        t, cnt = row_t(logits, has_frame, blk, nd, min_bin, max_bin)
        if t is None:
            continue
        w, _ = row_weights(t, cnt > 0, frac, mass, floor)
        bf = blk.to(torch.float32)
        if w is not None:
            bf = bf * w.reshape(-1, 1)
            any_w = True
        acc = bf if acc is None else acc + bf
    if acc is None or not any_w:
        return None
    return acc


def effective_sample_size(mask):
    """Kish's (sum w)^2 / sum w^2 over the mask's ROW weights, from the REALISED mask.

    WHY IT IS LOGGED, AND WHY FROM THE REALISED MASK. This weighting trades constraints per pose
    against focus with the same parameter count on either side, so a setting that quietly collapses
    it has cut supervision per pose by the same factor. Computing it from the mask that was
    actually used -- after coordinate validity and the scored blocks have zeroed rows -- keeps it an
    upper bound on nothing: a closed-form from frac and mass alone would overstate the focus
    whenever a top-k row turns out to carry no usable targets.
    """
    r = mask.to(torch.float64).sum(-1)
    nz = r[r > 0]
    if nz.numel() == 0:
        return 0.0
    return float((nz.sum() ** 2) / (nz ** 2).sum())


def predicted_ess(n, frac, mass, floor):
    """The closed form 1/(mass^2/k + (1-mass)^2/(n-k)), for documenting a setting before a run.

    Reports what the weighting WOULD give on a block of n equally usable rows. The logged number
    comes from effective_sample_size on the realised mask instead; this is for banners.
    """
    k = min(n, max(int(floor), int(math.ceil(float(frac) * n))))
    if k >= n:
        return float(n)
    return 1.0 / (mass ** 2 / k + (1.0 - mass) ** 2 / (n - k))
