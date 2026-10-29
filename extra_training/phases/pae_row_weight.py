#!/usr/bin/env python
"""Row weighting for the PAE term: concentrate the loss on the anchor rows ipTM can read.

WHY THIS EXISTS. ipTM is max_i t_i over a pose's anchor rows, where t_i is row i's mean predicted
cross-chain TM weight. Our structures carry 29-801 anchor rows (median 332), so selection reads ONE
row while the PAE term grades all of them equally. These functions build a per-row multiplier that
tilts the term toward the rows selection actually reads.

WHAT IT DELIBERATELY DOES NOT DO. It never touches the LABEL -- every pair is still trained toward
its own true error bin. That is the whole reason to prefer this to a ranking loss on ipTM: a
softmax ranking loss is invariant to adding a constant to every score, so it cannot learn the
absolute confidence level, and it can be lowered by distorting ipTM away from calibration. This
cannot. It only moves where capacity goes.

WHY IT LIVES IN ITS OWN FILE. phase_k_train.py installs these through a hook, where they would be
nested closures that no test can import. The identity in row_weights -- frac 1.0 must give uniform
weights, so the term is the unweighted one bit for bit -- is the correctness gate for the whole
change, and a gate that cannot be imported is not a gate. See run_logging/test_pae_row_weight.py.
"""
import math

import torch

from protenix.model.sample_confidence import calculate_normalization, get_bin_centers


def row_t(logits, has_frame, scored_mask, min_bin=0.0, max_bin=32.0):
    """t_i for every anchor row: ipTM before the max is taken. Returns [N_frame].

    Replicates sample_confidence.calculate_iptm rather than paraphrasing it -- same bin centres,
    same d0 = 1.24*(max(N,19)-15)^(1/3) - 1.8, same per-bin weight 1/(1+(c/d0)^2) -- so that
    max(row_t(...)) equals calculate_iptm(...) when scored_mask is the full different-chain mask.
    test_pae_row_weight.py asserts exactly that against Protenix's own function.

    THE ONE DELIBERATE DIFFERENCE is the denominator. calculate_iptm averages a row over every
    different-chain target; this averages over the SCORED ones, the same set the masked PAE term is
    computed on. Ranking rows by one target set while training them on another would weight rows by
    a quantity the loss never sees.

    Args:
        logits: [..., N_sample, N_token, N_token, n_bins] raw PAE head output.
        has_frame: [N_token] bool. Rows are returned in has_frame order, which is the order
            PAELoss.calculate_label returns its [N_frame, N_token] pair mask in.
        scored_mask: [N_frame, N_token] weighting for the targets of each row, or None to average
            over every target. Only its support and relative values matter.
    """
    hf = has_frame.reshape(-1).bool()
    if int(hf.sum()) == 0:
        return None
    prob = torch.softmax(logits[..., hf, :, :].float(), dim=-1)
    while prob.dim() > 3:          # collapse N_sample: one pair mask is shared across samples
        prob = prob.mean(dim=0)
    d0 = calculate_normalization(int(hf.numel()))
    centres = get_bin_centers(min_bin, max_bin, prob.shape[-1]).to(prob.device, prob.dtype)
    tt = (prob * (1.0 / (1.0 + (centres / d0) ** 2))).sum(-1)       # [N_frame, N_token]
    if scored_mask is None:
        return tt.mean(-1)
    mm = scored_mask.to(tt.dtype)
    return (tt * mm).sum(-1) / mm.sum(-1).clamp_min(1.0)


def row_weights(t, frac, mass, floor):
    """Per-row multiplier with mean 1 over rows. Returns (w, k); w is None when uniform.

    The top k = max(floor, ceil(frac * n)) rows by t share `mass` of the loss and the remaining
    n - k rows share 1 - mass. Mean 1 keeps the term on the scale of the unweighted one, so the
    curve stays readable against arms that did not use this.

    NONE MEANS UNIFORM, NOT DISABLED. When k >= n every row is in the top set, the weights are all
    1, and multiplying by them is a no-op -- so returning None makes --pae-row-frac 1.0 reproduce
    the unweighted term exactly rather than to within float error. That identity is the gate.

    MASS IS NOT 1.0 BY DESIGN. At mass 1.0 the rows outside the top set get zero weight, which
    leaves them unsupervised and free to drift over-confident -- and since ipTM is a max, an
    over-confident row is precisely the one that gets selected. Keeping a floor of weight on them
    costs one parameter and removes that failure mode.
    """
    n = int(t.shape[-1])
    if n == 0:
        return None, 0
    k = min(n, max(int(floor), int(math.ceil(float(frac) * n))))
    if k >= n:
        return None, k
    w = torch.full((n,), (1.0 - mass) * n / (n - k), device=t.device, dtype=torch.float32)
    w[torch.topk(t, k).indices] = mass * n / k
    return w, k


def effective_sample_size(w, n):
    """Kish's (sum w)^2 / sum w^2: how many rows the weighted term effectively trains on.

    WHY IT IS LOGGED. This weighting trades constraints per pose against focus, with the same
    parameter count on either side. At frac 0.2 / mass 0.8 it is 0.31n -- about 103 of 332 -- and a
    setting that quietly collapses it to single digits has cut the supervision per pose by 30x,
    which is the overfitting risk in the whole idea. It makes the knob readable instead of a guess.
    """
    if w is None:
        return float(n)
    return float((w.sum() ** 2) / (w ** 2).sum())


def predicted_ess(n, frac, mass, floor):
    """effective_sample_size without building the tensor: 1/(mass^2/k + (1-mass)^2/(n-k))."""
    k = min(n, max(int(floor), int(math.ceil(float(frac) * n))))
    if k >= n:
        return float(n)
    return 1.0 / (mass ** 2 / k + (1.0 - mass) ** 2 / (n - k))
