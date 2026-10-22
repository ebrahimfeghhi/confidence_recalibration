#!/usr/bin/env python
"""PAIRWISE RANKING LOSS on the statistic selection actually uses.

WHY. Selection is argmax of per-interface chain_pair_iptm over the poses of an interface; DockQ is
whatever that one pose scores. The FT3 loss instead minimises PAE -- a per-pair accuracy objective
averaged over pairs and poses. Those are different functionals of the same logits, and five arms
showed they move independently: held-out PAE fell 0.18-0.29 in every arm while the ORDERING went
nowhere or backwards. Measured mean within-interface Spearman(ipTM, DockQ): shipped head 0.196,
pose-share arm 0.271 after 50 epochs, full head 0.078 -- far BELOW untrained. Training PAE taught
the big arms to rank worse.

So this trains the ordering directly: for two poses of one interface, the one with the higher DockQ
should get the higher ipTM.

    L = - sum_ij (s_i - s_j)(dq_i - dq_j) / sum_ij |dq_i - dq_j|

over unordered pose pairs of the same interface, s = chain_pair_iptm, dq = DockQ. It is the
normalised covariance between score gaps and quality gaps: negative when the head's ordering leans
the right way.

WHY LINEAR AND NOT softplus. The first version used softplus(-(s_i - s_j) * sign(dq_i - dq_j)),
the RankNet logistic loss. Measured on 5,400 real within-seed pairs, that was a mistake of units:
the median |ipTM gap| is 0.0081 and the 90th percentile 0.034, while softplus only bends over
arguments of order 1. At a margin of 0.02 the gradient magnitude is 0.495 for a correct pair and
0.505 for an inverted one -- indistinguishable. Every property softplus is chosen for (saturating
on pairs already right, capping badly inverted ones) was inactive, and the loss was a straight
line through ln2 with the curvature sitting entirely outside the data. So it is written as the
straight line it already was.

A multiplier would move the bend onto the data (sigma ~ 30 stops pushing at a gap of 0.1, just
above the 90th percentile). Per-group standardisation would do it adaptively, and was rejected:
the implied multiplier ranges 12 to 1,262 across groups because the divisor is a standard
deviation of five points, so it amplifies hardest exactly where the head has no opinion.

WHAT LINEAR GIVES UP is any stopping point -- it pays to widen the gap forever. Two things bound
the damage: ipTM is structurally in [0,1], and the per-pose gradients sum to zero, so the group
MEAN is never pushed, only the spread. The spread is therefore the thing to watch, and it is
logged per epoch; if it inflates, sigma and softplus come back.

THE DockQ GAP IS THE WEIGHT, and with a linear term it folds in exactly: w_ij * sign(dq_i - dq_j)
IS (dq_i - dq_j). So the weighted loss needs no sign function. It also fixes a real defect of the
unweighted form, whose per-pose gradient is the pose's RANK: two poses at DockQ 0.136 and 0.124
are near-identical in quality but the rank form pushes the last-placed one twice as hard. The
weighted form pushes each pose by (dq_i - mean dq), which treats them alike.

THE MEMORY CONSTRAINT IS WHY THIS IS TWO PASSES. phase_k_train runs forward AND backward one pose
at a time on purpose: two fp32 pair-logit tensors at 3,400 tokens x 64 bins are ~5.5 GiB, and an
earlier version that held them across a yield overlapped the peaks for no reason. A pairwise loss
over 5 poses would need 5 graphs live at once.

It does not. The gradient factors:

    dL/dtheta = sum_i (dL/ds_i) * (ds_i/dtheta)

dL/ds_i is a scalar per pose, computable from the 5 ipTM VALUES alone. So:

  pass A  forward all 5 poses under no_grad, collect s_i                (no graph, +1 forward/pose)
  coeffs  autograd on a 5-element leaf tensor gives c_i = dL/ds_i exactly
  pass B  the EXISTING per-pose forward adds alpha_rank * c_i * s_i to its own backward

One graph at a time, exact gradient, and the only new cost is pass A's forward -- measured at
roughly a third of a pose's GPU time (gpu_forward 0.5m vs gpu_backward 1.0m per 450 poses).

PASS A AND PASS B MUST PRODUCE THE SAME s_i, or c_i was computed for a different function than the
one being differentiated. The confidence head has dropout, so the RNG state is captured before pass
A and restored before pass B, and the trainer re-checks |s_A - s_B| per pose against
--rank-scalar-tol. That check is not decoration: silently drifting scalars would give a wrong
gradient that still trains and still looks plausible.

ipTM IS COMPUTED THE SAME WAY THE EVALUATOR COMPUTES IT -- sample_confidence.calculate_chain_based_ptm
over softmax(PAE) with get_bin_params(configs.loss.pae), which is exactly what
phase_j_rescore_cache.pair_iptm_only does and what phase_k_eval's self-check validates against the
fold's stored ipTM to 2e-3. A loss that optimised a near-miss of the selection statistic would be a
different experiment.
"""
import collections
import csv
import os
import re

import torch

# `10fj_AB` -> pdb `10fj`, chain letters `A`,`B`. write_job_cif letters model chains A, B, C... in
# asym_id order, so a letter's alphabet position IS its asym index -- the same join phase_k_train
# does for --pae-scored-pairs and phase_k_eval does for chain_roles.
IFACE = re.compile(r"^(?P<pdb>[0-9a-zA-Z]+)_(?P<c1>[A-Z])(?P<c2>[A-Z])$")


def parse_iface(iface):
    """(pdb, (asym_lo, asym_hi)) or None. The pair is ORDERED LOW-HIGH so it is hashable one way."""
    m = IFACE.match(iface.strip())
    if not m:
        return None
    a = ord(m.group("c1")) - ord("A")
    b = ord(m.group("c2")) - ord("A")
    return m.group("pdb"), (min(a, b), max(a, b))


def load_labels(path):
    """{(pdb, seed, asym_pair): {sample: dockq}} from a fold's results.csv.

    KEYED BY INTERFACE, NOT STRUCTURE. A training structure can hold several scored interfaces
    (1,028 over the 488 structures), and reading one interface's ipTM against another's DockQ is
    the exact error phase_k_eval refuses to run. The asym pair in the key is what makes the join
    explicit here too.
    """
    out = collections.defaultdict(dict)
    bad = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            p = parse_iface(r.get("iface", ""))
            if p is None:
                bad += 1
                continue
            pdb, pair = p
            try:
                out[(pdb, int(r["seed"]), pair)][int(r["sample"])] = float(r["dockq"])
            except (ValueError, KeyError, TypeError):
                bad += 1
    return dict(out), bad


def iptm(pae_logits, feat, configs, pair):
    """chain_pair_iptm[pair] for ONE pose, differentiable when pae_logits requires grad.

    Mirrors phase_j_rescore_cache.pair_iptm_only, minus its torch.no_grad: same softmax in fp32,
    same calculate_chain_based_ptm, same bin params. fp32 because the bins are weighted and summed
    over N_token^2 and bf16 accumulation there is not what the reference path does.
    """
    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator
    pae_prob = torch.softmax(pae_logits.float(), dim=-1)
    token_is_ligand = torch.zeros_like(feat["asym_id"]).scatter_add(
        0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0
    out = autocasting_disable_decorator(True)(sample_confidence.calculate_chain_based_ptm)(
        pae_prob, has_frame=feat["has_frame"].bool(), asym_id=feat["asym_id"],
        token_is_ligand=token_is_ligand,
        **sample_confidence.get_bin_params(configs.loss.pae))
    cpi = out["chain_pair_iptm"]
    if cpi.dim() == 3:
        cpi = cpi[0]
    i, j = pair
    if i >= cpi.shape[0] or j >= cpi.shape[1]:
        return None
    return cpi[i, j]


def pair_terms(s_vals, dq_vals, weight_by_gap=True, dq_tol=1e-6):
    """(loss, coeffs, n_pairs) for one interface's poses.

    s_vals/dq_vals are plain floats. The loss is rebuilt on a LEAF TENSOR and differentiated by
    autograd rather than by the closed form n*(dq_i - mean dq)/sum|ddq|. The closed form is right
    only when NO pair is dropped, and equal-DockQ pairs are dropped; autograd stays correct either
    way and costs microseconds on a 5-element vector.

    A pose pair whose DockQ values are equal carries NO ordering information and is dropped: asking
    the head to separate two equally-good poses is noise with a gradient.
    """
    n = len(s_vals)
    s = torch.tensor(s_vals, dtype=torch.float64, requires_grad=True)
    num, den, npair = s.new_zeros(()), 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = dq_vals[i] - dq_vals[j]
            if abs(d) <= dq_tol:
                continue
            # LINEAR. With the gap as the weight, w * sign(d) collapses to d itself.
            w = abs(d) if weight_by_gap else 1.0
            num = num - (s[i] - s[j]) * (d if weight_by_gap else (1.0 if d > 0 else -1.0))
            den += w
            npair += 1
    if npair == 0:
        return 0.0, [0.0] * n, 0
    loss = num / den
    loss.backward()
    return float(loss.detach()), [float(g) for g in s.grad], npair


def coeffs(dq_vals, weight_by_gap=True, dq_tol=1e-6):
    """(c, n_pairs) where c_i = dL/ds_i -- FROM THE LABELS ALONE, no scores needed.

    THIS IS THE WHOLE REASON THE TRAINER NEEDS NO SECOND FORWARD PASS. The loss is linear in the
    scores, L = sum_i c_i s_i, so its derivative contains no score:

        c_i = - sum_{j: (i,j) valid} (dq_i - dq_j) / sum_{(i,j) valid} |dq_i - dq_j|

    Verified: pair_terms returns byte-identical coefficients for score vectors as different as
    [0.70, 0.69, 0.69, 0.69, 0.66], [0.1, 0.9, 0.5, 0.2, 0.7] and all-zeros.

    Computed by handing pair_terms a dummy score vector rather than by writing the closed form out
    again. The closed form above is right only when NO pair is dropped, and equal-DockQ pairs ARE
    dropped; routing through the same autograd path that computes the loss keeps the two
    definitions from drifting apart, and costs microseconds on five elements.
    """
    _, c, npair = pair_terms([0.0] * len(dq_vals), dq_vals,
                             weight_by_gap=weight_by_gap, dq_tol=dq_tol)
    return c, npair


def value(s_vals, c):
    """The loss itself, L = sum_i c_i s_i. For logging; no second evaluation of anything."""
    return float(sum(a * b for a, b in zip(c, s_vals)))
