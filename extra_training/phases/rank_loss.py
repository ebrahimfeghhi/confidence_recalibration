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

TWO FORMS LIVE HERE. `coeffs` is the LINEAR loss above. `coeffs_softplus` is the SCALED LOGISTIC,
and it is now the recommended one. The history matters because both mistakes were real.

THE FIRST SOFTPLUS ATTEMPT WAS A UNITS ERROR, not a shape error. It used
softplus(-(s_i - s_j) * sign(dq_i - dq_j)) with no temperature. Measured on 5,400 real within-seed
pairs, the median |ipTM gap| was 0.0081 and the 90th percentile 0.034 -- that 0.0081 is a 5,400-pair
sample taken before DockQ ties were dropped and should not be quoted as the within-seed scale; the
full training set after the tie filter gives 0.0036 (measure_tau.py --pairs within). Either way the
argument below is unchanged, because softplus only bends
over arguments of order 1. At a margin of 0.02 the gradient magnitude was 0.495 for a correct pair
against 0.505 for an inverted one -- indistinguishable. Every property softplus is chosen for was
inactive and the loss was a straight line through ln2. That is why it was replaced by the straight
line it already was, which was the right call for that parameterisation.

THE LINEAR LOSS THEN FAILED FOR A DEEPER REASON. Being linear in the scores, L = sum_i c_i s_i, its
score-space gradient is a CONSTANT: it never shrinks, so it has no stopping point and cannot tell
an already-correct ordering from an incorrect one when deciding how hard to push a pair. Scaling
all scores by lambda scales L by lambda, so WIDENING THE SPREAD PAYS WITHOUT LIMIT, and it pays far
better than reordering: at the real within-seed gap scale, saturating the spread to [0,1] is worth
~71x more loss reduction than going from fully-inverted to fully-correct order. Dropping the DockQ
weighting to pure sign does NOT fix this -- sign only changes the constants, and spread still pays
exactly proportionally. Measured consequence in run ntwjfp (alpha_rank 250, trainable readout):
spread 0.0114 -> 0.0371 over 8 epochs while held-out rank loss went +0.00018 -> +0.00288, i.e. the
ordering on held-out interfaces got WORSE while the training objective improved.

Two qualifications, both fair and neither a rescue: the constancy is with respect to the SCORES,
not the parameters, so d/dtheta can still change or vanish; and 71x is specific to that score-range
comparison. The argument survives both -- increasing an already-correct margin can dominate the
benefit of correcting small-margin inversions.

THE SCALED SOFTPLUS FIXES IT. A pair's gradient weight becomes sigmoid(-m/tau), which decays to 0
once the pair is comfortably correct and saturates at 1 (not beyond) when badly inverted. So
spreading buys a bounded amount and then nothing, while every inverted pair keeps generating
full-strength gradient until it is fixed. The loss floor is 0, reachable only by ordering every
pair.

TAU IS MEASURED, NOT CHOSEN -- see run_logging/measure_tau.py. Over the full training set
(training/fold/results.csv: 497 structures, 1,028 interfaces, 1,056,035 CROSS-SEED pairs after
dropping DockQ ties) the median |ipTM gap| is 0.0305, which is the default. Cross-seed pairs are
8.5x wider than the within-seed 0.0036, so the within-seed figure does not transfer. At tau=0.0305
inverted pairs are 43.7% of pairs BY COUNT and take 74.5% of the PAIRWISE SCORE-SPACE GRADIENT
MASS, against 41.7% under the DockQ-weighted linear loss -- a redistribution of +32.8 points.
(The figures here were 81.6 / 40.4 / +41.3 until measure_tau.py's allocation block was found to
be weighting pairs by |dipTM| where the loss weights them by |dDockQ|. The redistribution is
real and smaller than was claimed.) That
redirection is the entire point.

TWO THINGS THAT MEASUREMENT IS NOT, both of which an earlier version of it got wrong. It is not
the parameter gradient: per-pair mass is |d_i - d_j| sigmoid(-m/tau) * 2/W for the softplus and
|d_i - d_j|/W for the linear form, and parameter Jacobians and cross-pose cancellation are not in
it (a probe once measured 65x cancellation between per-pose and accumulated norms). And the
baseline is NOT the raw inversion count: with the gap as the weight the linear loss does not give
every pair equal gradient magnitude, so 43.7% is the right comparison only for
--no-rank-weight-by-gap, where the figures are 44.7% -> 72.7%.

ONE GLOBAL TAU, NOT PER-INTERFACE. Per-interface median gap spans ~1,700x, which sounds fatal but
is not, because the failure modes are asymmetric: wide interfaces have their CORRECT pairs saturate
(right -- already ordered) while their inversions still push, and tight interfaces look like ties
only because the head has no opinion there. A per-interface tau would divide by that interface's
own spread, amplifying hardest where there is least signal -- the same defect that got per-group
standardisation rejected (implied multiplier 12x to 1,262x across groups).

THE 2*tau PREFACTOR IS DELIBERATE. sigmoid(-m/tau) carries an implicit 1/tau, so without it
switching forms would silently multiply the gradient by ~33 at tau=0.0305 -- changing the objective
AND the effective learning rate at once. With it, the score-space gradient AT A TIED PAIR exactly
reproduces the linear loss's, so only the shape changes. (The 2 is because sigmoid(0) = 1/2.)

WHAT THE SOFTPLUS GIVES UP is the label-only coefficients, and that is not a coincidence: a loss
whose gradient ignores the current scores is exactly a loss that is linear in them, hence one with
no stopping point. The convenience and the defect are the same property. So pass A below is
REQUIRED for this form -- see THE MEMORY CONSTRAINT.

The spread remains the thing to watch, and it is logged per epoch.

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

dL/ds_i is a scalar per pose, so only the GROUP'S ipTM VALUES are needed to compute it -- never the
graphs. So:

  pass A  forward every pose in the group under no_grad, collect s_i   (no graph, +1 forward/pose)
  coeffs  autograd on an n-element leaf tensor gives c_i = dL/ds_i exactly
  pass B  the EXISTING per-pose forward adds alpha_rank * c_i * s_i to its own backward

One graph at a time, exact gradient, and the only new cost is pass A's forward -- measured at
roughly a third of a pose's GPU time (gpu_forward 0.5m vs gpu_backward 1.0m per 450 poses).

PASS A IS REQUIRED FOR THE SOFTPLUS AND SKIPPED FOR THE LINEAR FORM. The linear c_i depend on the
DockQ labels alone, so the trainer computes them before the model runs; a pass A there was dead
weight and was removed in 04503a9e after the review of cd8e6ab7 caught it. The softplus c_i contain
sigmoid(-m/tau), which needs the current margins, so pass A comes back for that form only.

PASS A AND PASS B MUST PRODUCE THE SAME s_i, or c_i was computed for a different function than the
one being differentiated. The confidence head has dropout, so the RNG state is captured before pass
A and restored before pass B. Silently drifting scalars would give a wrong gradient that still
trains and still looks plausible.

CROSS-SEED GROUPS SPAN RANKS, AND THAT IS FREE. build_epoch hands every rank the SAME structures in
the same order, differing only in which seed of each it takes, so the poses of one structure's
several seeds are already in flight simultaneously across the ranks. Pass A therefore ends in one
all_gather of a few floats per rank, after which every rank holds the whole group's scores and
computes identical coefficients; each rank then backprops only its own poses. No extra trunks are
co-resident on any GPU, whatever the group size.

HOW MANY SEEDS PER GROUP IS A MEASURED TRADEOFF. Agreement between a subsample's gradient direction
and the full 10-seed group's, as cosine over the poses they share (400 interfaces, 8 draws each):

     3 seeds (15 poses,   75 pairs): mean 0.856  median 0.949  p10 0.544
     5 seeds (25 poses,  250 pairs): mean 0.925  median 0.981  p10 0.823
    10 seeds (50 poses, 1125 pairs): exact by construction

The median 3-seed group is already close, but one group in ten points substantially wrong. Taking
all ten seeds of a structure into one group is exact and costs nothing extra in batch size -- a
step is ~20 units either way, so it buys exactness by covering 2 structures per step instead of 6.
That trades structure diversity per step for an exact ranking signal, which is the intended
configuration for the cross-seed arm.

ipTM IS COMPUTED THE SAME WAY THE EVALUATOR COMPUTES IT -- sample_confidence.calculate_chain_based_ptm
over softmax(PAE) with get_bin_params(configs.loss.pae), which is exactly what
phase_j_rescore_cache.pair_iptm_only does and what phase_k_eval's self-check validates against the
fold's stored ipTM to 2e-3. A loss that optimised a near-miss of the selection statistic would be a
different experiment.
"""
import collections
import csv
import math
import os
import re

import torch

# `10fj_AB` -> pdb `10fj`, REFERENCE chains `A`,`B`. NOT model chains: results.csv names an
# interface by the chains of the REFERENCE structure, while chain_pair_iptm is indexed by the
# model's asym ids. The two differ for most interfaces -- 7sfw_EO is reference E/O and model C/A.
IFACE = re.compile(r"^(?P<pdb>[0-9a-zA-Z]+)_(?P<c1>[A-Za-z])(?P<c2>[A-Za-z])$")


def load_roles(path):
    """{pdb: {ref_chain: model_chain}} from a fold's chain_roles.csv, the same join the trainer
    uses for --pae-scored-pairs and phase_k_eval uses for its own."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["pdb"], {})[r["ref_chain"]] = r["model_chain"]
    return out


def parse_iface(iface, roles):
    """(pdb, (asym_lo, asym_hi)) in MODEL asym indices, or None.

    THE JOIN IS NOT OPTIONAL AND NOT A LETTER CAST. The first version of this function read the
    iface letters as asym indices directly -- `7sfw_EO` became (4, 14). The real model pair is
    (0, 2), because write_job_cif letters model chains A, B, C... in asym_id order and
    chain_roles.csv maps reference chain -> model chain. Measured consequence on the overfit30
    pool: of 770 scored (pdb, seed, interface) groups the trainer would look up, the letter cast
    found 190, and the ones it did find matched by coincidence where the letters happened to
    collide. That is the error phase_k_eval refuses outright -- reading one interface's ipTM
    against another's DockQ -- arriving through the labels instead.

    NORMALISED LOW-HIGH so the tuple is hashable one way, matching phase_k_train's _scored_asym.
    chain_pair_iptm is symmetric (phase_k_eval reads cpi[ai, aj] in the pose's own order and
    reproduces the fold's stored ipTM to 2e-3 on every row), so the order never changes a value.
    """
    m = IFACE.match(iface.strip())
    if not m:
        return None
    rm = roles.get(m.group("pdb"))
    if not rm:
        return None
    m1, m2 = rm.get(m.group("c1")), rm.get(m.group("c2"))
    if m1 is None or m2 is None:
        return None
    a = ord(m1) - ord("A")
    b = ord(m2) - ord("A")
    return m.group("pdb"), (min(a, b), max(a, b))


def load_labels(path, roles_path):
    """{(pdb, seed, model_asym_pair): {sample: dockq}} from a fold's results.csv.

    `roles_path` is the fold's chain_roles.csv, and it is REQUIRED, not a convenience: the iface
    column names REFERENCE chains while the key must be model asym ids. See parse_iface for what
    the letter cast cost.

    KEYED BY INTERFACE, NOT STRUCTURE. A training structure can hold several scored interfaces
    (1,028 over the 488 structures), and reading one interface's ipTM against another's DockQ is
    the exact error phase_k_eval refuses to run.
    """
    roles = load_roles(roles_path)
    out = collections.defaultdict(dict)
    bad = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            p = parse_iface(r.get("iface", ""), roles)
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


DEFAULT_TAU = 0.0305
"""Median |ipTM gap| over CROSS-SEED pairs of the full training set.

training/fold/results.csv, 497 structures, 1,028 interfaces, 1,056,035 pairs after dropping DockQ
ties. Reproduce with `python -I run_logging/measure_tau.py`. For reference the 20-structure
overfit30 subset gives 0.0426 and val10 gives 0.0313; the training number is the one to use,
because tau shapes the training gradient and taking it from val would let the eval set in.
"""


DEFAULT_SOFTMAX_TAU = 0.05
"""Softmax temperature for the EXPECTED-DockQ form. NOT the same quantity as DEFAULT_TAU.

DEFAULT_TAU is a margin scale in ipTM units -- the measured median cross-seed gap -- and it sets
where the softplus stops caring. This is a temperature dividing the scores before a softmax, and
it sets how much of the probability mass sits on the current leaders. They are both called tau and
they are not interchangeable: 0.0305 here would be a far sharper selection than 0.05, and 0.05 in
the softplus would be a 1.6x wider margin. The flag is shared, so the banner prints which meaning
is in force.
"""


def expected_dockq(s_vals, dq_vals, tau=DEFAULT_SOFTMAX_TAU):
    """(loss, coeffs, n_poses) for the EXPECTED-DockQ form.

        p = softmax(s / tau)          over EVERY pose in the group
        L = - sum_i p_i * dq_i        the expected DockQ of the pose the model would select

    WHY THIS EXISTS. Every other form in this file is pairwise, and a pairwise loss spends its
    gradient in proportion to how many pairs a pose appears in -- which is the same for a pose
    ranked 1st and a pose ranked 90th. Deployment takes the argmax and nothing else, so ordering
    the bottom of the list is work that cannot pay. The softmax is a differentiable stand-in for
    that argmax: as tau falls it concentrates on the current leaders, and in the limit L is the
    DockQ of the selected pose.

    THERE ARE NO PAIRS. The returned count is POSES, not pairs. `weight_by_gap` and
    `pairs` has nothing to act on here -- the softmax is over the whole group by
    construction, which is what "all poses competing for selection" means -- so group_coeffs
    ignores both for this form and says so in the log rather than appearing to honour them.

    THE FLOOR IS NOT ZERO. L is in [-max(dq), -min(dq)], so a perfect model scores -max(dq) for
    that group, not 0. The softplus log line's "0 is the FLOOR" claim is false for this form and
    is suppressed.

    THE GRADIENT, closed form:
        dL/ds_i = -(p_i / tau) * (dq_i - sum_j p_j dq_j)
    A pose is pushed up when its DockQ beats the current expected DockQ, down when it does not,
    and the push is scaled by p_i -- so a pose with no chance of being selected gets no gradient.
    test_rank_loss.py checks this against autograd.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau!r}")
    n = len(s_vals)
    if n != len(dq_vals):
        raise ValueError(f"length mismatch: {n} scores, {len(dq_vals)} labels")
    if n == 0:
        return 0.0, [], 0
    z = [float(s) / tau for s in s_vals]
    zmax = max(z)                      # shift before exp; at tau 0.02 a score of 0.9 is z = 45
    ex = [math.exp(v - zmax) for v in z]
    tot = sum(ex)
    p = [v / tot for v in ex]
    ebar = sum(pi * float(d) for pi, d in zip(p, dq_vals))
    loss = -ebar
    c = [-(pi / tau) * (float(d) - ebar) for pi, d in zip(p, dq_vals)]
    return loss, c, n


def pair_terms_softplus(s_vals, dq_vals, tau=DEFAULT_TAU, weight_by_gap=True, dq_tol=1e-6):
    """(loss, coeffs, n_pairs) for the SCALED LOGISTIC form.

        m_ij = sign(dq_i - dq_j) * (s_i - s_j)              signed margin; > 0 means right order
        L    = (2 tau / W) * sum_ij w_ij softplus(-m_ij / tau)
        W    = sum_ij w_ij,     w_ij = |dq_i - dq_j|  (or 1 when weight_by_gap is False)

    and so the gradient weight on a pair is sigmoid(-m_ij / tau): ~1 when badly inverted, 1/2 at a
    tie, decaying to 0 once correct by a few tau.

    THE 2*tau PREFACTOR MAKES THE TIE GRADIENT MATCH THE LINEAR LOSS. Differentiating,

        dL/ds_i = -(2/W) sum_j w_ij sign(dq_i - dq_j) sigmoid(-m_ij / tau)

    and with w_ij = |dq_i - dq_j| the product w*sign collapses to (dq_i - dq_j), so at m = 0 where
    sigmoid is 1/2 this is -(1/W) sum_j (dq_i - dq_j) -- exactly pair_terms' coefficient. Without
    the prefactor the whole gradient would carry a spurious 1/tau, about 33x at the default tau,
    and a form change would be inseparable from a learning-rate change. test_rank_loss.py asserts
    this agreement.

    SAME LEAF-TENSOR AUTOGRAD as pair_terms, for the same reason: the closed form above is right
    only when no pair is dropped, and DockQ ties ARE dropped. Routing both forms through autograd
    keeps the loss and its derivative from drifting apart.

    UNLIKE pair_terms' COEFFICIENTS, THESE DEPEND ON s_vals. That is the point -- it is what lets
    the loss stop pushing a pair it has already ordered -- and it is why this form needs pass A.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau!r}")
    n = len(s_vals)
    s = torch.tensor(s_vals, dtype=torch.float64, requires_grad=True)
    num, den, npair = s.new_zeros(()), 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = dq_vals[i] - dq_vals[j]
            if abs(d) <= dq_tol:
                continue
            w = abs(d) if weight_by_gap else 1.0
            m = (s[i] - s[j]) * (1.0 if d > 0 else -1.0)
            num = num + w * torch.nn.functional.softplus(-m / tau)
            den += w
            npair += 1
    if npair == 0:
        return 0.0, [0.0] * n, 0
    loss = (2.0 * tau / den) * num
    loss.backward()
    return float(loss.detach()), [float(g) for g in s.grad], npair


def coeffs_softplus(s_vals, dq_vals, tau=DEFAULT_TAU, weight_by_gap=True, dq_tol=1e-6):
    """(c, n_pairs) where c_i = dL/ds_i for the softplus form. REQUIRES the current scores.

    The linear `coeffs` takes labels only; this one cannot, and the asymmetry is the whole design
    argument. A gradient that ignores the current scores is a gradient that is constant in them,
    which is a loss that is linear in them, which is a loss with no stopping point. Needing s_vals
    here is the price of having somewhere to stop.
    """
    _, c, npair = pair_terms_softplus(s_vals, dq_vals, tau=tau,
                                      weight_by_gap=weight_by_gap, dq_tol=dq_tol)
    return c, npair


PAIR_FAMILIES = ("cross", "within", "all")


def pair_included(ki, kj, pairs):
    """Does the pair of group keys (seed, sample) belong to the `pairs` family?

    THE FAMILY IS PART OF THE OBJECTIVE, NOT A FILTER ON ONE. Cross-seed and within-seed pairs
    sit at different ipTM scales -- over the 497-structure training set after the DockQ-tie filter,
    median |gap| is 0.0305 cross-seed against 0.0036 within-seed, 8.5x; on overfit30, 0.0426
    against 0.0137. Reproduce with `measure_tau.py --pairs {cross,within}` against the pool being
    TRAINED on, never against val.

    ONE TAU CAN SERVE BOTH, which an earlier version of this note denied. At tau 0.0305 an
    overfit30-median within-seed pair weights 0.376 correct against 0.624 inverted (1.66x), and
    even at the full training set's tighter 0.0036 it is 0.467 / 0.533 with both directions
    pushing toward the better-DockQ pose. A wide tau costs saturation, not gradient. What the
    within-seed family actually lacks is SHARE -- 7.9% of pairs, 7.0% of pairwise gradient mass --
    which `within_weight` addresses directly and measurably. Reach for a second tau only to change
    the correct/inverted RATIO, not to make within-seed pairs count for more.
    """
    if pairs not in PAIR_FAMILIES:
        raise ValueError(f"unknown pair family {pairs!r}, expected one of {PAIR_FAMILIES}")
    if pairs == "all":
        return True
    same_seed = (ki[0] == kj[0])
    return (not same_seed) if pairs == "cross" else same_seed


def pair_straddles(di, dj, thresh):
    """Does this pair cross the acceptability threshold -- exactly one pose above it?

    thresh <= 0 turns the restriction OFF and every pair passes, which is the default and what
    every run before this flag did.

    WHY RESTRICT TO THESE PAIRS. The reported criterion is the FRACTION of interfaces whose
    selected pose clears DockQ 0.23, and that number changes only when selection moves a pose
    from one side of 0.23 to the other. A pair of two acceptable poses, or of two unacceptable
    ones, can be reordered freely without moving the fraction at all. Ordering every straddling
    pair correctly is exactly equivalent to never selecting an unacceptable pose when an
    acceptable one is present in the group, which is the whole of the headroom that the fraction
    metric can see.

    WHAT IT COSTS. Mean DockQ is NOT invariant to the pairs this drops: promoting 0.61 over 0.38
    raises mean DockQ and leaves the fraction untouched, and this filter discards that gradient.
    So the arm optimises the fraction and treats mean DockQ as a free variable, which is a real
    change of objective and not merely a sparser estimate of the same one. The two numbers are
    reported side by side for that reason.

    THE THRESHOLD IS A DockQ THRESHOLD, applied to the LABELS, never to the scores. It therefore
    partitions pairs by a fixed property of the data: the set of included pairs does not move as
    the model trains, so unlike rank_decay_weights this does not make the objective a function of
    the model's current ordering and cannot oscillate.
    """
    if not thresh or thresh <= 0:
        return True
    # >=, NOT >, because the reported metric is `1.0 if best_dockq >= 0.23 else 0.0`
    # (rank_rho.py). DockQ is stored to 3-4 decimals and 26 rows sit at exactly 0.2300, so a pose
    # on the line is "acceptable" to the criterion; a `>` here would call it unacceptable and the
    # filter would be optimising a threshold the scoreboard does not use. Worth 116 of 143,680
    # pairs -- immaterial to any result, but there is no reason for the two to disagree.
    return (di >= thresh) != (dj >= thresh)


def rank_decay_weights(sv, k):
    """v(r) = exp(-(r-1)/k) over the group's CURRENT ipTM order, r = 1 for the top pose.

    Returns a list aligned with sv, or None when k <= 0 (decay off).

    WHY IT EXISTS. Selection is argmax ipTM over the group, and the selected pose changes only
    when some pose's ipTM crosses the current top's -- so the pairs (top, j) are the entire
    selection-relevant subspace. Measured on valfull at the shipped head, those pairs are 1.33%
    of all pairs and carry 1.56% of the loss's weight, while ordering them all correctly is
    equivalent to the top pose being the best pose, i.e. the FULL oracle (0.5831 -> 0.6926 DockQ
    on 92 interfaces). The objective therefore spends ~98% of its weight on orderings deployment
    never consults.

    WHY A DECAY RATHER THAN A SPIKE ON RANK 1. The weight would then be a step function of the
    scores: whenever the argmax flips, the objective's identity changes discontinuously and the
    run can oscillate -- demote A, B becomes top, demote B, A returns. With k = 2 a rank-1/rank-2
    swap moves their weights 1.00 <-> 0.61 instead of 1.00 <-> 0. k -> 0 recovers the pure star,
    k -> inf recovers the unweighted objective, so k interpolates between them.

    THE PAIR WEIGHT USES max(), NOT the product -- see group_coeffs. A product would annihilate
    the (rank 1, deep pose) pairs, and those are exactly the ones carrying the headroom: the
    best-DockQ pose sits at a median rank of 68 of 150.

    STOP-GRADIENT BY CONSTRUCTION. The ranks come from the detached float scores, so the weight
    is treated as a constant and the loss is not the exact gradient of any fixed objective. That
    is deliberate and is the same choice every ranking method of this family makes; it is
    recorded here so it is not mistaken for an oversight.
    """
    if k is None or k <= 0:
        return None
    order = sorted(range(len(sv)), key=lambda i: -sv[i])
    v = [0.0] * len(sv)
    for r, i in enumerate(order):
        v[i] = math.exp(-r / float(k))
    return v


def group_coeffs(s_by_key, dq_by_key, form="softplus", tau=DEFAULT_TAU,
                 weight_by_gap=True, dq_tol=1e-6, pairs="cross", within_weight=1.0,
                 rank_decay_k=0.0, straddle=0.0):
    """Coefficients for ONE interface over poses drawn from SEVERAL SEEDS.

    s_by_key / dq_by_key are {(seed, sample): float} over the whole group -- after pass A's
    all_gather, every rank holds the same dicts and so computes the same coefficients.

    Returns ({(seed, sample): c_i}, n_pairs, loss).

    CROSS-SEED BY DEFAULT; `pairs` selects the family -- see pair_included. Deployment takes the
    argmax over ALL poses of an interface regardless of seed, so cross-seed ordering is the
    objective and within-seed ordering is a proxy for it, and the five earliest arms trained the
    within-seed proxy alone without beating shipped. Those arms ran the LINEAR form with no tau at
    all, so they do not test a within-seed softplus; "within" exists to make that a separable
    experiment rather than an assumption.

    `within_weight` SCALES WITHIN-SEED PAIRS UNDER pairs="all", in both the numerator and the
    normaliser W, so the result stays a weighted mean and the coefficients still sum to zero.
    It exists because within-seed pairs are under-REPRESENTED, not under-GRADIENTED: at tau
    0.0305 a median within-seed pair on overfit30 (gap 0.0137) already gets 0.376 if correctly
    ordered against 0.624 if inverted, which is real discrimination. The deficit is share --
    6.7% of the pairwise gradient mass -- and share is what this fixes. A multiplier is a better
    knob for that than a second tau: it is monotone in the quantity being targeted, so it can be
    set from a measurement, while tau changes the shape of the weighting and its effect on mass
    share is not monotone. Measured on overfit30 at tau 0.0305, within-seed mass share goes
    6.7% -> 17.6% -> 36.0% -> 58.0% at L = 1, 3, 8, 20, while mass reaching inverted pairs falls
    only 82.7% -> 80.2% at L=8.

    NOTE THE SEED COUNT MOVES THE BASELINE. Within-seed pairs are 10*C(5,2) of C(50,2) = 8.2% of
    an interface at the 10 training seeds, but only 2.7% at the 30 seeds the held-out evaluation
    uses. L is calibrated against the TRAINING shape.

    PAIRS ARE DROPPED, NOT ZEROED, when excluded -- they leave the normaliser W as well as the
    numerator, so the loss stays a mean over the pairs it actually uses.
    """
    keys = sorted(set(s_by_key) & set(dq_by_key))
    if len(keys) < 2:
        return {}, 0, 0.0
    sv = [float(s_by_key[k]) for k in keys]
    dv = [float(dq_by_key[k]) for k in keys]

    n = len(keys)
    # EXPECTED-DockQ IS PAIR-FREE and so leaves before the pair loop. It needs no DockQ-tie
    # dropping (a tie is two poses with equal utility, which the expectation handles), and it
    # cannot honour `pairs` or weight_by_gap -- see expected_dockq. Differentiated by
    # autograd here, like the other forms, so the value and its gradient cannot drift apart;
    # the closed form is checked against this in the tests.
    if form == "expected":
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau!r}")
        s = torch.tensor(sv, dtype=torch.float64, requires_grad=True)
        d = torch.tensor(dv, dtype=torch.float64)
        loss = -(torch.softmax(s / tau, dim=0) * d).sum()
        loss.backward()
        return ({k: float(g) for k, g in zip(keys, s.grad)}, n, float(loss.detach()))

    if tau <= 0 and form == "softplus":
        raise ValueError(f"tau must be positive, got {tau!r}")
    s = torch.tensor(sv, dtype=torch.float64, requires_grad=True)
    _vdec = rank_decay_weights(sv, rank_decay_k)
    num, den, npair = s.new_zeros(()), 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if not pair_included(keys[i], keys[j], pairs):
                continue
            d = dv[i] - dv[j]
            if abs(d) <= dq_tol:
                continue
            if not pair_straddles(dv[i], dv[j], straddle):
                continue
            w = abs(d) if weight_by_gap else 1.0
            if keys[i][0] == keys[j][0]:
                w *= within_weight
            if _vdec is not None:
                w *= max(_vdec[i], _vdec[j])
            if form == "softplus":
                m = (s[i] - s[j]) * (1.0 if d > 0 else -1.0)
                num = num + w * torch.nn.functional.softplus(-m / tau)
            elif form == "linear":
                # w, NOT d. The old form folded the gap weighting straight into the numerator as
                # `d`, which equalled sign(d)*w only while w was exactly |d|. within_weight and
                # the rank decay break that: they entered `den` but not here, so each knob became
                # a uniform GAIN on the ranking term rather than a reweighting -- measured as
                # every coefficient scaled by one constant, 0.239x at L=8, i.e. the opposite of
                # the "L cannot act as a second alpha_rank" property the flag promises.
                num = num - (s[i] - s[j]) * (1.0 if d > 0 else -1.0) * w
            else:
                raise ValueError(f"unknown form {form!r}")
            den += w
            npair += 1
    # den == 0 WITH npair > 0 BECAME REACHABLE WITH THE NEW KNOBS. It used to be impossible:
    # w was |dDockQ| and every counted pair had |dDockQ| > dq_tol, so npair > 0 implied den > 0.
    # within_weight=0 (the natural way to write "drop the within-seed family") and a tiny
    # rank_decay_k (whose weights underflow to exactly 0.0 away from the top pose) both break it.
    # Unguarded, this path divided by zero: group_coeffs returned inf/nan COEFFICIENTS -- which
    # the pass-A non-finite check does not see, because it tests the scores, not the coefficients
    # -- while group_loss raised ZeroDivisionError. The two implementations must never disagree,
    # so both now report the group as carrying no weighted pair.
    if npair == 0 or den <= 0.0:
        return {}, 0, 0.0
    loss = (2.0 * tau / den) * num if form == "softplus" else num / den
    loss.backward()
    return ({k: float(g) for k, g in zip(keys, s.grad)}, npair, float(loss.detach()))


def group_loss(s_by_key, dq_by_key, form="softplus", tau=DEFAULT_TAU,
               weight_by_gap=True, dq_tol=1e-6, pairs="cross", within_weight=1.0,
               rank_decay_k=0.0, straddle=0.0):
    """The loss VALUE for one interface, in plain floats. No autograd, no graph.

    Same objective and same normalisation as group_coeffs -- asserted equal in test_rank_loss.py --
    but it returns only (loss, n_pairs) and never the coefficients.

    WHY A SECOND PATH EXISTS. The held-out diagnostic needs the value for every interface at every
    epoch: 18 interfaces x ~1,225 cross-seed pairs x 2 (trained and shipped) is ~44,000 terms per
    epoch, and routing those through a traced leaf tensor to then discard the gradient made
    recomputing a 50-epoch history take minutes. Training still uses group_coeffs, because there
    the gradient IS the point and the autograd path is what keeps the loss and its derivative from
    drifting apart.
    """
    keys = sorted(set(s_by_key) & set(dq_by_key))
    if len(keys) < 2:
        return 0.0, 0
    # PAIR-FREE FORM, value only. Same closed form the trainer differentiates through autograd.
    if form == "expected":
        _l, _c, _n = expected_dockq([s_by_key[k] for k in keys], [dq_by_key[k] for k in keys],
                                   tau=tau)
        return _l, _n
    if tau <= 0 and form == "softplus":
        raise ValueError(f"tau must be positive, got {tau!r}")
    _vdec = rank_decay_weights([float(s_by_key[k]) for k in keys], rank_decay_k)
    _ix = {k: i for i, k in enumerate(keys)}
    num, den, npair = 0.0, 0.0, 0
    for i in range(len(keys)):
        ki = keys[i]
        si, di = float(s_by_key[ki]), float(dq_by_key[ki])
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if not pair_included(ki, kj, pairs):
                continue
            d = di - float(dq_by_key[kj])
            if abs(d) <= dq_tol:
                continue
            if not pair_straddles(di, float(dq_by_key[kj]), straddle):
                continue
            w = abs(d) if weight_by_gap else 1.0
            if ki[0] == kj[0]:
                w *= within_weight
            if _vdec is not None:
                w *= max(_vdec[_ix[ki]], _vdec[_ix[kj]])
            if form == "softplus":
                x = -((si - float(s_by_key[kj])) * (1.0 if d > 0 else -1.0)) / tau
                # log1p(exp(x)) without overflowing for large x, which happens at a badly
                # inverted pair: at x = 700 math.exp overflows and the honest answer is x.
                num += w * (x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x)))
            elif form == "linear":
                # w, NOT d -- see group_coeffs. The two paths shared the bug identically, which
                # is exactly why their agreement test could not see it.
                num -= (si - float(s_by_key[kj])) * (1.0 if d > 0 else -1.0) * w
            else:
                raise ValueError(f"unknown form {form!r}")
            den += w
            npair += 1
    # den == 0 with npair > 0 -- see group_coeffs. Both paths must answer identically, and
    # before this guard group_coeffs returned inf coefficients here while this one raised.
    if npair == 0 or den <= 0.0:
        return 0.0, 0
    return ((2.0 * tau / den) * num if form == "softplus" else num / den), npair
