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
pairs, the median |ipTM gap| is 0.0081 and the 90th percentile 0.034, while softplus only bends
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
3.9x wider than the within-seed 0.0081, so the within-seed figure does not transfer. At tau=0.0305
inverted pairs are 43.7% of pairs BY COUNT and take 81.6% of the PAIRWISE SCORE-SPACE GRADIENT
MASS, against 40.4% under the DockQ-weighted linear loss -- a redistribution of +41.3 points. That
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
overfit30 subset gives 0.0429 and val10 gives 0.0313; the training number is the one to use,
because tau shapes the training gradient and taking it from val would let the eval set in.
"""


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


def group_coeffs(s_by_key, dq_by_key, form="softplus", tau=DEFAULT_TAU,
                 weight_by_gap=True, dq_tol=1e-6, cross_seed_only=True):
    """Coefficients for ONE interface over poses drawn from SEVERAL SEEDS.

    s_by_key / dq_by_key are {(seed, sample): float} over the whole group -- after pass A's
    all_gather, every rank holds the same dicts and so computes the same coefficients.

    Returns ({(seed, sample): c_i}, n_pairs, loss).

    CROSS-SEED ONLY, BY DEFAULT. Within-seed pairs are the thing the five earlier arms already
    trained and failed to improve, and they are 3.9x tighter than cross-seed pairs so a single tau
    cannot serve both. More to the point, deployment selects argmax over ALL poses of an interface
    regardless of seed, so cross-seed ordering is the objective and within-seed ordering is a
    proxy for it. Pass cross_seed_only=False to include both.

    PAIRS ARE DROPPED, NOT ZEROED, when excluded -- they leave the normaliser W as well as the
    numerator, so the loss stays a mean over the pairs it actually uses.
    """
    keys = sorted(set(s_by_key) & set(dq_by_key))
    if len(keys) < 2:
        return {}, 0, 0.0
    sv = [float(s_by_key[k]) for k in keys]
    dv = [float(dq_by_key[k]) for k in keys]

    n = len(keys)
    if tau <= 0 and form == "softplus":
        raise ValueError(f"tau must be positive, got {tau!r}")
    s = torch.tensor(sv, dtype=torch.float64, requires_grad=True)
    num, den, npair = s.new_zeros(()), 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if cross_seed_only and keys[i][0] == keys[j][0]:
                continue
            d = dv[i] - dv[j]
            if abs(d) <= dq_tol:
                continue
            w = abs(d) if weight_by_gap else 1.0
            if form == "softplus":
                m = (s[i] - s[j]) * (1.0 if d > 0 else -1.0)
                num = num + w * torch.nn.functional.softplus(-m / tau)
            elif form == "linear":
                num = num - (s[i] - s[j]) * (d if weight_by_gap else (1.0 if d > 0 else -1.0))
            else:
                raise ValueError(f"unknown form {form!r}")
            den += w
            npair += 1
    if npair == 0:
        return {}, 0, 0.0
    loss = (2.0 * tau / den) * num if form == "softplus" else num / den
    loss.backward()
    return ({k: float(g) for k, g in zip(keys, s.grad)}, npair, float(loss.detach()))
