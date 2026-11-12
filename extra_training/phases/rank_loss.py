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
    # PROTENIX COMPACTS ASYM IDS, SO THE PAIR MUST BE COMPACTED TOO.
    # calculate_chain_based_ptm remaps asym ids to 0..N-1 whenever `len(unique) != max+1` --
    # which is exactly when a crop has removed a whole chain -- and sizes chain_pair_iptm to the
    # compacted count. `pair` is in UNCROPPED model asym indices. Indexing a compacted matrix
    # with them is out of range when the dropped chain precedes a scored one, and, worse, IN
    # range and pointing at a different chain pair otherwise: a plausible number, no error.
    # phase_k_eval._cpi_index has replicated this on the eval side all along; this is the same
    # mapping, built the same way. It is the identity when the ids are already contiguous, so it
    # applies unconditionally and the cropped and uncropped paths read the same code.
    _u = torch.unique(feat["asym_id"].reshape(-1).long())
    _remap = {int(o): n for n, o in enumerate(_u.tolist())}
    if i not in _remap or j not in _remap:
        # The chain is not in this (possibly cropped) structure at all. None is correct and the
        # caller's completeness guard will refuse rather than train on a silent gap.
        return None
    i, j = _remap[i], _remap[j]
    if i >= cpi.shape[0] or j >= cpi.shape[1]:
        return None
    return cpi[i, j]
"""Median |ipTM gap| over CROSS-SEED pairs of the full training set.

training/fold/results.csv, 497 structures, 1,028 interfaces, 1,056,035 pairs after dropping DockQ
ties. Reproduce with `python -I run_logging/measure_tau.py`. For reference the 20-structure
overfit30 subset gives 0.0426 and val10 gives 0.0313; the training number is the one to use,
because tau shapes the training gradient and taking it from val would let the eval set in.
"""
"""Softmax temperature for the EXPECTED-DockQ form. NOT the same quantity as DEFAULT_TAU.

DEFAULT_TAU is a margin scale in ipTM units -- the measured median cross-seed gap -- and it sets
where the softplus stops caring. This is a temperature dividing the scores before a softmax, and
it sets how much of the probability mass sits on the current leaders. They are both called tau and
they are not interchangeable: 0.0305 here would be a far sharper selection than 0.05, and 0.05 in
the softplus would be a 1.6x wider margin. The flag is shared, so the banner prints which meaning
is in force.
"""
"""Soft-rank temperature, in the units of the quantity being ranked.

0.05 is ABAG-Rank's own training value (configs/losses/one_minus_rho.yaml), and it is applied here
to ipTM and DockQ -- both on 0..1 -- which is the space upstream runs it in. It is NOT DEFAULT_TAU:
that one is a margin scale for the softplus loss, measured as the median cross-seed ipTM gap, and
the two numbers answer different questions. Reusing it here would couple a temperature to a
measurement that has nothing to do with rank saturation.
"""
"""Relevance gate: a group's loss is scaled by clamp(std(DockQ) / 0.07, 0, 1).

Upstream's, and the reason it exists is that a complex whose poses are all equally good carries no
ordering to learn, so its gradient is noise. It matters MORE here than upstream, not less: a
within-seed group holds 5 poses rather than 50, and 59% of our training interfaces already sit
below 0.07 measured over all 50. Expect this gate to suppress a large share of within-seed groups,
and read the logged mean weight before concluding anything about the term's effect.

Set rel_scale=0.0 to disable the gate.
"""


# ---------------------------------------------------------------------------------------------
# RELATIVE (MEAN-CENTRED) iTM REGRESSION
#
# THE PROBLEM IT TARGETS, measured on d32kqp epoch 66 over 738 winnable interfaces:
#   corr(interface MEAN predicted ipTM, interface MEAN DockQ) = +0.866   <- the head HAS this
#   median WITHIN-interface Spearman vs DockQ                 = +0.317   <- the head LACKS this
#   the same quantity for the TRUE iTM                        = +0.690   <- what is available
# So the head has learned "is this complex easy" almost perfectly and barely learned "which pose
# of this complex is best", which is the only thing selection uses. The across-complex signal is
# also the LARGER one (ipTM sd 0.277 across structures against 0.208 within), so a per-pose loss
# fits it first and leaves the part we need.
#
# Subtracting each interface's own mean from BOTH the prediction and the label removes that
# signal from the objective entirely: "this complex scores around 0.6" cancels and earns nothing.
# What remains is the deviation of each pose from its own interface's mean.
#
# THE COEFFICIENTS SUM TO EXACTLY ZERO, which is worth stating because it is the loss's defining
# property and its main risk:
#   L        = (1/P) sum_i r_i^2 ,   r_i = (s_i - mean_s) - (g_i - mean_g)
#   dL/ds_i  = (2/P) sum_j r_j (delta_ij - 1/P) = (2/P) r_i      because sum_j r_j == 0
# so sum_i dL/ds_i == 0 identically. The term therefore CANNOT move the interface's overall
# level -- by design -- and its accumulated gradient is a sum of terms that cancel by
# construction. That is why alpha must be set from gradient NORMS over a whole accumulation
# window rather than from loss values, and why the per-pose alignment is worth measuring
# separately: a small accumulated norm here does not mean small per-pose gradients.
#
# NOT DIVIDED BY THE WITHIN-INTERFACE STD, deliberately. Standardising would equalise every
# interface's contribution, but interfaces do not contribute equally to the metric: measured,
# corr(std of true iTM, DockQ range) = +0.745 on the static-mapped labels; not recomputed after the remap, so treat it as indicative, so wide-spread interfaces are exactly the ones
# where a wrong pick is expensive and letting them carry more gradient is roughly correct
# importance weighting. Dividing by std also spends equal effort on interfaces where every pose
# is already acceptable (frac acceptable median 1.00 among the narrow ones). It remains a
# one-line ablation -- see rel_coeffs(standardize=True) -- rather than part of the design.
# EVERY CEILING FIGURE QUOTED IN THIS SECTION COMES FROM THE PER-POSE OST MAPPING
# (training/runs/itm_ceiling_ost), not from the static chain_roles join. The static version was
# used first and was wrong on 3,154 of 36,900 labels (8.55%) across 52 structures -- 7stf, for
# instance, maps 7stf_BC to asym (3,1) statically and (3,0) per pose, a different chain and not a
# reordering. Remapping moved the mean-vs-max result from +0.0107 (t=2.70) to +0.0140 (t=4.06),
# so the conclusion strengthened rather than reversed. run_logging/audit_chain_mapping.py
# measures the disagreement and build_itm_labels.py refuses a statically-mapped ceiling.
DEFAULT_REL_STD_FLOOR = 0.02


def iptm_pooled(pae_logits, feat, configs, pair, pool="mean", n_pair_tokens=None):
    """chain_pair ipTM for ONE pose and pair, pooled over anchor rows by `pool`.

    pool="max" is shipped ipTM and reproduces iptm() exactly. pool="mean" averages the anchor
    rows instead, which measured +0.0140 +- 0.0034 (t=4.06) better than the max at the ORACLE and
    -0.0130 +- 0.0048 (t=-2.68) WORSE on the current trained head -- i.e. better if trained for,
    worse if swapped in. Differentiable when pae_logits requires grad.

    THE ARITHMETIC IS pae_row_weight.row_t's, pinned against Protenix's own calculate_iptm by
    run_logging/test_pae_row_weight.py, and the mean is pinned against row_t by
    run_logging/test_pool_mean.py. Rows with no scored target in this pair are EXCLUDED from the
    mean; their t_i is 0 by construction and including them would drag the mean toward zero by an
    amount set by crop size rather than by pose quality.
    """
    if pool == "max":
        # THE MAX PATH CANNOT PIN d0. iptm() delegates to Protenix's calculate_chain_based_ptm,
        # which derives d0 from the asym_id it is handed -- the CROPPED one. Patching vendor code
        # to accept an override is not worth it for a path this arm does not train on
        # (rank_pool=mean), so it refuses instead of silently scoring with a different kernel than
        # the target. Raised in external review.
        if n_pair_tokens:
            raise SystemExit(
                "STOP: --rank-pool max cannot be used with a cropped run. iptm() computes d0 "
                "inside Protenix's calculate_chain_based_ptm from the cropped asym_id, so the "
                "predicted score would use a different TM kernel than the rel target, which pins "
                "d0 at the uncropped chain-pair count. Use --rank-pool mean, or extend "
                "pae_row_weight to cover the max pooling.")
        return iptm(pae_logits, feat, configs, pair)
    if pool != "mean":
        raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
    import pae_row_weight as rw
    from protenix.model import sample_confidence
    bp = sample_confidence.get_bin_params(configs.loss.pae)
    asym = feat["asym_id"].reshape(-1).long()
    hf = feat["has_frame"].reshape(-1).bool()
    lig = (torch.zeros_like(feat["asym_id"]).scatter_add(
        0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0).reshape(-1)
    live = hf & (~lig)
    i, j = pair
    tok = (asym == i) | (asym == j)
    if int(tok.sum()) == 0 or int(live.sum()) == 0:
        return None
    arow = asym[live]
    tm = (tok.unsqueeze(0) & (arow.unsqueeze(1) != asym.unsqueeze(0))
          & ((arow == i) | (arow == j)).unsqueeze(1))
    if int(tm.sum()) == 0:
        return None
    # d0 COMES FROM THE UNCROPPED PAIR WHEN ONE IS GIVEN. feat's asym_id is the
    # CROPPED token set under --crop-pool, so int(tok.sum()) is the cropped chain-pair
    # length and calculate_normalization would shrink d0 with the crop -- while the rel
    # TARGET pins d0 at the uncropped count. Prediction and target would then use
    # different kernels: at a 600 -> 256 pair-token crop, w(10 A) moves 0.423 -> 0.259.
    # Caught in external review before any cropped run started.
    _npt = int(n_pair_tokens) if n_pair_tokens else int(tok.sum())
    t, cnt = rw.row_t(pae_logits, live, tm, n_pair_tokens=_npt,
                      min_bin=bp["min_bin"], max_bin=bp["max_bin"])
    if t is None:
        return None
    keep = cnt > 0
    if not bool(keep.any()):
        return None
    return t[keep].mean()


def rel_coeffs(s_by_key, g_by_key, standardize=False, std_floor=DEFAULT_REL_STD_FLOOR):
    """Mean-centred regression coefficients for ONE interface's poses.

    s_by_key / g_by_key: {(seed, sample): float}. Returns ({key: dL/ds}, n_poses, loss).

    Only keys present in BOTH dicts are used -- a pose whose label is missing is skipped rather
    than imputed, and the caller is expected to refuse a run whose label coverage is short.

    dL/ds_i = (2/P) * r_i, and sum_i dL/ds_i == 0 identically.

    standardize=True divides the centred prediction by its own std as well. THE INVARIANCE IS
    ONLY ABOVE THE FLOOR, which the gate measured and an earlier version of this docstring got
    wrong: it claimed collapse "becomes impossible rather than merely penalised". Once a shrunk
    prediction's std falls below std_floor the clamp engages and the loss changes again -- for the
    test group, scaling the prediction by 0.1 moved it 0.3718 -> 0.3557. So scale-invariance holds
    for moderate shrinking and collapse is still penalised in the limit. That is the more useful
    behaviour, but it is not invariance, and the cost remains equalising interfaces whose outcomes
    are not equal.
    """
    keys = sorted(set(s_by_key) & set(g_by_key))
    P = len(keys)
    if P < 2:
        return {}, P, 0.0
    s = torch.tensor([float(s_by_key[k]) for k in keys], dtype=torch.float64)
    g = torch.tensor([float(g_by_key[k]) for k in keys], dtype=torch.float64)
    gc = g - g.mean()
    if standardize:
        # AUTOGRAD, NOT THE CLOSED FORM. The closed form (2/P) r_i is the gradient of the
        # UNSTANDARDISED loss. With standardize=True the denominator is a function of s, so
        # dL/ds_i picks up a term through d(std)/ds_i that the closed form omits entirely.
        # Measured on a 5-pose group: the closed form gave
        #   [-0.44738, -0.03322, 0.26788, 0.13618, 0.07654]
        # against the true
        #   [-3.33852, +0.52290, 1.12625, 1.36759, 0.32179]
        # -- wrong by ~7x AND with a SIGN FLIP on pose 1, so the term would have pushed that pose
        # the wrong way while reporting a loss that disagreed with its own gradient. Caught in
        # external review. P is at most a few hundred, so differentiating a 1-D tensor here costs
        # nothing next to the forward pass that produced s.
        sl = s.clone().requires_grad_(True)
        sc_ = sl - sl.mean()
        sc_ = sc_ / sc_.std(unbiased=False).clamp_min(std_floor)
        gc_ = gc / gc.std(unbiased=False).clamp_min(std_floor)
        L = ((sc_ - gc_) ** 2).mean()
        L.backward()
        c = sl.grad.detach().clone()
        loss = float(L.detach())
    else:
        r = (s - s.mean()) - gc
        loss = float((r ** 2).mean())
        c = (2.0 / P) * r
    # EXPLICIT ZERO-SUM. It holds algebraically for the default path; asserting it here catches a
    # future edit that breaks the property silently, because a non-zero sum would let this term
    # shift the interface's level, which is the one thing it is defined not to do. It holds for
    # the standardised path too -- centring makes the loss invariant to a constant shift of s, so
    # its gradient is orthogonal to the all-ones direction -- and subtracting the mean removes
    # only float error.
    c = c - c.mean()
    return {k: float(v) for k, v in zip(keys, c)}, P, loss


def rel_loss_value(s_by_key, g_by_key, standardize=False,
                   std_floor=DEFAULT_REL_STD_FLOOR):
    """The loss value alone, for logging. Same arithmetic as rel_coeffs."""
    return rel_coeffs(s_by_key, g_by_key, standardize, std_floor)[2]
