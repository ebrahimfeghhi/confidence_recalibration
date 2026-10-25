"""PHASE K -- the FT3 finetune: train the confidence head on our own pose distribution.

WHAT THIS OPTIMISES, AND WHAT IT IS JUDGED ON. The head predicts pLDDT, PAE, PDE and resolved. Only
PAE matters for the thing we are measured on: sample_confidence.py builds EVERY ipTM number out of
the PAE logits (`pae_prob = logits_to_score(pae_logits, ...)`, then `calculate_iptm(pae_prob, ...)`
and `calculate_chain_based_ptm(pae_prob, ...)`), and `chain_pair_iptm` is what selects a pose. So
`alpha_pae` is not a tuning knob here -- at Protenix's shipped default of 0.0 the PAE term is
multiplied by zero, ipTM cannot move, and the whole experiment is a no-op whose total loss still
falls because pLDDT, PDE and resolved keep training. Their own comment says `# or 1 in finetuning
stage 3`. This script REFUSES to start if any of the four weights is zero.

The other three terms stay on as regularisers: they keep the head honest about per-atom and
distance-error confidence instead of letting it distort everything to fit PAE.

THE TARGET, measured 2026-10-15 over 92 validation interfaces at 100 seeds
(analysis/val_selection_metric.py): selecting by max pair_iptm gives mean DockQ 0.5775 and
frac-acceptable 0.7283, against an oracle of 0.7171 / 0.8696. So there is 0.1392 of mean DockQ
available to a better ranker AT ZERO EXTRA COMPUTE, and that gap is the deliverable. Note ipTM
selection PEAKS near 25 seeds and DECAYS to 100 while the oracle keeps climbing -- more sampling
produces better poses and a greater chance confidence prefers a confidently wrong one.

WHY THE FORWARD PASS IS IMPORTED, NOT WRITTEN. phase_j_replay.py already assembles the model, sets
train_confidence_only, builds the loss, runs the head and backpropagates, and it is the GATE that
proved a cached unit reproduces live FT3 exactly -- 0.00e+00 on the label and on every loss term over
5 units and 25 poses. A trainer that rebuilt that path could drift from the thing that was verified,
so this imports the same helpers the replay and the rescorer use.

WHAT IS FROZEN, ASSERTED RATHER THAN ASSUMED. 19.8M of 464.4M parameters (4.3%) are the confidence
head. protenix.py's train_confidence_only puts the trunk in eval mode, but it does NOT tell the
optimiser what it may touch, so the optimiser is given the head's parameters by module-tree name and
--grad-check-every steps re-assert that no gradient has appeared outside it. The replay measured the
expected answer: 165 head tensors with a gradient, 0 outside.

ONE UNIT IS ONE (structure, seed) WITH ITS 5 POSES. The dataset item is per STRUCTURE and is
expensive to build, so units are grouped by structure and the item is built once per structure --
which also means the unit order is not shuffled freely; see --shuffle.
"""
import argparse
import collections
import copy
import queue
import threading
import csv
import datetime as _dt
import functools
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))
print = functools.partial(print, flush=True)

TERMS = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss", "loss")


# THE COMMIT THIS PROCESS IS ACTUALLY RUNNING, read once at import and never again.
#
# It used to be read at save time, which is wrong in the one case that matters: run 9mfb4y launched
# at 1aedd04b, HEAD moved four commits while it ran, and resume_ep3.pt therefore recorded commit
# 7b744e7a for weights produced entirely by 1aedd04b's code. Python had the old module in memory;
# only the git working tree had moved. A provenance field that reports the tree instead of the
# running code is worse than no field, because it will be believed.
_COMMIT_AT_START = None


def _git_commit():
    """The commit THIS PROCESS loaded its code from, cached at first call during startup."""
    global _COMMIT_AT_START
    if _COMMIT_AT_START is not None:
        return _COMMIT_AT_START
    import subprocess
    try:
        _COMMIT_AT_START = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        _COMMIT_AT_START = "unknown"
    return _COMMIT_AT_START


def head_param_names(model):
    """The parameters FT3 may move, taken from the model's own module tree.

    Not a hand-written list: the confidence head is its own module, so its name prefix is the honest
    definition and a rename upstream surfaces as "no module named *confidence_head" rather than as a
    silently empty optimiser.
    """
    prefixes = [n for n, _m in model.named_modules() if n.endswith("confidence_head")]
    if not prefixes:
        raise SystemExit("STOP: the model has no module named *confidence_head, so the set of "
                         "trainable parameters cannot be established from the module tree.")
    names = [n for n, _p in model.named_parameters()
             if any(n == p or n.startswith(p + ".") for p in prefixes)]
    if not names:
        raise SystemExit(f"STOP: {prefixes} matched no parameters; the optimiser would be empty.")
    return prefixes, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--labels", default="", help="default <fold>/labels")
    ap.add_argument("--out", default=f"{ROOT}/training/ft3", help="checkpoints and trajectory")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/training/prep/job_bioassembly")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa")
    ap.add_argument("--seq-index", default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--crop-size", type=int, default=3400)
    ap.add_argument("--alpha-pae", type=float, default=1.0,
                    help="FT3 wants 1. At Protenix's shipped 0.0 the PAE term is multiplied by zero "
                         "and ipTM -- the only number this project is judged on -- cannot move.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Protenix ships TWO: configs_base.py has 0.0018 and their own "
                         "finetune_demo.sh passes 0.001. FT3 is a finetune, so 0.001 is the default "
                         "here. Neither was tuned for training 4.3%% of the model at batch 256, so "
                         "--lr-probe exists to measure it rather than inherit it.")
    ap.add_argument("--warmup-steps", type=int, default=20,
                    help="SHORT BECAUSE THE RUN IS SHORT. Protenix's finetune_demo.sh warms up over "
                         "2,000 steps of a 100,000-step run (2%%). At batch 256 three epochs of our "
                         "cache is 292 optimiser steps, so 2,000 would be 685%% of the run and the "
                         "learning rate would never reach its set value. 20 is 6.8%%.")
    ap.add_argument("--lr-final", type=float, default=0.0,
                    help="COSINE DECAY TARGET. 0 (default) keeps the historical schedule: warm up "
                         "over --warmup-steps and then hold --lr flat for the whole run, which is "
                         "what every arm before this one did. Set it and the lr follows "
                         "lr_final + (lr - lr_final)*(1+cos(pi*p))/2 for p the fraction of the "
                         "run's EXPECTED optimiser steps completed after warmup, so it reaches "
                         "--lr-final on the last step. WHY IT EXISTS: 8fkrsu and zg5fnw both sat "
                         "at a fixed distance from their L2-SP anchor for five epochs (5.55 and "
                         "~20) while still MOVING 3-12 per epoch, with gradient norms steady and "
                         "training loss flat -- run_logging/weight_travel.py reports 8fkrsu's "
                         "straightness at 0.227 against 0.462 for a free random walk of the same "
                         "steps, i.e. confined and retracing. That is an optimiser oscillating at "
                         "the limit its step size can resolve, and neither run ever decayed: lr "
                         "was constant 0.01 from step 20 to the end because no decay existed.")
    ap.add_argument("--grad-probe", action="store_true",
                    help="MEASURE, DON'T GUESS. Run EXACTLY ONE accumulation window -- the unit "
                         "count is derived as accum_local/nsample, not asked for, because a probe "
                         "that runs past the step mixes gradients measured at two different "
                         "parameter states and the whole point is to compare the two terms at "
                         "IDENTICAL weights. For each pose take the PAE "
                         "backward and the ranking backward SEPARATELY at identical weights, then "
                         "report per-tensor gradient norms, the cosine between the two, and "
                         "whether every trainable tensor received a finite non-zero ranking "
                         "gradient. Then take one real step and exit 0.\n"
                         "WHY THIS AND NOT A SHORT RUN. --alpha-rank cannot be set from loss "
                         "values: a ranking loss of 0.01-0.25 against a PAE loss of 2.3 says "
                         "nothing about their gradient magnitudes on the weights. Comparing "
                         "max grad_norm across two short runs is also wrong, because the combined "
                         "norm can be SMALLER than either when the gradients oppose -- which is "
                         "exactly what the cosine answers. And it is the only check that the "
                         "gradient reaches the trainable matrices THROUGH "
                         "calculate_chain_based_ptm at all; every offline test so far has "
                         "validated the label side, which cannot establish that.\n"
                         "Two scale constants chosen without this (--l2-sp 1e-2 on 10,240 "
                         "parameters, and softplus's implicit sigma=1) were both wrong.")
    ap.add_argument("--rank-loss", action="store_true",
                    help="PAIRWISE RANKING LOSS on chain_pair_iptm, the statistic selection "
                         "actually uses. The FT3 PAE loss is a per-pair accuracy objective; "
                         "selection is an argmax of ipTM over an interface's poses. Those are "
                         "different functionals of the same logits and they move independently: "
                         "measured mean within-interface Spearman(ipTM, DockQ) is 0.196 for the "
                         "shipped head, 0.271 for the pose-share arm at epoch 50, and 0.078 for "
                         "the full head at epoch 30 -- BELOW untrained. Training PAE taught the "
                         "big arms to rank worse. This trains the ordering instead. Needs DockQ "
                         "labels (--rank-labels) and --pae-scored-pairs, which is what names the "
                         "interface whose ipTM is read.")
    ap.add_argument("--alpha-rank", type=float, default=1.0,
                    help="weight on the ranking term. It is ADDED to the PAE term, so "
                         "--alpha-pae 0 --rank-loss trains the ordering alone and "
                         "--alpha-pae 1 --rank-loss trains both. NOT TUNED: the two terms are "
                         "different quantities (a PAE loss near 2.3 against a softplus ranking "
                         "loss whose tied value is ln2 = 0.693), so this ratio is a guess until a "
                         "sweep says otherwise.")
    ap.add_argument("--rank-labels", default=None,
                    help="results.csv carrying per-pose DockQ for the TRAINING poses. Defaults to "
                         "<fold>/results.csv, which holds 51,350 labelled poses over 1,028 "
                         "interfaces. Keyed (pdb, seed, asym pair, sample) -- per INTERFACE, not "
                         "per structure, because a structure holds several and reading one "
                         "interface's ipTM against another's DockQ is the error phase_k_eval "
                         "refuses to run.")
    ap.add_argument("--rank-weight-by-gap", dest="rank_weight_by_gap",
                    action="store_true", default=True,
                    help="weight each pose pair by |DockQ_i - DockQ_j| (default). Separating 0.1 "
                         "from 0.7 then matters more than separating 0.41 from 0.43, which is "
                         "what the metric cares about: frac_acc is a threshold at 0.23 and mean "
                         "DockQ is driven by the big misses.")
    ap.add_argument("--no-rank-weight-by-gap", dest="rank_weight_by_gap",
                    action="store_false", help="weight every pose pair equally instead.")
    ap.add_argument("--rank-form", choices=("linear", "softplus", "expected"), default="linear",
                    help="SHAPE of the ranking term. linear and softplus are PAIRWISE; expected "
                         "is not.\n"
                         "expected is L = -sum_i softmax(ipTM/tau)_i * DockQ_i over every pose of "
                         "the interface: the expected DockQ of the pose the model would SELECT. "
                         "It exists because a pairwise loss spends gradient in proportion to how "
                         "many pairs a pose appears in, which is the same for rank 1 and rank 90, "
                         "while deployment takes the argmax and nothing else -- so ordering the "
                         "bottom of the list is work that cannot pay. h7msge is the evidence: "
                         "held-out pairwise loss reached a run best and Spearman went +0.075 over "
                         "shipped, while selected DockQ stayed flat-to-down and frac acceptable "
                         "never beat shipped once in 37 epochs. The softmax is a differentiable "
                         "argmax, so tau sets how much mass sits on the current leaders; it has "
                         "NO PAIRS, so --rank-weight-by-gap and the cross-seed pair filter have "
                         "nothing to act on and are reported as inapplicable rather than silently "
                         "appearing to work. Its floor is -max(DockQ) for the group, not 0.\n"
                         "linear (default, what every arm so far ran) is "
                         "-(s_i - s_j)(dq_i - dq_j)/sum|ddq|. Its coefficients come from the "
                         "LABELS ALONE, so it needs no scoring pass -- and that convenience IS "
                         "its defect: a gradient that ignores the current scores is constant in "
                         "them, so the loss has no stopping point and cannot tell an "
                         "already-correct pair from an inverted one. Scaling all scores by lambda "
                         "scales the loss by lambda, so WIDENING THE SPREAD PAYS WITHOUT LIMIT, "
                         "and at the real gap scale spreading is worth ~71x more than reordering. "
                         "Measured in ntwjfp: spread 0.0114 -> 0.0371 over 8 epochs while held-out "
                         "rank loss went +0.00018 -> +0.00288 -- the training objective improved "
                         "while held-out ORDERING got worse.\n"
                         "softplus is (2 tau/W) sum w_ij softplus(-m_ij/tau), whose per-pair "
                         "gradient weight sigmoid(-m/tau) decays to 0 once a pair is comfortably "
                         "correct and caps at 1 when badly inverted. Spreading then buys a bounded "
                         "amount and stops. Costs the scoring pass (--rank-form softplus implies "
                         "one), which is exactly the price of having somewhere to stop.")
    ap.add_argument("--rank-tau", type=float, default=None, metavar="TAU",
                    help="softplus temperature, the margin scale over which the loss goes from "
                         "'push hard' to 'satisfied'. Defaults to rank_loss.DEFAULT_TAU = 0.0305, "
                         "the MEASURED median |ipTM gap| over cross-seed pairs of the full "
                         "training set (497 structures, 1,028 interfaces, 1,056,035 pairs); "
                         "reproduce with run_logging/measure_tau.py. NOT a free parameter: tau=1 "
                         "is the units error the first softplus attempt made (gradient 0.495 for a "
                         "correct pair vs 0.505 for an inverted one -- inert), and tau far below "
                         "the gap scale ignores every pair that is not an outright inversion. "
                         "The within-seed median is 0.0036 and does NOT transfer: cross-seed pairs "
                         "are 8.5x wider. (An earlier 0.0081 was measured before DockQ ties "
                         "were dropped; reproduce either with measure_tau.py --pairs within.) "
                         "The scale is POPULATION-SPECIFIC and must come from the pool being "
                         "TRAINED on: overfit30 gives within 0.0137 / cross 0.0426, while "
                         "valfull gives within 0.0011 -- setting tau from val would let the "
                         "eval set shape the training gradient.")
    ap.add_argument("--rank-pairs", choices=("cross", "within", "all"), default="cross",
                    metavar="FAMILY",
                    help="WHICH PAIRS THE LOSS FORMS INSIDE A GROUP. This is the OBJECTIVE; "
                         "--rank-cross-seed is the STEP SHAPE, and they are easy to confuse. "
                         "--rank-cross-seed decides which poses land in one optimiser step (all "
                         "seeds of a structure, so the group is the whole interface); "
                         "--rank-pairs decides which pairs drawn from that group enter the loss. "
                         "'cross' pairs poses from DIFFERENT seeds and is the deployed objective, "
                         "since selection is argmax ipTM over every pose regardless of seed. "
                         "'within' pairs poses from the SAME seed -- a separate family at a "
                         "separate scale, NOT a subset reading of the same one: over the 497-"
                         "structure training set the median |ipTM gap| is 0.0036 within-seed "
                         "against 0.0305 cross-seed, 8.5x, so --rank-tau MUST be re-measured for "
                         "it (measure_tau.py --pairs within, against the TRAINING pool; overfit30 "
                         "gives 0.0137). Leaving tau at the cross-seed value puts every "
                         "within-seed pair at m/tau ~ 0, where sigmoid gives a correct pair and "
                         "an inverted one the same weight and the term is inert. 'all' takes "
                         "both at ONE tau, which by the same argument can only suit one of them; "
                         "it is here for completeness, not as a recommendation. NOTE the five "
                         "earliest arms trained within-seed pairs and did not beat shipped, but "
                         "they ran the LINEAR form with no tau at all, so they do not test this.")
    ap.add_argument("--rank-topk-decay", type=float, default=0.0, metavar="K",
                    help="weight each pair by max(v(r_i), v(r_j)) with v(r) = exp(-(r-1)/K) over "
                         "the group's CURRENT ipTM order, r=1 the top pose. 0 disables it. "
                         "WHY: selection is argmax ipTM over the group, so the selected pose "
                         "changes only when some pose crosses the current top -- the pairs "
                         "(top, j) are the ENTIRE selection-relevant subspace. Measured on "
                         "valfull at the shipped head they are 1.33%% of pairs and 1.56%% of the "
                         "loss's weight, while ordering them all correctly is identical to the "
                         "top pose being the best pose, i.e. the full oracle (DockQ 0.5831 -> "
                         "0.6926 over 92 interfaces). MAX, NOT PRODUCT: a product would "
                         "annihilate the (rank 1, deep pose) pairs, which are the ones carrying "
                         "the headroom -- the best-DockQ pose sits at a median rank of 68 of "
                         "150. K INTERPOLATES: K->0 is the pure star (rank-1 pairs only), K->inf "
                         "is the undecayed objective, and K=2 puts 39.3%% of the weight on the "
                         "star and 92.8%% inside the top 5 while a rank-1/rank-2 swap moves "
                         "their weights 1.00 <-> 0.61 rather than 1.00 <-> 0. A spike on the "
                         "argmax would make the objective's identity change discontinuously "
                         "whenever the top pose flips, which invites oscillation. STOP-GRADIENT: "
                         "the ranks come from detached scores, so this is not the exact gradient "
                         "of a fixed objective -- deliberate, and recorded so it is not read as "
                         "an oversight.")
    ap.add_argument("--rank-straddle", type=float, default=0.0, metavar="DOCKQ",
                    help="keep ONLY pairs that straddle this DockQ threshold -- exactly one pose "
                         "above it, one below. 0 disables it (every run before this flag). "
                         "WHY: the reported criterion is the FRACTION of interfaces whose "
                         "selected pose clears 0.23, and that fraction moves only when selection "
                         "crosses the threshold. Two acceptable poses, or two unacceptable ones, "
                         "can be reordered freely without changing it. COST, stated rather than "
                         "hidden: mean DockQ is NOT invariant to the dropped pairs -- promoting "
                         "0.61 over 0.38 raises the mean and leaves the fraction alone -- so this "
                         "optimises the fraction and treats mean DockQ as a free variable. That "
                         "is a change of objective, not a sparser estimate of the same one. "
                         "The threshold applies to LABELS, never scores, so the included set is "
                         "fixed by the data and cannot oscillate as the model trains. "
                         "Needs its own tau: measure_tau.py --straddle.")
    ap.add_argument("--rank-within-weight", type=float, default=1.0, metavar="L",
                    help="with --rank-pairs all, multiply WITHIN-SEED pairs by L in both the "
                         "numerator and the normaliser W, so the loss stays a weighted mean and "
                         "L cannot act as a second --alpha-rank. Fixes a SHARE problem, not a "
                         "gradient-strength one: at tau 0.0305 a median within-seed pair on "
                         "overfit30 (gap 0.0137) already gets 0.376 correctly ordered against "
                         "0.624 inverted, which discriminates fine -- but within-seed pairs are "
                         "only 7.9%% of pairs and 7.0%% of the pairwise gradient mass, so "
                         "deployment-relevant orderings inside a seed are barely trained. "
                         "CALIBRATED ON THE FULL TRAINING SET, not on the pool being trained, "
                         "so the knob does not move when the pool does -- the same rule "
                         "DEFAULT_TAU follows. Within-seed mass share by L at tau 0.0270: "
                         "1 -> 7.0%%, 3 -> 18.3%%, 8 -> 37.1%%, 20 -> 59.3%%, while mass reaching "
                         "INVERTED pairs falls only 75.0%% -> 71.6%% at L=8. overfit30 gives "
                         "6.7/17.6/36.0/58.0, within a point, so nothing hinges on the pool. "
                         "Reproduce with run_logging/within_weight.py. Ignored unless "
                         "--rank-pairs is 'all' (under 'within' every pair is scaled alike and L "
                         "normalises away).")
    ap.add_argument("--rank-cross-seed", action="store_true",
                    help="RANK ACROSS SEEDS, not just within one. Deployment selects argmax of "
                         "ipTM over ALL poses of an interface regardless of seed, so cross-seed "
                         "ordering IS the objective and within-seed ordering is a proxy. Measured "
                         "variance split: ipTM is 82.3%% between-seed while DockQ is 54.6%% "
                         "pose-level.\n"
                         "This regroups the epoch so that ALL seeds of a structure land in one "
                         "optimiser step, which makes the ranking group the whole interface (50 "
                         "poses, ~1,225 cross-seed pairs) rather than a 3-seed sample. Measured "
                         "agreement between a subsample's gradient direction and the full group's, "
                         "as cosine over shared poses (400 interfaces, 8 draws each): 3 seeds mean "
                         "0.856 with p10 0.544, 5 seeds mean 0.925 with p10 0.823, 10 seeds exact. "
                         "Costs nothing in batch size -- a step is ~20 units either way -- so it "
                         "trades structures per step for an exact ranking signal.")
    ap.add_argument("--rank-structs-per-step", type=int, default=2, metavar="S",
                    help="with --rank-cross-seed, how many STRUCTURES share an optimiser step. "
                         "S structures x all their seeds x --nsample poses is the step. Default 2 "
                         "= 2 x 10 x 5 = 100 poses, matching the ~100 the current schedule already "
                         "uses. S x seeds need not divide the rank count; the short rank rotates "
                         "and the totals are checked at startup rather than being allowed to "
                         "deadlock.")
    ap.add_argument("--l2-sp", type=float, default=0.0, metavar="LAMBDA",
                    help="L2-SP: add LAMBDA * ||W - W_shipped||^2 to the loss, penalising DISTANCE "
                         "FROM THE SHIPPED HEAD rather than distance from zero. Plain weight decay "
                         "pulls toward zero, which is not a prior anyone holds about this head -- "
                         "the shipped weights are. This is the continuous form of --train-only and "
                         "--lora-rank: those forbid motion outright, this prices it. The anchor is "
                         "written to <out>/l2sp_anchor.pt on a fresh run and read back on a resume, "
                         "because a resumed run's current weights are NOT the shipped ones. "
                         "SCALE: the full head drifted ||dW||^2 = 67,052 over 30 epochs (mdv9x7), "
                         "so LAMBDA 1e-5 prices that drift at ~0.67 against a PAE loss near 2.0. "
                         "LAMBDA IS NOT TUNED; it is the hyperparameter of the experiment.")
    ap.add_argument("--l2-sp-exclude", default="",
                    help="comma-separated REGEXES matched against trainable parameter names; "
                         "matching tensors stay TRAINABLE but are left OUT of the L2-SP anchor.\n"
                         "THE CASE THIS EXISTS FOR is a tensor whose init is zero. L2-SP prices "
                         "distance from the SHIPPED weights, which is a prior worth holding for a "
                         "pretrained matrix; for a newly added zero-init projection the same "
                         "penalty is plain decay toward zero, i.e. it penalises the very term "
                         "being added. --frame-direction's frame_dir_proj and pose_perblock's "
                         "orientation copies are exactly that: zero-init by construction and "
                         "verified exactly zero at install.\n"
                         "IT DOES NOT MAKE THEM UNREGULARISED BY ACCIDENT -- excluded tensors "
                         "are named in the startup log and the count is reported, because an "
                         "anchor silently covering fewer tensors than the reader believes is the "
                         "same failure class as a --train-only that creates parameters and never "
                         "trains them.")
    ap.add_argument("--train-only", default="",
                    help="SELECTIVE FINETUNING: comma-separated REGEXES matched against parameter "
                         "names inside the confidence head; only matching tensors train and "
                         "everything else is frozen. Empty (default) trains the whole head.\n"
                         "POSE-FOCUSED ADAPTATION is the motivating case: "
                         "'linear_no_bias_d,linear_no_bias_d_wo_onehot,linear_no_bias_pae' is "
                         "13,312 parameters -- 0.07%% of the head -- and is the COMPLETE pose "
                         "pathway plus the PAE readout. Pose enters the head exactly once, as "
                         "cdist over representative atoms feeding those two distance embeddings "
                         "(both, not either), and leaves as linear_no_bias_pae(pae_ln(z_pair)). "
                         "Freezing the four pairformer blocks keeps the pretrained structural "
                         "representation and adapts only how geometry ENTERS and how PAE is READ "
                         "OUT -- a constraint on which computation may change, not merely on how "
                         "much. Add 'pae_ln' to make the readout's scale and shift learnable too "
                         "(+256 parameters).")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="LoRA RANK; 0 (default) trains the head the way every run so far has. "
                         "Above 0, every matched weight is FROZEN and learns a rank-r additive "
                         "update instead: W + (alpha/r) B@A. B is zero-initialised, so step 0 is "
                         "bit-identical to the shipped head and LoRA-on against LoRA-off is a "
                         "clean single variable. The point is capacity: full FT3 moves 19.8M "
                         "parameters on 488 structures, and the 30-structure anchor showed what "
                         "that does -- train PAE -38%% while held-out PAE rose 51%%. Rank 8 over "
                         "the whole head is ~4.4%% of its parameters.")
    ap.add_argument("--lora-alpha", type=float, default=0.0,
                    help="LoRA scaling numerator; the update is multiplied by alpha/rank. 0 "
                         "(default) means 2*rank, the usual choice, so the effective scale is 2.")
    ap.add_argument("--lora-targets", default="",
                    help="comma-separated REGEXES matched against module names inside the "
                         "confidence head; empty adapts every module with a 2-D weight. Matching "
                         "on the name rather than the class is deliberate -- 54 modules are "
                         "Linear/LinearNoBias but the ~96 matrices in tri_mul_* and tri_att_* are "
                         "a different class and hold 1.57M parameters between them.")
    ap.add_argument("--lora-dropout", type=float, default=0.0,
                    help="dropout on the adapter input only; the frozen path is untouched.")
    ap.add_argument("--structures-per-step", type=int, default=0,
                    help="K: WHOLE STRUCTURES PER OPTIMISER STEP. Each rank takes ONE seed of each "
                         "of the same K structures, so a step is K x world x --nsample poses. 0 "
                         "derives K from --accum as the nearest whole number of structures, which "
                         "is what every existing launch script wants: --accum 85 over 3 ranks at "
                         "nsample 5 resolves to K=6 and 90 poses per step.")
    ap.add_argument("--accum", type=int, default=256,
                    help="POSES PER OPTIMISER STEP -- the effective batch. The micro-batch is forced "
                         "to 1: token counts run 263-3,400 across our structures, so poses cannot be "
                         "stacked into one tensor without padding the head's pair representations. "
                         "256 at the user's direction. For reference, Protenix's own finetune runs "
                         "one structure per step with iters_to_accumulate=1 and scales by world_size; "
                         "256 is AF3's paper figure, for full-model training across many devices. At "
                         "256 an epoch of our 25,000 poses is 97 steps.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--eval-every-epoch", action="store_true",
                    help="score the validation metric at the end of every epoch. THE METRIC IS THE "
                         "RESULT, not the loss: mean DockQ and frac-acceptable of the pose selected "
                         "by max pair_iptm, against the 0.5775 / 0.7283 baseline at 100 seeds.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="also score every N optimiser steps. 0 disables. A full pass is 5,300 "
                         "validation units at 0.82 s/unit = ~73 min, MEASURED on GATE 1 2026-10-16 (73 units/min over 5,300). An earlier 0.187 s/unit here came from a small probe of small structures with the model already warm and was 4.4x optimistic, so at 97 steps/epoch this "
                         "is normally left off in favour of --eval-every-epoch.")
    ap.add_argument("--train-eval-pool", default="",
                    help="score the head on THIS pool of TRAINING interfaces at epoch boundaries, "
                         "in addition to (or instead of) validation. The question `is the model "
                         "learning at all` is about the training set, and --eval-every-epoch only "
                         "knows how to score validation. Uses the run's own --fold and --seeds, so "
                         "it scores exactly the poses it trained on. Sharded over every card the "
                         "run was given, with the other ranks parked at a barrier, same as the "
                         "validation path.")
    ap.add_argument("--train-eval-every", type=int, default=2,
                    help="epochs between training-set evaluations. Ignored without "
                         "--train-eval-pool.")
    ap.add_argument("--train-eval-fold", default="",
                    help="score --train-eval-pool against THIS fold instead of --fold. The point "
                         "is to watch a held-out signal every few epochs without paying for the "
                         "whole validation set: a 10-structure validation pool is ~1,000 units "
                         "against 5,300, so minutes rather than ~25. Defaults to --fold, which is "
                         "the in-sample training-pool behaviour. Pair it with "
                         "--train-eval-seed-base, because a different fold was folded at "
                         "different seeds, and with --train-eval-budget, which otherwise defaults "
                         "to the count of THIS run's --seeds and would silently score a tenth of "
                         "a 100-seed fold.")
    ap.add_argument("--train-eval-seed-base", type=int, default=-1,
                    help="the first seed of --train-eval-fold. -1 (default) uses this run's own "
                         "--seeds, which is right only when the pool lives in --fold.")
    ap.add_argument("--train-eval-budget", type=int, default=0,
                    help="seeds per interface for the training-set metric. 0 derives it from "
                         "--seeds, which is the only honest default: the pool has no more depth "
                         "than the run trained on.")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop when this many consecutive epochs fail to beat the best epoch-mean "
                         "TRAINING loss by --early-stop-min-delta. 0 disables. The decision is made "
                         "from the cross-rank mean, so every rank reaches it identically -- a stop "
                         "that differed between ranks would deadlock them at the next barrier.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.002,
                    help="how much an epoch must beat the running best by to count as progress. "
                         "The loss falls ~0.03/epoch EARLY and the rule compares against the best "
                         "so far rather than the previous epoch, because individual epochs rise "
                         "while the trend falls (+0.0255 at ep5, +0.0154 at ep13). 0.01 was the "
                         "first default and it was too coarse: late in a run the per-epoch gain "
                         "drops to ~0.003, which is real progress the rule cannot see, so the "
                         "strike counter can only ever run out. It ended krah8z at ep29 and owdyyp "
                         "at ep57, both still trending down, and krah8z's ep28 missed resetting "
                         "the counter by 0.0001 (1.9674 against a 1.9673 threshold). 0.002 is "
                         "below the late-run gain and still above the epoch-to-epoch noise.")
    ap.add_argument("--frame-direction", action="store_true",
                    help="ADD A THIRD POSE PATHWAY to the confidence head: for each ordered pair "
                         "(i, j), the UNIT direction of j in i's backbone frame, through one "
                         "zero-initialised LinearNoBias(3 -> c_z) added into z_pair beside the two "
                         "distance paths that are already there. See docs/pose_pathways.md. The "
                         "head's existing view of the pose is cdist over ONE representative atom "
                         "per token, so the N and C atoms that define a residue's frame are not in "
                         "its input at all and the frame's orientation is ABSENT, not merely "
                         "implicit -- while PAE is defined on exactly that orientation. Zero-init "
                         "means step 0 is bit-identical to the head without it, but it adds a "
                         "trainable tensor, so an existing resume bundle CANNOT be loaded and the "
                         "evaluator must be given --frame-direction too (this passes it). "
                         "phases/phase_k_frame_direction_check.py checks the geometry on CPU.")
    ap.add_argument("--pose-perblock", action="store_true",
                    help="PATHWAY A: re-inject the pose embedding before EVERY pairformer block "
                         "instead of only before the first. One zero-initialised "
                         "LinearNoBias(num_bins -> c_z) per injected block -- 3 matrices at "
                         "39x128 = 14,976 parameters on a 4-block stack. Adds ACCESS, not "
                         "information: the one-hot is the same tensor each time, so the "
                         "hypothesis is that the pose signal degrades crossing the stack, which "
                         "with 4 blocks is thin and is why --frame-direction was built first. "
                         "Zero-init makes step 0 bit-identical, but it adds trainable tensors, so "
                         "an existing resume bundle CANNOT be loaded and the evaluator must be "
                         "given --pose-perblock too (this passes it). See docs/pose_pathways.md.")
    ap.add_argument("--pose-perblock-tied", action="store_true",
                    help="share ONE matrix across every injected block instead of one each. The "
                         "sharper test of A's actual hypothesis -- the identical embedding "
                         "re-presented at each depth -- where untied also lets each depth learn a "
                         "different projection of distance, so a win is ambiguous between the "
                         "two. NOT equivalent to scaling a single injection: each block "
                         "transforms z before the next add. Requires --pose-perblock.")
    ap.add_argument("--pose-perblock-copy", action="store_true",
                    help="UNTIED, BUT EVERY DEPTH STARTS AS A COPY OF BLOCK 0's INJECTION. Each "
                         "injected block gets its own linear_no_bias_d AND "
                         "linear_no_bias_d_wo_onehot, initialised to block 0's pretrained weights "
                         "-- 3 x (9,984 + 256) = 30,720 new parameters on a 4-block stack. "
                         "STEP 0 IS BIT-IDENTICAL TO --pose-perblock-share's STEP 0, not to the "
                         "shipped head, and that is the point: against a share arm the ONLY "
                         "variable is capacity, where the zero-init --pose-perblock would change "
                         "the starting function too. Verified at install against block 0's "
                         "weights exactly, not approximately. Mutually exclusive with "
                         "--pose-perblock-tied and --pose-perblock-share, and requires "
                         "--pose-perblock. THE COPIES ARE NAMED pose_perblock_linear_no_bias_d*, "
                         "so a --train-only that does not match them creates them and never "
                         "trains them -- pass e.g. "
                         "--train-only 'linear_no_bias_d,pose_perblock'.")
    ap.add_argument("--pose-perblock-share", action="store_true",
                    help="SHARE BLOCK 0's OWN PRETRAINED INJECTION at every pairformer block "
                         "instead of adding new matrices: blocks 1..n-1 re-apply "
                         "linear_no_bias_d AND linear_no_bias_d_wo_onehot verbatim, both terms, "
                         "because block 0 adds both. ZERO new parameters -- the whole experiment "
                         "is one matrix pair applied at four depths, which is the literal form of "
                         "'tie the weights so they are the same across all 4'. Requires "
                         "--pose-perblock and is mutually exclusive with --pose-perblock-tied. "
                         "STEP 0 IS NOT BIT-IDENTICAL to the shipped head: blocks 1..n-1 gain an "
                         "injection they never had, from weights that are already non-zero. That "
                         "is inherent to the mode, so this arm is a different ARCHITECTURE "
                         "evaluated from the shipped weights, not a one-variable change to them. "
                         "Pair it with --l2-sp to price how far that shared injection may move.")
    ap.add_argument("--allow-unmasked-fallback", action="store_true",
                    help="accept training structures that --pae-scored-pairs does not name, whose "
                         "poses then train the UNMASKED PAE term. Refused by default: the running "
                         "guard only fires while nothing has been masked yet, so after one "
                         "successful unit any number of silent fallbacks can follow, and review "
                         "reproduced exactly that. Use it only when the mismatch is intended.")
    ap.add_argument("--prefetch-workers", type=int, default=1,
                    help="HOW MANY crop workers run concurrently. One is not enough: epochs 7 "
                         "and 8 hit a 100%% prefetch hit rate and still reported 11-16 min per "
                         "epoch WAITING on the worker, because a single thread needs ~0.7 s per "
                         "unit against ~0.84 s of GPU and the GIL eats that margin. The crop did "
                         "not disappear, it moved from the main thread into the wait. Workers "
                         "take units round-robin, so results arrive out of order and the holding "
                         "buffer is sized for it.")
    ap.add_argument("--prefetch-units", type=int, default=0,
                    help="prepare this many units ahead on a worker thread, overlapping the CPU "
                         "crop with the GPU's work on the previous unit. 0 disables it. Measured "
                         "on epoch 5 of the full set: crop_slice is 23.0 min of a 73.1 min epoch "
                         "with the GPU idle throughout, and GPU busy time is only ~20 min of 73. "
                         "The worker never touches CUDA -- it hands back host tensors and the main "
                         "thread moves them -- so only one thread ever issues CUDA work.")
    ap.add_argument("--val-loss-every", type=int, default=0,
                    help="EPOCHS BETWEEN HELD-OUT PAE PASSES; 0 is off. This is the monitor the "
                         "run had no way to compute: measured 2026-10-20, the lr 1e-2 anchor's "
                         "training PAE fell 38%% from epoch 1 to 142 while its val10 PAE ROSE 51%% "
                         "(2.8290 -> 4.2639), and nothing in the run could see it -- the DockQ "
                         "metric stayed inside its noise band the whole time because ipTM barely "
                         "orders poses within an interface (rho 0.20 even for the shipped head). "
                         "At 500 poses it costs ~3 min against a 37-min full-set epoch.")
    ap.add_argument("--val-loss-fold", default="", help="fold root for the held-out pass")
    ap.add_argument("--val-loss-labels", default="", help="defaults to <val-loss-fold>/labels")
    ap.add_argument("--val-loss-selection", default="", help="selection csv; needs a `tokens` column")
    ap.add_argument("--val-loss-pool", default="", help="pool csv for the PAE mask on the held-out side")
    ap.add_argument("--val-loss-preflight", default="", help="that fold's alignment preflight marker")
    ap.add_argument("--val-loss-seeds", default="", help="e.g. 400000-400009")
    ap.add_argument("--val-loss-prep", default="",
                    help="prep root holding job_indices.csv, job_bioassembly and job_cifs for that "
                         "fold. The dataset config defaults to the TRAINING prep, and pointing it "
                         "at the wrong one fails loudly -- 10 structures with no datapoint -- "
                         "rather than measuring the wrong molecules.")
    ap.add_argument("--val-loss-card", default="",
                    help="card for the held-out pass. Empty shares the trainer's; it loads a second "
                         "464M-parameter model, so the free-memory floor below applies.")
    ap.add_argument("--crop-all-poses", action="store_true",
                    help="DEFINE THE CROP FROM THE UNION OVER ALL 5 POSES instead of pose 0. The "
                         "union is what interface_crop_tokens has always done and it exists as "
                         "insurance: Protenix permutes symmetric chains per pose, so in principle "
                         "pose 1's interface can involve a copy pose 0's does not. In practice it "
                         "never has -- the function's own docstring records the largest "
                         "|d(pose i) - d(pose 0)| over token representative atoms as 0.000000 A "
                         "across all 491 training structures x all 5 poses, with 77 of them having "
                         "at least two chains of equal token count, i.e. the case that could "
                         "differ. Re-checked on the current labels before this flag was added: "
                         "masks identical on every unit sampled. The union costs 5x the work for "
                         "that insurance -- a cKDTree build, a query_pairs iterated as a Python "
                         "set, and a per-token Python loop, five times -- measured at 17 ms "
                         "against 4 ms for pose 0 on a 128-token unit. So the default is pose 0 "
                         "and this flag buys the insurance back. ANY LABEL REBUILD THAT COULD "
                         "PERMUTE must be followed by phases/phase_k_crop_consistency.py, which "
                         "is the gate that would catch a divergence; that gate still compares "
                         "against the union regardless of this flag.")
    ap.add_argument("--manifest-shard", type=int, default=-1,
                    help="TAKE ONLY THIS SHARD of --manifest-selection, INTERLEAVED (i::n), with "
                         "--manifest-nshards. -1, the default, takes everything and is what every "
                         "existing caller gets, so adding this changes no behaviour unless asked "
                         "for. INTERLEAVED and not contiguous because the selection csv is "
                         "ordered and contiguous blocks hand one shard all the large complexes -- "
                         "phase_k_eval_sharded.py's own note, and the reason its speedup would "
                         "otherwise be set by the slowest block. Exists for the held-out PAE "
                         "monitor, which was a SINGLE-card child while two cards sat at a "
                         "barrier: 12.8 min of a 46.0 min epoch cycle in j583i8.")
    ap.add_argument("--manifest-nshards", type=int, default=0,
                    help="how many shards --manifest-shard is one of. 0 disables sharding.")
    ap.add_argument("--loss-only", action="store_true",
                    help="FORWARD AND LOSS ONLY, no backward, no optimiser, no weight change -- one "
                         "pass over the manifest. For measuring a saved head's loss on a HELD-OUT "
                         "fold, which nothing else could do: the eval path reports DockQ selection "
                         "and never the loss, so a 37%% drop in training PAE had no held-out "
                         "counterpart to compare against. Runs in the trainer rather than a second "
                         "script so the crop, the --pae-scored-pairs mask, the --pae-only stub and "
                         "the loss weights are the SAME code -- a reimplementation would drift and "
                         "the two numbers would stop being comparable, which is the only thing this "
                         "measurement is for. Writes a trajectory like any run, so the pose-weighted "
                         "epoch PAE is computed from it by the identical formula.")
    ap.add_argument("--pae-only", action="store_true",
                    help="train the PAE term ALONE: pLDDT, PDE and resolved are stubbed out, not "
                         "computed, not trained. ipTM derives entirely from PAE, and at the shipped "
                         "weights PAE is only ~30%% of the loss (pLDDT 42%%, PDE 27%%), so most of "
                         "the gradient goes to outputs the metric never reads. THE RISK IS REAL "
                         "THOUGH: all four outputs come off one shared confidence head, so the "
                         "auxiliary terms may be regularising the representation PAE reads from. "
                         "Composes with --pae-scored-pairs, which narrows WHICH PAE pairs count.")
    ap.add_argument("--pae-scored-pairs", default="",
                    help="a pool CSV. The PAE term is then computed ONLY on token pairs that cross "
                         "a chain pair this pool scores -- not on intra-chain pairs, and not on "
                         "chain pairs nothing grades (H-L above all). chain_pair_iptm reads exactly "
                         "one cross-block per interface, while the unmasked term averages over "
                         "every pair in the crop, so most of the PAE gradient is spent on blocks "
                         "the metric never looks at. Masks the loss, not the crop: same tokens, "
                         "same forward pass, no extra compute.")
    ap.add_argument("--pae-scored-roles", default="",
                    help="chain_roles.csv for --pae-scored-pairs. Defaults to <fold>/chain_roles.csv. "
                         "Only the SET of scored pairs matters here, and that set is unchanged by "
                         "swapping the two antibody chains, so the chemically-identical-chain "
                         "mislabelling that makes roles unusable for SCORING does not reach it.")
    ap.add_argument("--limit-units", type=int, default=0, help="debug: cap the number of units")
    ap.add_argument("--shuffle", default="structure",
                    choices=["structure", "none"],
                    help="`structure` shuffles the ORDER OF STRUCTURES and the units within each, "
                         "which is as much as can be shuffled while building each structure's "
                         "dataset item once. `none` is deterministic for debugging.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="500000-500009",
                    help="the seed range the manifest expects per structure, inclusive.")
    ap.add_argument("--manifest-selection",
                    default=f"{ROOT}/train_val_splits/cache_train_structures.csv",
                    help="THE FROZEN POPULATION. The manifest is this times --seeds, not whatever "
                         "label files happen to exist; a label that was never written must be an "
                         "error, not a silently smaller training set.")
    ap.add_argument("--manifest-skip", default="9mx1",
                    help="comma-separated pdb ids the manifest must NOT expect, each a deliberate "
                         "documented exclusion. 9mx1 is 2,941 residues against Protenix's "
                         "2,560-token limit, so it cannot be folded at all.")
    ap.add_argument("--preload-items", type=int, default=0,
                    help="how many STRUCTURE dataset items to hold in memory at once. 0 keeps one, "
                         "which rebuilds every structure every epoch. MEASURED 2026-10-15: an item "
                         "takes 3.6-10.4 s to build (avg 6.8) and occupies 331-886 MB (avg 518), so "
                         "500 structures cost ~57 min per epoch to rebuild and ~253 GB to hold. With "
                         "837 GB free that fits, and every epoch after the first saves the 57 min. "
                         "Use 0 for a single-epoch run, where there is nothing to amortise.")
    ap.add_argument("--max-nonfinite", type=int, default=10,
                    help="how many non-finite-gradient steps to tolerate before stopping. Each one "
                         "drops its gradients and takes no step; past this the data or the learning "
                         "rate is wrong and continuing trains on noise.")
    ap.add_argument("--max-rss-gb", type=float, default=300.0,
                    help="HARD CAP ON THIS PROCESS'S OWN MEMORY, checked before every item is added "
                         "to the cache. The user's constraint is that our processes do not exceed "
                         "300 GB, and --preload-min-free-gb answers a different question: it looks "
                         "at MemAvailable, so on an idle box it would let one run take 700 GB. Past "
                         "this the cache stops growing and evicts; the run never fails for it.")
    ap.add_argument("--preload-min-free-gb", type=float, default=200.0,
                    help="FLOOR ON FREE RAM, checked before every item is added to the cache -- not "
                         "once at startup. The cache GROWS to ~253 GB over the first epoch (500 "
                         "items at a measured 331-886 MB, avg 518), so a startup check would pass "
                         "and the box could still be driven out of memory later, taking the fold "
                         "workers with it. When MemAvailable falls below this the cache stops "
                         "growing and evicts instead; it never fails the run.")
    ap.add_argument("--grad-check-every", type=int, default=25,
                    help="re-assert that no gradient has appeared outside the confidence head")
    ap.add_argument("--save-every", type=int, default=250, help="optimiser steps per checkpoint")
    ap.add_argument("--mem-log", action="store_true",
                    help="print live GPU memory after every pose, with the token count. For "
                         "diagnosing whether memory ACCUMULATES (a retained reference) or merely "
                         "peaks (a capacity limit) -- three peak-reducing changes did not move when "
                         "the run died, which is not what a capacity limit looks like.")
    ap.add_argument("--drop-pde", action="store_true",
                    help="do not compute or train the PDE term. ipTM -- the number the project is "
                         "judged on -- is derived ENTIRELY from PAE, so PDE contributes nothing to "
                         "it while costing the same memory: both are [N_token, N_token, 64] and the "
                         "loss builds several tensors of that shape per term, one of them an int64 "
                         "one-hot. Zeroing its weight does NOT help, because aggregate_losses calls "
                         "each loss_fn BEFORE it reads the weight, so the module itself is replaced "
                         "with a stub. This is a deviation from Protenix's FT3 recipe, which trains "
                         "all four terms, and the auxiliary signal may regularise the shared "
                         "pairformer -- so it is a memory decision, not a free one.")
    ap.add_argument("--blocks-per-ckpt", type=int, default=1,
                    help="ACTIVATION CHECKPOINTING over the head's pairformer blocks. Chunking "
                         "shrinks the SCRATCH inside one operation; this discards the activations "
                         "kept BETWEEN blocks and recomputes them in the backward pass, which is "
                         "the other half of the memory and the larger half. 1 checkpoints every "
                         "block (most saved, about 30%% more compute). Protenix supports it "
                         "natively on PairformerStack and it engages only when gradients are "
                         "enabled, so the evaluator is unaffected. 0 disables it.")
    ap.add_argument("--alpha-confidence", type=float, default=1.0,
                    help="weight on the four confidence losses. Protenix ships 1e-4 because in full "
                         "AF3-style training these terms sit beside the diffusion and distogram "
                         "losses and must not perturb them. FT3 freezes the trunk and optimises "
                         "NOTHING ELSE, so that weight is a constant on the whole objective -- it "
                         "moves no minimum, but it is not harmless either: it pushed per-parameter "
                         "gradients to ~1.8e-07, at or below AdamW's eps of 1e-8, so eps dominated "
                         "the denominator and damped the typical step ~12x (MEASURED: a step at "
                         "lr 1e-3 moved parameters by 8.1e-05). Default 1.0, at the user's "
                         "direction on 2026-10-16, so lr means what it says.")
    ap.add_argument("--clip-norm", type=float, default=25.0,
                    help="gradient-norm clip. THIS SCALES WITH --alpha-confidence and must be set "
                         "with it. At the old 1e-4 weight the norm was 7.9e-04 against a threshold "
                         "of 1.0, so clipping never fired and was decorative. At weight 1.0 the "
                         "same gradients are norm 7.9 (median) and 11.8 (max, measured over this "
                         "run's first 8 steps), so a threshold of 1.0 would fire on EVERY step and "
                         "silently turn this into normalised gradient descent. 25 is ~3x the "
                         "measured median: it catches a genuine spike and never binds on a normal "
                         "step. It is a judgement call, derived from measurement rather than picked "
                         "in advance, and it should be revisited if the gradient scale changes.")
    ap.add_argument("--eval-cards", default="",
                    help="cards to SHARD the per-epoch evaluation over, comma separated. Default: "
                         "every card this run was given, since the other ranks are parked at a "
                         "barrier throughout and their cards would otherwise idle for the whole "
                         "~73 min pass. One card falls back to the unsharded evaluator.")
    ap.add_argument("--dist-timeout-min", type=float, default=240.0,
                    help="COLLECTIVE TIMEOUT IN MINUTES, and it must exceed the per-epoch "
                         "evaluation. torch's default is 30 min; the evaluation at --eval-budget "
                         "100 covers 5,300 validation units and takes ~73 min, measured. Rank 0 "
                         "evaluates while the other ranks wait at a barrier, so on the default the "
                         "barrier would time out and kill the run at the FIRST epoch boundary -- "
                         "after hours of correct training, with the failure looking like a network "
                         "fault rather than a timeout someone chose. 240 min leaves room for an "
                         "evaluation that runs long under contention.")
    ap.add_argument("--debug-save-rank-heads", action="store_true",
                    help="EVERY rank also writes head_final_rank<r>.pt. Off by default and not for "
                         "training runs: the replicas are supposed to be identical, so these files "
                         "are redundant in normal use. phase_k_train_ddp_selftest.py turns it on to "
                         "CHECK that they are identical after several steps, which is the only way "
                         "to catch replicas drifting apart -- a single-rank checkpoint cannot show "
                         "it, and drift is exactly what a missed all-reduce would cause.")
    ap.add_argument("--dist-backend", default="gloo", choices=("gloo", "nccl"),
                    help="the collective backend for multi-GPU. DEFAULT gloo, NOT nccl, because "
                         "NCCL is broken on this box: a 20-line two-rank all_reduce on cards 1 and "
                         "3 fails with `CUDA error: an illegal memory access was encountered`, with "
                         "P2P enabled and disabled alike, on NCCL 2.26.2+cuda12.2 under a torch "
                         "built for 12.8 and driver 595.99.02. It is not this script -- the same "
                         "failure reproduces with no model loaded. gloo is a CPU backend, so the "
                         "gradients are staged through host memory: measured 60 ms per 79 MB "
                         "all-reduce, about 6 s per epoch over ~98 steps, against an epoch of "
                         "hours. Switch to nccl if the driver/NCCL pairing is ever fixed.")
    ap.add_argument("--nsample", type=int, default=5,
                    help="poses per unit, as the fold wrote them. Only asserted under multi-GPU, "
                         "where a unit with a different count would shift one rank's optimiser step "
                         "onto a different pose and deadlock the next all-reduce.")
    ap.add_argument("--lr-probe", default="",
                    help="comma-separated learning rates. Runs --probe-steps steps at each from the "
                         "SAME initial weights and reports the loss trend, then exits without "
                         "training. This is how --lr is chosen instead of inherited.")
    ap.add_argument("--probe-steps", type=int, default=30)
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="accept a label whose fingerprint differs ONLY in the `code` field, with "
                         "cif, bio, idx, trunk, loss and schema all matching. The data provenance "
                         "is what determines whether a label is CORRECT; the code hash additionally "
                         "says it is REPRODUCIBLE, and it covers whole files, so editing a docstring "
                         "in phase_j_build_cache.py invalidates all 5,000 labels. OFF by default: a "
                         "drift in any other field, or several fields, is still fatal, and every "
                         "accepted drift is printed with both hashes.")
    ap.add_argument("--eval-script", default=f"{ROOT}/phases/phase_k_eval.py")
    ap.add_argument("--eval-device", default="",
                    help="CUDA index for the epoch-end evaluation. The subprocess would otherwise "
                         "INHERIT CUDA_VISIBLE_DEVICES from the trainer and land on the same card "
                         "while the trainer is still resident and holding its memory -- the "
                         "evaluation does not get the trainer's allocator back. Give it an idle "
                         "card. Empty means share, and then the memory is checked first and the "
                         "evaluation is refused rather than risking an OOM mid-run.")
    ap.add_argument("--eval-min-free-mib", type=int, default=20000,
                    help="when the evaluation must share the training card, how much VRAM has to be "
                         "free for it to run. The eval loads a second 464M-parameter model; "
                         "measured at ~16 GB, so 20 GB is the floor.")
    ap.add_argument("--eval-budget", type=int, default=100,
                    help="seeds per interface for the epoch-end metric. 100 is the full budget and "
                         "the one the 0.5775 / 0.7283 baseline is quoted at; a pass is ~73 min, measured.")
    ap.add_argument("--resume", default="",
                    help="a resume bundle written at an epoch boundary (resume_ep<N>.pt). Restores "
                         "the head's weights AND AdamW's moment estimates AND the step counter, so "
                         "the warmup position and the per-parameter adaptive scales continue rather "
                         "than restarting -- a weights-only warm start throws both away. Training "
                         "picks up at epoch N+1, which reproduces the right data order because the "
                         "per-epoch reshuffle is random.Random(seed + ep) and depends on nothing "
                         "else. REFUSES a bundle whose world size, seed, accum or nsample differ "
                         "from this run's: all four change which poses land in which batch, so the "
                         "optimiser state would belong to a different problem.")
    ap.add_argument("--minimal-items", action="store_true",
                    help="ACCEPTED AND IGNORED -- this is now the only path, and --alignment-preflight is required whether or not it is passed. Kept so the launch scripts that pass it keep working. MEASURED "
                         "2026-10-17 by recording every key access during a real "
                         "calculate_losses call on four structures spanning 128-1,581 tokens and "
                         "2-5 chains: the loss reads nine feature keys and two label keys, and all "
                         "eleven are already in the cached label npz -- feat_ds and label_dict "
                         "contribute nothing. phase_k_minimal_item_equivalence.py then showed the "
                         "two paths give bit-identical losses on 50 of 50 comparisons. The item is "
                         "the run's memory: ~518 MB each, 164 per rank, so 82-95 GB per rank and "
                         "~264 GB across three, which is what breached the box's free-RAM budget, "
                         "stopped the cache growing at --max-rss-gb and degraded the pace from 2.8 "
                         "to 3.4 min/step as structures began to be rebuilt. The label npz is "
                         "0.11 MiB. REQUIRES --alignment-preflight, because check_alignment is the "
                         "one other thing the item was used for and it cannot run without one.")
    ap.add_argument("--alignment-preflight", default="",
                    help="path to the marker phase_k_alignment_preflight.py writes once it has "
                         "confirmed, for every structure in the manifest, that the trunk's features "
                         "agree atom for atom with the dataset's. --minimal-items refuses without "
                         "it: that check is the one failure that would pair a pose with another "
                         "molecule's label, and skipping it silently is not a trade worth making.")
    ap.add_argument("--resume-allow-drift", action="store_true",
                    help="BLANKET override: accept EVERY setting that differs from the bundle. "
                         "Prefer --resume-override, which names the fields and so still refuses "
                         "the difference you did not expect. Both record identically.")
    ap.add_argument("--resume-override", default="",
                    help="comma-separated GUARD FIELDS this resume may change, e.g. "
                         "'lr,warmup_steps'. A difference in any field NOT named here is still "
                         "refused, which is the whole point: a blanket --resume-allow-drift also "
                         "accepts the change you did not intend. Naming a field that does not "
                         "actually differ is an error too -- a stale override silently weakens "
                         "the guard on the next resume that reuses the command line.")
    ap.add_argument("--resume-note", default="",
                    help="why this resume deviates, recorded beside the accepted fields in "
                         "resume_drift.json and carried forward in every bundle this run writes. "
                         "Required when --resume-override or --resume-allow-drift accepts "
                         "anything: an unexplained deviation is the one a later reader cannot "
                         "tell from a mistake.")
    ap.add_argument("--crop-margin", type=float, default=0.0,
                    help="interface-plus-margin cropping: train on the window within this many "
                         "angstroms of an interface token instead of the whole structure. 0 "
                         "(default) is the whole structure. TRAINING ONLY -- validation is never "
                         "cropped, so the metric stays comparable with every historical number. "
                         "At 10 the population keeps a median 51.1%% of tokens (333 of 641, all "
                         "491 structures, 0 with no definable crop) for a MEASURED 3.35x on the "
                         "head and ~1.99x on a whole step, landing near Protenix's own "
                         "train_crop_size of 256. Requires --minimal-items.")
    ap.add_argument("--crop-contact", type=float, default=5.0,
                    help="a token is an interface token when any of its heavy atoms lies within "
                         "this many angstroms of a heavy atom on a DIFFERENT CHAIN, by ground-truth "
                         "coordinates. EVERY chain pair counts, the antibody's own heavy-light "
                         "packing included; restricting to antibody-antigen pairs is a separate "
                         "experiment needing role annotation this does not read.")
    ap.add_argument("--crop-log", default="",
                    help="csv of per-unit crop sizes; defaults to <out>/crop_sizes.csv when cropping")
    ap.add_argument("--require-replay", default="",
                    help="path to a replay result that must say PASS before training starts. "
                         "Cached training is only FT3 if the cache reproduces it; see "
                         "phase_j_replay.py. Pass 'skip' to train without the gate and say why.")
    a = ap.parse_args()
    a.labels = a.labels or os.path.join(a.fold, "labels")

    # CROPPING IS ONLY COHERENT ON THE MINIMAL PATH. The dataset item carries a FULL-SIZE
    # label_dict, so cropping the prediction while the loss reads full-size masks would either
    # raise on a shape or line up by accident and score poses against the wrong residues. The
    # minimal path builds the loss inputs from the same cached label the crop slices, so both
    # sides are cropped together or neither is.
    if a.crop_margin > 0 and not a.minimal_items:
        raise SystemExit(
            "STOP: --crop-margin needs --minimal-items. The dataset item's label_dict is built at "
            "full size, so cropping only the prediction would score poses against the wrong "
            "residues. Pass --minimal-items with its --alignment-preflight marker.")
    if a.crop_margin < 0:
        raise SystemExit("STOP: --crop-margin cannot be negative.")
    if a.val_loss_every:
        # CHECKED AT LAUNCH, NOT AT THE FIRST EPOCH BOUNDARY. On the full training set that
        # boundary is 37 minutes in, and a monitor that fails there is a monitor nobody has for
        # the one epoch where it would have mattered most.
        if a.loss_only:
            raise SystemExit("STOP: --val-loss-every inside --loss-only would have a measurement "
                             "measure itself, recursively.")
        _need = {"--val-loss-fold": a.val_loss_fold,
                 "--val-loss-selection": a.val_loss_selection,
                 "--val-loss-pool": a.val_loss_pool,
                 "--val-loss-preflight": a.val_loss_preflight,
                 "--val-loss-seeds": a.val_loss_seeds}
        _unset = sorted(k for k, v in _need.items() if not v)
        if _unset:
            raise SystemExit(f"STOP: --val-loss-every {a.val_loss_every} needs {_unset}. A "
                             f"held-out pass with no fold, selection, mask pool, preflight marker "
                             f"or seed range has nothing to measure.")
        _labels = a.val_loss_labels or os.path.join(a.val_loss_fold, "labels")
        _paths = {"--val-loss-fold": a.val_loss_fold, "labels": _labels,
                  "--val-loss-selection": a.val_loss_selection,
                  "--val-loss-pool": a.val_loss_pool,
                  "--val-loss-preflight": a.val_loss_preflight}
        if a.val_loss_prep:
            _paths["--val-loss-prep"] = a.val_loss_prep
        _gone = sorted(f"{k} -> {v}" for k, v in _paths.items() if not os.path.exists(v))
        if _gone:
            raise SystemExit("STOP: the held-out PAE monitor points at path(s) that do not "
                             "exist:\n  " + "\n  ".join(_gone))
        # THE LABELS MUST COVER THE SEED RANGE. An empty or partial label directory makes the
        # subprocess fail every epoch with "the training set is INCOMPLETE", which is correct but
        # 37 minutes too late to be useful. validation/fold/labels was EMPTY until 2026-10-20.
        try:
            _lo, _hi = (int(x) for x in a.val_loss_seeds.split("-"))
        except ValueError:
            raise SystemExit(f"STOP: --val-loss-seeds {a.val_loss_seeds!r} is not <lo>-<hi>.")
        with open(a.val_loss_selection) as _fh:
            _pdbs = sorted({r["pdb"] for r in csv.DictReader(_fh)
                            if r.get("tokens", "").isdigit()})
        if not _pdbs:
            raise SystemExit(f"STOP: {a.val_loss_selection} yielded 0 structure(s). Rows are kept "
                             f"only when `tokens` is a digit -- a selection csv without that "
                             f"column drops every row silently.")
        _miss = [f"{q}/free_{sd}" for q in _pdbs for sd in range(_lo, _hi + 1)
                 if not os.path.exists(os.path.join(_labels, q, f"free_{sd}.npz"))]
        if _miss:
            raise SystemExit(
                f"STOP: {len(_miss)} of {len(_pdbs) * (_hi - _lo + 1)} held-out label(s) are "
                f"missing under {_labels}, e.g. {_miss[:4]}. Build them with "
                f"phases/phase_j_build_cache.py --fold {a.val_loss_fold}; see "
                f"analysis/oneoff/01_build_val10_labels.sh.")
        print(f"held-out PAE monitor: every {a.val_loss_every} epoch(s), {len(_pdbs)} structure(s) "
              f"x seeds {_lo}-{_hi} = {len(_pdbs) * (_hi - _lo + 1)} unit(s), masked by "
              f"{a.val_loss_pool}")

    if a.loss_only and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # DEFENCE IN DEPTH, on the child's side of the same bug. The parent now strips the
        # distributed environment before launching this, but an inherited WORLD_SIZE used to make
        # this process join a group whose members were all already present -- it hung inside
        # init_process_group holding a card while the parent sat at a barrier waiting for it, which
        # reads as "training stopped" with no error anywhere. Refusing is a message; hanging is not.
        raise SystemExit(
            f"STOP: --loss-only is a single-process measurement but WORLD_SIZE is "
            f"{os.environ['WORLD_SIZE']}. Launched under torchrun, or with the parent run's "
            f"distributed environment inherited, it would block in init_process_group. Clear "
            f"WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT, or run it outside torchrun.")

    if a.loss_only:
        # A MEASUREMENT MUST NOT LOOK LIKE A RUN. Everything that would persist a head is refused
        # rather than quietly skipped: a loss-only pass that dropped a head_step*.pt or a
        # resume_ep*.pt into a run directory would leave a checkpoint nobody could tell apart from
        # a trained one, and the whole point of the mode is that the weights do not move.
        if a.epochs > 1:
            raise SystemExit(f"STOP: --loss-only is one pass over the manifest; --epochs "
                             f"{a.epochs:g} would measure the same frozen head {a.epochs:g} times "
                             f"and report the mean as though it were an epoch curve.")
        if a.save_every:
            raise SystemExit("STOP: --loss-only with --save-every would write checkpoints of a "
                             "head it never trained. Drop --save-every.")
        if a.warmup_steps:
            raise SystemExit("STOP: --loss-only with --warmup-steps is meaningless -- there is no "
                             "optimiser to warm up, and the logged lr would imply one ran.")
    if a.crop_log and a.crop_margin == 0:
        raise SystemExit("STOP: --crop-log was given but --crop-margin is 0, so nothing is cropped.")

    # PIN THE COMMIT NOW, before any work, so every bundle this process writes records the code it
    # is running rather than whatever HEAD has become hours later. Caching it lazily at the first
    # epoch boundary was not enough: that is hours in, and the tree can already have moved.
    _git_commit()

    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold. Without it the trunk's coordinates are "
                         "silently NaN -- phase_j_replay.py's guard fired on this first.")

    # THE GATE. A cached-training run is only an FT3 run if the cached halves reproduce what the live
    # code computes. That is phase_j_replay.py's job and it PASSES; requiring its result here is what
    # stops a future run from quietly skipping it.
    # EXEMPT UNDER --loss-only, AND NOT BECAUSE IT IS INCONVENIENT. The gate asserts that cached
    # TRAINING reproduces FT3, and its log is about the fold being trained on. A loss-only pass
    # trains nothing, and the fold it measures is usually a different one -- passing the training
    # fold's replay log there would assert something about validation trunks that nobody checked,
    # which is worse than not asking. The replay that matters for a measurement is the shipped-head
    # self-check phase_k_eval already runs against results.csv on the same fold.
    if a.require_replay != "skip" and not a.loss_only:
        if not a.require_replay:
            raise SystemExit(
                "STOP: --require-replay not given. Run\n"
                "  CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold python phases/phase_j_replay.py "
                "--unit <pdb>/free_<seed> > replay.log\n"
                "and pass that log, or pass --require-replay skip with a reason recorded.")
        if not os.path.exists(a.require_replay):
            raise SystemExit(f"STOP: no replay result at {a.require_replay}")
        txt = open(a.require_replay).read()
        if "PASS:" not in txt:
            raise SystemExit(f"STOP: {a.require_replay} does not say PASS. Cached training would be "
                             f"optimising something other than FT3.")
        print(f"replay gate: PASS ({a.require_replay})")

    import torch
    import phase_j_dataset as DS
    import phase_j_build_cache as BC
    import phase_j_rescore_cache as RC
    from protenix.model.loss import ProtenixLoss
    from protenix.utils.permutation.permutation import SymmetricPermutation

    os.makedirs(a.out, exist_ok=True)

    # ---- MULTI-GPU: SYNCHRONOUS REPLICAS, NOT DistributedDataParallel ----
    #
    # DDP wraps a module's forward, and ours is RC.confidence(model, configs, ...) over a FROZEN
    # trunk. DDP's hooks would both fight that custom call and expect gradients for the 444M
    # parameters that deliberately have none. So every rank holds an identical replica -- same
    # checkpoint, hence identical initial parameters -- and only the HEAD gradients are summed
    # across ranks before each step.
    #
    # WHY A SUM AND NOT A MEAN. Each pose is already scaled by the global accumulation at
    # backward(), so summing the ranks' accumulated gradients reconstructs exactly the mean over all
    # --accum poses that one GPU would have computed. After the all-reduce every rank holds
    # bit-identical gradients, so the clip, the non-finite decision, the AdamW state and therefore
    # the parameters stay in lockstep with no further communication and no parameter broadcast.
    #
    # RANKS MUST TAKE THE SAME NUMBER OF STEPS or they deadlock on the all-reduce. That is enforced
    # two ways: shards are whole structures truncated to an equal count (shard_by_structure), and
    # each unit is asserted to hold the same number of poses as every other.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    dist = None
    if world > 1:
        import torch.distributed as dist
        import datetime
        dist.init_process_group(backend=a.dist_backend,
                                timeout=datetime.timedelta(minutes=a.dist_timeout_min))
        torch.cuda.set_device(local_rank)
        print(f"[rank {rank}/{world}] cuda:{local_rank} backend={a.dist_backend} "
              f"timeout={a.dist_timeout_min:.0f} min", flush=True)
    dev = torch.device(f"cuda:{local_rank}") if world > 1 else torch.device("cuda")

    # The effective batch is what the ranks actually sum to, which is not --accum when the world
    # size does not divide it. It is reported rather than silently assumed, and it is what the loss
    # is scaled by, so the gradient is a true mean over the poses that contributed.
    # THE STEP IS K WHOLE STRUCTURES, ONE SEED OF EACH PER RANK. Every seed of a structure crops
    # to the IDENTICAL token count -- the interface crop is defined on the ground-truth interface,
    # not on the predicted pose, and 0 of 488 structures disagree across their ten seeds -- so
    # giving each rank one seed of the same K structures makes the per-step work equal across
    # ranks BY CONSTRUCTION rather than on average. --accum is honoured as a REQUEST: K is the
    # nearest whole number of structures to it.
    structs_per_step = a.structures_per_step or max(
        1, int(round(a.accum / float(world * a.nsample))))
    accum_local = structs_per_step * a.nsample
    accum_eff = accum_local * world
    print(f"  accum: {a.accum} requested -> K={structs_per_step} structure(s)/step x {world} "
          f"rank(s) x {a.nsample} pose(s) = {accum_eff} poses per step ({accum_local}/rank)")
    model, configs = RC.build_model(a.ckpt_dir, a.model_name, dev)
    configs.train_confidence_only = True
    configs.loss.weight.alpha_pae = float(a.alpha_pae)
    configs.loss.weight.alpha_confidence = float(a.alpha_confidence)

    # THE HEAD WE ARE TRAINING MUST BE IN TRAINING MODE.
    #
    # build_model() comes from phase_j_rescore_cache, which exists to RESCORE, so it ends with
    # model.eval(). Training through it left confidence_head.training == False, and Protenix's
    # confidence head has an inference-only memory optimisation keyed on exactly that flag
    # (modules/confidence.py:233): above PROTENIX_CONF_OFFLOAD_TOKENS tokens (default 2000) and when
    # `not self.training`, it moves pae and pde to the CPU and empties the cache. The loss then gets
    # two of its four logit tensors on the CPU while the labels are on the GPU and dies inside
    # Protenix's PDE loss as "cuda:2 and cpu" -- pointing at the loss rather than at the mode flag.
    # Data-dependent, so it killed two runs 20-24 min in, once the shuffle reached a structure over
    # 2000 tokens (9dzf).
    #
    # Switching the mode changes NO ARITHMETIC, which is why the steps already taken were valid:
    # the head's four Dropout modules all have p=0 and it holds no BatchNorm, so train() and eval()
    # compute the same function. The trunk stays in eval, which is what FT3 requires.
    model.confidence_head.train()
    # ACTIVATION CHECKPOINTING ON THE HEAD'S PAIR STACK. Set here rather than at construction,
    # because build_model comes from the rescoring helper. PairformerStack reads this attribute at
    # forward time and ignores it when grad is off, so the evaluator is unaffected.
    # 0 DISABLES IT, rather than silently leaving whatever the config set. The first version was
    # `if a.blocks_per_ckpt:`, which skipped the assignment on 0 -- and Protenix's config already
    # sets blocks_per_ckpt=1, so checkpointing was ALWAYS on, --blocks-per-ckpt 1 was a no-op, and
    # --blocks-per-ckpt 0 did not turn it off. Every conclusion drawn about what checkpointing cost
    # or saved was drawn from two identical configurations.
    model.confidence_head.pairformer_stack.blocks_per_ckpt = (a.blocks_per_ckpt
                                                              if a.blocks_per_ckpt else None)
    _nblk = len(getattr(model.confidence_head.pairformer_stack, "blocks", []) or [])
    print(f"pair stack: {_nblk} block(s), blocks_per_ckpt="
          f"{model.confidence_head.pairformer_stack.blocks_per_ckpt}, chunking off")
    print(f"confidence_head.training = {model.confidence_head.training} "
          f"(trunk stays eval; train_confidence_only={configs.train_confidence_only})")
    # THE THIRD POSE PATHWAY, INSTALLED BEFORE head_param_names() READS THE MODULE TREE. That
    # function derives the trainable set from the tree rather than from a hand-written list, so the
    # new projection is picked up automatically -- but only if it exists by then. Installed after
    # .train() and the blocks_per_ckpt assignment so neither is applied to a module that is about
    # to be wrapped.
    if a.frame_direction:
        import pose_direction
        pose_direction.install(model)
    if a.pose_perblock:
        import pose_perblock
        pose_perblock.install(model, tied=a.pose_perblock_tied,
                              share_pretrained=a.pose_perblock_share,
                              copy_pretrained=a.pose_perblock_copy)
    elif a.pose_perblock_tied or a.pose_perblock_share or a.pose_perblock_copy:
        raise SystemExit("STOP: --pose-perblock-tied/-share/-copy without --pose-perblock does "
                         "nothing. They select how A shares its weights, not whether A is "
                         "installed.")
    loss_fn = ProtenixLoss(configs)
    if a.drop_pde:
        # A STUB IN PLACE OF THE MODULE, because a zero weight would still run it.
        class _NoPDE(torch.nn.Module):
            def forward(self, *_args, **_kw):
                return torch.zeros((), device=dev)
        loss_fn.pde_loss = _NoPDE()
        print("PDE term DROPPED: not computed, not trained. ipTM comes only from PAE.")
    _dropped_terms = set()
    if a.drop_pde:
        _dropped_terms.add("pde_loss")
    if a.pae_only:
        # DO NOT COMPUTE THE ALL-ATOM LDDT WE THEN THROW AWAY.
        #
        # ProtenixLoss computes per_atom_lddt unconditionally -- `if all(x in pred_dict for x in
        # ["plddt","pde","pae","resolved"])`, which is always true because the head returns all
        # four -- and those values feed ONLY the plddt_loss lambda, which --pae-only stubs to
        # zeros. So it is pure waste. MEASURED on an idle card: 3.4 ms per pose at 11,504 atoms,
        # about 1% of the 0.300 s/pose, but up to 1.66 GB of TRANSIENT allocation, which is the
        # actual reason to remove it -- that headroom is what a larger crop or a larger accum
        # would need. Found in review.
        #
        # Patched at the module where loss.py looks it up, and ONLY under --pae-only, where the
        # consumer is already a stub that ignores both return values. The zeros are shaped from
        # pred_coordinate so nothing downstream can trip over them if it ever does read.
        import protenix.model.loss as _PL
        _orig_lddt = _PL.calculate_atom_bespoke_lddt

        def _no_lddt(pred_coordinate, true_coordinate, *_args, **_kw):
            z = torch.zeros(pred_coordinate.shape[:-1], dtype=pred_coordinate.dtype,
                            device=pred_coordinate.device)
            return z, z
        _PL.calculate_atom_bespoke_lddt = _no_lddt
        print("  pae-only: all-atom lddt SKIPPED (feeds only the stubbed pLDDT term); "
              "measured 3.4 ms and up to 1.66 GB per pose")
        # SAME STUB TRICK as --drop-pde, for the same reason: a zero weight still runs the module
        # and still pays for it. Replacing the module is what actually removes the term.
        #
        # IT ANSWERS ANY ENTRY POINT, not just forward(). ProtenixLoss does NOT call these three
        # uniformly: pde_loss and resolved_loss go through __call__, but pLDDT is reached by a
        # NAMED METHOD -- `self.plddt_loss.forward_given_atom_lddt(...)` (loss.py:1771). A stub
        # carrying only forward() raised AttributeError at pose 0 of unit 0, so the first version
        # of this flag could not run a single step. Generalising from --drop-pde, which happens to
        # be the easy case, is what produced that.
        class _Zero(torch.nn.Module):
            def forward(self, *_args, **_kw):
                return torch.zeros((), device=dev)

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if name.startswith("_"):
                        raise
                    return self.forward

        # CHECKED AGAINST THE REAL MODULE, not with hasattr on the stub. hasattr(loss_fn, name)
        # only proved the attribute existed; it said nothing about the calling convention, which
        # is precisely what broke. This asserts the stub answers every public callable the module
        # being replaced exposes, so a library that grows a new named entry point fails loudly.
        for _name in ("plddt_loss", "pde_loss", "resolved_loss"):
            _orig = getattr(loss_fn, _name, None)
            if _orig is None:
                raise SystemExit(f"STOP: --pae-only cannot drop {_name}: ProtenixLoss has no such "
                                 f"submodule. It has {sorted(k for k in dir(loss_fn) if 'loss' in k)}. "
                                 f"Dropping nothing while announcing PAE-only would mislabel the run.")
            _stub = _Zero()
            _base = set(dir(torch.nn.Module))
            _api = [m for m in dir(type(_orig))
                    if not m.startswith("_") and m not in _base and callable(getattr(_orig, m, None))]
            _missing = [m for m in _api if not callable(getattr(_stub, m, None))]
            if _missing:
                raise SystemExit(f"STOP: the --pae-only stub cannot stand in for {_name}: it does "
                                 f"not answer {_missing}, which {type(_orig).__name__} exposes and "
                                 f"the loss may call. A stub that is missing the entry point the "
                                 f"caller uses dies mid-run, or worse, silently skips a term.")
            setattr(loss_fn, _name, _stub)
            _dropped_terms.add(_name)
            print(f"  dropped {_name} ({type(_orig).__name__}); stub answers {sorted(_api)}")
        print("PAE-ONLY: pLDDT, PDE and resolved DROPPED -- not computed, not trained. "
              "Every gradient now goes to the one output ipTM reads.")
    _pae_keep = {"m": None}
    _scored_asym = {}
    _pae_mask_stats = {"masked": 0, "unmasked": 0}
    if a.pae_scored_pairs:
        import csv as _csv
        _roles_p = a.pae_scored_roles or os.path.join(a.fold, "chain_roles.csv")
        _roles = {}
        with open(_roles_p) as _fh:
            for _r in _csv.DictReader(_fh):
                _roles.setdefault(_r["pdb"], {})[_r["ref_chain"]] = _r["model_chain"]
        # Model chains are lettered A, B, C... in asym_id order by write_job_cif, so the letter's
        # position in the alphabet IS the asym index -- the same join phase_k_eval does.
        _miss = []
        with open(a.pae_scored_pairs) as _fh:
            for _r in _csv.DictReader(_fh):
                _m1 = _roles.get(_r["pdb"], {}).get(_r["chain1"])
                _m2 = _roles.get(_r["pdb"], {}).get(_r["chain2"])
                if _m1 is None or _m2 is None:
                    _miss.append(f"{_r['pdb']}:{_r['chain1']}/{_r['chain2']}")
                    continue
                # NORMALISED LOW-HIGH. chain_roles maps ref_chain -> model_chain, and the
                # pool's (chain1, chain2) order is the REFERENCE order, so the model-chain pair
                # comes out descending for most interfaces -- 46 of 77 on the overfit30 pool.
                # _set_pae_mask symmetrises explicitly so it never cared, but --rank-loss looks up
                # DockQ labels by this tuple and rank_loss.parse_iface normalises to (min, max):
                # unnormalised, the lookup found 9 of 77 pairs and the other 68 interfaces would
                # have paid a forward for zero ranking gradient. chain_pair_iptm is symmetric --
                # phase_k_eval reads cpi[ai, aj] in the pose's own order and reproduces the fold's
                # stored ipTM to 2e-3 on every row, descending ones included -- so normalising
                # changes no value, only the key.
                _a1 = ord(_m1) - ord("A")
                _a2 = ord(_m2) - ord("A")
                _scored_asym.setdefault(_r["pdb"], set()).add((min(_a1, _a2), max(_a1, _a2)))
        if _miss:
            raise SystemExit(f"STOP: {len(_miss)} pool interface(s) have no role mapping, e.g. "
                             f"{_miss[:4]}. The PAE mask would silently drop their blocks.")
        _orig_calc = loss_fn.pae_loss.calculate_label

        def _masked_calc(*args, **kw):
            sq, bins, pm = _orig_calc(*args, **kw)
            m = _pae_keep["m"]
            if m is not None:
                pm = pm * m.to(pm.dtype)
            return sq, bins, pm

        loss_fn.pae_loss.calculate_label = _masked_calc
        print(f"PAE term MASKED to scored chain pairs: {len(_scored_asym)} structure(s), "
              f"{sum(len(v) for v in _scored_asym.values())} interface(s), from "
              f"{a.pae_scored_pairs}")

    # RANKING LOSS SETUP. Refused rather than silently degraded when its inputs are absent: without
    # --pae-scored-pairs there is no interface to read ipTM for, and without labels there is no
    # ordering to learn. Both failures would otherwise train the PAE term alone and look like a
    # ranking experiment.
    # UNITS PER STEP, so the ranking term is a mean over units the way cum/accum_eff is a mean
    # over poses. accum_eff counts poses; each unit contributes --nsample of them.
    _rank_lab = {}
    _rank_div = max(1, accum_eff // max(1, a.nsample))
    _rank_seen = {"loss": 0.0, "units": 0}
    # CROSS-SEED STATE. _xs_coef holds dL/ds for every pose of the CURRENT step, keyed
    # (pdb, seed, sample, pair) -- a pose's coefficient depends on every other pose of its
    # interface, which is why it cannot be built per unit the way the linear one is. _xs_div is the
    # number of interface-groups in the step, so the term is a mean over groups; it is a one-element
    # list because pass A writes it and the pose loop reads it from a different frame.
    _xs_coef, _xs_div, _xs_stat = {}, [1], {"groups": 0, "pairs": 0, "loss": 0.0, "steps": 0,
                                             "last_groups": 0, "skipped": 0}
    # THE CURRENT STEP'S EXPECTED POSE COUNT. accum_local is a constant only when every
    # step holds the same number of units, which --rank-cross-seed breaks: S structures x
    # their seeds need not divide `world`, so the short rank rotates and the quota varies
    # by step. Driving the boundary from the quota keeps every rank stepping on the same
    # iteration -- a mismatch there HANGS on the first all-reduce rather than failing.
    _step_poses = [accum_local]
    # AND THE GLOBAL COUNT, which is what the PAE term's mean is taken over. accum_eff is
    # K*nsample*world = 90 here, but a cross-seed step holds S*seeds*nsample = 100 poses across the
    # ranks, so dividing by accum_eff would scale the PAE gradient by 100/90. Set once per epoch
    # from the actual step shape; equal to accum_eff when --rank-cross-seed is off.
    _step_poses_g = [accum_eff]
    # THE COSINE HORIZON, in optimiser steps, filled in just before train_loop from the same
    # `_expect` the startup banner prints. A list because train_loop closes over it and the value
    # is not known until the unit list has been flattened and sharded.
    _lr_horizon = [0]
    _rank_tau = float(a.rank_tau) if a.rank_tau is not None else None
    _probe = {"rows": [], "units": 0, "poses": 0, "sum": {}, "finite": {}, "nonzero": {}}
    if a.grad_probe:
        import rank_loss as RKL_probe
        if not a.rank_loss:
            raise SystemExit("STOP: --grad-probe needs --rank-loss; there is nothing to compare "
                             "the PAE gradient against.")
    if a.rank_loss and a.loss_only:
        # SAME REASON. Nothing below is needed for a measurement, and --rank-labels points at the
        # training fold while this subprocess reads the validation one.
        print("  --loss-only: ranking setup skipped (the term does not run in a measurement)")
    elif a.rank_loss:
        import rank_loss as RKL
        if not _scored_asym:
            raise SystemExit("STOP: --rank-loss needs --pae-scored-pairs. The ranking term reads "
                             "chain_pair_iptm for a NAMED interface; with no scored pair there is "
                             "nothing to rank and no join to the DockQ labels.")
        _rl_p = a.rank_labels or os.path.join(a.fold, "results.csv")
        if not os.path.exists(_rl_p):
            raise SystemExit(f"STOP: --rank-loss found no DockQ labels at {_rl_p}. Pass "
                             f"--rank-labels explicitly.")
        _rank_lab, _rl_bad = RKL.load_labels(_rl_p, _roles_p)
        if not _rank_lab:
            raise SystemExit(f"STOP: {_rl_p} yielded no usable (pdb, seed, pair) -> DockQ groups.")
        # THE LABELS MUST COVER THE INTERFACES THIS RUN TRAINS, checked here and not at the first
        # pose. A structure whose labels are missing would contribute zero ranking gradient while
        # still paying a full forward, so the arm would be a partial experiment reported as a whole
        # one -- the same class of error as the LoRA bundles that trained four epochs with no
        # held-out points.
        # COMPLETENESS, NOT MERE PRESENCE. The first version checked only that a scored
        # structure appeared SOMEWHERE in the labels, which a structure with one labelled seed out
        # of ten would pass -- and its unlabelled units would then pay a forward and contribute no
        # ranking gradient, making this a partial experiment reported as a whole one. That is the
        # same class as the LoRA bundles that trained four epochs with zero held-out points.
        # Review of cd8e6ab7 pointed out the guard was weaker than the doc claimed. So every
        # (pdb, seed, interface) this run will train must carry --nsample finite DockQ values at
        # the expected sample indices.
        _rl_pdbs = {k[0] for k in _rank_lab}
        _rl_missing = sorted(set(_scored_asym) - _rl_pdbs)
        import rank_loss as _RKL_cfg
        _tau_eff = _rank_tau if _rank_tau is not None else _RKL_cfg.DEFAULT_TAU
        print(f"RANKING LOSS on chain_pair_iptm: form {a.rank_form}, alpha {a.alpha_rank:g}, "
              f"weight_by_gap={a.rank_weight_by_gap}, {len(_rank_lab)} labelled "
              f"(pdb, seed, interface) group(s) from {_rl_p}"
              + (f", {_rl_bad} unparsed row(s)" if _rl_bad else ""))
        # VALIDATED BEFORE ANYTHING IS PRINTED. tau <= 0 is refused in three places while these
        # two floats took nan, inf and negatives silently: --rank-topk-decay nan makes every
        # weight nan and the `> 0` banner test below is FALSE for nan, so the log would not even
        # mention the decay while the loss trained on nan; --rank-within-weight -1 drives den
        # negative and flips the loss's sign.
        for _nm, _vv in (("--rank-within-weight", a.rank_within_weight),
                         ("--rank-topk-decay", a.rank_topk_decay),
                         ("--rank-straddle", a.rank_straddle)):
            if not math.isfinite(_vv) or _vv < 0:
                raise SystemExit(f"STOP: {_nm} {_vv!r} must be a finite number >= 0.")
        if a.rank_within_weight == 0 and a.rank_pairs == "all":
            print("  NOTE --rank-within-weight 0 drops the within-seed family from the WEIGHTING "
                  "while --rank-pairs all still forms its pairs. Groups whose poses all come "
                  "from one seed then carry no weighted pair and contribute nothing, which is "
                  "handled (rank_loss guards den == 0) but means this is NOT the same run as "
                  "--rank-pairs cross.")
        # THE BANNERS MUST NOT DESCRIBE A PATH THAT IS NOT TAKEN. These three flags are read only
        # by group_coeffs on the pass-A cross-seed path. The linear form without --rank-cross-seed
        # goes through RKL.coeffs, which takes labels alone and honours none of them, and the
        # expected form returns before the pair loop. Printing "CROSS-SEED pairs only -- the
        # deployed objective" while training within-seed linear pairs is how a run gets written up
        # as the wrong experiment, so a non-default setting on a path that ignores it is refused
        # rather than announced.
        _ignores = (a.rank_form == "expected") or not a.rank_cross_seed
        _nondefault = [f"--rank-pairs {a.rank_pairs}" for _ in (1,) if a.rank_pairs != "cross"] \
            + [f"--rank-within-weight {a.rank_within_weight:g}" for _ in (1,)
               if a.rank_within_weight != 1.0] \
            + [f"--rank-topk-decay {a.rank_topk_decay:g}" for _ in (1,) if a.rank_topk_decay != 0.0] \
            + [f"--rank-straddle {a.rank_straddle:g}" for _ in (1,) if a.rank_straddle != 0.0]
        if _ignores and _nondefault:
            raise SystemExit(
                "STOP: " + ", ".join(_nondefault) + " cannot take effect here. "
                + ("--rank-form expected is pair-free -- group_coeffs returns before the pair "
                   "loop, so the family, the within-seed weight and the rank decay are all "
                   "inapplicable."
                   if a.rank_form == "expected" else
                   "without --rank-cross-seed the loss goes through the per-unit path "
                   "(rank_loss.coeffs), which takes labels alone and honours none of these three. "
                   "Add --rank-cross-seed, or drop these flags.")
                + " Refused rather than warned: the startup banner would otherwise name an "
                  "objective the run is not training.")

        # THE FAMILY IS NAMED BEFORE THE TAU, because the tau is only interpretable against it.
        _fam_txt = {"cross": "CROSS-SEED pairs only (different seeds) -- the deployed objective",
                    "within": "WITHIN-SEED pairs only (same seed) -- a PROXY for selection, not "
                              "the deployed objective",
                    "all": "BOTH families at ONE tau"}[a.rank_pairs]
        print(f"  PAIR FAMILY: {_fam_txt}")
        if a.rank_pairs == "all":
            print(f"  within-seed pairs weighted x{a.rank_within_weight:g} in numerator and W "
                  f"(full training set at tau 0.0270: L=1 leaves within-seed at 7.0% of pairwise "
                  f"gradient mass, L=3 18.3%, L=8 37.1%, L=20 59.3%)")
        if a.rank_straddle > 0:
            print(f"  STRADDLE-ONLY at DockQ {a.rank_straddle:g}: a pair counts only when exactly "
                  f"one of its two poses is above {a.rank_straddle:g}. This is the subspace the "
                  f"FRACTION-ACCEPTABLE criterion can see; mean DockQ is deliberately left as a "
                  f"free variable and both are reported.")
        if a.rank_topk_decay > 0:
            print(f"  TOP-k RANK DECAY K={a.rank_topk_decay:g}: pair weight x "
                  f"max(v(r_i), v(r_j)), v(r)=exp(-(r-1)/K) over the current ipTM order. "
                  f"K=2 puts ~39% of weight on pairs touching rank 1 (1.6% undecayed) and ~93% "
                  "inside the top 5. Ranks are taken from DETACHED scores, so the weight is a "
                  "constant in the backward and the loss is not the exact gradient of a fixed "
                  "objective.")
        elif a.rank_within_weight != 1.0:
            print(f"  NOTE --rank-within-weight {a.rank_within_weight:g} is INERT under "
                  f"--rank-pairs {a.rank_pairs}: it scales one family against the other, and "
                  "this run has only one family. Not an error; stated so the log cannot be read "
                  "as if the weighting were in force.")
        # NO STOP HERE, AND THE EARLIER ONE WAS WRONG. It refused --rank-pairs within/all
        # without an explicit --rank-tau, on the grounds that at the cross-seed tau a within-seed
        # pair "sits at m/tau ~ 0 where the softplus gives a correct pair and an inverted one the
        # same weight, so the term is inert". The premise is right and the conclusion is not.
        # At overfit30's within-seed median gap of 0.0137 against tau 0.0305 the weights are
        # 0.376 correct / 0.624 inverted -- real discrimination; even at the full training set's
        # 0.0036 they are 0.467 / 0.533, and BOTH PUSH THE SAME WAY, toward the better-DockQ
        # pose. They add, they do not cancel. What a large tau costs is SATURATION: the loss
        # never stops pushing pairs it has already ordered, so the term degenerates toward the
        # linear loss. That is what the --rank-tau help means by the original units error being
        # "inert" -- the softplus NONLINEARITY is inert, not the gradient. Within-seed pairs are
        # under-weighted (6.7% of pairwise gradient mass on overfit30), which is a question for
        # --rank-within-weight, not for tau. So this prints the regime and lets the run proceed.
        if a.rank_pairs != "cross" and a.rank_form == "softplus":
            _r = 0.0137 / _tau_eff          # overfit30 within-seed median; the training pool's
            _sg = 1.0 / (1.0 + math.exp(-_r))
            print(f"  within-seed pairs at tau {_tau_eff:g}: a median-gap pair (0.0137 on "
                  f"overfit30) gets weight {1 - _sg:.3f} if correctly ordered and {_sg:.3f} if "
                  f"inverted ({_sg / (1 - _sg):.2f}x). Re-measure for another pool with "
                  "measure_tau.py --pairs within --restrict-to <pool>/structures.csv; the full "
                  "training set gives 0.0036 and valfull 0.0011, so this ratio is "
                  "POPULATION-SPECIFIC and tau must never be set from val.")
        if a.rank_form == "softplus":
            print(f"  softplus tau {_tau_eff:g}"
                  + ("  (rank_loss.DEFAULT_TAU, the measured median cross-seed |ipTM gap| over "
                     "the full training set)" if _rank_tau is None else "  (given)"))
        elif a.rank_form == "expected":
            # TWO DIFFERENT QUANTITIES SHARE THE --rank-tau FLAG, so say which one is in force.
            # A margin scale of 0.0305 and a softmax temperature of 0.05 are not comparable, and
            # defaulting silently to DEFAULT_TAU here would be a 1.6x sharper selection than
            # intended with nothing in the log to show it.
            _tau_sm = _rank_tau if _rank_tau is not None else _RKL_cfg.DEFAULT_SOFTMAX_TAU
            print(f"  EXPECTED-DockQ, softmax temperature tau {_tau_sm:g}"
                  + ("  (rank_loss.DEFAULT_SOFTMAX_TAU)" if _rank_tau is None else "  (given)")
                  + "\n  L = -sum_i softmax(ipTM/tau)_i * DockQ_i over ALL poses of the "
                    "interface; the floor is -max(DockQ) of the group, NOT 0. PAIR-FREE: "
                    f"--rank-weight-by-gap={a.rank_weight_by_gap} and the cross-seed pair filter "
                    "are INAPPLICABLE and are not applied.")
        if a.rank_cross_seed:
            print(f"  CROSS-SEED GROUPS: {a.rank_structs_per_step} structure(s) x all seeds per "
                  f"step; a pose's coefficient uses every other pose of its interface, so pass A "
                  f"scores the whole step before the first backward")
        elif a.rank_form in ("softplus", "expected"):
            raise SystemExit(
                f"STOP: --rank-form {a.rank_form} without --rank-cross-seed is not implemented. "
                "Its coefficients need pass A, which is wired to the cross-seed step grouping; "
                "running it within-seed would silently fall through to the linear per-unit path "
                "and train a different objective than the flags say. For the expected form this "
                "would also defeat its purpose outright -- the softmax has to run over ALL poses "
                "competing for selection, and within one seed it would range over 5 poses rather "
                "than 50.")
        if _rl_missing:
            raise SystemExit(
                f"STOP: {len(_rl_missing)} scored structure(s) have no DockQ labels in {_rl_p}, "
                f"e.g. {_rl_missing[:6]}. Their poses would pay a forward and contribute no "
                f"ranking gradient, making this a partial experiment.")
        # COVERAGE IS CHECKED AFTER THE MANIFEST IS BUILT, not here: the expected key set is
        # (pdb, seed, pair) over the units this run will actually train, and `want` does not exist
        # yet at this point in main(). An earlier version referenced it here anyway and would have
        # raised NameError at startup the first time --rank-loss was passed. Found in review of
        # a3086ef. See _rank_check_coverage below.

    def _rank_key(pdb, arm_seed_or_int, pair):
        """THE label key, built in exactly one place.

        A unit's identifier is the label/trunk basename -- "free_500000", not 500000 -- because
        that is what names the files. The guard built its keys from the manifest, where seeds are
        plain ints, while the lookup in run_unit did int(arm_seed) on the basename and raised
        ValueError on the first unit. Both sides were internally consistent and disagreed with
        each other, which is the same shape as the reference-vs-model chain bug: two independent
        key constructions. There is now one, so they cannot diverge again.
        """
        sd = arm_seed_or_int
        if isinstance(sd, str):
            sd = sd.rsplit("_", 1)[-1]
        return (pdb, int(sd), pair)

    def _rank_check_coverage(want_units):
        """Every (pdb, seed, interface) this run will train must carry --nsample finite DockQ.

        DRIVEN FROM THE EXPECTED SET, NOT FROM THE LABELS. Two earlier versions were weaker and
        both would have passed the run that was actually broken: one checked only that a structure
        appeared SOMEWHERE in the labels (a pdb-level test cannot see a per-pair hole), and one
        iterated over the labels and skipped keys absent from _scored_asym (so a scored pair with
        no labels at all was invisible). The reference-vs-model chain bug survived both. The
        expected set is built from the manifest and the scored pairs, and set difference does the
        work.
        """
        # NOT UNDER --loss-only. The held-out monitor is a subprocess on the VALIDATION fold
        # (seeds 400000+), it inherits rank_loss=True only so the resume guard matches, and the
        # ranking term never runs there -- a measurement takes no step. Demanding training-fold
        # labels for validation units is a check it cannot satisfy, and it killed the held-out
        # pass on every epoch of s7dh7o: "540 group(s) have NO labels, e.g. ('7qu2', 400000,
        # (0, 2))". FIFTH instance of this class -- --train-only, LoRA, the evaluator's
        # --pose-perblock-share, the ranking guard fields, now a check of my own that a
        # measurement cannot pass.
        # UNCHANGED BY --rank-cross-seed, and that is not an oversight. Cross-seed grouping
        # changes which poses are COMPARED, not which labels must exist: every (pdb, seed,
        # interface) the run touches still needs its --nsample DockQ values, and the groups are
        # built from the same manifest units this set is built from. A group missing a seed's
        # labels is therefore already caught here, before pass A ever runs.
        if not a.rank_loss or a.loss_only:
            return
        expected = {_rank_key(pdb, sd, pr) for pdb, sd in want_units
                    for pr in (_scored_asym.get(pdb) or ())}
        if not expected:
            raise SystemExit("STOP: --rank-loss expected zero (pdb, seed, interface) groups. "
                             "Either the manifest or --pae-scored-pairs selects nothing.")
        missing = sorted(expected - set(_rank_lab))
        short, nonfinite = [], []
        for k in sorted(expected & set(_rank_lab)):
            m = _rank_lab[k]
            if sorted(m) != list(range(a.nsample)):
                short.append((k, sorted(m)))
            if not all(math.isfinite(v) for v in m.values()):
                nonfinite.append(k)
        if missing or short or nonfinite:
            raise SystemExit(
                f"STOP: --rank-loss label coverage is incomplete for the {len(expected)} "
                f"(pdb, seed, interface) group(s) this run trains.\n"
                + (f"  {len(missing)} group(s) have NO labels, e.g. {missing[:6]}\n"
                   if missing else "")
                + (f"  {len(short)} group(s) lack samples 0..{a.nsample - 1}, e.g. {short[:4]}\n"
                   if short else "")
                + (f"  {len(nonfinite)} group(s) carry a non-finite DockQ, e.g. "
                   f"{nonfinite[:4]}\n" if nonfinite else "")
                + "  Those poses would pay a forward and contribute no ranking gradient.")
        if is_main:
            print(f"  ranking label coverage: all {len(expected)} (pdb, seed, interface) group(s) "
                  f"hold {a.nsample} finite DockQ value(s)")

    def _check_eval_pool():
        """Stat the training-eval pool now, not at the first epoch boundary.

        eval_checkpoint shells out to phase_k_eval.py, which hard-stops on a missing trunk or a
        pool interface with no scored pose. That SystemExit lands on RANK 0 ONLY, while ranks 1
        and 2 sit in dist.barrier() until --dist-timeout-min, so a bad pool costs hours of
        training and then a four-hour zombie. The default fold pool really does contain a
        structure the manifest skips (9mx1 has no trunk at all), so this is one wrong path away.
        """
        if not a.train_eval_pool:
            return
        # THE POOL'S OWN FOLD, matching what eval_checkpoint will pass. Checking against --fold
        # when --train-eval-fold overrides it rejected a perfectly good validation pool for
        # having no trunks in the TRAINING fold, which is exactly where it should not have any.
        _f = a.train_eval_fold or a.fold
        if "," in _f:
            raise SystemExit(f"STOP: --train-eval-pool needs a single fold; got {_f!r}. "
                             f"phase_k_eval treats --fold as one directory, so a comma-separated "
                             f"root would be read as a literal path.")
        need = [a.train_eval_pool, os.path.join(_f, "results.csv"),
                os.path.join(_f, "chain_roles.csv")]
        gone = [f for f in need if not os.path.exists(f)]
        if gone:
            raise SystemExit(f"STOP: --train-eval-pool needs {gone}, which do not exist.")
        import csv as _csv2
        # --manifest-skip describes the TRAINING manifest, so it says nothing about a pool in
        # another fold; skipping a structure from training is not a reason it cannot be scored.
        skip = ({x.strip() for x in (a.manifest_skip or "").split(",") if x.strip()}
                if not a.train_eval_fold else set())
        pool_pdbs = {r["pdb"] for r in _csv2.DictReader(open(a.train_eval_pool))}
        bad = sorted(pool_pdbs & skip)
        if bad:
            raise SystemExit(f"STOP: --train-eval-pool names {bad}, which --manifest-skip excludes "
                             f"from training. The evaluator hard-stops on a structure with no "
                             f"trunk, on rank 0, with the other ranks parked at a barrier.")
        notrunk = sorted(p for p in pool_pdbs
                         if not os.path.isdir(os.path.join(_f, "trunks", p)))
        if notrunk:
            raise SystemExit(f"STOP: --train-eval-pool names {len(notrunk)} structure(s) with no "
                             f"trunk directory, e.g. {notrunk[:4]}.")
        print(f"train-eval pool: {len(pool_pdbs)} structure(s), all present, none skipped")

    def _pae_mask_coverage(units):
        """Every structure this run trains on must be named by the pool, checked BEFORE training.

        The running guard below only fires while NOTHING has been masked yet, so once a single
        unit masks successfully an unlimited number of later units can fall back to the UNMASKED
        PAE term in silence. Review reproduced one masked call followed by 100 unmasked ones with
        no error. A per-structure coverage check at startup closes that: a structure missing from
        the pool is a manifest/pool mismatch, which is a setup error, and it is cheap to find here
        rather than impossible to notice later.
        """
        if not _scored_asym:
            return
        want = sorted({t[0] for t in units})
        gone = [p for p in want if not _scored_asym.get(p)]
        if not gone:
            print(f"  PAE mask: all {len(want)} training structure(s) are named by "
                  f"{a.pae_scored_pairs}")
            return
        if not a.allow_unmasked_fallback:
            raise SystemExit(
                f"STOP: {len(gone)} of {len(want)} training structure(s) are NOT in "
                f"{a.pae_scored_pairs}, so their poses would train the UNMASKED PAE term while "
                f"this run is labelled masked: {gone[:8]}. Fix the pool, drop them from "
                f"--manifest-selection, or pass --allow-unmasked-fallback to accept it "
                f"deliberately -- it is recorded in args.json either way.")
        print(f"  PAE mask: ACCEPTING UNMASKED FALLBACK for {len(gone)} of {len(want)} "
              f"structure(s) (--allow-unmasked-fallback): {gone[:8]}")

    def _pae_mask_check():
        """A mask that never fires is a baseline arm wearing the masked arm's label.

        Both fallbacks below are silent by design -- they keep training rather than dying on one
        odd structure. But if NOTHING is ever masked, the startup banner has announced an objective
        the run is not training. The classic way in is pointing --pae-scored-pairs at the wrong
        pool (the validation one, say): the banner reports 92 structures, every training unit then
        misses, and the run trains the unmasked term under the masked arm's name.
        """
        st = _pae_mask_stats
        if _scored_asym and st["masked"] == 0 and st["unmasked"] >= 20:
            raise SystemExit(
                f"STOP: --pae-scored-pairs matched NO unit in the first {st['unmasked']} seen. "
                f"The pool names {len(_scored_asym)} structure(s), none of which this run trains "
                f"on, so every pose fell back to the UNMASKED PAE term while the run is labelled "
                f"masked. Check that {a.pae_scored_pairs} covers --manifest-selection.")

    def _set_pae_mask(pdb, asym, has_frame):
        """[N_frame, N_token] over the CROPPED tokens: keep only scored cross-chain entries.

        Falls back to no mask -- never to an all-zero one. An empty mask would make PAELoss divide
        by eps and return a meaningless number rather than fail, which is the worst outcome.
        """
        if not _scored_asym:
            return
        pairs = _scored_asym.get(pdb)
        if not pairs:
            _pae_keep["m"] = None
            _pae_mask_stats["unmasked"] += 1
            _pae_mask_check()
            return
        aa = asym.reshape(-1).long()
        fa = aa[has_frame.reshape(-1).bool()]
        keep = torch.zeros((fa.numel(), aa.numel()), dtype=torch.bool, device=aa.device)
        for x, y in pairs:
            keep |= ((fa[:, None] == x) & (aa[None, :] == y))
            keep |= ((fa[:, None] == y) & (aa[None, :] == x))
        if not bool(keep.any()):
            _pae_keep["m"] = None
            _pae_mask_stats["unmasked"] += 1
            _pae_mask_check()
            return
        _pae_keep["m"] = keep
        _pae_mask_stats["masked"] += 1

    w = {k: loss_fn.loss_weight[k] for k in ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")}
    print("loss weights: " + "  ".join(
        f"{k.replace('_loss','')} {v:g}" + (" [DROPPED]" if k in _dropped_terms else "")
        for k, v in w.items()))
    _check_eval_pool()
    zero = [k for k, v in w.items() if not v and k not in _dropped_terms]
    # THE GUARD'S PREMISE IS "at 0 the ipTM cannot move", AND A RANKING TERM FALSIFIES IT. This
    # check predates --rank-loss, when PAE was the only path from a weight to chain_pair_iptm. The
    # ranking term differentiates ipTM directly, so alpha_pae 0 with --rank-loss is a RANKING-ONLY
    # arm, which is a deliberate configuration and not the silent no-op the guard was written to
    # catch. Every other zero-weight term is still refused, and pae_loss at 0 WITHOUT a ranking
    # term still is too -- that really would train nothing the project is judged on.
    if zero == ["pae_loss"] and a.rank_loss and float(a.alpha_rank) > 0:
        print(f"  alpha_pae 0 WITH --rank-loss: this is a RANKING-ONLY arm. No PAE gradient is "
              f"taken; chain_pair_iptm moves through the ranking term alone. The held-out monitor "
              f"still measures at alpha_pae 1, so val10 PAE stays comparable to every other arm.")
        zero = []
    if zero:
        raise SystemExit(f"STOP: {', '.join(zero)} carries weight 0, so that term cannot train. "
                         f"FT3 wants alpha_pae = 1 (Protenix's own comment, configs_base.py:402); "
                         f"at 0 the ipTM this project is judged on cannot move. If you meant a "
                         f"ranking-only arm, pass --rank-loss: the ranking term differentiates "
                         f"ipTM directly and the guard allows alpha_pae 0 in that case.")

    prefixes, trainable = head_param_names(model)
    # LoRA REPLACES THE TRAINABLE SET, it does not add to it. Applied to the confidence head only
    # -- head_param_names already established that module is the honest boundary -- and after the
    # head is built so the adapters inherit its device and dtype.
    if a.train_only:
        import re as _re
        _pats = [t.strip() for t in a.train_only.split(",") if t.strip()]
        _pfx = prefixes[0]
        _rel = {n: n[len(_pfx) + 1:] for n in trainable}
        _kept = [n for n in trainable if any(_re.search(_p, _rel[n]) for _p in _pats)]
        if not _kept:
            raise SystemExit(
                f"STOP: --train-only matched no parameter, so the optimiser would be empty and "
                f"the run would report success while training nothing.\n"
                f"  patterns: {_pats}\n"
                f"  head has {len(trainable)} tensor(s), e.g. "
                f"{[_rel[n] for n in trainable[:6]]}")
        _before = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
        trainable = _kept
        _after = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
        print(f"--train-only {_pats}: {len(_kept)} of {len(_rel)} head tensor(s), "
              f"{_after:,} of {_before:,} parameters ({100.0*_after/max(1,_before):.3f}% of the "
              f"head). Everything else is FROZEN.")
        for n in _kept:
            print(f"    {_rel[n]:46} {tuple(dict(model.named_parameters())[n].shape)}")

    # A LoRA BUNDLE HOLDS ADAPTERS, NOT HEAD TENSORS, and the held-out monitor spawns itself
    # without --lora-rank -- the running trainer builds that command from its in-memory arguments,
    # so adding the flag there cannot reach a trainer already running. Its trainable set was
    # therefore the 248 plain head tensors, none of which a rank-8 bundle contains, and every
    # measurement died on
    #     STOP: the bundle is missing 248 trainable tensor(s)
    # Arm m8rote trained three epochs and produced ZERO held-out points before this was caught --
    # the same failure as ggfhj5 epoch 1, which was --train-only instead of LoRA.
    #
    # So under --loss-only, inherit the bundle's own LoRA configuration. The guard already records
    # all four fields for exactly this reason ("resuming one as the other fails on the
    # missing-tensor check"). Only when this run named no rank of its own, so an explicit
    # --lora-rank still wins and a non-LoRA bundle is untouched.
    # THE SAME CLASS OF BUG, FOURTH INSTANCE. --train-only, then LoRA, then the evaluator's
    # --pose-perblock-share, each broke because a setting the trainer passes was not also inherited
    # or forwarded. The ranking fields are in the guard because they change the objective, so a
    # --loss-only monitor that does not name them is refused on a rank-loss bundle -- and the
    # monitor CANNOT name them, because a running trainer keeps sending the argument list its own
    # loaded code built. So inherit them here instead. The ranking term is skipped under
    # --loss-only anyway (there is no backward to add it to), so this only quiets the guard.
    if a.loss_only and a.resume and not a.rank_loss:
        try:
            _gr = (torch.load(a.resume, map_location="cpu", weights_only=False).get("guard")
                   or {})
        except Exception as _e:                              # noqa: BLE001
            _gr = {}
            print(f"  (could not read {a.resume} for its ranking config: {type(_e).__name__})")
        if _gr.get("rank_loss"):
            a.rank_loss = True
            a.alpha_rank = float(_gr.get("alpha_rank") or 0.0)
            a.rank_weight_by_gap = bool(_gr.get("rank_weight_by_gap", True))
            if is_main:
                print(f"  --loss-only: inheriting the bundle's ranking configuration "
                      f"(alpha_rank {a.alpha_rank:g}, "
                      f"weight_by_gap={a.rank_weight_by_gap}) so the guard matches; the ranking "
                      f"term itself does not run -- a measurement takes no step.")

    if a.loss_only and a.resume and not a.lora_rank:
        try:
            _g = (torch.load(a.resume, map_location="cpu", weights_only=False).get("guard")
                  or {})
        except Exception as _e:                              # noqa: BLE001
            _g = {}
            print(f"  (could not read {a.resume} for its LoRA config: {type(_e).__name__})")
        if int(_g.get("lora_rank") or 0):
            a.lora_rank = int(_g["lora_rank"])
            a.lora_alpha = float(_g.get("lora_alpha") or 0.0)
            a.lora_targets = _g.get("lora_targets") or ""
            a.lora_dropout = float(_g.get("lora_dropout") or 0.0)
            if is_main:
                print(f"  --loss-only: inheriting the bundle's LoRA configuration "
                      f"(rank {a.lora_rank}, alpha {a.lora_alpha:g}, dropout {a.lora_dropout:g}) "
                      f"-- it holds adapter tensors, so the model must carry the same adapters "
                      f"for its weights to load at all.")

    lora_report = None
    if a.lora_rank:
        import lora as LORA
        _hm = dict(model.named_modules())[prefixes[0]]
        _lnames, lora_report = LORA.apply_lora(
            _hm, a.lora_rank, a.lora_alpha or 2.0 * a.lora_rank,
            [t.strip() for t in a.lora_targets.split(",") if t.strip()], a.lora_dropout)
        trainable = [f"{prefixes[0]}.{n}" for n in _lnames]

        print(f"LoRA rank {a.lora_rank} alpha {a.lora_alpha or 2.0*a.lora_rank:g} "
              f"dropout {a.lora_dropout:g}: {lora_report['modules_adapted']} module(s) adapted, "
              f"{lora_report['lora_params']/1e6:.3f}M adapter parameters against "
              f"{lora_report['frozen_params']/1e6:.2f}M frozen "
              f"({100*lora_report['fraction_of_head']:.2f}% of the head)")
        if lora_report["skipped_rank_too_high"]:
            print(f"  {len(lora_report['skipped_rank_too_high'])} module(s) skipped: rank "
                  f"{a.lora_rank} exceeds a dimension, e.g. "
                  f"{lora_report['skipped_rank_too_high'][:3]}")
        if lora_report["adapted_with_bias"]:
            print(f"  NOTE {len(lora_report['adapted_with_bias'])} adapted module(s) have a bias; "
                  f"it stays FROZEN -- the adapter is the only thing that moves.")
    head_ids = {id(p) for n, p in model.named_parameters() if n in set(trainable)}
    n_head = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
    n_all = sum(p.numel() for _n, p in model.named_parameters())
    print(f"trainable: {len(trainable)} tensor(s), {n_head/1e6:.1f}M of {n_all/1e6:.1f}M parameters "
          f"({n_head/n_all:.1%}), under {prefixes}")
    for n, p in model.named_parameters():
        p.requires_grad_(id(p) in head_ids)

    _head_names = {n for n, _p in model.named_parameters()
                   if any(n == pr or n.startswith(pr + ".") for pr in prefixes)}

    def _head_ckpt_tensors():
        """The head checkpoint, ALWAYS in the shipped head's tensor names and shapes.

        Under LoRA the wrapper renames `blk.linear.weight` to `blk.linear.base.weight`, so
        saving named_parameters() straight would write a file phase_k_eval cannot load -- it
        knows nothing about adapters. Merging here keeps every downstream scorer unchanged and
        makes a LoRA checkpoint indistinguishable from a full-finetune one. Without LoRA this is
        exactly the dict that was saved before.
        """
        if a.lora_rank:
            import lora as LORA
            _pfx = prefixes[0]
            return {f"{_pfx}.{k}": v
                    for k, v in LORA.merged_head_tensors(
                        dict(model.named_modules())[_pfx]).items()}
        # EVERY HEAD TENSOR, NOT JUST THE TRAINABLE ONES. phase_k_eval requires SET EQUALITY
        # between the checkpoint's keys and the model's head parameters -- deliberately, so a
        # checkpoint from a different architecture cannot load with tensors left at their
        # initial values. Under --train-only the trainable set is three tensors, and writing
        # only those would be refused at the first eval. Identical to the old behaviour when the
        # whole head trains, which is every run so far.
        return {n: pp.detach().cpu() for n, pp in model.named_parameters()
                if n in _head_names}

    # NO EXPERIMENT TRACKER. Runs are recorded by the files this script already writes --
    # trajectory.csv, metric_ep<N>.json, args.json -- and rendered by run_logging/render_report.py.
    # wandb was removed on 2026-10-16 at the user's direction: it needed an account, it put the
    # record somewhere the repo cannot read, and instrumenting the trainer for a viewer means a
    # restart whenever the viewer changes. The records outlive whatever reads them.

    # ---- THE MANIFEST COMES FROM THE FROZEN SELECTION, NOT FROM WHATEVER FILES EXIST ----
    # GLOBBING THE LABEL DIRECTORY CANNOT ESTABLISH COMPLETENESS. It did: the manifest was built by
    # globbing *.npz and then checking that each DISCOVERED label had a trunk, so a label that was
    # never written simply did not appear and the training set shrank with no error. Review round 8.
    # The population is now the selection csv times the seed range, minus named exclusions, and a
    # missing label is fatal.
    roots = [r.strip() for r in a.fold.split(",") if r.strip()]
    for r in roots:
        if not os.path.isdir(r):
            raise SystemExit(f"STOP: --fold root {r} does not exist")
    label_dirs = [x.strip() for x in a.labels.split(",") if x.strip()]
    lo_s, hi_s = (int(x) for x in a.seeds.split("-"))
    skip = {x.strip().lower() for x in a.manifest_skip.split(",") if x.strip()}
    with open(a.manifest_selection) as fh:
        sel_pdbs = [r["pdb"].strip().lower() for r in csv.DictReader(fh)]
    if a.manifest_nshards and a.manifest_nshards > 1:
        if not (0 <= a.manifest_shard < a.manifest_nshards):
            raise SystemExit(f"STOP: --manifest-shard {a.manifest_shard} is not in "
                             f"[0, {a.manifest_nshards}).")
    want = [(p, sd) for p in sel_pdbs if p not in skip for sd in range(lo_s, hi_s + 1)]
    _rank_check_coverage(want)
    print(f"manifest: {len(sel_pdbs)} selected structure(s) minus {sorted(skip) or 'none'} "
          f"x seeds {lo_s}-{hi_s} = {len(want)} intended unit(s), from {a.manifest_selection}")

    manifest, no_label, no_trunk = [], [], []
    for pdb, sd in want:
        arm_seed = f"free_{sd}"
        lp = next((os.path.join(d, pdb, f"{arm_seed}.npz") for d in label_dirs
                   if os.path.exists(os.path.join(d, pdb, f"{arm_seed}.npz"))), None)
        if lp is None:
            no_label.append(f"{pdb}/{arm_seed}"); continue
        cands = [os.path.join(r, "trunks", pdb, f"{arm_seed}.npz") for r in roots
                 if os.path.exists(os.path.join(r, "trunks", pdb, f"{arm_seed}.npz"))]
        if not cands:
            no_trunk.append(f"{pdb}/{arm_seed}"); continue
        manifest.append((pdb, lp, cands))        # ALL candidates; provenance picks below
    if no_label or no_trunk:
        raise SystemExit(
            f"STOP: the training set is INCOMPLETE.\n"
            f"  {len(no_label)} intended unit(s) have no label, e.g. {no_label[:5]}\n"
            f"  {len(no_trunk)} have no trunk under {roots}, e.g. {no_trunk[:5]}\n"
            f"  Build them, or name a deliberate exclusion in --manifest-skip and record it in "
            f"docs/build_cache_exclusions.md.")

    bypdb = {}
    for pdb, lp, cands in manifest:
        bypdb.setdefault(pdb, []).append((lp, cands))
    order = sorted(bypdb)
    rng = random.Random(a.seed)
    if a.shuffle == "structure":
        rng.shuffle(order)
        for pp in order:
            rng.shuffle(bypdb[pp])
    flat = [(pp, lp, cands) for pp in order for lp, cands in bypdb[pp]]
    if a.limit_units:
        flat = flat[:a.limit_units]
    # THE POPULATION, AS ONE HASH. The resume guard compares this rather than the arguments that
    # produced it: --manifest-selection, --manifest-skip, --seeds and --limit-units can all change
    # which units exist, and a guard that checked the flags would still miss a rewritten selection
    # csv. Review 2026-10-16 reproduced a resume accepted after the exclusions changed.
    # --eval-every IS PARSED AND NEVER READ. Found in review: the flag has no effect anywhere in
    # this file, so launch_lr_probe_full.sh asked for validation every 1,452 steps and would never
    # have produced a single pass. Refused rather than implemented: a second eval trigger has to
    # reproduce the barrier discipline of the epoch path exactly -- rank 0 evaluates while the
    # others wait -- and a mistake there deadlocks rather than fails. The two working triggers do
    # cover the need.
    if a.eval_every:
        raise SystemExit(
            f"STOP: --eval-every {a.eval_every} has no effect -- it is parsed and never read, so "
            f"it would promise a validation pass that never happens. Use --eval-every-epoch for "
            f"the validation set every epoch, or --train-eval-pool with --train-eval-every N "
            f"(optionally --train-eval-fold / --train-eval-seed-base / --train-eval-budget) to "
            f"score any pool every N epochs. Both are exercised by the current arms.")
    _manifest_digest = hashlib.sha256(
        "\n".join(f"{p}/{os.path.basename(lp)}" for p, lp, _c in sorted(flat)).encode()
    ).hexdigest()[:16]
    print(f"  manifest digest {_manifest_digest} over {len(flat)} unit(s)")
    _pae_mask_coverage(flat)
    # THE PREFLIGHT IS REQUIRED, AND IT MUST BE THIS POPULATION'S. A marker from a different
    # manifest proves nothing about the structures this run will train on.
    # UNCONDITIONAL NOW, NOT GATED ON --minimal-items. The dataset-item path is gone, so the
    # minimal path is the ONLY path and check_alignment can no longer run per structure. Leaving
    # this behind the flag meant omitting --minimal-items silently dropped the alignment check
    # altogether instead of falling back to the per-structure one that used to exist.
    if True:
        if not a.alignment_preflight:
            raise SystemExit(
                "STOP: --minimal-items needs --alignment-preflight. Skipping the full dataset item "
                "also skips check_alignment, which is what proves the trunk's features agree atom "
                "for atom with the dataset's -- the one failure that would pair a pose with another "
                "molecule's label. Run:\n"
                "  phases/phase_k_alignment_preflight.py --out <marker>\n"
                "and pass that marker.")
        if not os.path.exists(a.alignment_preflight):
            raise SystemExit(f"STOP: no alignment preflight marker at {a.alignment_preflight}")
        _pf = json.load(open(a.alignment_preflight))
        if _pf.get("manifest_digest") != _manifest_digest:
            raise SystemExit(
                f"STOP: the alignment preflight was run over manifest "
                f"{_pf.get('manifest_digest')}, this run is {_manifest_digest}. It says nothing "
                f"about the structures this run will train on; re-run it.")
        if not _pf.get("ok"):
            raise SystemExit(f"STOP: {a.alignment_preflight} does not record a pass.")
        # THE MARKER MUST COVER THE WHOLE MANIFEST, not just carry its digest. The digest is
        # computed over every unit even when the preflight ran with --limit, so a one-structure
        # debug marker used to pass both checks above and authorise all 491. Asserted on BOTH
        # sides: the preflight now refuses to write ok=true when incomplete, and this refuses a
        # marker that does not say so -- including one written before that change existed.
        if not _pf.get("complete"):
            _lim = _pf.get("limit")
            raise SystemExit(
                f"STOP: {a.alignment_preflight} does not record complete=true. It checked "
                f"{_pf.get('n_structures')} of {_pf.get('n_manifest_structures', '?')} structure(s)"
                + (f" (--limit {_lim})" if _lim else "")
                + ", so it says nothing about the rest. Re-run the preflight without --limit.")
        print(f"  alignment preflight: PASS over {_pf.get('n_structures')} structure(s), "
              f"digest {_pf.get('manifest_digest')}")
        print(f"  MINIMAL ITEMS: the dataset item is not built; the loss reads the cached label "
              f"only ({len(BC.LOSS_FEATS)} features + coordinate/coordinate_mask)")
    # THE SHARD IS APPLIED HERE, AFTER the manifest digest, the PAE mask coverage and the
    # alignment preflight have all run over the WHOLE population. Slicing sel_pdbs earlier -- the
    # first version of this -- made each child compute a digest over its own 17-18 structures and
    # compare it to a marker written for all 53, so every child exited with "the alignment
    # preflight was run over manifest X, this run is Y". All three shards would have failed, and
    # the any-shard-fails policy would then have dropped the held-out point EVERY epoch: a
    # permanently dead monitor that looks like a configuration problem. Caught in review.
    #
    # Filtering by STRUCTURE keeps every seed of a structure in one child, which is convenient but
    # is NOT what makes the pooled mean correct -- disjoint, complete pose coverage is, and that
    # is asserted by the parent against the expected population before it publishes a number.
    if a.manifest_nshards and a.manifest_nshards > 1:
        _all_pdbs = sorted({x[0] for x in flat})
        _mine = set(_all_pdbs[a.manifest_shard::a.manifest_nshards])
        _n_before = len(flat)
        flat = [x for x in flat if x[0] in _mine]
        print(f"  manifest SHARD {a.manifest_shard} of {a.manifest_nshards}: "
              f"{len(_mine)} of {len(_all_pdbs)} structure(s), {len(flat)} of {_n_before} unit(s), "
              f"interleaved -- digest and preflight above cover the FULL population")
        # MACHINE-READABLE, because the parent cannot derive the FULL population itself: units are
        # dropped for missing labels or trunks after the selection is read, so
        # structures x seeds overstates it. Only the child knows _n_before. The parent asserts the
        # shards agree on it and that they sum to it, which is what makes the pooled mean a mean
        # over the same population every epoch.
        print(f"SHARD COVERAGE shard={a.manifest_shard} nshards={a.manifest_nshards} "
              f"units_mine={len(flat)} units_all={_n_before} nsample={a.nsample}", flush=True)
    print(f"  resolved {len(flat)} unit(s) over {len(set(x[0] for x in flat))} structure(s) "
          f"from {len(roots)} fold root(s), shuffle={a.shuffle}, seed={a.seed}")

    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    cfg.train_confidence_only = True
    err = os.path.join(ROOT, "training", "prep", "dataset_errors")
    perm = SymmetricPermutation(cfg, error_dir=err)
    # THE SAME FINGERPRINT CONTEXT build_cache USES, imported rather than reimplemented. A
    # reimplementation that drifted would reject every label or accept every label; round 8 found
    # the latter, because `fingerprint` was required to be PRESENT and never compared.
    code_fp, loss_fp = BC.fingerprint_context(cfg)
    print(f"fingerprint context: code={code_fp} loss={loss_fp}")

    opt = torch.optim.AdamW([p for n, p in model.named_parameters() if n in set(trainable)],
                            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    init_state = copy.deepcopy({n: p.detach().clone() for n, p in model.named_parameters()
                                if n in set(trainable)})
    init_opt = copy.deepcopy(opt.state_dict())

    # THE RESUME, LOADED BY EVERY RANK. The bundle is written by rank 0 but all ranks must restore
    # it: they each hold their own copy of the head and their own optimiser, and a rank left at the
    # initial weights would contribute a gradient computed at a different point in parameter space
    # and quietly corrupt the averaged step.
    start_epoch, start_step = 0, 0
    _drift_history = []
    if a.resume:
        if not os.path.exists(a.resume):
            raise SystemExit(f"STOP: no resume bundle at {a.resume}")
        _b = torch.load(a.resume, map_location="cpu", weights_only=False)
        _g = _b.get("guard", {})
        # INHERITED ON EVERY RESUME, NOT ONLY A DRIFTING ONE. A clean continuation of a run that
        # was itself a fork is still a fork's descendant, and a chain that dropped its history at
        # the first non-drifting link would read as pristine two bundles later.
        _drift_history = list(_g.get("drift_history") or [])
        # WHAT MUST MATCH, in two groups. The first changes which poses land in which batch, so the
        # saved moments describe a different problem. The second changes the objective or the
        # schedule, so continuing would silently blend two experiments. Checking only the first four
        # let a resume through after the exclusions, the loss weights and the lr had all changed
        # (review 2026-10-16).
        _mine = {"world": world, "seed": a.seed, "accum": a.accum, "nsample": a.nsample,
                 "manifest_digest": _manifest_digest, "crop_size": a.crop_size,
                 "alpha_confidence": a.alpha_confidence, "alpha_pae": a.alpha_pae,
                 "drop_pde": a.drop_pde, "clip_norm": a.clip_norm,
                 "rank_loss": a.rank_loss, "alpha_rank": a.alpha_rank,
                 "rank_weight_by_gap": a.rank_weight_by_gap,
                 # THE FOUR CROSS-SEED FIELDS. Every one of them changes the objective or the step
                 # shape, so a resume that moved any of them silently would be a different
                 # experiment wearing the same run id. FOURTH time a field has been added to this
                 # comparison without being added to the saved bundle below -- so both sides are
                 # edited in the same patch, and test_guard_fields.py asserts they agree.
                 "rank_form": a.rank_form,
                 "rank_tau": (float(a.rank_tau) if a.rank_tau is not None else -1.0),
                 "rank_cross_seed": a.rank_cross_seed,
                 "rank_pairs": a.rank_pairs,
                 "rank_within_weight": a.rank_within_weight,
                 "rank_topk_decay": a.rank_topk_decay,
                 "rank_straddle": a.rank_straddle,
                 "rank_structs_per_step": a.rank_structs_per_step,
                 # L2-SP CHANGES THE OBJECTIVE and was missing here: a resume could silently move
                 # lambda and the drift guard would not notice. Caught in review of cd8e6ab7.
                 "l2_sp": a.l2_sp,
                 "l2_sp_exclude": a.l2_sp_exclude,
                 # AND SO DOES THE SCHEDULE. lr_final changes the lr at every step after warmup,
                 # so a resume that added or dropped the cosine would be a different experiment.
                 "lr_final": a.lr_final,
                 "lr": a.lr, "warmup_steps": a.warmup_steps,
                 # CROPPING CHANGES THE OBJECTIVE -- the loss averages over a different set of
                 # tokens and pairs -- so resuming across a crop change would blend two experiments
                 # into one curve, which is exactly what this guard exists to stop.
                 "crop_margin": a.crop_margin, "crop_contact": a.crop_contact,
                 # SAME REASON AS CROPPING. --pae-scored-pairs changes which pairs the PAE term
                 # averages over and --pae-only changes which terms exist at all, so resuming
                 # across either blends two objectives into one curve, carrying the first one's
                 # AdamW moments. These were added to the saved bundle and to _PRE_CROP but NOT
                 # here, which left the guard inert: it compared neither, and the _PRE_CROP
                 # defaults for them were never consulted.
                 "pae_scored_pairs": a.pae_scored_pairs, "pae_only": a.pae_only,
                 # NOT JUST THE OBJECTIVE -- THE PARAMETER SET. --frame-direction adds a trainable
                 # tensor, so a bundle written without it is missing one and the _missing check
                 # below refuses it anyway. It is compared here so the refusal names the setting
                 # that caused it instead of reporting an absent tensor.
                 "frame_direction": a.frame_direction,
                 # SAME REASON AS frame_direction: A adds trainable tensors, so a bundle written
                 # without it is missing some and the _missing check below refuses it anyway.
                 # Compared here so the refusal names the setting rather than an absent tensor.
                 "pose_perblock": a.pose_perblock,
                 "pose_perblock_tied": a.pose_perblock_tied,
                 "pose_perblock_share": a.pose_perblock_share,
                 "pose_perblock_copy": a.pose_perblock_copy,
                 # THE PARAMETER SET ITSELF. A bundle written at rank 8 holds 282 adapter
                 # tensors and one written without LoRA holds 248 head tensors; resuming one as
                 # the other fails on the missing-tensor check with a message about tensors
                 # rather than about the setting that changed.
                 "lora_rank": a.lora_rank, "lora_alpha": a.lora_alpha,
                 "lora_targets": a.lora_targets, "lora_dropout": a.lora_dropout,
                 "train_only": a.train_only}

        # A BUNDLE WRITTEN BY OLDER CODE IS NOT DRIFT. The guard grew from four fields to twelve
        # while run 9mfb4y was in flight, so its bundles carry only the original eight: the running
        # process had the old module loaded and kept writing the old shape while HEAD moved under
        # it. Reading the six newer fields as "<absent>" made every one of them a mismatch, which
        # would have reported six settings as changed when none had, and a resume as a FORK when it
        # is a continuation.
        #
        # So an absent field falls back to the run's own args.json, which sits beside the bundle and
        # does record them. That is the same evidence, written by the same process at launch.
        _side = os.path.join(os.path.dirname(os.path.abspath(a.resume)), "args.json")
        _fallback = {}
        if os.path.exists(_side):
            try:
                _fallback = json.load(open(_side))
            except Exception as e:
                print(f"  (could not read {_side}: {type(e).__name__}: {e})")

        # A FIELD THAT POSTDATES THE BUNDLE IS NOT AN UNKNOWN FIELD. crop_margin and crop_contact
        # joined the guard after run 9mfb4y was already in flight, so none of its bundles carry
        # them -- not resume_ep1/2/3, and not the ep4/ep5 ones being written tonight either, since
        # that process loaded its code before the fields existed. Reading them as "recorded
        # nowhere" made the _unknown hard-stop fire on every pre-crop bundle and refused the
        # resume outright, which --resume-allow-drift could not even override because the stop
        # runs first. I turned a safety check into a wall by adding two fields every existing
        # bundle lacks; review round 6 reproduced it.
        #
        # Absence is EVIDENCE here, not ignorance: a bundle written by code with no cropping was
        # written with cropping off. So these migrate to the pre-crop defaults and are compared
        # like any other field -- which means a resume that ENABLES cropping shows up as real
        # drift and needs --resume-allow-drift, because it genuinely changes the objective.
        _PRE_CROP = {"lora_rank": 0, "lora_alpha": 0.0, "lora_targets": "", "lora_dropout": 0.0,
                 "train_only": "",
                     "crop_margin": 0.0, "crop_contact": 5.0, "pae_scored_pairs": "",
                     "pae_only": False,
                     "frame_direction": False,
                     "pose_perblock": False, "pose_perblock_tied": False,
                     "pose_perblock_share": False, "pose_perblock_copy": False,
                     "rank_loss": False, "alpha_rank": 1.0,
                     "rank_weight_by_gap": True, "l2_sp": 0.0, "l2_sp_exclude": "",
                     # A BUNDLE PREDATING THESE RAN THE LINEAR WITHIN-SEED LOSS, which is what
                     # these defaults say. rank_tau -1.0 means "unset", matching how the
                     # comparison side encodes None.
                     "rank_form": "linear", "rank_tau": -1.0,
                     "rank_cross_seed": False, "rank_structs_per_step": 2,
                     # A BUNDLE PREDATING THESE THREE RAN CROSS-SEED PAIRS ONLY, AT NATURAL
                     # WEIGHT, WITH NO RANK DECAY -- which is what these defaults say. Their
                     # absence is evidence of the inert values, not of an unknown setting, so
                     # they belong here rather than in _unknown: the STOP at "recorded nowhere
                     # for this bundle" fires BEFORE --resume-allow-drift is consulted and so
                     # cannot be overridden, which would have made every pre-patch bundle --
                     # including the live zg5fnw's per-epoch held-out subprocess, which resumes
                     # its own bundle through this same file -- unresumable.
                     "rank_pairs": "cross", "rank_within_weight": 1.0, "rank_topk_decay": 0.0,
                     "rank_straddle": 0.0}
        _migrated = []

        _bad, _from_args, _unknown = {}, [], []
        for k, v in _mine.items():
            if k in _g:
                if _g[k] != v:
                    _bad[k] = (_g[k], v)
            elif k in _fallback:
                if _fallback[k] != v:
                    _bad[k] = (_fallback[k], v)
                else:
                    _from_args.append(k)
            elif k in _PRE_CROP:
                _migrated.append(k)
                if _PRE_CROP[k] != v:
                    _bad[k] = (f"{_PRE_CROP[k]} (pre-crop default)", v)
            else:
                _unknown.append(k)
        if _migrated:
            print(f"  resume guard: {len(_migrated)} field(s) postdate this bundle and are read as "
                  f"the pre-crop default: "
                  f"{', '.join(f'{k}={_PRE_CROP[k]}' for k in sorted(_migrated))}")
        if _from_args:
            print(f"  resume guard: {len(_from_args)} field(s) absent from the bundle and matched "
                  f"against {os.path.basename(_side)} instead: {', '.join(sorted(_from_args))}")
        if _unknown:
            # manifest_digest is the only field in neither place. It is a hash of the unit list,
            # which is determined by the selection file and the skip list -- and manifest_skip IS
            # in the bundle and is compared above. So the population is still checked, by its
            # inputs rather than by its hash. Said out loud rather than passed over, because an
            # unverified field is not a verified one.
            print(f"  resume guard: {len(_unknown)} field(s) in neither the bundle nor "
                  f"{os.path.basename(_side)}, so NOT compared: {', '.join(sorted(_unknown))}")
            for k in _unknown:
                if k != "manifest_digest":
                    raise SystemExit(
                        f"STOP: {k} is recorded nowhere for this bundle, so a resume cannot show "
                        f"the setting is unchanged. Relaunch from scratch instead.")
            print(f"    manifest_digest is a hash of the unit list; manifest_skip "
                  f"({_g.get('manifest_skip')!r}) and the selection file determine that list and "
                  f"manifest_skip is compared, so the population is checked by its inputs.")
        # NAMED OVERRIDES, NOT A BLANKET ONE. --resume-allow-drift accepts every difference,
        # including the one you did not know you had made -- which is how a resume that was meant
        # to change only the lr can also silently change the crop. --resume-override names the
        # fields, so an unnamed difference is still refused.
        _ovr = {f.strip() for f in a.resume_override.split(",") if f.strip()}
        _unknown_ovr = sorted(_ovr - set(_mine))
        if _unknown_ovr:
            raise SystemExit(
                f"STOP: --resume-override names {len(_unknown_ovr)} field(s) the guard does not "
                f"have: {_unknown_ovr}.\n"
                f"  A typo here protects nothing -- the field it was meant to allow stays refused "
                f"and the one it mistyped is not a field at all.\n"
                f"  The guard's fields are: {', '.join(sorted(_mine))}")
        # A STALE OVERRIDE IS ALSO AN ERROR. A name left on the command line after the setting it
        # covered went back to matching still disarms the guard for that field on the NEXT resume
        # that reuses the line, and nothing would say so.
        _stale = sorted(_ovr - set(_bad))
        if _stale and not a.resume_allow_drift:
            raise SystemExit(
                f"STOP: --resume-override names {len(_stale)} field(s) that do not differ from the "
                f"bundle: {_stale}.\n"
                f"  Drop them. Left in place they disarm the guard for those fields on the next "
                f"resume that reuses this command line.")
        _unnamed = {k: v for k, v in _bad.items() if k not in _ovr}
        if _unnamed and not a.resume_allow_drift:
            raise SystemExit(
                "STOP: this resume bundle was written under different settings:\n"
                + "".join(f"  {k}: bundle {b!r} vs this run {m!r}\n"
                          for k, (b, m) in _unnamed.items())
                + "  Changes to world/seed/accum/nsample/manifest_digest/crop_size move which poses\n"
                  "  land in which batch, so the saved AdamW moments belong to a different problem.\n"
                  "  Changes to the alphas, drop_pde, clip_norm, lr or warmup change the objective or\n"
                  "  the schedule, so continuing would blend two experiments into one curve.\n"
                  "  Match the settings, relaunch from scratch, or accept these deliberately with\n"
                  f"    --resume-override {','.join(sorted(_unnamed))} --resume-note '<why>'\n"
                  "  (--resume-allow-drift accepts everything at once, including what you did not\n"
                  "  intend). Either way it is recorded in resume_drift.json and carried forward in\n"
                  "  every bundle this run writes.")
        # AN UNEXPLAINED DEVIATION IS THE ONE A LATER READER CANNOT TELL FROM A MISTAKE --
        # EXCEPT UNDER --loss-only, WHICH IS NOT A DEVIATION AT ALL. A measurement does not
        # continue the bundle's schedule; it reads the bundle's weights, writes no checkpoint and
        # takes no step, so there is nothing for a later reader to mistake. The same reasoning
        # already exempts it from the epoch-count refusal below.
        #
        # THIS WAS LEARNED THE EXPENSIVE WAY, on the live run. The epoch-7 held-out pass died
        # with "accepts 5 changed setting(s) (lr, manifest_digest, pae_scored_pairs,
        # warmup_steps, world) and --resume-note is empty" -- a held-out fold legitimately
        # differs from the training bundle in all five. Passing the note from the subprocess
        # command does NOT fix it: the trainer that spawns that subprocess has its own code
        # loaded in memory and keeps sending the old argument list, while the subprocess reads
        # this file fresh from disk. Any mandatory flag added here breaks every RUNNING trainer's
        # monitor the moment the file is saved. A measurement must never be able to do that.
        if _bad and not a.resume_note.strip() and not a.loss_only:
            raise SystemExit(
                f"STOP: this resume accepts {len(_bad)} changed setting(s) "
                f"({', '.join(sorted(_bad))}) and --resume-note is empty.\n"
                f"  Say why in one line. It is written to resume_drift.json and travels in every "
                f"bundle this run writes, which is what lets the next reader tell a deliberate "
                f"fork from a mismatch nobody noticed.")
        if _bad:
            _via = "--resume-allow-drift" if (a.resume_allow_drift and _unnamed) \
                else "--resume-override"
            print(f"ACCEPTING RESUME DRIFT ({_via}); this run is a FORK of the bundle, "
                  f"not a continuation of it:")
            for k, (b, m) in _bad.items():
                print(f"  {k}: bundle {b!r} -> this run {m!r}"
                      + ("" if k in _ovr else "   <- NOT NAMED, accepted by the blanket flag"))
            print(f"  why: {a.resume_note.strip()}")
            # CARRIED FORWARD, NOT JUST PRINTED. A fork of a fork must still be readable as one,
            # so this run's deviation is appended to whatever history the bundle already carried
            # and the whole chain goes into every bundle this run writes.
            _drift_history.append({
                "from_bundle": os.path.abspath(a.resume),
                "bundle_commit": _g.get("commit", "?"),
                "bundle_written": _g.get("written", "?"),
                "bundle_epoch": int(_b["epoch"]), "bundle_step": int(_b["step"]),
                "accepted_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "via": _via,
                "note": a.resume_note.strip(),
                "fields": {k: {"bundle": b, "this_run": m} for k, (b, m) in _bad.items()},
                "named": sorted(_ovr & set(_bad)),
                "unnamed": sorted(_unnamed),
            })
            if is_main:
                _dp = os.path.join(a.out, "resume_drift.json")
                try:
                    with open(_dp, "w") as _fh:
                        json.dump(_drift_history, _fh, indent=1, default=str)
                    print(f"  recorded -> {_dp}")
                except Exception as _e:
                    print(f"  (could not write {_dp}: {type(_e).__name__}: {_e})")
        # NOT UNDER --loss-only. A measurement does not continue the bundle's schedule, it reads
        # the bundle's weights: "the bundle has 141 epochs done and --epochs is 1" is a refusal
        # about training that is not happening, and the epoch counter below is reset to 0 so the
        # single pass actually runs rather than falling into range(141, 1) and measuring nothing.
        if not a.loss_only and _b["epoch"] >= max(1, math.ceil(a.epochs)):
            raise SystemExit(f"STOP: the bundle has {_b['epoch']} epoch(s) done and --epochs is "
                             f"{a.epochs}; there is nothing left to run.")
        # A --train-only BUNDLE CARRIES ONLY THE TENSORS IT TRAINED, which is correct: the ones it
        # omits were never updated, so they are still the shipped head's. But the held-out monitor
        # spawns itself WITHOUT --train-only -- the running trainer builds that command from its
        # own in-memory arguments, so adding the flag there cannot reach a trainer already running
        # -- and its trainable set is therefore all 248. This guard then refused every measurement
        # of a --train-only run: "the bundle is missing 245 trainable tensor(s)". It killed panel
        # 2, the only panel that matters, on arm ggfhj5 at epoch 1.
        #
        # SO INHERIT THE BUNDLE'S OWN SET, and only under --loss-only, where there is no optimiser
        # to keep consistent and no checkpoint to write. THIS IS NOT A BLANKET WEAKENING: the
        # bundle must carry EXACTLY what its own recorded train_only selects, so a TRUNCATED
        # bundle -- the case this check exists for -- is still refused.
        _b_only = (_b.get("guard") or {}).get("train_only") or ""
        if a.loss_only and _b_only and not a.train_only:
            import re as _re_i
            _pfx_i = prefixes[0]
            _pats_i = [t.strip() for t in _b_only.split(",") if t.strip()]
            _want_i = {_n for _n in trainable
                       if any(_re_i.search(_p, _n[len(_pfx_i) + 1:]) for _p in _pats_i)}
            if _want_i and _want_i == set(_b["head"]):
                trainable = sorted(_want_i)
                if is_main:
                    print(f"  --loss-only: inheriting the bundle's --train-only '{_b_only}'. It "
                          f"carries exactly the {len(_want_i)} tensor(s) that selects; the rest of "
                          f"the head was never trained and stays at the shipped weights, which is "
                          f"the head this measures.")
        _tset = set(trainable)
        _missing = _tset - set(_b["head"])
        if _missing:
            raise SystemExit(f"STOP: the bundle is missing {len(_missing)} trainable tensor(s), "
                             f"e.g. {sorted(_missing)[:3]}")
        with torch.no_grad():
            for _n, _p in model.named_parameters():
                if _n in _tset:
                    _p.data.copy_(_b["head"][_n].to(_p.device, _p.dtype))
        # THE OPTIMISER IS NOT RESTORED FOR A MEASUREMENT. AdamW never steps under --loss-only, so
        # its moments are dead weight -- and after inheriting a 3-tensor set above, the live param
        # group still holds 248, so load_state_dict would raise on the size mismatch.
        if not a.loss_only:
            opt.load_state_dict(_b["opt"])
        start_epoch, start_step = int(_b["epoch"]), int(_b["step"])
        if a.loss_only:
            _measured_epoch, _measured_step = start_epoch, start_step
            start_epoch, start_step = 0, 0
        if is_main:
            # SAY WHAT ACTUALLY HAPPENED TO THE OPTIMISER. This line claimed "AdamW state for 298
            # tensor(s) restored" on a --loss-only pass that deliberately does NOT restore it --
            # a log that misreports its own run is worse than a quiet one.
            _opt_note = (f"AdamW state for {len(_b['opt'].get('state', {}))} tensor(s) restored"
                         if not a.loss_only else
                         "AdamW state NOT restored (measurement takes no step)")
            print(f"RESUMED from {a.resume}: {start_epoch} epoch(s) done, step {start_step}, "
                  f"{_opt_note}; "
                  f"continuing at epoch {start_epoch + 1}/{max(1, math.ceil(a.epochs))}")
            print(f"  bundle commit {_g.get('commit', '?')[:8]}, written {_g.get('written', '?')}")

    # THE L2-SP ANCHOR IS THE SHIPPED HEAD, AND ONLY A FRESH RUN STILL HOLDS IT. Set up AFTER the
    # resume block on purpose: at this point a fresh run's weights are untouched (init == shipped)
    # and can be snapshotted, while a resumed run's weights are 60 epochs of training away from
    # shipped and must read the anchor back from disk instead.
    _l2sp_anchor, _l2sp_params = None, []
    _l2sp_drift = {"sq": 0.0, "n": 0}
    if a.l2_sp:
        _anchor_p = os.path.join(a.out, "l2sp_anchor.pt")
        _tset_sp = set(trainable)
        # EXCLUSIONS ARE APPLIED HERE, BEFORE THE ANCHOR IS BUILT, so an excluded tensor never
        # enters _a_sd and the resume consistency check cannot be satisfied by a stale anchor
        # that still carries it. Reported by name: an anchor covering fewer tensors than the
        # reader believes is the bug this project keeps rediscovering.
        if a.l2_sp_exclude:
            import re as _re_sp
            _pats_sp = [t.strip() for t in a.l2_sp_exclude.split(",") if t.strip()]
            _pfx_sp = prefixes[0]
            _drop = {n for n in _tset_sp
                     if any(_re_sp.search(_p, n[len(_pfx_sp) + 1:]) for _p in _pats_sp)}
            if not _drop:
                raise SystemExit(
                    f"STOP: --l2-sp-exclude {_pats_sp} matched no trainable parameter, so it "
                    f"would silently do nothing while the record says the anchor was reduced.\n"
                    f"  trainable: {[n[len(_pfx_sp) + 1:] for n in sorted(_tset_sp)]}")
            if _drop == _tset_sp:
                raise SystemExit(
                    f"STOP: --l2-sp-exclude {_pats_sp} matched EVERY trainable parameter, which "
                    f"leaves no anchor at all. Drop --l2-sp instead of excluding everything, so "
                    f"the record says plainly that there is no anchor.")
            _tset_sp = _tset_sp - _drop
            if is_main:
                print(f"  --l2-sp-exclude {_pats_sp}: {len(_drop)} tensor(s) stay TRAINABLE but "
                      f"are OUT of the anchor, {len(_tset_sp)} anchored:")
                for n in sorted(_drop):
                    print(f"    excluded  {n[len(_pfx_sp) + 1:]}")
        _l2sp_params = [(n, pp) for n, pp in model.named_parameters() if n in _tset_sp]
        if a.resume:
            if not os.path.exists(_anchor_p):
                raise SystemExit(
                    f"STOP: --l2-sp resumed but {_anchor_p} is missing. The anchor is the SHIPPED "
                    f"head; this run's weights are not it, so there is nothing to penalise "
                    f"distance from. Copy the anchor from the original run's directory.")
            _a_sd = torch.load(_anchor_p, map_location="cpu", weights_only=True)
            if is_main:
                print(f"  --l2-sp {a.l2_sp:g}: anchor READ from {_anchor_p} "
                      f"({len(_a_sd)} tensor(s)) -- a resumed run's weights are not the shipped ones")
        else:
            _a_sd = {n: pp.detach().float().cpu().clone() for n, pp in _l2sp_params}
            if is_main:
                torch.save(_a_sd, _anchor_p)
                print(f"  --l2-sp {a.l2_sp:g}: anchor SNAPSHOT at init ({len(_a_sd)} tensor(s)) "
                      f"-> {_anchor_p}. Penalty is LAMBDA * ||W - W_shipped||^2; at the drift the "
                      f"full head showed over 30 epochs (67,052) that prices to "
                      f"{a.l2_sp * 67051.81:.3f} against a PAE loss near 2.0.")
        _missing_sp = [n for n, _pp in _l2sp_params if n not in _a_sd]
        if _missing_sp:
            raise SystemExit(f"STOP: --l2-sp anchor is missing {len(_missing_sp)} trainable "
                             f"tensor(s), e.g. {_missing_sp[:3]}")
        _l2sp_anchor = {n: _a_sd[n].to(pp.device, pp.dtype) for n, pp in _l2sp_params}

    # PERSIST THE TRAJECTORY PER STEP, not at the end. A run that dies at step 900 of 1000 must still
    # be analysable, and a loss curve is the only way to see a learning rate that is wrong.
    # CROP BOOKKEEPING, counted rather than assumed. An arm whose crop silently failed to apply on
    # many structures is not the arm it claims to be, and the retained fraction is the only record
    # that says which happened.
    crop_stats = {"n": 0, "tok": 0, "kept": 0, "frames": 0, "undefined": 0}
    crop_rows = []
    crop_cols = ["pdb", "unit", "n_tok", "kept_tok", "frames_cleared", "status"]
    # PER RANK, because each rank crops its own third of the units and only rank 0 was flushing:
    # the other two accumulated rows forever and wrote none, so the csv described a third of the
    # run while reading like the whole of it. Review round 6 caught that.
    crop_p = ""
    if a.crop_margin > 0:
        _base = a.crop_log or os.path.join(a.out, "crop_sizes.csv")
        _stem, _ext = os.path.splitext(_base)
        crop_p = f"{_stem}_rank{rank}{_ext or '.csv'}"

    traj_p = os.path.join(a.out, "trajectory.csv")
    # THE BATCH COLUMNS ARE THE CURVE; the bare TERMS are the single pose that happened to be last.
    # batch_<term> is the mean over ALL accum_eff poses in the step, reduced across every rank, and
    # batch_n is how many actually contributed -- accum_eff on a normal step, fewer on the partial
    # step an epoch ends with. Added after review 2026-10-16: reporting read the stepping pose's own
    # loss as the batch loss, which is 1 pose of 255, and no amount of reader-side work can
    # reconstruct the other ranks' poses because they were never written.
    traj_cols = ["wall_s", "step", "unit_i", "pdb", "unit", "pose", "lr"] + list(TERMS) + \
                ["batch_" + k for k in TERMS] + ["batch_n", "partial_step", "step_ok"] + \
                ["grad_norm", "outside_head", "epoch", "step_taken", "nonfinite_grad"]
    # the per-structure item build is the cost --preload-items exists to amortise, so it is measured
    # rather than assumed; reported at the end against the number of structures actually visited.
    # WHERE AN EPOCH'S WALL CLOCK ACTUALLY GOES, accumulated per rank and printed at every epoch
    # boundary. Always on: a perf_counter pair is ~100 ns against a 0.58 s pose, and the last time
    # this was guessed at rather than measured, 0.67 s of a 3.10 s step attributed to the collective
    # turned out to be straggler wait (b829879d). The point is to order the optimisations by
    # measurement instead of by which is easiest to write.
    #
    # READ EPOCH 2, NOT EPOCH 1. Two costs are first-touch only: resolve_trunk memoises
    # unit_fingerprint in a mutable default argument, so the full-trunk hash runs once per unit per
    # PROCESS, and the dataset item cache fills on first use. Epoch 1 pays both for all 4880 units
    # and is not representative of the 499 that follow.
    _T = collections.defaultdict(float)
    _TN = collections.defaultdict(int)

    def _tick(key, t0):
        _T[key] += time.perf_counter() - t0
        _TN[key] += 1

    # GPU TIME NEEDS CUDA EVENTS, NOT WALL CLOCK. A kernel launch returns before the kernel runs,
    # so perf_counter around .backward() times the launch and reports a few milliseconds for work
    # that takes hundreds. `pose_total` happens to be honest because float(met["loss"]) inside it
    # forces a device sync, but the forward/backward split is not, and that split is what decides
    # whether disabling activation checkpointing could pay.
    #
    # DRAINED ONCE PER STEP, not per pose. elapsed_time() requires the events to have completed, so
    # querying per pose would impose ~84 syncs per step and distort exactly what it measures. The
    # pairs are held and read at the step boundary, where the gradient reduction synchronises
    # anyway.
    _ev_pending = []

    def _gpu_span(key):
        if not torch.cuda.is_available():
            return None, None
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        _ev_pending.append((key, e0, e1))
        return e0, e1

    def _drain_gpu_spans():
        if not _ev_pending:
            return
        torch.cuda.synchronize()
        for key, e0, e1 in _ev_pending:
            _T[key] += e0.elapsed_time(e1) / 1000.0
            _TN[key] += 1
        _ev_pending.clear()
    nonfinite = {"n": 0}
    code_drift = {"n": 0, "warned": False}

    # ONE FINGERPRINT CACHE, SHARED BY BOTH CALLERS. unit_fingerprint SHAs the trunk (~205 MB),
    # the job cif and the bioassembly pickle. resolve_trunk memoised it privately in a default
    # argument, but run_unit re-asserts it per unit and did NOT, so a full SHA of all three ran on
    # every unit of every epoch: at 1620 units per rank that is roughly half of this rank's trunk
    # I/O spent re-proving something proved in epoch 1.
    #
    # THE SAFETY PROPERTY IS UNCHANGED. The re-assertion exists so the check cannot be skipped if
    # resolve_trunk is ever bypassed -- and a bypass leaves this cache without the key, so it still
    # computes and still asserts. What is skipped is only the SECOND and later computations of a
    # value that is a pure function of files this process has already read.
    #
    # RAM: one short string per (pdb, trunk path), 4880 keys on the full set -- about 1 MB.
    _fp_cache = {}
    # THE RESOLVED TRUNK PATH, per (pdb, label). resolve_trunk is a pure function of the label's
    # stored fingerprint and the candidate trunks' own, and the per-unit re-assertion in run_unit
    # still checks the chosen trunk every epoch -- so caching the CHOICE skips a label npz open
    # and a dict comparison, not a safety check.
    #
    # MEASURED: the one-unit-ahead prefetch calls resolve_trunk on the next unit, and epoch 1 of
    # x3k2m4 spent 8.1 min of 77.9 inside prefetch_issue doing exactly that -- my own overhead,
    # larger than the trunk read it was meant to hide. With the choice cached it is a dict lookup
    # plus one fadvise syscall.
    _trunk_choice = {}

    # ---- THE FINGERPRINT CACHE, PERSISTED BESIDE THE FOLD -------------------------------------
    #
    # WHY. The in-memory cache above saves ~8 min per epoch, but only after the FIRST epoch of each
    # PROCESS: it is not in the resume bundle, so every restart re-pays the hashing of 4,880
    # trunks, cifs and bioassembly pickles. Review caught this, and it matters more than it looks
    # -- this project restarted four times in one day, and a 500-epoch run will restart again.
    #
    # WHAT IT TRUSTS, AND WHAT THAT GIVES UP. unit_fingerprint is a pure function of SCHEMA,
    # code_fp, loss_fp, the job cif, the bioassembly pickle, the indices row and the trunk. A fresh
    # process currently re-reads all of those, which re-verifies them. Persisting replaces that
    # with a stat signature -- size and mtime_ns of every file the fingerprint covers -- plus an
    # exact match on code_fp and loss_fp. That is the same bargain phase_k_eval's --join-cache
    # already makes ("the cache key carries each trunk's size and mtime, so a refold or rebuild
    # invalidates it"), and it is a real reduction in checking: a file rewritten with its size and
    # mtime preserved would go unnoticed. Everything that rewrites these files -- a refold, a
    # rebuild, the 2026-10-14 geometry fix -- changes mtime.
    #
    # PER RANK, AND ALL RANKS' FILES ARE READ. Each rank sees a third of the units and the shuffle
    # moves that third every epoch, so one rank's file covers only part of the manifest. Writing
    # one file per rank and loading every one of them accumulates coverage across epochs and across
    # runs without any rank having to write what another measured.
    _FP_CACHE_VER = 1
    _fp_disk = os.path.join(roots[0], f"unit_fingerprints.r{rank}.json")

    def _fp_sig(pdb, trunk_p):
        out = []
        for path in (trunk_p,
                     os.path.join(a.mmcif_dir, f"{pdb}.cif"),
                     os.path.join(a.bio_dir, f"{pdb}.pkl.gz"),
                     a.indices):
            try:
                st = os.stat(path)
                out.append(f"{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                out.append("absent")
        return "|".join(out)

    def _fp_cache_load():
        kept = dropped = 0
        for fp_path in sorted(glob.glob(os.path.join(roots[0], "unit_fingerprints.r*.json"))):
            try:
                blob = json.load(open(fp_path))
            except (OSError, ValueError):
                continue
            # A STALE SCHEMA OR A DIFFERENT CODE/LOSS FINGERPRINT INVALIDATES THE WHOLE FILE, not
            # entry by entry: those three are in every fingerprint the file holds.
            if (blob.get("ver") != _FP_CACHE_VER or blob.get("code_fp") != code_fp
                    or blob.get("loss_fp") != loss_fp):
                dropped += len(blob.get("entries", {}))
                continue
            for key, ent in blob.get("entries", {}).items():
                pdb, _, tp = key.partition("\t")
                if not tp or (pdb, tp) in _fp_cache:
                    continue
                if ent.get("sig") == _fp_sig(pdb, tp):
                    _fp_cache[(pdb, tp)] = ent["fp"]
                    kept += 1
                else:
                    dropped += 1
        if kept or dropped:
            print(f"fingerprint cache: {kept} entry(ies) reused from {roots[0]}, {dropped} "
                  f"invalidated by a changed input, code_fp or schema", flush=True)
        return kept

    def _fp_cache_save():
        """Atomic, and only ever ADDS to what this rank already had on disk."""
        try:
            prev = json.load(open(_fp_disk))
            entries = (prev.get("entries", {})
                       if (prev.get("ver") == _FP_CACHE_VER and prev.get("code_fp") == code_fp
                           and prev.get("loss_fp") == loss_fp) else {})
        except (OSError, ValueError):
            entries = {}
        for (pdb, tp), fp in _fp_cache.items():
            entries[f"{pdb}\t{tp}"] = {"fp": fp, "sig": _fp_sig(pdb, tp)}
        tmp = _fp_disk + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump({"ver": _FP_CACHE_VER, "code_fp": code_fp, "loss_fp": loss_fp,
                           "entries": entries}, fh)
            os.replace(tmp, _fp_disk)
        except OSError as e:
            # A CACHE IS NOT WORTH A RUN. Reported and dropped.
            print(f"  WARNING: could not write {_fp_disk} ({e}); the next process re-hashes",
                  flush=True)

    # ---- UNIT PREFETCH: the CPU half of the crop, run ahead on a worker thread ---------------
    #
    # WHY. Measured on epoch 5: crop_slice is 23.0 min of a 73.1 min epoch and the GPU is idle for
    # all of it -- each rank prepares a unit on ONE core (90% of one core on a 288-core box),
    # then runs that unit's 5 poses on the GPU, then prepares the next. GPU busy time is ~20 min
    # of 73. The two phases take turns when they could overlap.
    #
    # NOT PARALLELISM -- OVERLAP. Threading the slice itself does not help: measured 0.088 s at
    # OMP=1, 0.084 at 4, 0.144 at 16. It is memory-bound numpy indexing, not BLAS. What helps is
    # doing it while the GPU is busy with the previous unit.
    #
    # THE WORKER NEVER TOUCHES CUDA. It calls slice_cached_unit with dev=cpu and hands back host
    # tensors; the main thread moves them. Only one thread ever issues CUDA work, which is the
    # whole reason this is safe to do with a thread rather than a process.
    #
    # BOUNDED AT --prefetch-units, default 1. Each cached slice is ~30 MB at our crop sizes, so a
    # depth of 1 costs ~30 MB per rank and a depth of 4 ~120 MB -- nothing against 250 GB, but the
    # queue is bounded anyway so a fast worker cannot run away with memory while the GPU lags.
    _pf_q = None
    _pf_thread = None
    _pf_stop = threading.Event()

    def _cpu_crop(pdb, lab_p, trunk_p):
        """The expensive CPU half, with no device in sight. Returns None if no crop is definable.

        TIMED IN FOUR STAGES, because this was a black box and every number anyone quoted about it
        had to be DERIVED from the epoch total. That is not good enough to tell the two
        explanations of a shrinking prefetch wait apart: the trunk read genuinely getting faster
        (a real win) and the same work merely moving from the main thread into this one (no win).
        Stage `cpucrop_trunk` falling while `cpucrop_tokens` and `cpucrop_slice` hold constant is
        the signature of the first. It also settles a gap nobody could close: a warm-cache bench
        on 10 units put the whole crop at 0.291 s while the live worker appeared to use ~0.485 s.

        The counters are shared with the main thread, which runs this inline on a prefetch miss.
        `+=` on a defaultdict is not atomic, so a simultaneous miss can lose one sample out of
        thousands. That is acceptable for a diagnostic and is NOT acceptable for anything the run
        branches on; nothing branches on these.
        """
        _t = time.perf_counter()
        lab_l = np.load(lab_p, allow_pickle=True)
        _tick("cpucrop_label", _t)
        _t = time.perf_counter()
        zc = np.load(trunk_p)
        _tick("cpucrop_trunk", _t)
        _t = time.perf_counter()
        kt, ka = BC.interface_crop_tokens(zc, lab_l, a.crop_contact, a.crop_margin,
                                          all_poses=a.crop_all_poses)
        _tick("cpucrop_tokens", _t)
        if kt is None:
            return ("nocrop", None, None)
        _t = time.perf_counter()
        sl = BC.slice_cached_unit(zc, lab_l, kt, ka, torch.device("cpu"), torch)
        _tick("cpucrop_slice", _t)
        return ("ok", kt, sl)

    def _pf_worker(units):
        # ROUND-ROBIN SLICES, so worker w takes units[w::W] and the queue is fed in roughly the
        # order the main loop consumes. Roughly, not exactly: with W workers the arrival order
        # can run ahead by up to W units, which is what _pf_pending absorbs.
        for _k, t in enumerate(units):
            if _pf_stop.is_set():
                break
            pdb_, lab_p_, cands_ = t
            try:
                trunk_p_ = resolve_trunk(pdb_, lab_p_, cands_)
                # THE READAHEAD BELONGS TO THE THREAD THAT DOES THE READING. _prefetch() is also
                # issued by the MAIN loop for unit i+1, but with --prefetch-units 2 this worker is
                # already 2-3 units past the main loop, so that hint lands on a trunk this thread
                # has ALREADY read -- it is issued behind the only consumer it exists to help, and
                # its own timer confirms it is free and useless there (prefetch_issue 0.0m/1460).
                # Measured on j583i8: the worker sits at 49% of one core with the device at
                # 553 MB/s and ~250 MB of z_trunk per unit, i.e. ~0.505 s/unit blocked in read()
                # against 0.715 s/unit of main-thread work -- so the WORKER sets the epoch length
                # and 6.7 of 24.1 min is the main thread waiting on it.
                if _k + 1 < len(units):
                    _n = units[_k + 1]
                    try:
                        _prefetch(resolve_trunk(_n[0], _n[1], _n[2]))
                    except BaseException:         # noqa: BLE001 -- a bad NEXT unit must not kill
                        pass                      # THIS one; it reports itself when reached.
                res = _cpu_crop(pdb_, lab_p_, trunk_p_)
            except BaseException as e:            # noqa: BLE001 -- see below
                # THE WORKER NEVER KILLS THE RUN. Anything it fails on is handed back as a miss
                # and the main thread redoes that unit inline, where the real error surfaces with
                # its own diagnostics. A prefetcher that can crash training is worse than none.
                res = ("error", type(e).__name__, str(e)[:120])
            while not _pf_stop.is_set():
                try:
                    _pf_q.put((lab_p_, res), timeout=1.0)
                    break
                except queue.Full:
                    continue

    _pf_pending = {}
    # HIT/MISS IS REPORTED, because "crop_slice did not move" is how the first version's total
    # failure went unnoticed -- the inline fallback ticks the same counter as the fast path.
    _pf_stat = {"hit": 0, "miss": 0, "wait_s": 0.0}

    def _pf_take(lab_p, budget_s=120.0):
        """The prefetched result for this unit, WAITING for the worker if it is not ready yet.

        IT MUST BLOCK, AND get_nowait() IS WHY THE FIRST VERSION DID NOTHING. Measured on the
        smoke test: crop_slice 0.70 s/unit with the prefetch on against 0.71 s/unit with it off.
        A non-blocking take gives up the instant the queue is empty, which at a cold start is a
        PERMANENT MISS CASCADE:

          unit 0  queue empty (worker just started) -> None -> main thread crops unit 0 inline,
                  0.70 s, while the worker crops the same unit 0
          unit 1  queue holds unit 0's result; k != lab_p so it is stashed and the loop goes round;
                  queue now empty -> None -> main thread crops unit 1 inline
          ...     the worker stays exactly one unit behind forever, every take misses, and a core
                  is burned recomputing what the main thread just computed

        The main thread never waited even once, so the worker never got the head start it needed
        to pull ahead. Blocking costs nothing against the alternative -- the worker is already
        partway through this very unit, and doing it inline means doing the whole thing -- and
        after one wait the main thread is on the GPU for ~0.84 s/unit while the worker fills the
        queue to --prefetch-units and every later take is a hit.

        THE BUDGET IS A DEADLOCK GUARD, NOT A TUNABLE. The worker emits exactly one result per
        unit, in the main loop's own order, and hands back ("error", ...) rather than dying, so
        the awaited result always comes. 120 s against a 0.7 s crop means the worker is gone;
        then this returns None and the unit is cropped inline exactly as before.

        IN ORDER, WITH A SMALL HOLDING BUFFER. Results that arrive early are held by lab_p rather
        than discarded.
        """
        if lab_p in _pf_pending:
            _pf_stat["hit"] += 1
            return _pf_pending.pop(lab_p)
        if _pf_q is None:
            return None
        _t_w = time.perf_counter()
        deadline = _t_w + budget_s
        while True:
            try:
                k, v = _pf_q.get(timeout=max(0.05, deadline - time.perf_counter()))
            except (queue.Empty, AttributeError):
                _pf_stat["miss"] += 1
                _pf_stat["wait_s"] += time.perf_counter() - _t_w
                return None
            if k == lab_p:
                _pf_stat["hit"] += 1
                _pf_stat["wait_s"] += time.perf_counter() - _t_w
                return v
            _pf_pending[k] = v
            # SIZED FOR THE WORKERS, not a flat 8. W workers racing can deliver up to ~W units
            # ahead of the one wanted, and evicting a result that has not been consumed yet
            # turns into a miss and an inline recrop -- the very work the prefetch exists to
            # avoid. Each held slice is ~30 MB, so this is tens of MB per rank.
            if len(_pf_pending) > max(16, 8 * max(1, a.prefetch_workers)):
                _pf_pending.pop(next(iter(_pf_pending)))

    def _pf_start(units):
        nonlocal _pf_q, _pf_thread
        if not a.prefetch_units:
            return
        _pf_stop.clear(); _pf_pending.clear()
        _pf_stat.update(hit=0, miss=0, wait_s=0.0)
        _w = max(1, a.prefetch_workers)
        # DEEP ENOUGH FOR EVERY WORKER TO BE IN FLIGHT. At maxsize 1 with W workers, W-1 of them
        # block on put() and the extra threads buy nothing.
        _pf_q = queue.Queue(maxsize=max(1, a.prefetch_units) * _w)
        _pf_thread = [threading.Thread(target=_pf_worker, args=(units[i::_w],), daemon=True)
                      for i in range(_w)]
        for _t in _pf_thread:
            _t.start()

    def _pf_end():
        nonlocal _pf_q, _pf_thread
        if _pf_thread is None:
            return
        _pf_stop.set()
        for _t in _pf_thread:
            # DRAIN BETWEEN JOINS, not once before them. A worker blocked on a full put() never
            # sees the stop flag, so draining once and then joining W threads deadlocks on the
            # second one as soon as the first refills the queue.
            while True:
                try:
                    _pf_q.get_nowait()
                except queue.Empty:
                    break
            _t.join(timeout=30)
        _pf_thread = None; _pf_q = None; _pf_pending.clear()

    def _to_dev(x):
        """Move a slice_cached_unit payload onto the device, structure preserved."""
        if torch.is_tensor(x):
            return x.to(dev, non_blocking=True)
        if isinstance(x, dict):
            return {k: _to_dev(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_to_dev(v) for v in x)
        return x

    def _prefetch(path):
        """Ask the kernel to read a trunk ahead, so its I/O overlaps the current unit's compute.

        posix_fadvise(WILLNEED) rather than a thread and a queue: it is one syscall, it cannot
        race anything, and it adds no Python-side buffer. The bytes land in PAGE CACHE, which is
        reclaimable -- so this does not raise this process's RSS against --max-rss-gb and does not
        lower MemAvailable in the way that matters to other users of the box. Under memory pressure
        the kernel simply drops the readahead and the subsequent read is a normal cold read.

        BOUNDED AT ONE UNIT AHEAD, deliberately. A deeper queue would have several ~205 MB trunks
        in flight per rank for a loader that consumes them strictly in order, which buys nothing
        and evicts cache that the three ranks are competing for.
        """
        if not hasattr(os, "posix_fadvise") or not path:
            return
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        except OSError:
            pass
        finally:
            os.close(fd)

    # LOADED HERE, after the helpers above are bound and before the first resolve_trunk call.
    # Calling it beside DS.build_configs raised UnboundLocalError: the name is local to main() and
    # that line runs ~280 lines before the def.
    _fp_cache_load()

    def resolve_trunk(pdb, lab_p, cands, want_fp_cache=None):
        _ck = (pdb, lab_p)
        if _ck in _trunk_choice:
            return _trunk_choice[_ck]
        want_fp_cache = _fp_cache if want_fp_cache is None else want_fp_cache
        """Pick the trunk whose PROVENANCE matches this label, not the first filename that exists.

        THE FIRST MATCHING FILENAME IS NOT THE RIGHT TRUNK. The multi-root lookup took it, so a
        label built from one root's trunk could be paired with another root's trunk of the same name
        -- and since the label is a PER-POSE permutation of its own trunk's poses, that pairs a pose
        with another run's permuted label and trains on it. Review round 8 reproduced exactly that
        with a fold_regen label and a fold trunk.

        The stored fingerprint names its trunk by content hash, so the trunk is chosen BY it. Two
        candidates matching is ambiguous and refused rather than resolved arbitrarily.
        """
        stored = str(np.asarray(np.load(lab_p, allow_pickle=True)["fingerprint"]))
        ok = []
        for tp in cands:
            key = (pdb, tp)
            if key not in want_fp_cache:
                _t0 = time.perf_counter()
                want_fp_cache[key] = BC.unit_fingerprint(a, pdb, tp, code_fp, loss_fp)
                _tick("fingerprint_hash", _t0)
            if want_fp_cache[key] == stored:
                ok.append(tp)
        if len(ok) == 1:
            _trunk_choice[_ck] = ok[0]
            return ok[0]
        if not ok:
            # WHICH FIELD DIFFERS IS THE ACTIONABLE PART. "stale" with no detail cannot be acted on,
            # and the fields are not equally serious: cif, bio, idx and trunk say the label was
            # built from DIFFERENT DATA, while code says only that the builder's source has changed.
            def fields(fp):
                out = {}
                for part in fp.split("|"):
                    if "=" in part:
                        k, v = part.split("=", 1); out[k] = v
                return out
            want_f = fields(want_fp_cache[(pdb, cands[0])])
            got_f = fields(stored)
            diff = sorted(k for k in set(want_f) | set(got_f) if want_f.get(k) != got_f.get(k))
            if diff == ["code"] and a.allow_code_drift:
                if not code_drift["warned"]:
                    print(f"  ACCEPTING CODE DRIFT (--allow-code-drift): labels were built by "
                          f"code={got_f.get('code')}, this run is code={want_f.get('code')}. Every "
                          f"data field matches -- cif, bio, idx, trunk, loss, schema -- so the "
                          f"labels are correct but not reproducible from this checkout.")
                    code_drift["warned"] = True
                code_drift["n"] += 1
                return cands[0]
            raise SystemExit(
                f"STOP: {lab_p} does not match its inputs. No trunk among {cands} reproduces its "
                f"fingerprint.\n  field(s) differing: {diff}\n"
                + "".join(f"    {k}: label {got_f.get(k)} vs now {want_f.get(k)}\n" for k in diff)
                + (f"  Only `code` differs, so the DATA provenance is intact; pass "
                   f"--allow-code-drift to accept that explicitly.\n" if diff == ["code"] else "")
                + f"  Otherwise rebuild with phase_j_build_cache.py --only {pdb}.")
        raise SystemExit(f"STOP: {len(ok)} trunks match {lab_p}'s fingerprint: {ok}. Ambiguous "
                         f"provenance is refused rather than resolved arbitrarily.")

    def run_unit(pdb, lab_p, trunk_p, score_only=False):
        """Forward + loss for every pose of one unit. Yields (pose_index, loss_tensor, metrics).

        score_only=True makes this PASS A: a no_grad forward per pose that yields
        {"pose": i, "iptm": {pair: value}} and nothing else -- no loss, no metrics, no backward,
        no graph. Needed only by --rank-form softplus, whose coefficients contain
        sigmoid(-m/tau) and therefore depend on the current scores. See rank_loss's
        THE MEMORY CONSTRAINT.

        Both paths are resolved by the manifest and asserted to exist before the run starts, so
        there is nothing here to skip.
        """
        arm_seed = os.path.basename(lab_p)[:-4]
        # THE UNIT'S SEED AS AN INT, built through _rank_key so there is ONE place that parses a
        # seed out of an arm name. The pair argument is unused by the seed field and is only there
        # because _rank_key returns the whole key.
        _xs_sd = _rank_key(pdb, arm_seed, (0, 0))[1] if a.rank_cross_seed else None
        # THE FULL DATASET ITEM IS NEVER BUILT, AND NOW CANNOT BE. The loss reads nine feature
        # keys and two label keys and all eleven live in the cached label npz -- 0.11 MiB against
        # the item's ~518 MB, bit-identical on 50 of 50 comparisons. It was already dead code
        # under --minimal-items; the K-structures-per-step order makes it unrunnable besides,
        # because every rank now touches EVERY structure each epoch, so an item cache would need
        # all 488 resident per rank where 164 already breached the box's free-RAM floor.
        _t_lab = time.perf_counter()
        lab = np.load(lab_p, allow_pickle=True)
        _tick("label_open", _t_lab)
        # THE CACHE'S PROVENANCE IS REQUIRED, NOT OPTIONAL. build_cache stores a `fingerprint` of
        # the job cif, bioassembly, index rows, trunk and code the label was built from, plus a
        # `schema`. Training read neither, so a label written by older code or from inputs that have
        # since changed would be trained on with nothing noticing. The lf_* loss features are the
        # same story: `if kk in lab` silently fell back to the LIVE features, which are not the ones
        # the label was built against -- exactly the substitution the replay gate exists to detect.
        for req in ("fingerprint", "schema", "label_coordinate", "label_coordinate_mask",
                    "sample_index"):
            if req not in lab:
                raise SystemExit(f"STOP: {lab_p} has no `{req}`. It was written by an older "
                                 f"build_cache; rebuild it before training.")
        if int(np.asarray(lab["schema"])) != int(BC.SCHEMA):
            raise SystemExit(f"STOP: {lab_p} is schema {int(np.asarray(lab['schema']))}, this code "
                             f"expects {int(BC.SCHEMA)}. Rebuild the cache.")
        # THE FINGERPRINT IS COMPARED, NOT MERELY PRESENT. resolve_trunk already proved one trunk
        # reproduces it; this re-asserts against the trunk actually being used, so the two cannot
        # come apart if resolve_trunk is ever bypassed.
        _fp_key = (pdb, trunk_p)
        if _fp_key not in _fp_cache:
            _t_fp = time.perf_counter()
            _fp_cache[_fp_key] = BC.unit_fingerprint(a, pdb, trunk_p, code_fp, loss_fp)
            _tick("fingerprint_hash", _t_fp)
        _want = _fp_cache[_fp_key]
        _got = str(np.asarray(lab["fingerprint"]))
        if _got != _want:
            _dw = dict(x.split("=", 1) for x in _want.split("|") if "=" in x)
            _dg = dict(x.split("=", 1) for x in _got.split("|") if "=" in x)
            _df = sorted(k for k in set(_dw) | set(_dg) if _dw.get(k) != _dg.get(k))
            if not (_df == ["code"] and a.allow_code_drift):
                raise SystemExit(f"STOP: {lab_p} fingerprint differs in {_df}. Rebuild it.")
        _missing_lf = [k for k in BC.LOSS_FEATS if f"lf_{k}" not in lab]
        if _missing_lf:
            raise SystemExit(f"STOP: {lab_p} stores no lf_ array for {_missing_lf}. Falling back to "
                             f"the live features would score the label against features it was not "
                             f"built with. Rebuild the cache.")
        # BOTH LOADERS, as the replay does. load_trunk gives the NUMPY features that
        # check_alignment compares against the dataset's -- it calls np.asarray, so handing it the
        # device tensors raises "can't convert cuda:0 device type tensor to numpy". load_unit gives
        # the device tensors the head consumes, and reverses the bf16-as-uint16 storage.
        # check_alignment ran once per structure in the preflight --minimal-items requires.
        # THE CROP, WHEN ASKED FOR. The head's tensors and the loss's features come out of
        # slice_cached_unit together, from one pair of masks, so there is no path on which one side
        # is cropped and the other is not -- the misalignment the re-indexing gate exists to catch.
        # interface_crop_tokens returns (None, None) when a crop is not definable (fewer than two
        # chains with usable coordinates, or no cross-chain contact), and this trains on the whole
        # structure then, counted, rather than guessing at a window.
        crop_slice = None
        if a.crop_margin > 0:
            # THE PREFETCHED SLICE, IF THE WORKER GOT THERE FIRST. A hit skips straight to the
            # device move; a miss falls through to the inline path below, so a worker that is
            # behind, failed, or disabled costs correctness nothing.
            # PASS A DOES NOT TOUCH THE PREFETCH QUEUE, and this is a correctness fix, not a
            # tuning choice. _pf_take's contract is "the worker emits exactly one result per unit,
            # in the main loop's own order", and it POPS from _pf_pending. Pass A walks the same
            # units pass B is about to walk, so taking there consumed each unit's result and left
            # pass B waiting out the full 120 s budget per unit before falling back to an inline
            # crop -- 100 units an epoch, so the run looked hung with flat CPU and 0% GPU. It was
            # not deadlocked, it was sleeping ~3 hours an epoch. Observed on probe dsswrj.
            #
            # Pass A crops inline instead, which leaves the worker in lockstep with pass B. The
            # cost is one extra crop per unit (~0.7 s) rather than an extra trunk read, since the
            # trunk's bytes are in page cache from the worker's own read moments earlier.
            _pre = _pf_take(lab_p) if (a.prefetch_units and not score_only) else None
            if _pre is not None and _pre[0] == "ok":
                _t1 = time.perf_counter()
                crop_slice = _to_dev(_pre[2])
                _tick("crop_slice", _t1)
                _T["cropped_tokens"] += float(_pre[1].sum())
                _TN["cropped_tokens"] += 1
                _TN["prefetch_hit"] += 1
                crop_stats["n"] += 1
                crop_stats["tok"] += int(_pre[1].shape[0])
                crop_stats["kept"] += int(_pre[1].sum())
                crop_stats["frames"] += int(crop_slice[8])
                crop_rows.append({"pdb": pdb, "unit": arm_seed, "n_tok": int(_pre[1].shape[0]),
                                  "kept_tok": int(_pre[1].sum()),
                                  "frames_cleared": int(crop_slice[8]), "status": "cropped"})
                _kt = None          # the inline branches below must not run
                _skip_inline_crop = True
            else:
                _skip_inline_crop = False
                if _pre is not None and _pre[0] == "error":
                    # REPORTED, NOT SWALLOWED. The inline path will hit the same unit and raise
                    # properly if it is a real problem; this says the worker saw it first.
                    print(f"  prefetch miss on {pdb}/{arm_seed}: {_pre[1]} {_pre[2]}", flush=True)
                _TN["prefetch_miss"] += 1
            if not _skip_inline_crop:
                # SEPARATELY, because they argue for different fixes: a trunk read that dominates
                # argues for caching the cropped tensors on disk, while crop PREPARATION dominating
                # argues for caching the crop indices and nothing else.
                _t0 = time.perf_counter()
                _zc = np.load(trunk_p)
                _kt, _ka = BC.interface_crop_tokens(_zc, lab, a.crop_contact, a.crop_margin,
                                                    all_poses=a.crop_all_poses)
                _tick("trunk_read_plus_crop_tokens", _t0)
                if _kt is None:
                    crop_stats["undefined"] += 1
                    crop_rows.append({"pdb": pdb, "unit": arm_seed, "n_tok": "", "kept_tok": "",
                                      "frames_cleared": "", "status": "no-crop-definable"})
                else:
                    _t1 = time.perf_counter()
                    crop_slice = BC.slice_cached_unit(_zc, lab, _kt, _ka, dev, torch)
                    _tick("crop_slice", _t1)
                    # THE CROPPED TOKEN COUNT, which is the workload. Equal structure counts per
                    # rank do not mean equal compute, and this is what shows it: the pair track is
                    # quadratic in tokens, so a rank handed the large end of 481-2,289 does
                    # several times the work for the same unit count.
                    #
                    # INSIDE THE else, NOT BESIDE IT. A patch left this block one level out, where
                    # a unit with no definable crop reaches _kt.sum() on None.
                    _T["cropped_tokens"] += float(_kt.sum())
                    _TN["cropped_tokens"] += 1
                    crop_stats["n"] += 1
                    crop_stats["tok"] += int(_kt.shape[0])
                    crop_stats["kept"] += int(_kt.sum())
                    crop_stats["frames"] += int(crop_slice[8])
                    crop_rows.append({"pdb": pdb, "unit": arm_seed, "n_tok": int(_kt.shape[0]),
                                      "kept_tok": int(_kt.sum()),
                                      "frames_cleared": int(crop_slice[8]), "status": "cropped"})
        if crop_slice is not None:
            (feat_head, s_inputs, s_trunk, zt, poses, _nc,
             crop_feat_loss, crop_labels, _nb) = crop_slice
        else:
            crop_feat_loss = crop_labels = None
            feat_head, s_inputs, s_trunk, zt, poses, _nc = RC.load_unit(trunk_p, dev)

        _t_fc = time.perf_counter()

        def to_dev(d):
            return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
        if crop_feat_loss is not None:
            # THE CROPPED LOSS FEATURES, from the same masks as the head's tensors. These are the
            # per-atom and per-token arrays the loss indexes with -- plddt_m_rep_atom_mask,
            # pae_rep_atom_mask, frame_atom_index, has_frame and the rest -- so a full-size copy
            # here against a cropped prediction is the silent misalignment, not a shape error.
            feat_cache = crop_feat_loss
        else:
            # ONE SHARED BUILDER, so this and the equivalence gate cannot drift apart -- two
            # readers of one record diverging is how the reporting bug shipped twice.
            feat_cache, _ = BC.minimal_loss_inputs(lab, 0, dev, torch)

        n_pose = poses.shape[0] if hasattr(poses, "shape") else len(poses)
        # FROM THE HEAD'S FEATURES, which carry asym_id on both paths. feat_cache does not, on the
        # minimal path: the loss never reads it, so it is not in the nine.
        n_tok = int(feat_head["asym_id"].shape[-1]) if "asym_id" in feat_head else -1
        # ONCE PER UNIT, BEFORE ANY POSE RUNS. From feat_cache, not feat_head: the trunk cache
        # carries has_frame but neither frame_atom_index nor pae_rep_atom_mask, and all three are
        # in LOSS_FEATS. Under cropping feat_cache IS crop_feat_loss, so these are the cropped
        # copies whose atom indices match the cropped coordinates the head is given.
        if a.frame_direction:
            import pose_direction
            pose_direction.set_unit(feat_cache["frame_atom_index"], feat_cache["has_frame"],
                                    feat_cache["pae_rep_atom_mask"], pdb=pdb)
        if world > 1 and n_pose != a.nsample:
            raise SystemExit(f"STOP: {pdb}/{arm_seed} holds {n_pose} pose(s), not --nsample "
                             f"{a.nsample}. Across ranks that shifts the optimiser step onto a "
                             f"different pose and the next all-reduce deadlocks.")
        # ONE POSE, IN A FUNCTION THAT RETURNS NUMBERS.
        #
        # This was inline in the generator, which held each pose's logits, prediction dict and labels
        # in its frame ACROSS the yield -- so the caller ran backward and then began the next forward
        # while the previous pose's pair tensors were still referenced. Two fp32 pair-logit tensors at
        # 3,400 tokens x 64 bins are about 5.5 GiB, so the peaks overlapped for no reason. Review round
        # 4 reproduced the object-lifetime behaviour and was explicit that this is OVERLAPPING PEAKS,
        # not unbounded accumulation -- correcting my own claim of a leak, which rested on dividing
        # 90.4 GB by 720 poses and finding a number that happened to match a tensor size.
        #
        # A called function's frame is destroyed on return, so every large tensor here is freed before
        # the next pose allocates. backward() happens inside and only Python floats come out; the
        # optimiser step stays with the caller, which owns the accumulation count.
        #
        # EVERYTHING FROM to_dev's DEFINITION TO HERE is per-unit loss-feature preparation, which
        # sat entirely in the unaccounted residual. Review was right that BC.minimal_loss_inputs
        # was the wrong suspect for it: with every one of the 4,860 units cropping successfully,
        # crop_feat_loss is always set and both the per-unit and per-pose minimal_loss_inputs
        # branches are dead code for this workload.
        _tick("loss_feature_prep", _t_fc)

        def _one_pose(i):
            x = poses[i:i + 1].to(dev).float() if torch.is_tensor(poses) \
                else torch.as_tensor(poses[i:i + 1], dtype=torch.float32, device=dev)
            _fe0, _fe1 = _gpu_span("gpu_forward")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plddt, pae, pde, resolved = model.run_confidence_head(
                    input_feature_dict=feat_head, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
                    pair_mask=None, x_pred_coords=x,
                    triangle_multiplicative=configs.triangle_multiplicative,
                    triangle_attention=configs.triangle_attention, inplace_safe=False,
                    # CHUNKING WAS TRIED AND REMOVED: --chunk-size 64 cost 5.8x in speed for no
                    # measurable memory, and Protenix's own config says chunk_size "should set to
                    # null for normal training".
                    chunk_size=None)
            if _fe1 is not None:
                _fe1.record()
            # WHERE THE HEAD'S OUTPUTS LANDED. Compared BY INDEX, because torch.device("cuda") is not
            # equal to torch.device("cuda:0") and the first version of this check fired on every pose
            # of a single-GPU run -- it killed the memory probe before it printed a single number.
            _di = dev.index if dev.index is not None else torch.cuda.current_device()
            _off = [n for n, t in (("plddt", plddt), ("pae", pae), ("pde", pde), ("resolved", resolved))
                    if t.device.type != "cuda" or (t.device.index or 0) != _di]
            if _off:
                raise SystemExit(
                    f"STOP: {pdb}/{arm_seed} pose {i}: the confidence head returned {_off} off "
                    f"cuda:{_di}. Protenix offloads pae/pde to the CPU above "
                    f"PROTENIX_CONF_OFFLOAD_TOKENS (default 2000) when the module is NOT in training "
                    f"mode; confidence_head.training is {model.confidence_head.training}.")
            pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                    "resolved": resolved.float(), "coordinate_mini": x}
            # EVERY NUMERIC ARRAY GOES TO THE DEVICE, not only those already tensors. A numpy array
            # left as numpy is converted by Protenix with torch.as_tensor, giving a CPU tensor, and
            # PDELoss then builds its bin boundaries on the CPU too, so the whole loss computes there.
            if crop_feat_loss is not None:
                # SLICED ONCE PER UNIT: the label coordinates for every pose were cropped together,
                # so per pose this only picks a row.
                _cc, _cm = crop_labels
                ld = {"coordinate": torch.as_tensor(_cc[i], dtype=torch.float32, device=dev),
                      "coordinate_mask": torch.as_tensor(_cm[i].astype(np.float32), device=dev)}
            else:
                _, ld = BC.minimal_loss_inputs(lab, i, dev, torch)
            _set_pae_mask(pdb, feat_head["asym_id"], feat_cache["has_frame"])
            # PASS A STOPS HERE. The scoring sweep needs the pose's ipTM and nothing else: no loss,
            # no metrics, no backward. Returning before calculate_losses is not an optimisation, it
            # is the point -- pass A must not touch the loss state that pass B reports, and it runs
            # inside torch.no_grad() so no graph is built for the forward it just did.
            #
            # THE SAME ipTM FUNCTION AS PASS B, deliberately. If the two disagreed, the coefficient
            # would have been computed for a different function than the one being differentiated.
            if score_only:
                import rank_loss as RKL
                _sc = {}
                for _p in sorted(_scored_asym.get(pdb) or ()):
                    _v = RKL.iptm(pred["pae"], feat_head, configs, _p)
                    if _v is not None:
                        _sc[_p] = float(_v.detach())
                del pred, x, plddt, pae, pde, resolved
                return {"pose": i, "iptm": _sc}
            try:
                cum, met = loss_fn.calculate_losses(feat_cache, pred, ld, mode="train")
            except RuntimeError as _e:
                if "same device" not in str(_e):
                    raise
                def _where(d, label):
                    out = []
                    for _k2, _v2 in sorted(d.items()):
                        if torch.is_tensor(_v2):
                            out.append(f"{_k2}:{_v2.device}")
                        elif isinstance(_v2, np.ndarray):
                            out.append(f"{_k2}:numpy[{_v2.dtype}]")
                    return f"  {label}: " + " ".join(out)
                print(f"DEVICE MISMATCH on {pdb}/{arm_seed} pose {i}, dev={dev}: {_e}", flush=True)
                print(_where(ld, "label_dict"), flush=True)
                print(_where(feat_cache, "feat"), flush=True)
                print(_where(pred, "pred"), flush=True)
                raise
            absent = [k for k in TERMS if k not in met and k not in _dropped_terms]
            if absent:
                raise SystemExit(f"STOP: the loss produced no {absent} for {pdb}/{arm_seed} pose {i}; "
                                 f"a term that stopped being produced reads as a lower loss, which is "
                                 f"the failure this check exists for.")
            if not math.isfinite(float(met["loss"])):
                raise SystemExit(f"STOP: {pdb}/{arm_seed} pose {i} loss is {float(met['loss'])}, not "
                                 f"finite. Training on it would destroy the head.")
            out = {k: (float(met[k]) if k in met else 0.0) for k in TERMS}
            out["loss"] = float(met["loss"])
            out["pose"] = i
            # SCALED BY THE EFFECTIVE BATCH, so summing the ranks reconstructs the full-batch mean.
            # NOT UNDER --loss-only: there are no gradients to accumulate, and calling backward
            # would be the one way this mode could change the weights it is supposed to measure.
            # TWO SOURCES OF c_i, AND ONLY ONE IS ACTIVE. Per-unit (_rank_c, from the labels
            # alone) for the within-seed linear form; per-step (_xs_coef, from pass A) for
            # --rank-cross-seed, where a pose's coefficient depends on every other pose of its
            # interface and the divisor is interface-groups per step rather than units.
            if a.rank_cross_seed:
                _cs_pairs = sorted({_k[3] for _k in _xs_coef
                                    if _k[0] == pdb and _k[1] == _xs_sd and _k[2] == i})
                _cs_src = [(_p, _xs_coef.get((pdb, _xs_sd, i, _p))) for _p in _cs_pairs]
                _cs_div = float(_xs_div[0])
            elif _rank_on:
                _cs_src = [(_p, _rank_c.get((i, _p)))
                           for _p in sorted({_k[1] for _k in _rank_c if _k[0] == i})]
                _cs_div = float(_rank_div)
            else:
                _cs_src = []
                _cs_div = 1.0
            # BOUND BEFORE THE BACKWARD BLOCK, not inside it. The first version computed this
            # below the --grad-probe block that reads it, so the probe raised NameError on its
            # first pose -- the same defect review found at line 1067 with `want`. One binding,
            # above both readers.
            if not a.loss_only and not score_only:
                _be0, _be1 = _gpu_span("gpu_backward")
                # UNDER --grad-probe: the two terms are measured with torch.autograd.grad,
                # which returns gradients WITHOUT touching any parameter's .grad. The first
                # version used .backward() with opt.zero_grad() around it, which wiped every
                # earlier pose's contribution to the open accumulation window -- so the claim that
                # the probe took the same step as a real run was false, and the step it did take
                # was built from one pose. Found in review of a3086ef.
                #
                # ACCUMULATED, NOT AVERAGED. The optimiser sees sum_i g_i, and ||sum_i g_i|| is
                # not sum_i ||g_i||: the ranking coefficients sum to zero inside a group, so
                # per-pose contributions can cancel heavily and a mean of per-pose norms would
                # overstate the ranking gradient at the step. Both are reported, and alpha is read
                # off the accumulated pair.
                # THE PROBE READS THE SAME COEFFICIENT SOURCE THE OBJECTIVE DOES. _rank_on is set
                # only on the per-unit linear path, so gating on it alone made the probe silently
                # measure nothing under --rank-cross-seed -- it would have reported a ranking
                # gradient of zero and the parity alpha as infinite.
                if a.grad_probe and _cs_src:
                    _rt = None
                    for _p, _cv in _cs_src:
                        if not _cv:
                            continue
                        _sp_ = RKL_probe.iptm(pred["pae"], feat_head, configs, _p)
                        if _sp_ is None:
                            continue
                        _tm = (a.alpha_rank * _cv / _cs_div) * _sp_
                        _rt = _tm if _rt is None else _rt + _tm
                    # THE COEFFICIENT THIS POSE WAS PUSHED WITH, recorded alongside the
                    # gradient it produced, so the two factors of
                    #     g_i = c_i * grad_theta s_i
                    # are measured ON THE SAME POSE IN THE SAME WINDOW. Without this the only
                    # coefficient figures available were val10 sum|c_i| against a training-set
                    # gradient probe, which cannot be divided -- raised in review, and correct.
                    # Summed over the pose's interfaces exactly as the term above is built, with
                    # the same alpha_rank and _cs_div scaling, so |g_rank| / this quantity is the
                    # realised sensitivity |grad_theta s_i|.
                    _cabs = sum(abs(a.alpha_rank * _cv / _cs_div)
                                for _p, _cv in _cs_src if _cv)
                    _pr_tensors = [pp for _n, pp in trainable_params]
                    _pr_norms = {}
                    for _nm, _t in (("pae", cum / _step_poses_g[0]), ("rank", _rt)):
                        if _t is None:
                            continue
                        _gs = torch.autograd.grad(_t, _pr_tensors, retain_graph=True,
                                                  allow_unused=True)
                        _flat = []
                        for (_n, _pp), _g in zip(trainable_params, _gs):
                            _gv = torch.zeros_like(_pp) if _g is None else _g.detach().float()
                            _probe["sum"].setdefault(_nm, {})
                            _prev = _probe["sum"][_nm].get(_n)
                            _probe["sum"][_nm][_n] = _gv.clone() if _prev is None else _prev + _gv
                            _probe["finite"][_nm] = (_probe["finite"].get(_nm, True)
                                                     and bool(torch.isfinite(_gv).all()))
                            _probe["nonzero"].setdefault(_nm, set())
                            if float(_gv.abs().sum()) > 0:
                                _probe["nonzero"][_nm].add(_n)
                            _flat.append(_gv.flatten())
                        _pr_norms[_nm] = torch.cat(_flat)
                    if "pae" in _pr_norms and "rank" in _pr_norms:
                        _vp, _vr = _pr_norms["pae"], _pr_norms["rank"]
                        _c = (float((_vp @ _vr) / (_vp.norm() * _vr.norm()))
                              if _vp.norm() > 0 and _vr.norm() > 0 else float("nan"))
                        _probe["rows"].append((pdb, arm_seed, i, float(_vp.norm()),
                                               float(_vr.norm()), _c, float(_cabs)))
                    _probe["poses"] += 1
                _obj = cum / _step_poses_g[0]
                # THE RANKING TERM, as c_i * s_i. Its gradient is the exact gradient of the unit's
                # pairwise loss contribution, because dL/dtheta = sum_i (dL/ds_i)(ds_i/dtheta) and
                # c_i IS dL/ds_i. Divided by units-per-step so the term is a mean over units, the
                # way cum/accum_eff is a mean over poses.
                if _cs_src:
                    import rank_loss as RKL
                    for _p, _cv in _cs_src:
                        if not _cv:
                            continue
                        _sr = RKL.iptm(pred["pae"], feat_head, configs, _p)
                        if _sr is None:
                            continue
                        _obj = _obj + (a.alpha_rank * _cv / _cs_div) * _sr
                        # RECONSTRUCTED FOR THE LINEAR FORM ONLY. L = sum_i c_i s_i holds exactly
                        # there, so accumulating c_i * s_i gives the loss with no second
                        # evaluation. The SOFTPLUS loss is NOT sum_i c_i s_i -- its value comes
                        # from group_coeffs in pass A and is reported out of _xs_stat instead.
                        if not a.rank_cross_seed:
                            _rank_seen["loss"] += float(_cv) * float(_sr.detach())
                _obj.backward()
                if _be1 is not None:
                    _be1.record()
            # Explicit, though the return would free them anyway: this is the function's whole purpose,
            # and naming them stops a later edit from quietly hoisting one into the enclosing scope.
            del cum, met, pred, ld, x, plddt, pae, pde, resolved
            return out

        # THE RANKING COEFFICIENTS, FROM THE LABELS ALONE. NO EXTRA FORWARD PASS.
        #
        # The loss is LINEAR in the scores, L = sum_i c_i * s_i, so dL/ds_i does not contain any
        # score at all -- only DockQ:
        #
        #     c_i = dL/ds_i = - sum_{j valid} (d_i - d_j) / sum_{(i,j) valid} |d_i - d_j|
        #
        # Verified: the coefficients are byte-identical for score vectors as different as
        # [0.70,0.69,0.69,0.69,0.66], [0.1,0.9,0.5,0.2,0.7] and all-zeros.
        #
        # THIS REPLACES A TWO-PASS SCHEME, and the correction is worth recording. The first version
        # of this loss used softplus, whose derivative sigmoid(-m) DOES depend on every score, so a
        # no-grad forward was needed to collect them before the real one -- a third more GPU time
        # per pose, plus save/restore of the dropout RNG so the two passes evaluated the same
        # function, plus a per-pose scalar equality check to catch it when they did not. Switching
        # the loss to linear made every bit of that dead, and it was not removed at the time:
        # the surrounding machinery was not re-derived when the function it served changed.
        # Review of cd8e6ab7 caught it.
        #
        # Each pose's existing forward already yields its own s_i with a graph, so the ranking term
        # rides along on the backward that was happening anyway. One graph at a time, exact
        # gradient, zero extra forwards.
        _rank_c, _rank_on = {}, False
        if a.rank_loss and not a.loss_only and not a.rank_cross_seed:
            import rank_loss as RKL
            _rk_pairs = sorted(_scored_asym.get(pdb) or ())
            _rk_lab = [(_p, _rank_lab.get(_rank_key(pdb, arm_seed, _p))) for _p in _rk_pairs]
            _rk_lab = [(_p, _d) for _p, _d in _rk_lab if _d]
            # AVERAGED OVER THE UNIT'S INTERFACES, so a 3-interface structure does not get three
            # times the weight of a 1-interface one.
            _nif = 0
            for _p, _dqmap in _rk_lab:
                _idx = [_i for _i in range(n_pose) if _i in _dqmap]
                if len(_idx) < 2:
                    continue
                _cs, _np_ = RKL.coeffs([_dqmap[_i] for _i in _idx],
                                       weight_by_gap=a.rank_weight_by_gap)
                if _np_ == 0:
                    continue
                _nif += 1
                for _i, _cv in zip(_idx, _cs):
                    _rank_c[(_i, _p)] = _rank_c.get((_i, _p), 0.0) + _cv
            if _nif:
                for _k in list(_rank_c):
                    _rank_c[_k] /= float(_nif)
                _rank_on = True
                _rank_seen["units"] += 1

        for i in range(n_pose):
            _t_pose = time.perf_counter()
            # NO GRAPH UNDER --loss-only. Skipping only the backward would still build the
            # autograd graph for every pose and free it unused -- correct, but it pays the
            # activation memory of training for a measurement that needs none.
            if a.loss_only or score_only:
                with torch.no_grad():
                    m = _one_pose(i)
            else:
                m = _one_pose(i)
            # PASS A'S POSES GET THEIR OWN BUCKET. Ticking them into pose_total would double-count
            # every pose and make the epoch accounting -- which is checked against the wall clock
            # and reported as a residual -- silently wrong.
            _tick("pose_a_total" if score_only else "pose_total", _t_pose)
            if a.mem_log:
                # AT A CONSISTENT BOUNDARY: after backward, and after the pose's tensors are gone.
                # allocated = live tensors. reserved = what the allocator holds, which is what
                # nvidia-smi shows -- which is why a high nvidia-smi number proves nothing by itself.
                print(f"MEM {pdb}/{arm_seed} pose {i} tokens {n_tok}: "
                      f"allocated {torch.cuda.memory_allocated(dev) / 2**30:.2f} GB  "
                      f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f}  "
                      f"reserved {torch.cuda.memory_reserved(dev) / 2**30:.2f}", flush=True)
            yield m

    def _xs_prepare(step_units, ep):
        """PASS A for one optimiser step: score every pose, share the scalars, build dL/ds.

        Fills _xs_coef for exactly the poses of `step_units` across ALL ranks, and sets _xs_div to
        the number of interface-groups the step contains. Called once per step, before that step's
        first backward.

        WHY A WHOLE STEP AT A TIME. With --rank-cross-seed a pose's coefficient contains
        sigmoid(-m_ij/tau) against every other pose of its interface, and build_epoch_cross_seed
        puts all of a structure's seeds inside one step. So the unit of scoring is the step, not
        the unit -- which is the whole reason the step boundary had to become explicit.

        THE TRUNK IS READ TWICE PER STEP, once here and once in pass B. That is deliberate: the
        alternative is holding every unit of the step resident, and at ~205 MB of trunk per unit
        and 20 units per step that is ~4 GB of NON-reclaimable RSS per rank against a box that
        other people are using. The second read comes from page cache, which is reclaimable and
        does not count against --max-rss-gb or MemAvailable in the way that matters.

        NO OPTIMISER STEP MAY HAPPEN BETWEEN THIS AND PASS B, or the scores were computed at
        different weights than the gradient is taken at. That is guaranteed by construction: this
        runs at a step boundary, immediately after the previous step was applied.

        THE RNG IS SAVED AND RESTORED because the confidence head has dropout, so pass A would
        otherwise advance the generator and pass B would evaluate a different function than the one
        the coefficients describe.
        """
        _xs_coef.clear()
        _xs_div[0] = 1
        if not (a.rank_cross_seed and a.rank_loss) or a.loss_only:
            return
        import rank_loss as RKL
        _rng_cpu = torch.get_rng_state()
        _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _t_a = time.perf_counter()
        local = {}
        try:
            for _pdb, _lab, _cands in step_units:
                _tp = resolve_trunk(_pdb, _lab, _cands)
                _sd = _rank_key(_pdb, os.path.basename(_lab)[:-4], (0, 0))[1]
                for _m in run_unit(_pdb, _lab, _tp, score_only=True):
                    for _pair, _v in _m["iptm"].items():
                        local[(_pdb, _sd, int(_m["pose"]), _pair)] = float(_v)
        finally:
            torch.set_rng_state(_rng_cpu)
            if _rng_cuda is not None:
                torch.cuda.set_rng_state_all(_rng_cuda)
        _tick("rank_pass_a", _t_a)

        # PASS A MUST SCORE EVERYTHING IT WAS ASKED TO, AND THIS ASSERTS IT RATHER THAN HOPING.
        #
        # RKL.iptm returns None when a requested asym index falls outside the pose's
        # chain_pair_iptm -- which cropping can cause, not only a bad roles join. Without this
        # check that None is silently dropped from `local`, group_coeffs then INTERSECTS the
        # surviving scores with the labels, and a group left with fewer than two poses is skipped
        # entirely. The objective would quietly stop being the full-group ranking loss it is
        # documented as, and in a ranking-only arm (--alpha-pae 0) the affected poses would
        # receive NO gradient at all while the run looked healthy. The label-coverage guard cannot
        # see this: it checks that DockQ exists, not that a score was produced.
        #
        # ASSERTED PER RANK, BEFORE THE GATHER. Every rank verifies its own units, so the merged
        # set is complete by construction -- there is no need to know the other ranks' units here.
        # Found by review of f01dec8a.
        _exp = {(_pdb, _rank_key(_pdb, os.path.basename(_lab)[:-4], (0, 0))[1], _i, _pair)
                for _pdb, _lab, _cands in step_units
                for _pair in (_scored_asym.get(_pdb) or ())
                for _i in range(a.nsample)}
        _got = set(local)
        if _exp != _got:
            _missing = sorted(_exp - _got)[:6]
            _extra = sorted(_got - _exp)[:6]
            raise SystemExit(
                f"STOP: pass A scored {len(_got)} of {len(_exp)} expected "
                f"(pdb, seed, pose, interface) key(s) for this step.\n"
                f"  missing e.g. {_missing}\n"
                f"  unexpected e.g. {_extra}\n"
                f"A missing score silently removes comparisons from the ranking objective, and "
                f"with --alpha-pae 0 it removes the affected poses' only gradient. The usual "
                f"cause is RKL.iptm returning None because an asym index falls outside the pose's "
                f"chain_pair_iptm -- check --pae-scored-pairs against the fold's chain_roles.csv "
                f"and whether the crop dropped a chain.")
        _nonfinite = sorted(k for k, v in local.items() if not math.isfinite(v))
        if _nonfinite:
            raise SystemExit(
                f"STOP: pass A produced {len(_nonfinite)} non-finite ipTM value(s), e.g. "
                f"{_nonfinite[:6]}. A non-finite score makes every coefficient in that group "
                f"non-finite, which would train on it rather than fail.")

        # EVERY RANK NEEDS THE WHOLE STEP. A few hundred floats, so the object collective's
        # pickling is irrelevant next to a 205 MB trunk read. All ranks then compute the SAME
        # coefficients from the SAME inputs, and each applies only its own poses -- so the
        # cross-rank gradient sum reconstructs one group contribution, not `world` of them.
        if world > 1:
            _t_g = time.perf_counter()
            _bag = [None] * world
            dist.all_gather_object(_bag, local)
            _tick("rank_gather", _t_g)
            allsc = {}
            for _d in _bag:
                allsc.update(_d or {})
        else:
            allsc = dict(local)

        # GROUPED BY (pdb, interface), which is what an interface IS. Keyed by (seed, sample)
        # inside the group so group_coeffs can tell cross-seed pairs from within-seed ones.
        groups = {}
        for (_pdb, _sd, _i, _pair), _v in allsc.items():
            groups.setdefault((_pdb, _pair), {})[(_sd, _i)] = _v
        n_groups = 0
        for (_pdb, _pair), s_by in groups.items():
            dq_by = {}
            for (_sd, _i) in s_by:
                _lab = _rank_lab.get(_rank_key(_pdb, _sd, _pair))
                if _lab is not None and _i in _lab:
                    dq_by[(_sd, _i)] = float(_lab[_i])
            if len(dq_by) < 2:
                continue
            # THE FORM IS PASSED THROUGH, NOT RE-DERIVED. This read
            #     form=("softplus" if a.rank_form == "softplus" else "linear")
            # which sent any third form to the LINEAR loss while every log line and the saved
            # guard still said otherwise -- a run that looks right and is the wrong experiment.
            # rank_loss.group_coeffs raises on an unknown form, so a typo fails loudly instead.
            c, npair, lval = RKL.group_coeffs(
                s_by, dq_by,
                form=a.rank_form,
                # THE DEFAULT DEPENDS ON THE FORM. DEFAULT_TAU is a margin scale in ipTM units;
                # the expected form's tau is a softmax temperature. Falling back to 0.0305 for
                # the expected form would be a sharper selection than the 0.05 default intends,
                # with nothing in the log to reveal it.
                tau=(_rank_tau if _rank_tau is not None else
                     (RKL.DEFAULT_SOFTMAX_TAU if a.rank_form == "expected" else RKL.DEFAULT_TAU)),
                weight_by_gap=a.rank_weight_by_gap,
                pairs=a.rank_pairs,
                within_weight=a.rank_within_weight,
                rank_decay_k=a.rank_topk_decay,
                straddle=a.rank_straddle)
            if npair == 0:
                continue
            n_groups += 1
            _xs_stat["pairs"] += npair
            _xs_stat["loss"] += lval
            for (_sd, _i), _cv in c.items():
                _xs_coef[(_pdb, _sd, _i, _pair)] = _cv
        _xs_div[0] = max(1, n_groups)
        # PER-STEP, not cumulative. Pass A is REPLICATED -- it fills _xs_coef for the poses of
        # step_units across ALL ranks -- so n_groups is identical on every rank and the skip
        # decision below needs no collective and cannot desynchronise them.
        _xs_stat["last_groups"] = n_groups
        _xs_stat["groups"] += n_groups
        _xs_stat["steps"] += 1

    def outside_head_grads():
        return sum(1 for n, p in model.named_parameters()
                   if n not in set(trainable) and p.grad is not None
                   and bool(p.grad.abs().sum() > 0))

    # ONE LIST, NAMES AND TENSORS TOGETHER, at main level so both _apply_step and the gradient
    # probe read the same objects in the same order. They were built independently before, and
    # `params` lived inside _apply_step where _one_pose could not see it -- the probe would have
    # raised NameError the first time it ran. Zipping two separately-built lists also risks
    # pairing a name with the wrong tensor. Found in review of a3086ef.
    trainable_params = [(n, pp) for n, pp in model.named_parameters() if n in set(trainable)]
    if a.rank_loss and not trainable_params:
        raise SystemExit("STOP: no trainable tensors found for the ranking term.")

    def _grad_probe_report(pr, names, a, step_taken, accum_local, world):
        """Report, then RAISE on failure. A safety gate that only prints is not a gate.

        WHAT IT CHECKS, and why each one exists:
          * every trainable tensor receives a FINITE, NON-ZERO ranking gradient. This is the only
            thing that establishes gradient flows through calculate_chain_based_ptm into the two
            matrices; every offline test validates the label side, which cannot. An earlier
            version computed finiteness and then left it out of the verdict, so a tensor holding
            both valid and non-finite values could pass on the non-zero check alone.
          * at least one optimiser step actually completed. The first probe ran 3 units = 15 poses
            against an accumulation window of 30, so it returned before any step was taken and
            still reported success.
          * the ACCUMULATED gradients, because that is what the optimiser sees. ||sum_i g_i|| is
            not sum_i ||g_i||, and the ranking coefficients sum to zero within a group, so per-pose
            contributions cancel. alpha is read off the accumulated pair; the per-pose spread is
            printed beside it for debugging only.

        Both points were raised in review of a3086ef and both were real.
        """
        bar = "=" * 78
        print(f"\n{bar}\n--grad-probe: {pr['poses']} pose(s), {pr['units']} unit(s), "
              f"alpha_rank {a.alpha_rank:g}, accumulation window {accum_local} pose(s)\n{bar}")
        fail = []

        for nm in ("pae", "rank"):
            got = pr["sum"].get(nm)
            if not got:
                fail.append(f"no {nm.upper()} gradient was measured at all")
                continue
            print(f"\n  {nm.upper()}, accumulated over the probe:")
            for n_ in names:
                g = got.get(n_)
                if g is None:
                    print(f"    {n_:<60} ABSENT")
                    continue
                print(f"    {n_:<60} |sum g| {float(g.norm()):.4e}  "
                      f"max|g| {float(g.abs().max()):.4e}")

        # THE TWO ACCUMULATED VECTORS, SUMMED ACROSS RANKS -- exactly what _apply_step does
        # before clipping and stepping. Each rank only ever saw its own units, so a per-rank
        # measurement is not the optimiser's gradient, and the ranking contributions cancel both
        # within a seed and across ranks. Found in review of 523013a.
        vecs = {}
        for nm in ("pae", "rank"):
            got = pr["sum"].get(nm) or {}
            fl = [got[n_].flatten() for n_ in names if got.get(n_) is not None]
            if not fl:
                continue
            v = torch.cat(fl)
            if world > 1:
                host = v.cpu()
                dist.all_reduce(host, op=dist.ReduceOp.SUM)
                v = host.to(v.device)
            vecs[nm] = v
        if world > 1:
            print(f"\n  (accumulated gradients SUMMED over {world} rank(s), as _apply_step does)")
        if "pae" in vecs and "rank" in vecs:
            vp, vr = vecs["pae"], vecs["rank"]
            np_, nr_ = float(vp.norm()), float(vr.norm())
            cos = float((vp @ vr) / (np_ * nr_)) if np_ > 0 and nr_ > 0 else float("nan")
            comb = float((vp + vr).norm())
            print(f"\n  ACCUMULATED (what the optimiser sees):")
            print(f"    |g_PAE|      {np_:.4e}")
            print(f"    |g_rank|     {nr_:.4e}   ratio {nr_ / max(1e-30, np_):.4f}")
            print(f"    cos(PAE,rank) {cos:+.4f}   -> " + (
                "they OPPOSE: alpha trades one objective for the other" if cos < -0.05 else
                "near-orthogonal: the ranking term is close to free" if abs(cos) <= 0.05 else
                "they AGREE: the ranking term reinforces the PAE direction"))
            print(f"    |g_PAE + g_rank| {comb:.4e}   "
                  f"(clip threshold {a.clip_norm:g}"
                  f"{', WOULD CLIP' if comb > a.clip_norm else ''})")
            if nr_ > 0:
                print(f"    for parity at the step: alpha_rank ~ "
                      f"{a.alpha_rank * np_ / nr_:.4g}")
            if pr["rows"]:
                pr_ratios = [r[4] / r[3] for r in pr["rows"] if r[3] > 0]
                pr_cos = [r[5] for r in pr["rows"] if r[5] == r[5]]
                if pr_ratios:
                    print(f"\n  per-pose, for comparison only: ratio mean "
                          f"{sum(pr_ratios)/len(pr_ratios):.4f} "
                          f"[{min(pr_ratios):.4f}, {max(pr_ratios):.4f}]"
                          + (f", cos mean {sum(pr_cos)/len(pr_cos):+.4f}" if pr_cos else ""))
                    print(f"  (a mean of per-pose ratios is NOT the accumulated ratio; the "
                          f"coefficients sum to zero per group so contributions cancel)")

            # ALIGNMENT: ||sum_i g_i|| / sum_i ||g_i||, THE ONE NUMBER THAT SEPARATES
            # CANCELLATION FROM LOST SENSITIVITY.
            #
            # A shrinking parameter gradient has two possible causes and they need different
            # fixes. Either the per-pose gradients still have real magnitude but point in
            # opposing directions and destroy each other when summed (CANCELLATION), or the poses
            # individually stopped producing gradient because the trainable parameters no longer
            # move ipTM much (SENSITIVITY). The accumulated norm alone cannot tell them apart,
            # which is exactly the ambiguity raised in review after the first plateau analysis:
            # 40% of pairs were still inverted and the score-level signal was GROWING while the
            # parameter gradient fell 8x, and I attributed that to cancellation without having
            # measured it.
            #
            # sum_i ||g_i|| is the total magnitude BEFORE any cancellation; ||sum_i g_i|| is what
            # the optimiser receives. Their ratio is 1 when every pose pulls the same way and
            # tends to ~1/sqrt(N) for N independent directions, so READ IT AGAINST THAT
            # BASELINE, not against 1. Tracking it across checkpoints answers the question:
            #   ratio falls while sum_i ||g_i|| holds up  -> CANCELLATION
            #   sum_i ||g_i|| itself collapses            -> SENSITIVITY
            #
            # Summed over ranks like everything else here, because each rank saw only its own
            # poses and a per-rank figure is not the optimiser's.
            # THE DECOMPOSITION, on the same poses and the same window:
            #     g_i = c_i * grad_theta s_i
            # so sum_i ||g_i|| / sum_i |c_i| is the realised mean sensitivity of ipTM to the
            # trainable weights. Tracked across checkpoints it separates the two causes of a
            # shrinking parameter gradient that the alignment ratio alone cannot:
            #     |c_i| falls, sensitivity flat  -> the loss stopped asking
            #     |c_i| flat, sensitivity falls  -> the weights lost leverage over ipTM
            # Previously the only coefficient numbers were val10 sum|c_i| against a training-set
            # gradient probe -- different poses, so the ratio was not formable. Raised in review.
            if pr["rows"] and len(pr["rows"][0]) > 6:
                _cl = sum(r[6] for r in pr["rows"])
                _gl = sum(r[4] for r in pr["rows"])
                _t = torch.tensor([_cl, _gl], dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                _cl, _gl = float(_t[0]), float(_t[1])
                if _cl > 0:
                    print(f"\n  rank DECOMPOSITION: sum_i |c_i| {_cl:.4e}   "
                          f"sum_i ||g_i|| {_gl:.4e}   "
                          f"mean sensitivity ||grad_theta s_i|| {_gl / _cl:.4e}")
                    print(f"  (track across checkpoints: |c_i| falling = the loss stopped "
                          f"asking; sensitivity falling = the weights lost leverage over ipTM)")

            for nm, vec in (("PAE", vecs.get("pae")), ("rank", vecs.get("rank"))):
                col = 3 if nm == "PAE" else 4
                loc = sum(r[col] for r in pr["rows"]) if pr["rows"] else 0.0
                tot = torch.tensor([loc], dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(tot, op=dist.ReduceOp.SUM)
                tot = float(tot[0])
                if vec is None or tot <= 0:
                    continue
                acc = float(vec.norm())
                # N IS THE ALL-REDUCED POSE COUNT, not this rank's count times the world size.
                # Ranks get different numbers of units when the split is uneven, so the latter
                # printed two different N (105 and 90) for the same accumulation window and so
                # two different 1/sqrt(N) baselines beside an identical ratio.
                _npt = torch.tensor([float(pr["poses"])], dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(_npt, op=dist.ReduceOp.SUM)
                n_poses = int(_npt[0])
                print(f"\n  {nm} ALIGNMENT over {n_poses} pose(s): "
                      f"||sum g_i|| {acc:.4e} / sum ||g_i|| {tot:.4e} = {acc / tot:.4f}"
                      f"   (1/sqrt(N) = {1.0 / math.sqrt(max(1, n_poses)):.4f} would be "
                      f"independent directions)")
        else:
            fail.append("could not form both accumulated gradient vectors")

        # THE ACCUMULATED NORM, NOT JUST THE PER-POSE ONES. Non-zero per-pose gradients can
        # cancel to nothing over the window -- the coefficients sum to zero inside each group --
        # and an alpha recommended off a zero accumulated norm is meaningless. Found in review
        # of 523013a.
        if "rank" in vecs:
            _rn = float(vecs["rank"].norm())
            if not math.isfinite(_rn):
                fail.append("the ACCUMULATED ranking gradient is non-finite")
            elif _rn == 0.0:
                fail.append("the ACCUMULATED ranking gradient is exactly zero: per-pose "
                            "contributions cancelled over the window, so no alpha can be read "
                            "from norm parity")
        if "pae" in vecs and "rank" in vecs:
            _vp2, _vr2 = vecs["pae"], vecs["rank"]
            if float(_vp2.norm()) == 0.0:
                fail.append("the ACCUMULATED PAE gradient is exactly zero")
            elif not math.isfinite(float((_vp2 @ _vr2) / max(1e-30, _vp2.norm() * _vr2.norm()))):
                fail.append("the PAE/ranking cosine is not finite")
        if not pr["finite"].get("rank", False):
            fail.append("the RANKING gradient contained non-finite values")
        if not pr["finite"].get("pae", False):
            fail.append("the PAE gradient contained non-finite values")
        nz = pr["nonzero"].get("rank") or set()
        absent = [n_ for n_ in names if n_ not in nz]
        if absent:
            fail.append(f"{len(absent)} trainable tensor(s) got NO non-zero ranking gradient: "
                        f"{absent}")
        if not step_taken:
            fail.append(f"no optimiser step completed: {pr['poses']} pose(s) probed against an "
                        f"accumulation window of {accum_local}. Raise --grad-probe.")

        print("\n  " + "-" * 74)
        if fail:
            print("  FAIL:")
            for f in fail:
                print(f"    - {f}")
            print(f"{bar}\n", flush=True)
            raise SystemExit("STOP: --grad-probe failed its checks (above). Do not launch. "
                             "A gate that only prints is not a gate.")
        print(f"  PASS: all {len(names)} trainable tensor(s) received a finite, non-zero ranking "
              f"gradient through calculate_chain_based_ptm, and an optimiser step completed.")
        print(f"{bar}\n", flush=True)

    def _apply_step(step):
        """Clip, assert, step. Returns (grad_norm, n_outside_head, ok).

        A FINITE LOSS DOES NOT MEAN FINITE GRADIENTS -- the loss is checked per pose already, and
        that is not the same assertion. error_if_nonfinite makes clip_grad_norm_ raise rather than
        hand inf or nan to AdamW, which would silently poison every head parameter through the
        exponential averages and never recover. The gradients are dropped and the run continues,
        counted, rather than the step being taken.
        """
        if a.loss_only:
            # THE WINDOW STILL CLOSES, so the trajectory keeps its step_taken==1 rows and the
            # pose-weighted epoch PAE is computed from them exactly as for a training run. What is
            # skipped is the clip, the all-reduce of gradients that do not exist, and AdamW.
            # grad_norm is reported as nan rather than 0.0: zero is a measurement, nan is "not
            # measured", and a 0.0 here would read as a vanished gradient in the usual summaries.
            return float("nan"), 0, True
        # A STEP WITH NO TASK GRADIENT MUST NOT BE TAKEN. --rank-straddle makes this reachable
        # for the first time: a group with no straddling pair is skipped at the `npair == 0`
        # continue above, and 52.3% of the training fold's structures have no straddling pair at
        # all, so with --rank-structs-per-step 2 about 27% of steps accumulate NOTHING from the
        # ranking term. With --alpha-pae 0 there is then no task gradient whatsoever, and the
        # L2-SP block below would be the step's ONLY gradient.
        #
        # THAT IS NOT A SMALL STEP, BECAUSE ADAMW IS SCALE-INVARIANT. Once exp_avg_sq has decayed
        # (beta2 0.95, ~20-step memory) a gradient 100x smaller does not give a step 100x smaller;
        # it gives a near-full lr-sized step straight back toward l2sp_anchor.pt. Measured on the
        # first attempt at this arm: 16 of 69 taken steps had grad_norm < 1e-3 against a typical
        # 0.01-0.20, the late ones clustered at ~2.5e-4 and rising as the weights drifted, i.e.
        # the anchor pulling harder the further the arm got. The lambda calibrated against
        # |g_rank| is simply not in force on those steps.
        #
        # Counted, not silent: the epoch summary prints how many steps were skipped this way.
        if (a.rank_loss and a.rank_cross_seed and not a.alpha_pae
                and _xs_stat.get("last_groups", 1) == 0):
            _xs_stat["skipped"] += 1
            opt.zero_grad(set_to_none=True)
            return float("nan"), 0, False
        params = [pp for _n, pp in trainable_params]
        if world > 1:
            # BEFORE THE CLIP, NOT AFTER. Clipping each rank's partial gradient to norm 1 and then
            # averaging is not the same function as clipping the summed gradient, so the multi-GPU
            # run would not reproduce the single-GPU one. A rank with no gradient for a parameter
            # still has to contribute a tensor, or the collective's shapes do not line up.
            for pp in params:
                if pp.grad is None:
                    pp.grad = torch.zeros_like(pp)
            # ONE COLLECTIVE, NOT 248. The head's gradients are flattened into a single buffer, so
            # the per-call latency is paid once per step rather than once per tensor. gloo is a CPU
            # backend, so that buffer is staged through host memory and copied back.
            # ONE SYNC PER STEP, BEFORE THE TIMER STARTS. Without it the collective's timer
            # absorbs every queued kernel this rank has not finished, and the "collective" number
            # becomes mostly local compute. One sync per 290-step epoch is not the per-operation
            # synchronisation that would distort the measurement.
            _drain_gpu_spans()
            _t0 = time.perf_counter()
            flat = torch.cat([pp.grad.reshape(-1) for pp in params])
            if a.dist_backend == "gloo":
                host = flat.cpu()
                dist.all_reduce(host, op=dist.ReduceOp.SUM)
                flat = host.to(flat.device)
            else:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            off = 0
            for pp in params:
                k = pp.grad.numel()
                pp.grad.copy_(flat[off:off + k].view_as(pp.grad))
                off += k
            # COLLECTIVE PLUS STRAGGLER WAIT, and the two cannot be separated from inside one rank:
            # a rank that arrives early blocks here until the slowest arrives. The per-rank
            # pose_total totals printed below are what separates them -- their spread IS the
            # straggler share.
            _tick("grad_collective_plus_wait", _t0)
        # L2-SP PENALTY, ADDED ONCE PER STEP. After the all-reduce, because that SUMS each rank's
        # gradient and adding before it would multiply the penalty by the world size; before the
        # clip, because the clip is meant to bound the total gradient actually applied.
        # d/dW [ lambda * ||W - W0||^2 ] = 2 * lambda * (W - W0).
        if a.l2_sp and _l2sp_anchor is not None:
            with torch.no_grad():
                _sq = 0.0
                for _n_sp, _p_sp in _l2sp_params:
                    _d_sp = _p_sp.detach() - _l2sp_anchor[_n_sp]
                    _p_sp.grad.add_(_d_sp, alpha=2.0 * a.l2_sp)
                    _sq += float((_d_sp * _d_sp).sum())
                _l2sp_drift["sq"] = _sq
                _l2sp_drift["n"] += 1
        try:
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_norm,
                                                      error_if_nonfinite=True))
        except RuntimeError as e:
            nonfinite["n"] += 1
            print(f"  NON-FINITE GRADIENT at step {step} ({e}); gradients dropped, no step taken "
                  f"({nonfinite['n']} so far)")
            opt.zero_grad(set_to_none=True)
            if nonfinite["n"] > a.max_nonfinite:
                raise SystemExit(f"STOP: {nonfinite['n']} non-finite gradient(s), over "
                                 f"--max-nonfinite {a.max_nonfinite}. The learning rate or the data "
                                 f"is wrong; continuing would train on noise.")
            return float("nan"), -1, False
        outside = -1
        if step % a.grad_check_every == 0:
            outside = outside_head_grads()
            if outside:
                raise SystemExit(
                    f"STOP: {outside} parameter(s) OUTSIDE the confidence head carry a non-zero "
                    f"gradient at step {step}. FT3 freezes the trunk; this is not FT3 any more.")
        opt.step()
        opt.zero_grad(set_to_none=True)
        return gn, outside, True

    # DERIVED FROM --seeds, NOT FROM --eval-budget. The validation fold holds 100 seeds per
    # interface; the training fold holds exactly the ones this run trained on, and asking the
    # evaluator for more than exist makes it refuse rather than quietly report a thinner number.
    def _seed_lo_hi(spec):
        """--seeds is `lo-hi` and nothing else, so parse it that way and say so.

        The first version accepted comma-separated ranges, which the rest of the file does not:
        the manifest builder does `lo, hi = (int(x) for x in a.seeds.split("-"))` and hard-fails
        on anything else. Worse, the paired --seed-base derivation took the first number of the
        first range, so `500000-500009,500020-500029` would have asked the evaluator for 20 seeds
        starting at 500000 -- half of them never trained on. Dead generality that implied a wrong
        answer is worse than no generality.
        """
        lo, hi = (int(x) for x in str(spec).split("-"))
        return lo, hi - lo + 1
    _train_eval_seed_base, _n_seeds = _seed_lo_hi(a.seeds)
    _train_eval_budget = a.train_eval_budget or _n_seeds

    def _record_val_loss(epoch, step, bundle, mean, n_poses):
        """Append one held-out point to val_loss.csv. Shared by the sharded and unsharded paths.

        RECORDING THE MEASUREMENT MUST NOT BE ABLE TO END THE RUN. A NameError on this exact
        bookkeeping killed epoch 1 of x3k2m4 after the pass had already succeeded -- the mean was
        computed and 78 minutes of training died on the append. The subprocess trajectories are the
        durable record either way, so a failure here is reported and dropped.
        """
        try:
            vp = os.path.join(a.out, "val_loss.csv")
            new_file = not os.path.exists(vp)
            with open(vp, "a", newline="") as fh:
                w = csv.writer(fh)
                if new_file:
                    w.writerow(["epoch", "step", "n_poses", "val_pae", "bundle"])
                w.writerow([epoch, step, n_poses, f"{mean:.6f}", os.path.basename(bundle)])
        except Exception as e:
            print(f"  WARNING: could not append to val_loss.csv ({type(e).__name__}: {e}); the "
                  f"number is in the line above", flush=True)

    def val_loss_checkpoint(epoch, bundle, step):
        """Held-out PAE for the bundle this epoch just wrote. Returns the pose mean, or None.

        A SUBPROCESS, for the same reason eval_checkpoint is one: --loss-only carries its own
        guards -- the alignment preflight, the mask-coverage check, the manifest completeness
        check, the refusal to write a checkpoint -- and running it as the script keeps all of
        them. It also means the held-out fold's dataset config, labels and crop live entirely in
        another process, so there is no path by which measuring can perturb the weights being
        measured. That property is asserted by the caller rather than assumed.

        IT READS THE BUNDLE FROM DISK, not the live parameters. The epoch boundary has already
        written resume_ep<N>.pt, which carries all 248 head tensors, so the thing measured is
        exactly the thing a later scorer would load -- not a copy made by a different code path.
        """
        import json
        import subprocess
        out = os.path.join(a.out, f"val_loss_ep{epoch}")
        labels = a.val_loss_labels or os.path.join(a.val_loss_fold, "labels")
        cmd = [sys.executable, "-I", os.path.abspath(__file__),
               # A MEASUREMENT IS NOT A FORK, but it resumes the bundle with a different fold,
               # selection, seed range and epoch count, so every one of those reads as drift.
               # --resume-note is mandatory whenever drift is accepted, and the subprocess cannot
               # be the one exception -- a silent exemption is how the val monitor would die on
               # the next restart with a message about a note nobody could have passed.
               "--loss-only", "--resume", bundle, "--resume-allow-drift",
               "--resume-note", (f"held-out PAE measurement of {os.path.basename(bundle)} at "
                                 f"epoch {epoch}; --loss-only reads the weights and writes no "
                                 f"checkpoint, so the bundle's schedule is not continued"),
               "--fold", a.val_loss_fold, "--labels", labels,
               "--manifest-selection", a.val_loss_selection,
               "--alignment-preflight", a.val_loss_preflight, "--minimal-items",
               "--pae-scored-pairs", a.val_loss_pool,
               "--seeds", a.val_loss_seeds,
               # THE RUN'S OWN GEOMETRY AND WEIGHTS. A different crop or a different alpha would
               # make the held-out number incomparable to the training curve it is read against,
               # which is the only thing it is for.
               "--crop-margin", str(a.crop_margin), "--crop-contact", str(a.crop_contact),
               "--crop-size", str(a.crop_size), "--accum", str(a.accum),
               # ALPHA_PAE 0 IS A TRAINING CHOICE, NOT A MEASUREMENT ONE. A ranking-only arm sets
               # --alpha-pae 0 so no PAE gradient is taken, but the held-out number this
               # subprocess exists to produce IS the PAE loss, and passing 0 through would report
               # val10 PAE as exactly 0.0 every epoch -- silently destroying the one diagnostic
               # that is comparable across every arm (shipped 2.9756). So the monitor always
               # measures at alpha_pae 1. SIXTH instance of the --loss-only inheritance class:
               # --train-only, LoRA, --pose-perblock-share, the ranking guard fields, my own
               # coverage guard, now the loss weight itself.
               "--nsample", str(a.nsample),
               "--alpha-pae", str(a.alpha_pae if float(a.alpha_pae) > 0 else 1.0),
               "--alpha-confidence", str(a.alpha_confidence), "--clip-norm", str(a.clip_norm),
               "--blocks-per-ckpt", str(a.blocks_per_ckpt),
               "--epochs", "1", "--save-every", "0", "--warmup-steps", "0",
               "--shuffle", "structure", "--out", out]
        if a.pae_only:
            cmd.append("--pae-only")
        if a.drop_pde:
            cmd.append("--drop-pde")
        if a.frame_direction:
            cmd.append("--frame-direction")
        if a.pose_perblock:
            cmd.append("--pose-perblock")
            if a.pose_perblock_tied:
                cmd.append("--pose-perblock-tied")
            if a.pose_perblock_share:
                cmd.append("--pose-perblock-share")
            # PASSED TO THE CHILD, like the other two. A monitor that built a DIFFERENT
            # architecture than the parent would report a held-out number for a model that does
            # not exist -- the --loss-only inheritance class, now seven instances.
            if a.pose_perblock_copy:
                cmd.append("--pose-perblock-copy")
        if a.val_loss_prep:
            cmd += ["--indices", os.path.join(a.val_loss_prep, "job_indices.csv"),
                    "--bio-dir", os.path.join(a.val_loss_prep, "job_bioassembly"),
                    "--mmcif-dir", os.path.join(a.val_loss_prep, "job_cifs")]
        # A SINGLE-PROCESS CHILD, SO THE PARENT'S DISTRIBUTED ENVIRONMENT MUST GO. This script
        # reads WORLD_SIZE/RANK/LOCAL_RANK at startup, so an inherited WORLD_SIZE=3 makes the
        # monitor try to join a process group that already has all three of its members -- it
        # hangs on init_process_group holding a card, and the parent is parked at a barrier
        # waiting for it. eval_checkpoint never hit this because phase_k_eval.py reads none of
        # these; this child is phase_k_train.py itself.
        env = {k: v for k, v in os.environ.items()
               if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK", "ROLE_RANK",
                            "ROLE_NAME", "LOCAL_WORLD_SIZE", "ROLE_WORLD_SIZE", "MASTER_ADDR",
                            "MASTER_PORT", "TORCHELASTIC_RESTART_COUNT",
                            "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
                            "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_ERROR_FILE",
                            "TORCH_NCCL_ASYNC_ERROR_HANDLING")
               and not k.startswith("TORCHELASTIC_")}
        env["LAYERNORM_TYPE"] = "openfold"
        env["WORLD_SIZE"] = "1"
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        if a.val_loss_card:
            env["CUDA_VISIBLE_DEVICES"] = str(a.val_loss_card)
        # SHARDED ACROSS THE EVAL CARDS when there is more than one. This pass was a SINGLE
        # single-card child while the other two cards sat at the epoch-boundary barrier: measured
        # on j583i8, 12.8 min of a 46.0 min cycle, against a 24.1 min epoch. The children are
        # independent processes with no collective between them -- deliberately NOT a distributed
        # child, because --loss-only refuses WORLD_SIZE>1 and the comment there records the hang
        # that refusal prevents.
        _cards = [c.strip() for c in (a.eval_cards or "").split(",") if c.strip()]
        if a.val_loss_card:
            _cards = [str(a.val_loss_card)]
        _nsh = len(_cards) if len(_cards) > 1 else 1
        _where = (f"{_nsh} shard(s) over card(s) {','.join(_cards)}" if _nsh > 1
                  else ('card ' + str(a.val_loss_card) if a.val_loss_card else 'the training card'))
        print(f"  held-out PAE for epoch {epoch} ({_where})", flush=True)
        if _nsh > 1:
            # EACH CHILD WRITES ITS OWN LOG FILE rather than a pipe the parent drains one at a
            # time. With stdout=PIPE and sequential communicate(), a later child that fills its
            # 64 KiB pipe buffer BLOCKS until the parent gets round to reading it -- serialising
            # the very passes this change exists to run in parallel. Files also keep the whole
            # diagnostic instead of the last four lines of a failure. Caught in review.
            procs = []
            os.makedirs(os.path.join(a.out, f"val_loss_ep{epoch}"), exist_ok=True)
            for _i, _c in enumerate(_cards):
                _env = dict(env)
                _env["CUDA_VISIBLE_DEVICES"] = _c
                _o = os.path.join(a.out, f"val_loss_ep{epoch}", f"shard{_i}")
                os.makedirs(_o, exist_ok=True)
                _lg = os.path.join(_o, "shard.log")
                _cmd = [x for x in cmd]
                _cmd[_cmd.index("--out") + 1] = _o
                _cmd += ["--manifest-shard", str(_i), "--manifest-nshards", str(_nsh)]
                _fh = open(_lg, "w")
                procs.append((_i, _o, _lg, _fh,
                              subprocess.Popen(_cmd, stdout=_fh, stderr=subprocess.STDOUT,
                                               text=True, env=_env)))
            _bad = []
            for _i, _o, _lg, _fh, _p in procs:
                _p.wait()
                try:
                    _fh.close()
                except Exception:                 # noqa: BLE001
                    pass
                if _p.returncode != 0:
                    try:
                        _tail = open(_lg).read().strip().splitlines()[-6:]
                    except OSError:
                        _tail = [f"(no readable log at {_lg})"]
                    _bad.append((_i, _tail))
            if _bad:
                # ANY shard failing voids the epoch's number. Pooling the survivors would report a
                # mean over a BIASED subset of the held-out set -- a different population than
                # every other epoch -- which is worse than no point at all, because it is silently
                # comparable-looking. The next epoch tries again and the gap shows in val_loss.csv.
                for _i, _tail in _bad:
                    print(f"  WARNING: held-out shard {_i} of epoch {epoch} failed:\n    "
                          + "\n    ".join(_tail), flush=True)
                print(f"  WARNING: epoch {epoch} has no held-out point; {len(_bad)} of {_nsh} "
                      f"shard(s) failed and a partial pool would be a different population.",
                      flush=True)
                return None
            # POOLED AS A PLAIN MEAN OVER POSES. Each shard owns whole structures, every pose
            # appears in exactly one shard exactly once, so concatenating the rows and taking one
            # mean IS the pose-weighted figure -- the same identity the unsharded path relies on.
            # EXACT POSE POPULATION, ASSERTED. Arithmetic alone does not make the pooled mean
            # right: a child can exit 0 having measured fewer poses than it owned, and the mean
            # over a short population would look entirely normal while being a different number
            # from every other epoch. So the identities are checked, not just the count.
            _vals, _ids, _cov = [], set(), {}
            for _i, _o, _lg, _fh, _p in procs:
                try:
                    with open(_lg) as fh:
                        for ln in fh:
                            if ln.startswith("SHARD COVERAGE "):
                                _cov[_i] = dict(kv.split("=", 1)
                                                for kv in ln.split()[2:] if "=" in kv)
                except OSError:
                    pass
                try:
                    with open(os.path.join(_o, "trajectory.csv")) as fh:
                        for row in csv.DictReader(fh):
                            if not row.get("pae_loss"):
                                continue
                            _k = (row.get("pdb"), row.get("unit"), row.get("pose"))
                            if _k in _ids:
                                print(f"  WARNING: epoch {epoch} held-out pose {_k} appears in "
                                      f"more than one shard; the shards are not disjoint and the "
                                      f"epoch has no held-out point", flush=True)
                                return None
                            _ids.add(_k)
                            _vals.append(float(row["pae_loss"]))
                except OSError as e:
                    print(f"  WARNING: held-out shard {_i} of epoch {epoch} wrote no readable "
                          f"trajectory ({e}); epoch {epoch} has no held-out point", flush=True)
                    return None
            if not _vals:
                print(f"  WARNING: the epoch-{epoch} held-out shards have no pae_loss rows",
                      flush=True)
                return None
            if len(_cov) != _nsh:
                print(f"  WARNING: epoch {epoch} got coverage lines from {len(_cov)} of {_nsh} "
                      f"shard(s); cannot prove the population is complete, so no held-out point",
                      flush=True)
                return None
            _alls = {int(c["units_all"]) for c in _cov.values()}
            _nsamp = {int(c["nsample"]) for c in _cov.values()}
            _sum_mine = sum(int(c["units_mine"]) for c in _cov.values())
            if len(_alls) != 1 or len(_nsamp) != 1:
                print(f"  WARNING: epoch {epoch} shards disagree on the full population "
                      f"(units_all={_alls}, nsample={_nsamp}); no held-out point", flush=True)
                return None
            _exp_units, _ns = _alls.pop(), _nsamp.pop()
            if _sum_mine != _exp_units:
                print(f"  WARNING: epoch {epoch} shards cover {_sum_mine} of {_exp_units} unit(s); "
                      f"the population is incomplete, so no held-out point", flush=True)
                return None
            _exp_poses = _exp_units * _ns
            if len(_vals) != _exp_poses or len(_ids) != _exp_poses:
                print(f"  WARNING: epoch {epoch} pooled {len(_vals)} pose(s) ({len(_ids)} unique) "
                      f"against {_exp_poses} expected ({_exp_units} unit(s) x {_ns}); no held-out "
                      f"point", flush=True)
                return None
            _mean = sum(_vals) / len(_vals)
            print(f"epoch {epoch} VALLOSS: pae {_mean:.4f}  poses {len(_vals)}", flush=True)
            _record_val_loss(epoch, step, bundle, _mean, len(_vals))
            return _mean
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            # LOUD, BUT NOT FATAL. Training is the run; a monitor that cannot compute its number
            # must not destroy 37 minutes of epoch because a card was momentarily full. The next
            # epoch tries again and the gap is visible in val_loss.csv.
            tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
            print(f"  WARNING: the epoch-{epoch} held-out PAE pass failed; training continues and "
                  f"this epoch has no held-out point:\n    " + "\n    ".join(tail), flush=True)
            return None
        tp = os.path.join(out, "trajectory.csv")
        vals = []
        try:
            with open(tp) as fh:
                for row in csv.DictReader(fh):
                    if row.get("pae_loss"):
                        vals.append(float(row["pae_loss"]))
        except OSError as e:
            print(f"  WARNING: the epoch-{epoch} held-out pass wrote no readable trajectory ({e})",
                  flush=True)
            return None
        if not vals:
            print(f"  WARNING: the epoch-{epoch} held-out trajectory has no pae_loss rows", flush=True)
            return None
        # EVERY POSE EXACTLY ONCE, so the plain mean IS the pose-weighted figure. Deliberately not
        # the step_taken==1 windows: --accum leaves a tail outside the last complete window, which
        # a training epoch re-traverses next epoch and this single pass never does.
        mean = sum(vals) / len(vals)
        # PARSEABLE, in the same shape as the TRAIN: line the plot already reads. PRINTED BEFORE
        # the csv append, so the number survives any failure in the bookkeeping that follows it.
        print(f"epoch {epoch} VALLOSS: pae {mean:.4f}  poses {len(vals)}", flush=True)
        # RECORDING THE MEASUREMENT MUST NOT BE ABLE TO END THE RUN. A NameError on this exact
        # line killed epoch 1 of x3k2m4 at 09:34 after the pass had already succeeded: `step` was
        # not in scope, the mean was computed, and 78 minutes of training died on the bookkeeping.
        # The subprocess's own trajectory.csv is the durable record either way -- that is where the
        # lost number was recovered from -- so a failure here is reported and dropped.
        _record_val_loss(epoch, step, bundle, mean, len(vals))
        return mean

    def eval_checkpoint(step, epoch, train_pool=""):
        """Save the head, score it, log it. Returns the metric dict.

        With `train_pool`, scores that pool of TRAINING interfaces using the run's own --fold and
        --seeds instead of the validation set, and writes metric_train_ep<N>.json. Everything else
        -- the subprocess, the sharding, the guards -- is the same path, because the two metrics
        have to be comparable and a second implementation would not stay that way.

        RUN AS A SUBPROCESS, not imported. phase_k_eval carries its own guards -- the pool-coverage
        check, the pose-to-sample join by coordinate identity, and the self-check that the SHIPPED
        head reproduces results.csv's own ipTM -- and running it as the script keeps all of them,
        including its refusal to report a number when a join is unverified. It also keeps a second
        model off this process's GPU memory.
        """
        import json
        import subprocess
        cp = os.path.join(a.out, f"head_ep{epoch}_step{step}.pt")
        if not is_main:
            return          # the caller already holds the other ranks at a barrier
        torch.save(_head_ckpt_tensors(), cp)
        kind = ("pool@" + os.path.basename((a.train_eval_fold or a.fold).rstrip("/"))
                if train_pool else "validation")
        mj = os.path.join(a.out, ("metric_train_ep%d.json" if train_pool else "metric_ep%d.json")
                          % epoch)
        extra = []
        if train_pool:
            # THE POOL'S OWN FOLD, which is not always this run's. Scoring a validation subset
            # here means --fold, --results, --roles and --seed-base all have to come from that
            # fold, not from the one being trained on: a results.csv from the wrong fold would
            # look up DockQ for interfaces it does not contain and the coverage check would fail,
            # and a wrong --seed-base would ask for seeds the fold never wrote.
            _tef = a.train_eval_fold or a.fold
            _seed_base = (a.train_eval_seed_base if a.train_eval_seed_base >= 0
                          else _train_eval_seed_base)
            extra = ["--fold", _tef,
                     "--results", os.path.join(_tef, "results.csv"),
                     "--pool", train_pool,
                     "--roles", os.path.join(_tef, "chain_roles.csv"),
                     "--seed-base", str(_seed_base)]
        # THE EVALUATOR MUST RUN THE SAME FUNCTION WE TRAINED. Without this it builds a head with
        # no frame_dir_proj, and its own guard refuses the checkpoint for holding a tensor the
        # model does not have -- a loud failure rather than a silently different metric, but a
        # failure either way. Appended outside the train_pool branch so it reaches the validation
        # path too; both the sharded driver (--extra) and the single-card path forward `extra`.
        if a.frame_direction:
            extra = extra + ["--frame-direction"]
        if a.pose_perblock:
            extra = extra + ["--pose-perblock"]
            if a.pose_perblock_tied:
                extra = extra + ["--pose-perblock-tied"]
            if a.pose_perblock_share:
                extra = extra + ["--pose-perblock-share"]
            # AND THE COPY MODE. This one ADDS tensors, so omitting it makes the evaluator's
            # state-dict guard refuse outright rather than silently score a different
            # architecture -- but passing it is still the fix, not relying on the refusal.
            if a.pose_perblock_copy:
                extra = extra + ["--pose-perblock-copy"]
        # THE THREE OPTIMISATIONS THE EVALUATOR ALREADY HAS, which this path never passed.
        #
        # --join-cache is the big one. The pose->sample join is a function of the trunk's own
        # coordinates and the pose cifs on disk, so for a fixed pool it returns the same answer
        # every time -- and phase_k_eval_sharded's docstring measured it at 3.6 min per shard.
        # Recomputing a constant on every eval is most of why a 10-structure pool took 8.9 min
        # when its 156 GB of trunks is only ~1.7 min of I/O. The cache key carries each trunk's
        # size and mtime, so a refold or rebuild invalidates it rather than silently reusing a
        # mapping to superseded poses -- which is the exact failure resolve_seed_dir exists to
        # prevent. It lives beside the POOL, not in the run directory, so the first eval of the
        # first arm pays for it and every later eval and every later arm is free.
        #
        # --pair-iptm-only computes chain_pair_iptm straight from PAE instead of building the
        # whole Protenix summary and discarding every other field. Measured bit-identical.
        # --prefetch overlaps the next unit's trunk read with the current unit's compute.
        _jc = os.path.join(os.path.dirname(os.path.abspath(train_pool or a.train_eval_pool or ".")),
                           "join_cache.json") if train_pool else os.path.join(a.out, "join_cache.json")
        extra = extra + ["--join-cache", _jc, "--pair-iptm-only", "--prefetch"]
        # SHARDED OVER EVERY CARD THE RUN WAS GIVEN. The other ranks are parked at a barrier for
        # the whole evaluation, so their cards are idle: at 73 min a pass, evaluating on one card
        # wasted ~47% of the run's wall clock with two cards doing nothing. The driver puts one
        # shard on each card -- ~6 GB beside a parked rank's ~6 GB on a 98 GB card -- and merges.
        cards = a.eval_cards or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        shard_work = os.path.join(a.out, ("train_eval_shards_ep%d" if train_pool
                                           else "eval_shards_ep%d") % epoch)
        budget = _train_eval_budget if train_pool else a.eval_budget
        if len([c for c in cards.split(",") if c.strip()]) > 1:
            cmd = [sys.executable, "-I",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "phase_k_eval_sharded.py"),
                   "--cards", cards, "--ckpt", cp, "--budget", str(budget),
                   "--work", shard_work, "--out", mj]
            if extra:
                cmd += ["--extra", " ".join(extra)]
        else:
            cmd = [sys.executable, "-I", a.eval_script, "--ckpt", cp,
                   "--budget", str(budget), "--out", mj, *extra]
        env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
        sharded = "phase_k_eval_sharded.py" in cmd[2]
        if sharded:
            print(f"  evaluating epoch {epoch} ({kind}) sharded over GPU(s) {cards}")
        elif a.eval_device:
            # ONLY ON THE SINGLE-CARD PATH. Pinning CUDA_VISIBLE_DEVICES here would hand the sharding
            # driver one card and silently undo the split.
            env["CUDA_VISIBLE_DEVICES"] = str(a.eval_device)
            print(f"  evaluating epoch {epoch} on GPU {a.eval_device} (the trainer keeps its card)")
        else:
            # SHARING MEANS CHECKING FIRST. The trainer is still resident and does not hand its
            # allocator back, so a second 464M-parameter model has to fit alongside it.
            free_mib = None
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
                idx = int(str(os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0])
                free_mib = int(q.stdout.strip().splitlines()[idx])
            except Exception:
                pass
            if free_mib is not None and free_mib < a.eval_min_free_mib:
                raise SystemExit(
                    f"STOP: the epoch-{epoch} evaluation would share the training card with only "
                    f"{free_mib} MiB free, under --eval-min-free-mib {a.eval_min_free_mib}. It "
                    f"loads a second 464M-parameter model (~16 GB) while the trainer stays "
                    f"resident. Pass --eval-device <idle gpu>, or raise the floor deliberately.")
            print(f"  evaluating epoch {epoch} on the TRAINING card, {free_mib} MiB free")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
            raise SystemExit(f"STOP: the epoch-{epoch} {kind} evaluation failed, so this "
                             f"checkpoint cannot be scored and a training curve without it is not "
                             f"interpretable:\n  " + "\n  ".join(tail))
        m = json.load(open(mj))
        if train_pool:
            print(f"  epoch {epoch} TRAIN: mean_dockq {m['mean_dockq']:.4f}  "
                  f"frac_acc {m['frac_acc']:.4f}  oracle {m['oracle_mean_dockq']:.4f} / "
                  f"{m['oracle_frac_acc']:.4f}  headroom {m['headroom']:.4f}  "
                  f"({m['n_interfaces']} interfaces)")
        else:
            print(f"  epoch {epoch}: mean_dockq {m['mean_dockq']:.4f} (baseline 0.5775)  "
                  f"frac_acc {m['frac_acc']:.4f} (baseline 0.7283)  headroom {m['headroom']:.4f}")
        return m

    def build_epoch(ep):
        """This epoch's unit sequence FOR THIS RANK: K structures per step, one seed of each.

        Every rank walks the SAME structures in the SAME order and differs only in WHICH seed of
        each it takes. Every seed of a structure crops to an identical token count -- the interface
        crop is defined on the ground-truth interface rather than the predicted pose, and 0 of 488
        structures disagree across their ten seeds -- so the ranks do equal work per step and the
        collective has nothing to wait for.

        THIS REPLACES --shuffle structure PLUS shard_by_structure, which handed each rank ten
        CONSECUTIVE seeds of ONE structure. Simulated on this run's own kept_tok values that order
        leaves a 111% median per-step spread (p90 186%) and spends 38% of the critical path in
        straggler wait -- epoch 6 of 2w5to5 burned 22.6 min of 65.4 in loss_stats_collective. This
        order measures 0.0% and 0.0%, and a step spans K structures rather than `world`, so the
        gradient is built from MORE structures than before, not fewer.

        `world` does not divide ten seeds, so floor(10/world) whole groups are used and the
        remainder -- one seed per structure at world 3 -- sits out. The rng is reseeded per epoch,
        so a different seed sits out each time rather than a fixed tenth of the data never being
        trained on.
        """
        by = {}
        for _p, _lp, _c in flat:
            by.setdefault(_p, []).append((_lp, _c))
        rng_e = random.Random(a.seed + ep)
        order = sorted(by)
        rng_e.shuffle(order)
        # A GROUP IS `world` SEEDS OF ONE STRUCTURE -- the unit of equal work.
        groups = []
        for _p in order:
            u = by[_p][:]
            rng_e.shuffle(u)
            for g in range(len(u) // world):
                groups.append([(_p, _lp, _c) for _lp, _c in u[g * world:(g + 1) * world]])
        rng_e.shuffle(groups)
        # THE REMAINDER IS KEPT, as a short final step. Truncating to a whole number of steps
        # drops units, and for TRAINING that is invisible -- 1,464 groups over K=6 divides
        # exactly -- but --loss-only runs world=1 over 300 units at K=17 and 300//17*17 is 289,
        # so a held-out pass would have measured 1,445 poses instead of 1,500 and the epoch
        # would have been marked incomplete for missing its expected count. The accumulator
        # already flushes a short step with the gradient scaled to the poses that contributed,
        # and on a measurement pass no step is taken at all.
        mine = []
        for i in range(0, len(groups), structs_per_step):
            for gi, grp in enumerate(groups[i:i + structs_per_step]):
                # ROTATED BY POSITION so a rank is not always handed the same seed slot.
                mine.append(grp[(rank + gi) % world])
        return mine

    def build_epoch_cross_seed(ep):
        """This epoch's unit sequence FOR THIS RANK, with ALL SEEDS OF A STRUCTURE IN ONE STEP.

        Returns (mine, quotas, n_units): the flat unit list, this rank's unit count per step, and
        the GLOBAL unit count per step. The step boundary comes from `quotas` rather than from a
        constant, because S structures x their seeds need not divide `world`; n_units is what the
        PAE term's mean has to be taken over, since accum_eff no longer describes the step.

        WHY THE REGROUPING. --rank-cross-seed ranks poses across seeds, and the coefficient for one
        pose depends on every other pose in its interface. So the whole interface has to be scored
        before any backward in the step, which means every seed of the structure must be inside the
        same optimiser step. build_epoch's normal order deliberately spreads a structure's ten
        seeds across ten different steps (one per rank per step), which is the opposite.

        THE STEP STAYS THE SAME SIZE. S=2 structures x 10 seeds x 5 poses = 100 poses, against the
        ~100 the normal schedule already uses (K=6 structures x 3 ranks x 5 poses). What changes is
        the SHAPE: 2 structures with all their seeds instead of ~6 structures with 3 seeds each. So
        the ranking signal becomes exact (the full ~1,225 cross-seed pairs of each interface rather
        than a 75-pair sample, measured cosine 1.000 vs 0.856) at the cost of each step's PAE
        gradient seeing a third as many structures.

        NO SEED SITS OUT. build_epoch chops seeds into whole groups of `world` and the remainder
        -- one seed per structure at world 3 -- does not train that epoch. Here a group is ALL the
        seeds, so every pose is used every epoch.

        THE SHORT RANK ROTATES. 20 units over 3 ranks is 7/7/6, and a rank that takes fewer units
        in every step would reach the end of the epoch having taken a different number of steps --
        which does not fail, it HANGS on the first mismatched all-reduce. So the extra units go to
        ranks (k + 0 .. k + rem - 1) mod world at step k, which gives every rank the same total
        whenever the step count is a multiple of `world`. When it is not, per-rank unit counts
        differ and the existing cross-rank count check refuses to start -- loudly, rather than
        deadlocking. UNITS ARE SHUFFLED WITHIN THE STEP so each rank gets a mix of the step's
        structures rather than one structure's seeds, which is what keeps per-rank token counts
        comparable; every seed of a structure crops to the same token count, so a balanced mix of
        structures is balanced work.
        """
        by = {}
        for _p, _lp, _c in flat:
            by.setdefault(_p, []).append((_lp, _c))
        rng_e = random.Random(a.seed + ep)
        order = sorted(by)
        rng_e.shuffle(order)
        S = max(1, int(a.rank_structs_per_step))
        steps = []
        for i in range(0, len(order), S):
            chunk = order[i:i + S]
            if len(chunk) < S:
                # A SHORT FINAL STEP IS DROPPED, not padded. Padding would repeat poses inside one
                # gradient, and a step holding fewer structures than S is still correct -- but the
                # unit count would differ from every other step and the quota bookkeeping that
                # keeps the ranks in lockstep is what matters more here than the last few units.
                break
            units = []
            for _p in chunk:
                u = by[_p][:]
                rng_e.shuffle(u)
                units += [(_p, _lp, _c) for _lp, _c in u]
            rng_e.shuffle(units)
            steps.append(units)
        if not steps:
            raise SystemExit(
                f"STOP: --rank-cross-seed with --rank-structs-per-step {S} needs at least {S} "
                f"structure(s); the manifest resolved {len(order)}.")
        n_units = len(steps[0])
        base, rem = divmod(n_units, world)
        mine, quotas = [], []
        for k, units in enumerate(steps):
            counts = [base + (1 if ((r - k) % world) < rem else 0) for r in range(world)]
            start = sum(counts[:rank])
            mine += units[start:start + counts[rank]]
            quotas.append(counts[rank])
        return mine, quotas, n_units

    def train_loop(steps_cap, lr, tag, write_traj, from_epoch=0, from_step=0):
        for g in opt.param_groups:
            g["lr"] = lr
        rows, t0, step, n_in_accum = [], time.time(), from_step, 0
        ep_loss = {"sum": 0.0, "n": 0}
        stop_state = {"best": float("inf"), "bad": 0}
        opt.zero_grad(set_to_none=True)
        # LOCAL SUMS OVER THE WINDOW, reduced once per step. Summed rather than averaged so the
        # reduction is exact regardless of how unevenly the ranks are filled.
        batch_sum = {k: 0.0 for k in TERMS}

        def _batch_means():
            """Mean of each loss term over EVERY pose in this step, across all ranks, and the count.

            ONE COLLECTIVE for the terms and the count together: len(TERMS)+1 floats, against the
            79 MB gradient buffer already reduced every step, so the cost is not measurable. The
            count is reduced rather than assumed to be n_local * world -- the shards are equal
            length so it should be, but a number the curve depends on should not be taken on trust.

            Resets the accumulator, so it must be called exactly once per step.
            """
            nonlocal n_in_accum
            vals = [batch_sum[k] for k in TERMS] + [float(n_in_accum)]
            if world > 1:
                t = torch.tensor(vals, dtype=torch.float64)
                if a.dist_backend == "gloo":
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                else:
                    t = t.to(dev)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    t = t.cpu()
                vals = [float(x) for x in t]
            n_tot = int(round(vals[-1]))
            # THE EPOCH'S LOSS, ACCUMULATED HERE because `vals` is already reduced over every rank
            # and is therefore bit-identical on all of them. Summing it anywhere rank-local would
            # give each rank its own epoch mean, and an early-stop decision that differs between
            # ranks leaves some in the next epoch and the rest at a barrier.
            ep_loss["sum"] += vals[TERMS.index("loss")]
            ep_loss["n"] += n_tot
            for k in batch_sum:
                batch_sum[k] = 0.0
            n_in_accum = 0
            means = {"batch_" + k: (vals[i] / n_tot if n_tot else None)
                     for i, k in enumerate(TERMS)}
            means["batch_n"] = n_tot
            return means, n_tot

        def _apply_partial(ctx=None):
            ctx = ctx or {}
            """Step on FEWER than accum_local poses, scaled by the poses that actually contributed.

            Every pose divided its loss by accum_eff, so a partial accumulation holds sum/accum_eff:
            a gradient scaled DOWN by n_contributing/accum_eff, not a mean over the poses present.
            The old code stepped on that directly, making the last step of an epoch a fraction of
            the size of every other step -- 1/256th of one, if a single pose was left. Multiplying by
            accum_eff/n_contributing restores the mean.

            The count is reduced across ranks rather than assumed to be n_local * world: the shards
            are equal length so it should be, but a step whose size depends on that assumption
            should not take it on trust.

            It also respects _apply_step's success flag. The old code incremented `step` whether or
            not a step was taken, so a non-finite gradient here advanced the step counter anyway.
            """
            nonlocal step
            # The count comes back from the same collective that reduces the loss sums, so the
            # partial step is logged with the same numbers it was taken on.
            means, n_tot = _batch_means()
            if n_tot <= 0:
                return
            scale = _step_poses_g[0] / n_tot
            tset = set(trainable)
            for _n, pp in model.named_parameters():
                if _n in tset and pp.grad is not None:
                    pp.grad.mul_(scale)
            print(f"  flushing a partial accumulation of {n_tot}/{_step_poses_g[0]} pose(s), gradient "
                  f"scaled by {scale:.4f} so the step is the same size as every other")
            _gn, _outside, ok = _apply_step(step)
            # LOGGED, because this step used to leave no row at all: the reader then spliced the
            # epoch's trailing poses into the next epoch's first window and dropped the run's final
            # remainder. partial_step marks it; batch_n says how many poses it covered.
            rows.append({"wall_s": round(time.time() - t0, 1), "step": step,
                         "unit_i": ctx.get("unit_i", -1), "pdb": ctx.get("pdb", ""),
                         "unit": ctx.get("unit", ""), "pose": ctx.get("pose", -1),
                         "lr": ctx.get("lr", float("nan")), "grad_norm": _gn,
                         "outside_head": _outside, "epoch": ctx.get("epoch", 0),
                         "step_taken": 1 if ok else 0, "nonfinite_grad": 0 if ok else 1,
                         "partial_step": 1, "step_ok": 1 if ok else 0, **means})
            if ok:
                step += 1
        def _save_resume(epochs_done, step_now):
            """Everything needed to continue: the head, AdamW's moments, the step, and a guard.

            UNDER --loss-only THIS WRITES NOTHING. AdamW never stepped, so the moments are the
            resumed bundle's or empty, and a bundle carrying them under a new epoch number would
            be a forgery of training that did not happen.

            THE OPTIMISER STATE IS THE POINT. AdamW carries a per-parameter exp_avg and exp_avg_sq,
            which is what makes its steps adaptive; restoring weights alone restarts those from zero
            and the first steps after a resume are effectively un-adapted. It roughly triples the
            file -- 76 MiB of weights, ~158 MiB of moments -- which is nothing against the 1.4 TB of
            trunks this run reads.

            The guard records what determines batch composition, so a resume into different settings
            is refused rather than silently training with moments that belong elsewhere.
            """
            if a.loss_only:
                return
            p = os.path.join(a.out, f"resume_ep{epochs_done}.pt")
            tmp = p + ".tmp"
            torch.save({"head": {n: pp.detach().cpu() for n, pp in model.named_parameters()
                                 if n in set(trainable)},
                        "opt": opt.state_dict(),
                        "epoch": epochs_done, "step": step_now,
                        "guard": {"drift_history": _drift_history,
                                  "lora_rank": a.lora_rank, "lora_alpha": a.lora_alpha,
                                  "lora_targets": a.lora_targets,
                                  "lora_dropout": a.lora_dropout,
                                  "train_only": a.train_only,
                                  "world": world, "seed": a.seed, "accum": a.accum,
                                  "nsample": a.nsample, "lr": a.lr,
                                  "lr_final": a.lr_final,
                                  "warmup_steps": a.warmup_steps, "epochs": a.epochs,
                                  "manifest_skip": a.manifest_skip,
                                  "manifest_digest": _manifest_digest,
                                  "crop_size": a.crop_size,
                                  "alpha_confidence": a.alpha_confidence,
                                  "pae_scored_pairs": a.pae_scored_pairs,
                                  "pae_only": a.pae_only,
                                  # SAVED, NOT ONLY COMPARED. frame_direction went into the
                                  # comparison side but not here, so a bundle carried no record of
                                  # it: a resume beside its args.json still matched, because the
                                  # guard falls back to that file, but a bundle moved on its own
                                  # fell through to the pre-crop default and read as C DISABLED.
                                  # Third time a guard field has been half-wired. Found in review.
                                  "frame_direction": a.frame_direction,
                                  "pose_perblock": a.pose_perblock,
                                  "pose_perblock_tied": a.pose_perblock_tied,
                                  # FOURTH TIME, five fields at once. pose_perblock_share, l2_sp
                                  # and the three ranking fields were all in the comparison side
                                  # and none of them here, so a bundle moved without its args.json
                                  # carried no record of the architecture OR the objective. Found
                                  # in review of 04503a9.
                                  "pose_perblock_share": a.pose_perblock_share,
                                  "pose_perblock_copy": a.pose_perblock_copy,
                                  "pose_perblock_copy": a.pose_perblock_copy,
                                  "l2_sp": a.l2_sp,
                                  "l2_sp_exclude": a.l2_sp_exclude,
                                  "rank_loss": a.rank_loss, "alpha_rank": a.alpha_rank,
                                  "rank_weight_by_gap": a.rank_weight_by_gap,
                                  "rank_form": a.rank_form,
                                  "rank_tau": (float(a.rank_tau)
                                               if a.rank_tau is not None else -1.0),
                                  "rank_cross_seed": a.rank_cross_seed,
                                  "rank_structs_per_step": a.rank_structs_per_step,
                                  # FIFTH TIME. rank_pairs, rank_within_weight and rank_topk_decay
                                  # went into the comparison side with a comment claiming both
                                  # sides were edited together; they were not, and
                                  # test_guard_fields.py said so. Found in review.
                                  "rank_pairs": a.rank_pairs,
                                  "rank_within_weight": a.rank_within_weight,
                                  "rank_topk_decay": a.rank_topk_decay,
                                  "rank_straddle": a.rank_straddle,
                                  "alpha_pae": a.alpha_pae, "drop_pde": a.drop_pde,
                                  "clip_norm": a.clip_norm,
                                  "crop_margin": a.crop_margin,
                                  "crop_contact": a.crop_contact,
                                  "commit": _git_commit(), "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}},
                       tmp)
            # ATOMIC: a bundle half-written by a crash at the epoch boundary would be the one thing
            # a reader reaches for, so it is renamed into place only once it is complete.
            os.replace(tmp, p)
            print(f"  saved {p} ({os.path.getsize(p)/2**20:.0f} MiB: head + AdamW moments + step)",
                  flush=True)

        # AN EPOCH LOOP, BECAUSE --epochs USED TO DO NOTHING. It scaled steps_cap while the loop
        # traversed `flat` exactly once, so a request for more steps than the data provides simply
        # stopped early -- review round 7 asked for 5 optimiser steps and got 2. The data is now
        # re-traversed, reshuffled each epoch so the order is not identical, until the step cap or
        # the epoch count is reached.
        n_epochs = max(1, math.ceil(a.epochs))
        done = False
        for ep in range(from_epoch, n_epochs):
            if done:
                break
            # THE EPOCH ORDER, AND WITH --rank-cross-seed ITS STEP SHAPE TOO. The cross-seed
            # builder returns a per-step unit quota because S structures x their seeds need not
            # divide `world`, so the step boundary cannot be the constant accum_local.
            ep_quotas = None
            if a.rank_cross_seed:
                ep_units, ep_quotas, _xs_nunits = build_epoch_cross_seed(ep)
                # EVERY STEP HOLDS THE SAME TOTAL unit count across ranks -- only the per-rank
                # split rotates -- so the global pose count per step is a constant.
                _step_poses_g[0] = _xs_nunits * a.nsample
            else:
                ep_units = build_epoch(ep)
            if world > 1:
                # THE RANKS MUST AGREE ON THE UNIT COUNT. If they do not, they take a different
                # number of optimiser steps and the first mismatched all-reduce hangs forever
                # rather than failing, so it is checked here where it can still be reported.
                _t = torch.tensor([len(ep_units)], device=dev)
                _mx, _mn = _t.clone(), _t.clone()
                dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
                dist.all_reduce(_mn, op=dist.ReduceOp.MIN)
                if int(_mx) != int(_mn):
                    raise SystemExit(f"STOP: rank unit counts disagree ({int(_mn)}..{int(_mx)}). "
                                     f"The ranks would take different numbers of steps and deadlock "
                                     f"on the first all-reduce.")
            # a fractional last epoch takes that fraction of the units
            if ep == n_epochs - 1 and a.epochs < n_epochs:
                frac = a.epochs - (n_epochs - 1)
                ep_units = ep_units[:max(1, int(len(ep_units) * frac))]
            print(f"  epoch {ep + 1}/{n_epochs}: {len(ep_units)} unit(s)")
            # STARTED PER EPOCH, over THIS epoch's shuffled order, so the worker walks the units
            # in the order the main loop will consume them.
            _pf_start(ep_units)
            _T0, _TN0 = dict(_T), dict(_TN)
            # PEAK MEMORY PER EPOCH, reset here so each epoch reports its own rather than the
            # high-water mark of the whole run. This is the number that decides whether activation
            # checkpointing can stay off: --mem-log answers it too but emits one line per pose,
            # 8,100 per rank per epoch.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ep_start_step = step
            ep_t0 = time.time()
            ep_loss["sum"], ep_loss["n"] = 0.0, 0
            # UNIT INDEX -> (step index, the step's whole unit list), so pass A can score a
            # step's poses before the first backward of that step. Built once per epoch.
            _xs_steps, _xs_at = [], {}
            if ep_quotas:
                _o = 0
                for _k, _q in enumerate(ep_quotas):
                    if _o >= len(ep_units):
                        break
                    _xs_steps.append(ep_units[_o:_o + _q])
                    _xs_at[_o] = _k
                    _o += _q
                if _o != len(ep_units):
                    raise SystemExit(
                        f"STOP: the cross-seed quotas sum to {_o} unit(s) but the epoch holds "
                        f"{len(ep_units)}. The step boundary would drift from the quota and the "
                        f"ranks would step at different times.")
            for ui, (pdb, lab_p, cands) in enumerate(ep_units):
                if ui in _xs_at:
                    _step_poses[0] = len(_xs_steps[_xs_at[ui]]) * a.nsample
                    _xs_prepare(_xs_steps[_xs_at[ui]], ep)
                trunk_p = resolve_trunk(pdb, lab_p, cands)
                # THE NEXT UNIT'S TRUNK, WARMED WHILE THIS ONE COMPUTES. Resolving it is cheap
                # after epoch 1 because the fingerprint is cached; on the first epoch it pays the
                # hash it would have paid a moment later anyway.
                if ui + 1 < len(ep_units):
                    _n_pdb, _n_lab, _n_cands = ep_units[ui + 1]
                    _t_pf = time.perf_counter()
                    # EXCLUSIVE OF THE HASH NESTED INSIDE IT. resolve_trunk computes
                    # unit_fingerprint, which accumulates into fingerprint_hash -- and because the
                    # prefetch reaches unit i+1 before the main loop does, it pays EVERY first-touch
                    # hash. Epoch 1 of x3k2m4 therefore reported fingerprint_hash 8.1m and
                    # prefetch_issue 8.1m as if they were separate costs when they are the same
                    # 8.1 minutes; across ranks the pairs were 8.1/8.1, 7.6/7.6, 8.5/8.6. The
                    # budget subtracted it twice and understated the residual by ~8 min.
                    _h0 = _T["fingerprint_hash"]
                    try:
                        _prefetch(resolve_trunk(_n_pdb, _n_lab, _n_cands))
                    except SystemExit:
                        # A BAD NEXT UNIT MUST NOT KILL THE CURRENT ONE HERE. resolve_trunk raises
                        # SystemExit on an unresolvable fingerprint, which is the right behaviour
                        # when that unit is reached for real -- one iteration later, with its own
                        # diagnostics. Swallowing it here only declines to prefetch.
                        pass
                    _tick("prefetch_issue", _t_pf)
                    _T["prefetch_issue"] -= _T["fingerprint_hash"] - _h0
                for _m in run_unit(pdb, lab_p, trunk_p):
                    i, met = _m["pose"], _m      # floats only; the tensors are already freed
                    cur_lr = lr * min(1.0, (step + 1) / max(1, a.warmup_steps))
                    # COSINE AFTER WARMUP. Driven by `step`, which resume restores, so a resumed
                    # run picks up the schedule where it left off instead of restarting the decay.
                    if a.lr_final > 0.0 and _lr_horizon[0] > a.warmup_steps and step >= a.warmup_steps:
                        _p = min(1.0, (step - a.warmup_steps)
                                 / float(_lr_horizon[0] - a.warmup_steps))
                        cur_lr = a.lr_final + 0.5 * (lr - a.lr_final) * (1.0 + math.cos(math.pi * _p))
                    for g in opt.param_groups:
                        g["lr"] = cur_lr
                    n_in_accum += 1
                    # EVERY POSE CONTRIBUTES TO THE WINDOW SUM, which is what the batch columns
                    # average. The per-pose TERMS stay in the row too, for provenance.
                    for _k in TERMS:
                        batch_sum[_k] += float(met[_k])
                    gn, outside, took, bad_grad = float("nan"), -1, 0, 0
                    means = {"batch_" + _k: None for _k in TERMS}
                    means["batch_n"] = 0
                    if n_in_accum >= _step_poses[0]:
                        # REDUCED BEFORE THE STEP IS TAKEN, so the numbers logged are the numbers
                        # the step was taken on even if the step is then refused. _batch_means
                        # resets the accumulator and n_in_accum, so neither is reset again below.
                        # TIMED SEPARATELY, because it is the step's FIRST collective and it was
                        # invisible. _batch_means all_reduces the loss statistics before
                        # _apply_step touches gradients, so an early-arriving rank blocks HERE, not
                        # at the gradient reduction -- and none of that wait appeared in
                        # grad_collective_plus_wait. That is why 0.6 min of 77.9 could not be read
                        # as "synchronisation is cheap": the expensive-looking collective was timed
                        # and the one that absorbs the straggler was not.
                        _t_bm = time.perf_counter()
                        means, _n_tot = _batch_means()
                        _tick("loss_stats_collective_plus_wait", _t_bm)
                        _t_as = time.perf_counter()
                        gn, outside, ok = _apply_step(step)
                        _tick("apply_step_total", _t_as)
                        took = 1
                        if not ok:
                            bad_grad = 1
                            rows.append({"wall_s": round(time.time() - t0, 1), "step": step,
                                         "unit_i": ui, "pdb": pdb,
                                         "unit": os.path.basename(lab_p)[:-4], "pose": i,
                                         "lr": cur_lr, "grad_norm": gn, "outside_head": outside,
                                         "epoch": ep + 1, "step_taken": 0, "nonfinite_grad": 1,
                                         "partial_step": 0, "step_ok": 0,
                                         **{k: met[k] for k in TERMS}, **means})
                            continue
                        step += 1
                        if a.save_every and step % a.save_every == 0 and write_traj:
                            cp = os.path.join(a.out, f"head_step{step}.pt")
                            torch.save(_head_ckpt_tensors(), cp)
                            print(f"  saved {cp}")
                    rows.append({"wall_s": round(time.time() - t0, 1), "step": step, "unit_i": ui,
                                 "pdb": pdb, "unit": os.path.basename(lab_p)[:-4], "pose": i,
                                 "lr": cur_lr, "grad_norm": gn, "outside_head": outside,
                                 "epoch": ep + 1, "step_taken": took,
                                 "nonfinite_grad": bad_grad,
                                 "partial_step": 0, "step_ok": 1 if took else 0,
                                 **{k: met[k] for k in TERMS}, **means})
                    if write_traj and len(rows) % 20 == 0:
                        _flush(traj_p, traj_cols, rows); rows = []
                    # THE CROP LOG FLUSHES ON EVERY RANK, and with the trajectory rather than only
                    # at the epoch boundary: a run that dies mid-epoch -- as this one did, on a
                    # hangup -- would otherwise leave no record of what it had cropped. Outside the
                    # write_traj branch because crop_p is per-rank and every rank owns its file.
                    if crop_p and len(crop_rows) >= 20:
                        _flush(crop_p, crop_cols, crop_rows); crop_rows.clear()
                    if steps_cap and step >= steps_cap:
                        done = True
                        break
                # --grad-probe: one unit done. Report after N of them and stop; this mode exists to
                # measure the gradient scale, not to train.
                if a.grad_probe:
                    _probe["units"] += 1
                    # EXACTLY ONE WINDOW. accum_local poses close the window and the optimiser
                    # steps; one unit further and the probe is summing g(theta_0) with
                    # g(theta_1), which is not the gradient of any single step. Found in review
                    # of 523013a, where the launcher asked for 8 units against a 6-unit window.
                    # THE WINDOW IS THE STEP, AND THE STEP IS NO LONGER A CONSTANT. accum_local
                    # // nsample is the unit count per step only when every step holds the same
                    # number; --rank-cross-seed rotates a 7/7/6 split, so the window has to come
                    # from _step_poses[0], which the epoch loop sets per step. Measuring one unit
                    # past the boundary sums g(theta_0) with g(theta_1), which is the gradient of
                    # no single step -- the defect review found in 523013a.
                    if _probe["units"] >= max(1, _step_poses[0] // max(1, a.nsample)):
                        # RAISES on failure, so the launcher's exit status carries the verdict.
                        _grad_probe_report(_probe, trainable, a, step > 0, accum_local, world)
                        _probe["done"] = True
                        return
                if done:
                    break
            # HOW LONG THE EPOCH TOOK, logged rather than left to be derived. Timing has been
            # estimated wrong four times on this pilot, every time from a median or a short probe
            # treated as a mean; an epoch that reports its own duration cannot be misquoted.
            _ep_min = (time.time() - ep_t0) / 60.0
            _ep_poses = step - ep_start_step
            print(f"  epoch {ep + 1} took {_ep_min:.1f} min for {len(ep_units)} unit(s), "
                  f"{_ep_poses} optimiser step(s)"
                  + (f", {_ep_min * 60 / max(1, len(ep_units) * a.nsample):.2f} s/pose"
                     if ep_units else ""))

            # THE PARTIAL ACCUMULATION IS FLUSHED HERE, BEFORE THE EARLY-STOP CHECK AND BEFORE
            # THE EVALUATION, SO THE HEAD THAT IS SCORED IS THE HEAD THAT IS SAVED AND THE EPOCH
            # LOSS COVERS EVERY POSE. The flush used to sit BELOW the stop check, so `ep_loss`
            # was read with the epoch's last partial step still unaccumulated: the rule averaged
            # 1,428 of 1,500 poses and never saw the remaining 72. That step is a handful of
            # structures and its loss ranged 1.21-2.78 across krah8z's 29 epochs, against ~1.99
            # for the epoch, so the two differed by up to 0.04 and disagreed in SIGN: epoch 29
            # read 1.9935 (up, and the eighth strike that ended the run) where the full epoch was
            # 1.9558, its lowest. The stop rule and the plotted curve were not the same number.
            #
            # Before that it sat after the EPOCH LOOP, which is after the last evaluation: at
            # production settings the run evaluated step 292, then took step 293, then wrote
            # head_final.pt -- so the shipped head was one optimiser step away from the only head
            # whose validation metric we had, and the number reported for the run did not belong to
            # the checkpoint the run produced.
            if n_in_accum:
                _apply_partial({"epoch": ep + 1, "lr": cur_lr, "pdb": pdb, "unit_i": ui,
                                "unit": os.path.basename(lab_p)[:-4], "pose": i})

            if a.early_stop_patience and ep_loss["n"]:
                _ep_loss = ep_loss["sum"] / ep_loss["n"]
                if _ep_loss < stop_state["best"] - a.early_stop_min_delta:
                    stop_state["best"], stop_state["bad"] = _ep_loss, 0
                else:
                    stop_state["bad"] += 1
                    # AGAINST THE BEST SO FAR, not the previous epoch: this loss rises on
                    # individual epochs while still trending down, so a previous-epoch comparison
                    # would stop a run that is still learning.
                    print(f"  early stop: epoch {ep + 1} loss {_ep_loss:.4f} did not beat "
                          f"{stop_state['best']:.4f} by {a.early_stop_min_delta:g} "
                          f"({stop_state['bad']}/{a.early_stop_patience} without progress)")
                if stop_state["bad"] >= a.early_stop_patience:
                    print(f"\nEARLY STOP after epoch {ep + 1}: {stop_state['bad']} consecutive "
                          f"epoch(s) failed to beat the best training loss "
                          f"({stop_state['best']:.4f}) by {a.early_stop_min_delta:g}. The epoch "
                          f"below still evaluates and still writes its resume bundle, so this "
                          f"stops the run without discarding the epoch that triggered it.")
                    done = True

            # THE RESUME BUNDLE, WRITTEN BEFORE THE EVALUATION. The evaluator is a subprocess that
            # does not touch these weights, so either side would be correct -- but writing first
            # means a crash IN the evaluator still leaves this epoch resumable, and the evaluator
            # has crashed before (OST chain mapping, a card with no free memory).
            # WHERE THIS EPOCH'S WALL CLOCK WENT, per rank, as deltas over the epoch rather than
            # run totals -- the first-touch costs (the per-unit fingerprint hash, the item cache)
            # land entirely in epoch 1 and would otherwise contaminate every later epoch's figure.
            _pf_end()
            # ANY EVENTS LEFT FROM THE EPOCH'S TAIL, so the last partial accumulation window's
            # GPU time is not carried silently into the next epoch's figures.
            _drain_gpu_spans()
            _ep_s = time.time() - ep_t0
            # THE PREFETCH WAIT IS A REAL BUCKET AND WAS IN NO BUCKET AT ALL. _pf_take is called
            # between the label_open tick and the start of crop_slice, so no timer was open across
            # it and its minutes landed only in the unexplained residual -- which is why, on this
            # arm, "residual" and "prefetch wait" happened to be the same 6.7 min and could be
            # mistaken for two separate costs. Publishing it means the residual means the same
            # thing in every arm instead of only in the ones without a pass A.
            _T["prefetch_wait"] = _pf_stat["wait_s"]
            _TN["prefetch_wait"] = _pf_stat["hit"] + _pf_stat["miss"]
            _parts = [(k, _T[k] - _T0.get(k, 0.0), _TN[k] - _TN0.get(k, 0)) for k in sorted(_T)]
            # cropped_tokens is a WORKLOAD counter, not a duration: reported as a mean so the
            # per-rank imbalance is readable directly. The pair track is quadratic in tokens, so
            # equal unit counts across ranks still means unequal compute.
            _shown = "  ".join(
                (f"{k} mean {v/c:.0f}" if k == "cropped_tokens" else f"{k} {v/60:.1f}m/{c}")
                for k, v, c in _parts if c)
            # THE RESIDUAL, COMPUTED HERE RATHER THAN BY HAND. The buckets below are mutually
            # exclusive and sum to accounted time; gpu_forward/gpu_backward are NOT included
            # because they are CUDA-event spans that overlap pose_total rather than adding to it.
            # Printing the residual stops a reader summing the wrong subset -- which is how an
            # ~8 min double-count of the fingerprint hash survived into a committed document.
            # EVERY EXCLUSIVE HOST BUCKET, so the printed residual is actually unexplained time.
            # It omitted prefetch_wait and grad_collective_plus_wait, and the three pass-A buckets
            # that only ranking arms have -- so the same residual meant 6.7 min of prefetch wait
            # on j583i8 and 10-12 min of pass A on zg5fnw and 8fkrsu, with nothing to say which.
            # cpucrop_* are deliberately NOT here: they are a breakdown of work already counted by
            # crop_slice and by the worker, so adding them would double count.
            _EXCL = ("pose_total", "crop_slice", "fingerprint_hash", "prefetch_issue",
                     "trunk_read_plus_crop_tokens", "label_open", "loss_feature_prep",
                     "loss_stats_collective_plus_wait", "apply_step_total",
                     "prefetch_wait", "grad_collective_plus_wait",
                     "rank_pass_a", "rank_gather", "pose_a_total")
            _acc = sum(v for k, v, c in _parts if k in _EXCL and c)
            print(f"  [rank {rank}] epoch {ep + 1} took {_ep_s/60:.1f} min -- {_shown}", flush=True)
            # DID THE PREFETCH ACTUALLY FIRE? crop_slice alone cannot answer it: a miss falls back
            # to the inline crop and ticks the same counter, which is exactly how the first
            # version's 100% miss rate looked identical to no prefetch at all.
            if a.prefetch_units:
                _pf_n = _pf_stat["hit"] + _pf_stat["miss"]
                print(f"  [rank {rank}] epoch {ep + 1} prefetch "
                      f"({max(1, a.prefetch_workers)} worker(s)): {_pf_stat['hit']}/{_pf_n} hit "
                      f"({100.0 * _pf_stat['hit'] / max(1, _pf_n):.1f}%), "
                      f"{_pf_stat['wait_s']/60:.1f} min waiting on the worker", flush=True)
            # THE DRIFT IS THE EXPERIMENT under --l2-sp, so it is reported, not inferred. The
            # full head reached ||dW||^2 = 67,052 unconstrained over 30 epochs; this says where
            # the penalised run actually sits.
            if a.l2_sp and _l2sp_drift["n"]:
                print(f"  [rank {rank}] epoch {ep + 1} l2-sp: ||W-W0||^2 = "
                      f"{_l2sp_drift['sq']:.1f}, penalty = {a.l2_sp * _l2sp_drift['sq']:.4f} "
                      f"(unconstrained full head reached 67,052 by epoch 30)", flush=True)
            # THE RANKING LOSS IS THE EXPERIMENT under --rank-loss. ZERO is the reference, not
            # ln2: this loss is a normalised covariance between score gaps and DockQ gaps, so
            # equal scores give exactly 0 and so does an ordering uncorrelated with DockQ.
            # NEGATIVE is correctly aligned, positive is inverted. ln2 = 0.6931 was the reference
            # for the softplus version and was left here when the loss went linear -- caught in
            # review of cd8e6ab7, before it misled a single training run.
            #
            # AND IT DOES NOT MEASURE INVERSIONS. c_i depends only on DockQ, so a pose already
            # ranked above everything still gets pushed up: the loss can fall by widening gaps
            # that were already correct, with the inverted fraction unchanged. The within-seed
            # inversion rate has to be read separately -- see run_logging/rank_rho.py.
            if a.rank_loss and a.rank_cross_seed and _xs_stat["groups"]:
                # THE SOFTPLUS LOSS IS NOT sum_i c_i s_i, so it cannot be reconstructed from the
                # coefficients the way the linear one is. group_coeffs returns the value in pass A
                # and it is accumulated there; 0 is the floor (every pair ordered), not "no
                # alignment".
                # THE PAIR FAMILY IS a.rank_pairs, NOT the string "cross-seed". That was
                # hardcoded here from a time when every arm formed cross-seed pairs, so x5frz6 --
                # which forms WITHIN-SEED pairs only -- printed "cross-seed" at every epoch while
                # the startup banner said the opposite. The pair count was the only thing that
                # gave it away: 6,656 against the ~86,600 a cross-seed objective would form.
                print(f"  [rank {rank}] epoch {ep + 1} rank-loss ({a.rank_form}, "
                      f"{a.rank_pairs}-seed pairs): "
                      f"{_xs_stat['loss'] / _xs_stat['groups']:+.5f} over "
                      f"{_xs_stat['groups']} interface-group(s) in {_xs_stat['steps']} step(s)"
                      + (f" [{_xs_stat['skipped']} step(s) SKIPPED: no straddling pair in either "
                         f"structure, so no task gradient -- not handed to AdamW]"
                         if _xs_stat.get("skipped") else "") + ", "
                      # THE COUNT IS POSES FOR THE EXPECTED FORM, NOT PAIRS, and its floor is
                      # -max(DockQ) rather than 0. Calling 150 poses "150 pair(s)" and claiming a
                      # zero floor would misreport the objective in the one line most likely to
                      # be read back later.
                      f"{_xs_stat['pairs']} "
                      + ("pose(s)" if a.rank_form == "expected" else "pair(s)")
                      + ("  (0 is the FLOOR -- every pair ordered)"
                         if a.rank_form == "softplus" else "")
                      + ("  (NEGATIVE: this is -expected DockQ, floor -max(DockQ) per group)"
                         if a.rank_form == "expected" else ""), flush=True)
                for _k in ("groups", "pairs", "steps", "skipped"):
                    _xs_stat[_k] = 0
                _xs_stat["loss"] = 0.0
            elif a.rank_loss and _rank_seen["units"]:
                _rk_mean = _rank_seen["loss"] / _rank_seen["units"]
                print(f"  [rank {rank}] epoch {ep + 1} rank-loss: {_rk_mean:+.5f} over "
                      f"{_rank_seen['units']} unit(s) (0 = no alignment, negative is better; "
                      f"NOT an inversion rate)", flush=True)
                _rank_seen["loss"] = 0.0
                _rank_seen["units"] = 0
            print(f"  [rank {rank}] epoch {ep + 1} accounted {_acc/60:.1f}m of {_ep_s/60:.1f}m, "
                  f"residual {(_ep_s - _acc)/60:.1f}m "
                  f"(exclusive buckets only; gpu_* overlap pose_total)", flush=True)
            if torch.cuda.is_available():
                _gb = 1 << 30
                print(f"  [rank {rank}] epoch {ep + 1} memory: peak allocated "
                      f"{torch.cuda.max_memory_allocated()/_gb:.2f} GB, peak reserved "
                      f"{torch.cuda.max_memory_reserved()/_gb:.2f} GB, "
                      f"blocks_per_ckpt={model.confidence_head.pairformer_stack.blocks_per_ckpt}",
                      flush=True)
            # EVERY RANK SAVES ITS OWN, at the boundary rather than at exit: x3k2m4 died after
            # epoch 1 and would have discarded all 4,880 hashes it had just computed.
            _fp_cache_save()
            # HAND THE ALLOCATOR'S CACHE BACK BEFORE THE VAL HOOKS RUN. The ranks are parked at a
            # barrier for the held-out pass and the DockQ eval, not computing -- but PyTorch holds
            # freed activation blocks as *reserved*, so a parked rank still looks ~20-30 GB wide to
            # the subprocesses that have to fit beside it. Measured at the p100 crop: allocated
            # 2.28 GB against reserved 30.66, so ~28 GB per card is cached-but-free.
            #
            # THIS MATTERS MORE WITHOUT ACTIVATION CHECKPOINTING, where the stored activations push
            # the peak up and therefore the reserved pool with it. Costs a few hundred ms of
            # re-allocation on the next epoch's first poses, at an epoch boundary, once per epoch.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_main and write_traj:
                _save_resume(ep + 1, step)
            # NO write_traj IN THIS CONDITION. It is `is_main`, so including it meant ranks 1
            # and 2 skipped the whole block INCLUDING both barriers, while rank 0 entered them.
            # dist.barrier() is untagged, so rank 0's two calls paired with the other ranks' two
            # train-eval calls and rank 0 then blocked forever on its third -- a deadlock at the
            # first epoch boundary, not a dropped measurement. Every rank enters the barriers;
            # only rank 0 does the work inside, exactly as the two evaluation paths below.
            if (a.val_loss_every and (ep + 1) % a.val_loss_every == 0
                    and step > ep_start_step):
                # SAME BARRIER DISCIPLINE as the two evaluation paths: one measuring process, the
                # other ranks parked for its duration. Without the second barrier rank 0 re-enters
                # training a whole pass behind the others.
                if world > 1:
                    dist.barrier()
                if is_main:
                    _flush(traj_p, traj_cols, rows); rows.clear()
                    _bundle = os.path.join(a.out, f"resume_ep{ep + 1}.pt")
                    if not os.path.exists(_bundle):
                        print(f"  WARNING: no {_bundle}; skipping the held-out PAE for this epoch")
                    else:
                        # THE WEIGHTS MUST BE UNTOUCHED, ASSERTED NOT ASSUMED. The measuring
                        # process reads a bundle off disk and shares nothing with this one, so it
                        # cannot reach these tensors -- but "cannot" is the kind of claim that
                        # stops being true after an edit, and a monitor that silently perturbed
                        # the run it monitors would invalidate every number after it.
                        _before = hashlib.sha256()
                        for _n2 in trainable:
                            _before.update(dict(model.named_parameters())[_n2]
                                           .detach().cpu().numpy().tobytes())
                        val_loss_checkpoint(ep + 1, _bundle, step)
                        _after = hashlib.sha256()
                        for _n2 in trainable:
                            _after.update(dict(model.named_parameters())[_n2]
                                          .detach().cpu().numpy().tobytes())
                        if _before.digest() != _after.digest():
                            raise SystemExit(
                                f"STOP: the head changed across the epoch-{ep + 1} held-out PAE "
                                f"pass. It runs as a subprocess precisely so it cannot do that, so "
                                f"something now shares state with it and every number after this "
                                f"point is suspect.")
                if world > 1:
                    dist.barrier()
            if a.eval_every_epoch and step > ep_start_step:
                # RANK 0 EVALUATES, THE OTHERS WAIT. The evaluator is a separate process holding
                # its own model, so letting every rank launch one would put three on the cards at
                # once. The barriers keep the ranks together across the ~16 min it takes; without
                # the second one, rank 0 would re-enter training a whole evaluation behind.
                if world > 1:
                    dist.barrier()
                if is_main:
                    _flush(traj_p, traj_cols, rows); rows = []
                    eval_checkpoint(step, ep + 1)
                if world > 1:
                    dist.barrier()
            if (a.train_eval_pool and a.train_eval_every
                    and (ep + 1) % a.train_eval_every == 0 and step > ep_start_step):
                # SAME BARRIER DISCIPLINE as the validation path above: one evaluator, sharded
                # over every card, the other ranks parked for its duration.
                if world > 1:
                    dist.barrier()
                if is_main:
                    _flush(traj_p, traj_cols, rows); rows.clear()
                    eval_checkpoint(step, ep + 1, train_pool=a.train_eval_pool)
                if world > 1:
                    dist.barrier()
        # NOTHING MAY BE LEFT HERE. Every epoch flushes its own remainder above, before evaluating,
        # so a non-zero accumulator at this point would mean a path out of the loop that skipped the
        # flush -- and the saved head would again differ from the evaluated one. Flushed and reported
        # rather than dropped, because dropping poses silently is what the flush was added to stop.
        if n_in_accum:
            print(f"  WARNING: {n_in_accum} pose(s) still accumulated after the epoch loop; the "
                  f"evaluated head is NOT the saved head for this run")
            _apply_partial({"epoch": ep + 1, "lr": cur_lr, "pdb": pdb, "unit_i": ui,
                            "unit": os.path.basename(lab_p)[:-4], "pose": i})
        if write_traj:
            _flush(traj_p, traj_cols, rows)
        if crop_p and crop_rows:        # every rank, its own file
            _flush(crop_p, crop_cols, crop_rows)
            crop_rows.clear()
        return rows

    def _flush(path, cols, rows):
        if not rows:
            return
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            if new:
                wr.writeheader()
            wr.writerows(rows)

    # ---- LR PROBE: measure the learning rate instead of inheriting one tuned for a different job
    if a.lr_probe:
        lrs = [float(x) for x in a.lr_probe.split(",")]
        print(f"\nLR PROBE: {a.probe_steps} optimiser step(s) at each of {lrs}, all from the SAME "
              f"initial weights. Protenix's 0.0018 default is for finetuning the whole model; this "
              f"moves 4.3% of it, so the default is not transferable.")
        out = []
        for lr in lrs:
            for n, p in model.named_parameters():
                if n in set(trainable):
                    p.data.copy_(init_state[n])
            opt.load_state_dict(copy.deepcopy(init_opt))
            rows = train_loop(a.probe_steps, lr, f"probe{lr}", write_traj=False)
            if not rows:
                print(f"  lr {lr:g}: no rows"); continue
            # THE BATCH LOSS ON STEPS ACTUALLY TAKEN, not every per-pose row. `rows` is one entry
            # per POSE on rank 0; averaging those compares learning rates on rank 0's slice of a
            # handful of structures, and review 2026-10-16 built a case where the global batch loss
            # fell while rank 0's pose losses rose, so this recommended LOWERING the lr. The
            # cross-rank batch mean is in the row the optimiser stepped on.
            _st = [r for r in rows if r.get("step_taken") and r.get("step_ok")
                   and r.get("batch_loss") is not None]
            if not _st:
                print(f"  lr {lr:g}: no completed optimiser step to judge; skipped"); continue
            k = max(1, len(_st) // 5)
            first = sum(r["batch_loss"] for r in _st[:k]) / k
            last = sum(r["batch_loss"] for r in _st[-k:]) / k
            fin = all(math.isfinite(r["batch_loss"]) for r in _st)
            out.append((lr, first, last, last - first, fin))
            print(f"  lr {lr:<9g} loss first{k} {first:.6f} -> last{k} {last:.6f} "
                  f"(delta {last-first:+.6f}) finite={fin}")
        with open(os.path.join(a.out, "lr_probe.csv"), "w", newline="") as fh:
            wr = csv.writer(fh, lineterminator="\n")
            wr.writerow(["lr", "loss_first", "loss_last", "delta", "all_finite"])
            wr.writerows(out)
        print(f"\nwrote {os.path.join(a.out, 'lr_probe.csv')}")
        print("A PROBE IS NOT A RESULT. It shows which rates are stable and which diverge over a few "
              "dozen steps; it cannot show which generalises. Pick from it, then judge the run on "
              "analysis/val_selection_metric.py, never on this loss.")
        return 0

    # THE ESTIMATE USES --nsample, not a hardcoded 5. It was 5 because every unit has 5 poses in
    # this pilot, which is a fact about the data and not about the code.
    total_poses = int(len(flat) * a.nsample * a.epochs)
    # PER RANK. Each rank sees total_poses/world poses and steps every accum_local of them, so the
    # step count is total_poses/(world*accum_local) = total_poses/accum_eff either way, and a
    # multi-GPU run does the same number of optimiser steps as a single-GPU one -- in a third of the
    # wall clock, rather than a third of the training.
    # THE CAP IS FOR DEBUGGING ONLY, AND IS OFF FOR REAL TRAINING. total_poses // accum_eff counts
    # full accumulations, but the loop ALSO flushes a partial one at every epoch boundary, so the
    # step counter advances faster than the cap anticipates and the last epoch could be cut short.
    # Review 2026-10-16 reproduced three epochs of ten poses running only 24 of 30. The epoch loop
    # already knows when it is finished; `--limit-units` and the lr probe remain for capped runs.
    # 9mfb4y escaped by one step (cap 481, 480 taken) only because total_poses counts all 491
    # structures while the loop runs the 489 that survive shard truncation -- two errors cancelling.
    _full = (len(flat) // max(1, world)) * a.nsample // accum_local
    _expect = (_full + 1) * max(1, math.ceil(a.epochs))
    steps_cap = 0
    _lr_horizon[0] = _expect
    _sched = ("flat after warmup (no decay)" if a.lr_final <= 0.0 else
              f"COSINE {a.lr:g} -> {a.lr_final:g} over steps {a.warmup_steps}..{_expect}")
    print(f"\ntraining: lr {a.lr:g}, warmup {a.warmup_steps}, accum {a.accum}, "
          f"~{_expect} optimiser step(s) for {a.epochs} epoch(s) "
          f"({_full} full + 1 partial per epoch); no step cap, the epoch loop decides")
    print(f"  lr schedule: {_sched}")
    print(f"trajectory -> {traj_p}")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    train_loop(steps_cap, a.lr, "train", write_traj=is_main,
               from_epoch=start_epoch, from_step=start_step)
    # --grad-probe STOPS HERE. The return inside train_loop only exits train_loop; main() then
    # ran on through the summaries to the unconditional head_final.pt save, so a probe wrote a
    # checkpoint indistinguishable from a trained one WHILE PRINTING that it had not. Found in
    # review of 523013a. It also corrects the claim: the probe DOES take one real optimiser step,
    # by design -- what it must not do is leave a head behind that looks like an arm's.
    if a.grad_probe:
        if is_main:
            print(f"  --grad-probe complete. ONE optimiser step was taken (deliberately) and NO "
                  f"checkpoint is written: {a.out} holds no head_*.pt from this invocation, so it "
                  f"cannot be mistaken for a trained arm.", flush=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    # THE CROP SUMMARY BELONGS AFTER TRAINING. It was written above the train_loop call, so it read
    # every counter at zero and printed nothing at all -- an arm would have reported no retained
    # fraction and no fallback count, which is exactly the bookkeeping it exists to provide.
    # Review round 6 caught it.
    if a.crop_margin > 0:
        if crop_p and crop_rows:
            _flush(crop_p, crop_cols, crop_rows)
            crop_rows.clear()
        # ACROSS EVERY RANK, not just this one. Each rank crops its own third of the units, so a
        # single rank's counters describe a third of the run. One collective, at the very end.
        _keys = ("n", "tok", "kept", "frames", "undefined")
        _cs = dict(crop_stats)
        if world > 1:
            _t = torch.tensor([float(crop_stats[k]) for k in _keys], dtype=torch.float64)
            dist.all_reduce(_t, op=dist.ReduceOp.SUM)
            _cs = {k: int(v) for k, v in zip(_keys, _t.tolist())}
        if is_main:
            if _cs["n"]:
                print(f"\ncrop: contact {a.crop_contact:.0f} A, margin {a.crop_margin:.0f} A over "
                      f"{_cs['n']} unit(s) across {world} rank(s): kept {_cs['kept']}/{_cs['tok']} "
                      f"token(s) ({_cs['kept'] / _cs['tok'] * 100:.1f}%), "
                      f"{_cs['frames']} frame(s) cleared for want of their atoms")
            if _cs["undefined"]:
                print(f"  {_cs['undefined']} unit(s) had NO DEFINABLE CROP -- fewer than two "
                      f"chains with usable coordinates, or no cross-chain contact -- and trained "
                      f"on the whole structure, logged with status=no-crop-definable.")
            if crop_p:
                print(f"crop sizes -> {crop_p} (one file per rank; this is rank {rank}'s)")

    if a.pae_scored_pairs:
        # ACROSS EVERY RANK, same reason as the crop summary: each rank masks its own third of the
        # units, so one rank's counters describe a third of the run. These were being counted and
        # printed NOWHERE, which made a mask that covered nothing indistinguishable in the log from
        # one that covered everything.
        _mk = ("masked", "unmasked")
        _ms = dict(_pae_mask_stats)
        if world > 1:
            _t = torch.tensor([float(_pae_mask_stats[k]) for k in _mk], dtype=torch.float64)
            dist.all_reduce(_t, op=dist.ReduceOp.SUM)
            _ms = {k: int(v) for k, v in zip(_mk, _t.tolist())}
        if is_main:
            _tot = _ms["masked"] + _ms["unmasked"]
            print(f"\nPAE mask: {_ms['masked']}/{_tot} pose-unit(s) masked to scored chain pairs "
                  f"({_ms['masked'] / _tot * 100:.1f}%) across {world} rank(s), from "
                  f"{a.pae_scored_pairs}")
            if _ms["unmasked"]:
                print(f"  {_ms['unmasked']} fell back to the UNMASKED PAE term -- the structure is "
                      f"absent from the pool, or no scored pair survived the crop. Those poses "
                      f"trained the ungraded blocks too.")

    if code_drift["n"]:
        print(f"\nCODE DRIFT ACCEPTED on {code_drift['n']} unit(s) via --allow-code-drift. The "
              f"labels' data provenance matched in every field; only the builder's source hash "
              f"differed. This run is NOT reproducible from a clean checkout of that code.")
    # RANK 0 WRITES THE CHECKPOINT. Every rank holds the same parameters after the all-reduce, so
    # the content would be identical, but two processes writing one path concurrently can interleave
    # and leave a truncated file -- the smoke test showed both ranks reporting the same save.
    cp = os.path.join(a.out, "head_final.pt")
    if a.loss_only:
        # NOTHING TRAINED, SO NOTHING IS SAVED. --save-every is already refused for this mode, but
        # this save is unconditional and wrote head_final.pt on the first --loss-only pass: a file
        # indistinguishable from a trained head, holding the resumed bundle's weights under a name
        # that says otherwise. The measurement's output is the trajectory.
        _src = a.resume or "the shipped head"
        print(f"\n--loss-only: no checkpoint written -- {cp} would duplicate {_src} under a name "
              f"implying it was trained")
    elif is_main or a.debug_save_rank_heads:
        import torch as _t
        _sd = {n: p.detach().cpu() for n, p in model.named_parameters() if n in set(trainable)}
        if is_main:
            _t.save(_sd, cp)
            print(f"\nsaved {cp}")
        if a.debug_save_rank_heads:
            _rp = os.path.join(a.out, f"head_final_rank{rank}.pt")
            _t.save(_sd, _rp)
            print(f"saved {_rp}")
    if a.loss_only:
        print("THE LOSS IS THE RESULT HERE, and only that: this mode measures a frozen head's loss "
              "on the fold it was pointed at. Aggregate the per-pose pae_loss column of "
              f"{os.path.join(a.out, 'trajectory.csv')} -- every pose is measured exactly once, so "
              "the plain mean over all rows is the pose-weighted figure. Do NOT use the "
              "step_taken==1 windows: --accum leaves a tail of poses outside the last complete "
              "window, which a training run re-traverses next epoch and this single pass does not.")
    else:
        # NO HARDCODED TARGET. This printed "the target is mean DockQ 0.5775 and frac-acceptable
        # 0.7283 at 100 seeds" after every run, which are the FULL validation pool's numbers --
        # wrong for any run scored on a subset, and val10's are 0.2626 / 0.5000. The same string
        # was hardcoded in phase_k_eval and is now derived there from the pool actually scored, so
        # the baseline belongs in that output, not in a constant here.
        print("THE LOSS IS NOT THE RESULT. Score this checkpoint with phases/phase_k_eval.py, which "
              "reports the shipped head's baseline over the pool it scored alongside the metric.")
    # Tear the group down so the ranks exit together rather than on NCCL's timeout.
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
