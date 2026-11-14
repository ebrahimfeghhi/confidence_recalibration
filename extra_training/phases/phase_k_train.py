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
    ap.add_argument("--lr-decay-start-epoch", type=float, default=0.0,
                    help="HOLD --lr FLAT UNTIL THIS EPOCH, then begin decaying toward --lr-final. "
                         "0 (default) keeps the historical behaviour: decay starts the moment "
                         "warmup ends. Expressed in EPOCHS, not steps, because that is how the "
                         "run's own curves are read; it is converted using the run's expected "
                         "steps-per-epoch, the same horizon the cosine already uses.")
    ap.add_argument("--lr-decay-end-epoch", type=float, default=0.0,
                    help="REACH --lr-final BY THIS EPOCH and hold it flat afterwards. 0 (default) "
                         "spreads the decay over the whole run, which is what every arm before "
                         "this one did. Setting it shortens the decay window, so the lr lands at "
                         "--lr-final early and the remaining epochs train at that floor.")
    ap.add_argument("--lr-decay-shape", default="cosine", choices=["cosine", "exponential"],
                    help="COSINE (default) matches the existing schedule exactly, so an arm that "
                         "leaves the two epoch flags at 0 is bit-identical to before. EXPONENTIAL "
                         "is geometric in the lr, lr*(lr_final/lr)**p, which is the better shape "
                         "when the span is large: 1e-2 -> 1e-5 is a 1000x drop, and cosine spends "
                         "most of the window in its top decade.")
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
    ap.add_argument("--lr-steps", default="",
                    help="A STAIRCASE schedule, EPOCH:LR,EPOCH:LR -- e.g. "
                         "20:1e-3,40:1e-4 holds --lr until epoch 20, then 1e-3 until epoch "
                         "40, then 1e-4. TAKES PRECEDENCE over --lr-final and the decay "
                         "window/shape, which interpolate continuously and so cannot "
                         "express a hold-then-drop. Warmup still applies to the first "
                         "steps. Empty (default) leaves every existing arm bit-identical.")
    ap.add_argument("--save-init", default="",
                    help="write the head to this path BEFORE any optimiser step, then exit. This "
                         "is what step-zero parity needs and nothing else produced it: every "
                         "other head save happens after at least one step, and both pose "
                         "injections are zero-init, so a head saved after one step is no longer "
                         "bit-identical to the un-injected one. With it, phase_k_eval can be "
                         "handed a genuine in-block checkpoint whose PAE and ipTM MUST reproduce "
                         "the shipped head exactly -- which tests the evaluator's install path, "
                         "its key-set guard and the zero-init claim in one comparison, on the "
                         "real CLI rather than in a test harness. Refuses under WORLD_SIZE > 1: "
                         "it exits the process, and exiting one rank of three parks the others "
                         "on a collective.")
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
    ap.add_argument("--rank-pool", choices=("max", "mean"), default="max",
                    help="HOW A POSE'S ipTM IS POOLED OVER ANCHOR ROWS, in BOTH pass A and the "
                         "pass-B term, so the quantity trained is the quantity scored. max is "
                         "shipped ipTM. mean measured +0.0140 +- 0.0034 (t=4.06) better than the "
                         # ESCAPED. argparse runs every help string through `% params`, so a bare
                         # "67.7% of" is read as the conversion "% o" and `--help` died with
                         # "TypeError: %o format: an integer is required, not dict" -- the whole
                         # help, not just this entry. Present at 72148ac3 and found while adding
                         # the two flags below, which nobody could otherwise have read.
                         "max at the ORACLE on 738 winnable interfaces (67.7%% of oracle headroom "
                         "against 58.6%%) and -0.0130 +- 0.0048 (t=-2.68) WORSE when swapped onto "
                         "the current trained head -- better if trained for, worse if swapped in. "
                         "It also gives gradient to every row rather than to the argmax alone, "
                         "which removes the moving-target problem the max has.")
    ap.add_argument("--rel-labels", default="",
                    help="directory of <pdb>.npz true-iTM labels from "
                         "run_logging/build_itm_labels.py, keyed (seed, sample, asym_x, asym_y) "
                         "-> g_mean, g_max. REQUIRED by --rank-form rel, which regresses on them "
                         "instead of on DockQ.")
    ap.add_argument("--rel-std-floor", type=float, default=None, metavar="EPS",
                    help="the clamp on the std each side of --rel-standardize is divided by. "
                         "Defaults to rank_loss.DEFAULT_REL_STD_FLOOR = 0.02.\n"
                         "WHY YOU WANT IT NEAR ZERO. Above the floor the standardised loss is "
                         "exactly 2*(1 - corr), so shrinking the predictions changes nothing -- "
                         "the division undoes it -- and only the ORDER can improve the loss. "
                         "That is the whole point of standardising. The floor breaks it: below "
                         "0.02 the clamp stops normalising, the prediction contributes ~0, and "
                         "the loss falls to 1.0 -- which BEATS an uncorrelated ordering (2.0) "
                         "and any anti-correlated one (up to 4.0). So the floor creates exactly "
                         "the hedge standardising was meant to remove, for a head whose ordering "
                         "is worse than corr +0.5. At 1e-8 the division is only guarding against "
                         "0/0 and the invariance holds all the way down.\n"
                         "ON THE TRUTH SIDE it is harmless either way: a flat-truth interface "
                         "has a numerator that is zero too, and --rel-sd-weight already gives it "
                         "a weight of ~0.")
    ap.add_argument("--rel-sd-weight", action="store_true",
                    help="MULTIPLY each interface's rel loss and gradient by the GROUND TRUTH's "
                         "own within-interface std. Intended with --rel-standardize, which "
                         "divides both sides by their std and so makes every interface count "
                         "equally -- measured, corr(within-interface std of the true iTM, DockQ "
                         "range) = +0.745, so the wide-spread interfaces are exactly where a "
                         "wrong pick is expensive and the narrow ones have frac acceptable "
                         "median 1.00. This restores that ordering of importance on top of the "
                         "scale invariance. SD, NOT VARIANCE: the target is already divided by "
                         "sigma_g, so sigma_g^2 would nearly cancel the standardisation. The "
                         "weight comes from g, which does not depend on the parameters, so it "
                         "scales the gradient exactly; it is NOT floored, because a genuinely "
                         "flat interface should contribute nothing. ALPHA MUST BE RE-MEASURED "
                         "with --grad-norm-every: scaling the loss scales its gradient.")
    ap.add_argument("--rel-g-weight", type=float, default=0.0, metavar="POW",
                    help="PER-POSE WEIGHT BY THE RAW GROUND TRUTH SCORE, as g^POW. 0 (default) "
                         "is off; 1.0 weights each pose by its true iTM, 2.0 by its square if "
                         "the linear form is too gentle. The two-pass standardisation is "
                         "unchanged -- the prediction is still z-scored by its own mean and std "
                         "over the interface and the label by its own -- and the weight enters "
                         "ONLY the final reduction, as sum(w * err^2) / sum(w) instead of "
                         "mean(err^2), so the loss stays on the scale it had before and does not "
                         "inflate by the average weight. The point is that misplacing a pose the "
                         "truth rates highly should cost more than misplacing a bad one, while "
                         "the shift and scale invariance of the standardised form is kept. THE "
                         "RAW g, NOT THE Z-SCORED ONE: the z-scored target is centred, so half "
                         "its values are negative and it would SUBTRACT the error on "
                         "below-average poses. REQUIRES --rel-standardize, and composes with "
                         "--rel-sd-weight, which stays applied on top. ALPHA MUST BE "
                         "RE-MEASURED with --grad-norm-every.")
    ap.add_argument("--rel-g-sigmoid", default=None, metavar="MID:WIDTH",
                    help="PER-POSE WEIGHT AS A SOFT THRESHOLD ON THE RAW GROUND TRUTH, "
                         "w_i = sigmoid((g_i - MID) / WIDTH), in place of --rel-g-weight's g^POW. "
                         "0.23:0.1 gives ~0.2 at g=0.1, 0.5 at 0.23, ~0.75 at 0.35 and ~1 above "
                         "0.5. The reduction is the same weighted mean, sum(w * err^2) / sum(w), "
                         "and the two passes and --rel-sd-weight are unchanged. WHY A THRESHOLD "
                         "AND NOT THE SCORE: the true iTM here is BIMODAL -- measured over "
                         "1,117,590 poses, p10 0.086, p25 0.220, p50 0.670, p90 0.853 -- so "
                         "weighting by g itself spends its contrast separating 0.67 from 0.85, "
                         "two poses that are both fine, while a midpoint at 0.23 sits in the "
                         "valley and asks whether the pose is good at all. REQUIRES "
                         "--rel-standardize and is mutually exclusive with --rel-g-weight. ALPHA "
                         "MUST BE RE-MEASURED with --grad-norm-every.")
    ap.add_argument("--rel-balance-accept", type=float, default=0.0, metavar="DOCKQ",
                    help="PER-LIST CLASS BALANCE. Keep every pose, split each interface's poses "
                         "into acceptable (DockQ >= DOCKQ) and not, and scale the per-pose "
                         "weights so each GROUP carries half that interface's loss: "
                         "0.5/n_acceptable each for the acceptable poses, 0.5/n_unacceptable "
                         "each for the rest. 0 (default) is off; 0.23 is the acceptable "
                         "threshold used everywhere else in this project. In a 2-of-75 list that "
                         "is 0.25 per good pose against 0.00685 per bad one, where the "
                         "unweighted mean hands the two good poses 2.7%% of the gradient between "
                         "them -- and 151 of the 1,688 training interfaces are that shape. "
                         "BALANCED PER LIST, NOT ACROSS THE SET, so a 1-of-75 list and a "
                         "40-of-75 list are each half-and-half and nothing depends on which "
                         "lists the epoch dealt. THE SPLIT IS DockQ while the regression target "
                         "is the true iTM, so the DockQ cache must cover the group exactly or "
                         "the run refuses. REQUIRES --rel-standardize; mutually exclusive with "
                         "--rel-g-weight and --rel-g-sigmoid, which are different per-pose "
                         "weights. ALPHA MUST BE RE-MEASURED with --grad-norm-every.")
    ap.add_argument("--pae-split-accept", type=float, default=0.0, metavar="DOCKQ",
                    help="DIAGNOSTIC ONLY: report the per-epoch PAE loss SPLIT BY POSE GROUP on "
                         "the cropped training batches. Each pose's scored interfaces are looked "
                         "up in the DockQ labels; a pose whose interfaces are ALL acceptable "
                         "(DockQ >= DOCKQ) goes in one group, all unacceptable in the other, and "
                         "the plain per-pose mean of the labelled pae_loss is printed for each "
                         "with its pose count and their difference. 0 (default) is off; 0.23 is "
                         "the acceptable threshold used everywhere else in this project. WHAT IT "
                         "ANSWERS: whether both groups are being fit or one is being sacrificed "
                         "-- acceptable-pose PAE falling while unacceptable-pose PAE does not (or "
                         "the reverse) is the signal, and the aggregate pae_loss in "
                         "trajectory.csv cannot show it. TAKES NO GRADIENT and is NOT gated on "
                         "--rank-loss or --rank-cross-seed, because the arm it exists for is "
                         "PAE-only. NEEDS --pae-scored-pairs (there is no interface to classify "
                         "without it) and DockQ labels (--rank-labels, else <fold>/results.csv); "
                         "both are refused at STARTUP.")
    ap.add_argument("--rank-diag-only", action="store_true",
                    help="RUN THE PER-POSE SCORING AND ITS DIAGNOSTICS WITHOUT TRAINING ON THEM. "
                         "Every ranking diagnostic -- within-interface Spearman, score spread, "
                         "in-sample top-1, the rank of the first acceptable pose, the "
                         "acceptable-minus-unacceptable score gap -- needs a predicted score per "
                         "pose, and that scoring only ran when the ranking loss was on. So a "
                         "PAE-only arm could not be measured on the quantity it is meant to "
                         "improve. This runs the scoring and leaves the loss alone: _xs_coef is "
                         "never populated, so no coefficient reaches the pose loop and no term is "
                         "added. --alpha-rank 0 would also zero the term, but the run's args.json "
                         "and banner would then say rank_loss: True, which is the wrong record. "
                         "Needs --rank-cross-seed and --rel-labels; refuses together with "
                         "--rank-loss, which already does the scoring.")
    ap.add_argument("--rel-standardize", action="store_true",
                    help="ABLATION. Also divide each interface's centred prediction and label by "
                         "its own std (floored), which makes the loss invariant to shrinking the "
                         "scores so collapse becomes impossible rather than merely penalised. OFF "
                         "by default because it equalises interfaces whose outcomes are NOT equal: "
                         "measured, corr(within-interface std of the true iTM, DockQ range) = "
                         "+0.745, so wide-spread interfaces are exactly where a wrong pick is "
                         "expensive, and the narrow ones have frac acceptable median 1.00.")
    ap.add_argument("--rank-form", choices=("rel",), default="rel",
                    help="THE RANKING TERM, and there is exactly one. rel is MEAN-CENTRED "
                         "REGRESSION ON THE TRUE iTM: L = (1/P) sum_i "
                         "[(s_i - mean_s) - (g_i - mean_g)]^2 over an INTERFACE's poses, with g "
                         "from --rel-labels and the means taken over that interface's whole "
                         "group.\n"
                         "WHY THIS ONE. The head's failure is measured to be positional, not "
                         "collapse: on d32kqp epoch 66, corr(interface MEAN ipTM, interface MEAN "
                         "DockQ) = +0.866 while the median WITHIN-interface Spearman is +0.317 "
                         "against +0.690 for the true iTM. Subtracting each interface's own mean "
                         "from both sides removes the across-complex signal the head already has, "
                         "so the only thing left to earn is the within-complex ordering that "
                         "selection actually uses.\n"
                         "THE GROUP IS EVERY POSE OF THE INTERFACE, across all its seeds -- 15 "
                         "seeds x --nsample 5 = 75 on a deep structure -- so poses sharing a seed "
                         "are centred and compared alongside poses from every other seed. "
                         "WITHIN-SEED POSES ARE PART OF THE LOSS. A single-seed structure gets a "
                         "real 5-pose group on the same footing.\n"
                         "ALPHA COMES FROM GRADIENT NORMS, NEVER FROM LOSS VALUES: the "
                         "coefficients sum to EXACTLY zero, so this term cannot move an "
                         "interface's level and its accumulated gradient cancels by construction. "
                         "Use --grad-probe.\n"
                         "SIX OTHER FORMS WERE DELETED at the user's direction -- linear, "
                         "softplus, expected, spearman, distance, spearman_distance -- along with "
                         "the nine flags that only served them: --rank-weight-by-gap, "
                         "--rank-tau, --rank-pairs, --rank-within-weight, --rank-topk-decay, "
                         "--rank-straddle, --rank-spearman-tau, --rank-dist-weight and "
                         "--rank-rel-scale. --rank-pairs was the one that mattered: its `cross` "
                         "default EXCLUDED the within-seed family from the pairwise forms, so "
                         "deleting it removes the only mechanism that ever filtered those "
                         "comparisons out. The choice is kept as a one-value argument rather than "
                         "removed so a bundle's recorded rank_form still compares.")
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
    ap.add_argument("--pae-entropy", type=float, default=0.0, metavar="ALPHA",
                    help="TEST-TIME ADAPTATION: REPLACE the labelled PAE cross-entropy with "
                         "ALPHA * the mean Shannon entropy of the head's OWN predicted PAE "
                         "distribution, H = -sum_b p_b log p_b over the 64 distance-error bins. "
                         "0 (default) changes nothing.\n"
                         "NO GROUND TRUTH ENTERS THE OBJECTIVE. The label npz is still opened --"
                         " the unit pipeline and its coverage gates are built around it and a "
                         "structure with no label is refused long before this point -- but `ld` "
                         "is not read when this is on, so the gradient depends only on the "
                         "model's own output. That is what makes it runnable on a pool whose "
                         "DockQ nobody knows.\n"
                         "THE POINT IS THE DEPLOYMENT SETTING, not a better training loss. At "
                         "inference you hold the complex and its 50 poses and no labels at all, "
                         "so an objective that needs none can adapt to the very structures being "
                         "selected over. Adapting on the pool you then score is TRANSDUCTIVE by "
                         "construction and that is legitimate for TTA -- but it is NOT a held-out "
                         "measurement, and a number from it may not be compared with the "
                         "train-on-846/score-on-141 numbers in paper_results.md without saying "
                         "so.\n"
                         "IT HAS A DEGENERATE OPTIMUM. Entropy is minimised by putting all mass "
                         "in one bin regardless of which bin, so this can sharpen a confidently "
                         "WRONG prediction exactly as happily as a right one, and nothing in the "
                         "objective prefers a good pose's score over a bad one's. Watch "
                         "pae_entropy against the held-out selection metrics, not on its own: "
                         "entropy falling while rho falls is the collapse, not progress.\n"
                         "MASKED TO THE SCORED CROSS-CHAIN PAIRS when --pae-scored-pairs supplies "
                         "a mask, since those are the pairs ipTM is computed from; otherwise over "
                         "every token pair, and the startup line says which.")
    ap.add_argument("--lora-keep-trainable", default="",
                    help="comma-separated REGEXES matched against head PARAMETER names that stay "
                         "FULLY TRAINABLE alongside the adapters, instead of being frozen.\n"
                         "WHY THIS IS NEEDED AT ALL. apply_lora freezes every parameter of the "
                         "head and then phase_k_train sets `trainable` to the adapter names "
                         "ALONE, so --train-only does not survive LoRA and anything LoRA did not "
                         "adapt is frozen outright. Two classes of tensor are then lost: a module "
                         "whose smaller dimension is below --lora-rank is skipped by apply_lora "
                         "and frozen, and any parameter that is not a 2-D `weight` is never "
                         "adaptable in the first place.\n"
                         "THE CASE IT EXISTS FOR is the pose pathway. frame_dir_proj is (256, 3) "
                         "and pose_perblock_linear_no_bias_d_wo_onehot is (256, 1), so at rank 8 "
                         "BOTH are skipped for rank-too-high and would train not at all -- the "
                         "terms this architecture adds, frozen at their zero init, which is the "
                         "same as not having them. All ten pose tensors together are 33,792 "
                         "parameters, 0.6%% of the head's effective capacity, so keeping them "
                         "dense costs essentially nothing.\n"
                         "IT IS UNIONED BEFORE --lean-single, not after, so a pose tensor that is "
                         "genuinely gradient-free is still dropped and --skip-dead-ops still "
                         "refuses a module it would stub. This widens what MAY train; it never "
                         "overrides what provably does not.")
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
    ap.add_argument("--eval-pool", "--train-eval-pool", dest="train_eval_pool", default="",
                    help="score the head on THIS pool of TRAINING interfaces at epoch boundaries, "
                         "in addition to (or instead of) validation. The question `is the model "
                         "learning at all` is about the training set, and --eval-every-epoch only "
                         "knows how to score validation. Uses the run's own --fold and --seeds, so "
                         "it scores exactly the poses it trained on. Sharded over every card the "
                         "run was given, with the other ranks parked at a barrier, same as the "
                         "validation path.")
    ap.add_argument("--eval-pool-every", "--train-eval-every", dest="train_eval_every", type=int, default=2,
                    help="epochs between training-set evaluations. Ignored without "
                         "--train-eval-pool.")
    ap.add_argument("--extra-eval", action="append", default=[], metavar="SPEC",
                    help="AN ADDITIONAL eval pass at its own cadence, repeatable. The "
                         "trainer historically had exactly ONE: --eval-pool and "
                         "--train-eval-pool are argparse aliases for a single dest, so "
                         "\"cropped validation every 2 epochs AND uncropped validation and "
                         "in-sample every 6\" was not expressible and section 12 recorded "
                         "in-sample as simply unmeasured for the uncropped arm.\n"
                         "SPEC is comma-separated key=value with keys: name (required, "
                         "names metric_<name>_ep<N>.json and <name>_eval_shards_ep<N>), "
                         "pool (required), fold, seed_base, budget, every (required), "
                         "Example:\n"
                         "  --extra-eval 'name=valfull,pool=training/valfull/pool.csv,"
                         "fold=validation/fold,seed_base=400000,budget=30,every=6'\n"
                         "Each pass costs a full sharded evaluation, so every=1 on three "
                         "specs triples that cost; the cadence is the knob.")
    ap.add_argument("--eval-pool-fold", "--train-eval-fold", dest="train_eval_fold", default="",
                    help="score --train-eval-pool against THIS fold instead of --fold. The point "
                         "is to watch a held-out signal every few epochs without paying for the "
                         "whole validation set: a 10-structure validation pool is ~1,000 units "
                         "against 5,300, so minutes rather than ~25. Defaults to --fold, which is "
                         "the in-sample training-pool behaviour. Pair it with "
                         "--train-eval-seed-base, because a different fold was folded at "
                         "different seeds, and with --train-eval-budget, which otherwise defaults "
                         "to the count of THIS run's --seeds and would silently score a tenth of "
                         "a 100-seed fold.")
    ap.add_argument("--eval-pool-seed-base", "--train-eval-seed-base", dest="train_eval_seed_base", type=int, default=-1,
                    help="the first seed of --train-eval-fold. -1 (default) uses this run's own "
                         "--seeds, which is right only when the pool lives in --fold.")
    ap.add_argument("--eval-pool-budget", "--train-eval-budget", dest="train_eval_budget", type=int, default=0,
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
                         "buffer is sized for it. HISTORY, READ THIS BEFORE RAISING IT: 1083ed10 "
                         "set 3 everywhere and was reverted in 7033430b because ggfhj5 lost epoch "
                         "12 to what looked like a deadlock -- every rank at 0%% GPU in "
                         "futex_wait, nine minutes with no trajectory write. The mechanism was "
                         "found later and FIXED: _pf_pending evicted unconsumed results, and "
                         "since a worker emits each unit exactly once, the next take waited the "
                         "whole 120 s budget for something that could never arrive, then the unit "
                         "after it did the same. Evictions are now recorded and such a take "
                         "returns immediately; a take issued after every worker has finished "
                         "returns immediately too. run_logging/prefetch_deadlock_repro.py is the "
                         "CPU-only regression test and must stay at 0 stalls for the fixed path.")
    ap.add_argument("--prefetch-units", type=int, default=0,
                    help="prepare this many units ahead on a worker thread, overlapping the CPU "
                         "crop with the GPU's work on the previous unit. 0 disables it. Measured "
                         "on epoch 5 of the full set: crop_slice is 23.0 min of a 73.1 min epoch "
                         "with the GPU idle throughout, and GPU busy time is only ~20 min of 73. "
                         "The worker never touches CUDA -- it hands back host tensors and the main "
                         "thread moves them -- so only one thread ever issues CUDA work.")
    ap.add_argument("--trunk-memmap", action="store_true",
                    help="MAP z_trunk INSTEAD OF READING IT INTO A HEAP BUFFER. The trunk npz's "
                         "members are STORED, not deflated, and z_trunk.npy is the first member at "
                         "zip header_offset 0 -- verified on three real trunks -- so np.memmap can "
                         "address it in place and the host-to-device copy reads page cache "
                         "directly instead of a buffer np.load just filled. That removes one pass "
                         "over ~243 MB per unit, ~1.1 TB per rank per epoch, worth an estimated "
                         "~6 min of a 232 min epoch. z_trunk is ~99%% of the bytes and is the only "
                         "member mapped; the rest are at or under ~1.5 MB where the extra syscalls "
                         "cost more than the copy. OFF BY DEFAULT AND UNTESTED ON A GPU: "
                         "phases/test_trunk_memmap.py gates the two paths bit-identical on CPU "
                         "over real trunks, bf16 compared through a uint16 view, and any member "
                         "that cannot be mapped falls back to np.load silently. NOTE it moves the "
                         "disk read out of trunk_npz_read and into trunk_h2d, because a mapping is "
                         "lazy -- read their SUM when comparing against a run without it.")
    ap.add_argument("--trunk-cache-gb", type=float, default=0.0,
                    help="HOLD A STEP'S UNCROPPED TRUNKS AS HOST ARRAYS, up to this many GiB per "
                         "rank; 0 is off. The cross-seed objective runs TWO passes over every "
                         "step's units -- a no_grad scoring pass whose scores the relative-iTM "
                         "coefficients need, then the backward pass -- so each unit's ~243 MB was "
                         "read and copied to the device TWICE per step (4,600 load_unit calls for "
                         "2,300 units, rank 0, epoch 3 of mqird6). Caching the HOST arrays for the "
                         "span of one step halves the reads, and --prefetch-workers threads warm "
                         "them while the GPU runs, which puts most of the rest behind compute. "
                         "Estimated 10-20 min of a 232 min epoch. KEYED BY TRUNK PATH AND NEVER "
                         "POPPED, unlike the cropped prefetch: a pop queue walked by both passes "
                         "is the dsswrj bug that cost ~3 h an epoch. Purely advisory -- every miss "
                         "falls back to the ordinary read, so there is no wait and nothing to "
                         "deadlock on -- and REFUSES rather than evicts when full, because every "
                         "entry in a step is wanted twice. Ignored when cropping is on (the crop "
                         "prefetch already covers that path) or off a cuda device (host arrays "
                         "would be ALIASED, not copied). 4.0 fits about half a 30-unit step's "
                         "~7.3 GB, so expect a hit rate near 75%% there and read `full` for how "
                         "many were turned away; the epoch line's rate is 50%% when the warmers "
                         "contributed nothing and 100%% when every unit was warmed before pass A. "
                         "CHANGES NO NUMBER the run produces, and is UNTESTED ON A GPU.")
    ap.add_argument("--prefix-cache-gb", type=float, default=0.0,
                    help="HOLD THE FROZEN PREFIX OF THE CONFIDENCE HEAD FOR ONE STEP, up to this "
                         "many GiB of HOST memory per rank; 0 is off. Different cache from "
                         "--trunk-cache-gb: that one holds the INPUTS (z_trunk et al) so they are "
                         "read once instead of twice, this one holds an ACTIVATION from partway "
                         "THROUGH the head so pass B does not recompute it. Under "
                         "--pose-inblock --lean-single the deepest point at which z is a function "
                         "of frozen parameters alone is INSIDE the injected block, after "
                         "tri_mul_in and before the pose enters -- the same boundary the injection "
                         "is placed at, seen from the other side -- because --train-only freezes "
                         "blocks 0-2 AND that block's two tri-muls. So a hit skips three whole "
                         "blocks plus two of the last block's five pair operations, which is most "
                         "of a forward: gpu_forward was 69.7 min over 23,000 calls on mqird6 epoch "
                         "6, ~34.9 per pass, and the prefix is ~87%% of it. Expect ~25 min off a "
                         "222 min epoch, net of the D2H on the fill and the H2D on the hit. "
                         "SCOPE IS ONE OPTIMISER STEP: pass A fills, pass B hits, the step "
                         "boundary drops it -- the epoch's poses are distinct, so nothing is "
                         "reused across steps and a persistent cache would be ~6 TB an epoch "
                         "against a 3,072 GiB quota. ON THE HOST, because z is [N, N, 256] bf16 = "
                         "174 MB at the median 583 tokens and a 50-pose step is ~8.7 GB typical / "
                         "~26 GB at the p90, which no card has spare. CAPPED IN BYTES WITH THE "
                         "RESERVATION TAKEN BEFORE THE COPY, so the cap bounds RSS and not just "
                         "the dict; a pose that does not fit RECOMPUTES. Purely advisory -- every "
                         "miss recomputes the prefix, so NO NUMBER the run produces depends on a "
                         "hit. REFUSES to enable unless --pose-inblock is on, --lean-single is on, "
                         "every prefix tensor reads requires_grad=False at runtime and every "
                         "dropout rate in the prefix blocks is 0.0. Validated on CPU by "
                         "run_logging/test_prefix_cache.py; UNTESTED ON A GPU.")
    ap.add_argument("--pose-readout", action="store_true",
                    help="THE FIFTH INJECTION SITE: pose distance (one-hot + raw) and, with "
                         "--frame-direction, orientation, added after pae_ln and upstream of the "
                         "PAE readout -- and unlike --pose-inblock-pae this one works ALONGSIDE "
                         "--pose-perblock, because it READS the geometry that wrapper already "
                         "published for this forward instead of needing the in-block wrapper's "
                         "own stash. Zero-init and verified zero, so step 0 is bit-identical. The "
                         "tensors are pose_readout_d, pose_readout_d_wo_onehot and "
                         "pose_readout_frame_dir; a --train-only must match `pose_readout` to "
                         "reach them.")
    ap.add_argument("--pose-readout-copy-distance", action="store_true",
                    help="Start --pose-readout's two DISTANCE arms at block 0's pretrained "
                         "linear_no_bias_d/_wo_onehot instead of zero. STEP 0 IS THEN NOT "
                         "BIT-IDENTICAL: those weights add into z_pair inside the stack where "
                         "later blocks re-normalise, while this site adds after pae_ln with "
                         "nothing downstream to rescale it. Shapes permit it, semantics do not "
                         "match. The orientation arm stays zero either way.")
    ap.add_argument("--skip-dead-ops", action="store_true",
                    help="DO NOT RUN CONFIDENCE-HEAD MODULES WHOSE OUTPUT IS EXACTLY ZERO. The "
                         "shipped checkpoint carries pairformer modules whose EVERY tensor sits in "
                         "a band 4.904e-37..4.907e-37 -- a fixed exponent with only the low "
                         "mantissa bits varying, not a trained weight and not a denormal (fp32's "
                         "smallest normal is 1.175e-38). Two such factors multiplied underflow to "
                         "zero, so the module emits EXACTLY 0.0: measured on the shipped head, "
                         "NINE of the SIXTEEN O(N^3) pair operations (block 0 entirely, blocks "
                         "1-2's two triangle attentions, block 3's tri_att_end) plus block 1's "
                         "pair_transition. Every one is consumed as z + module(z), so adding its "
                         "zero is the identity on z and NOT RUNNING IT IS BIT-IDENTICAL "
                         "ARITHMETIC, not an approximation -- the module object is replaced by a "
                         "husk returning zeros_like(z) and the block's own forward is untouched, "
                         "so there is no third mirror of PairformerBlock.forward to go stale. "
                         "DECIDED BY MEASUREMENT AT EVERY INSTALL, never from a list: a module is "
                         "skipped only if every parameter is below 1e-30 AND running it returns "
                         "exactly 0.0 at the input's shape and dtype, at |z| x1 and x1e3, in "
                         "train and eval mode and under bf16 autocast. A hardcoded list would go "
                         "stale against a different checkpoint, and a threshold alone would call "
                         "a FRESH head fully dead (its residual branches are zero-init). The "
                         "husks share the originals' parameters by reference, so the head's "
                         "state-dict key set, --train-only's name matching and the evaluator's "
                         "set-equality guard are all unchanged, and the weights are re-checked "
                         "against the floor at every step boundary. REFUSES rather than degrading "
                         "if a module it would skip is in the TRAINABLE set -- a stubbed module "
                         "gets no gradient at all (.grad stays None rather than becoming zeros), "
                         "which changes what the optimiser and the gradient collective see, so "
                         "the resolution is to drop it from --train-only. NOTE this arm's "
                         "--train-only includes block 3's tri_att_end, which IS dead, so the "
                         "refusal fires until that pattern changes. COMPLEMENTARY to "
                         "--prefix-cache-gb, not redundant with it: the cache removes the frozen "
                         "prefix from pass B, this removes half of pass A, which the cache leaves "
                         "running in full. Validated on CPU by run_logging/test_dead_ops.py; "
                         "UNTESTED ON A GPU.")
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
    ap.add_argument("--val-loss-shard-retries", type=int, default=2,
                    help="how many times to re-run a FAILED held-out PAE shard before voiding "
                         "the epoch's number. The shards read a bundle already on disk and the "
                         "weights cannot move while they run, so a re-run measures exactly the "
                         "same model as its siblings; 0 restores the old void-on-any-failure "
                         "behaviour. tv6hue epoch 4 lost two of three shards to a transient "
                         "CUDA context-creation failure and voided a point worth 2.1 min of work.")
    ap.add_argument("--val-loss-uncropped", action="store_true",
                    help="run the held-out PAE pass on the FULL structure instead of the training "
                         "crop. OFF BY DEFAULT and it must stay off mid-comparison. WHY ANYONE "
                         "WANTS IT: evaluation should match the deployment setting, and the DockQ "
                         "eval is already uncropped -- so today the two held-out passes see "
                         "DIFFERENT INPUTS for the same poses, which is the only reason they "
                         "cannot share one forward. Uncropping this one is the prerequisite for "
                         "merging them, worth ~4-5 min of a ~10 min eval phase. WHAT IT COSTS, "
                         "and it is not small: the pass inherits the run's crop precisely so the "
                         "held-out number is comparable to the TRAINING curve it is read against, "
                         "which is the only thing it is for. Uncropped, train-vs-val PAE stops "
                         "being apples to apples and the generalisation-gap readings (copy's gap "
                         "wider than share's by +0.0067 +- 0.0017 over 12 matched epochs) are no "
                         "longer computable the same way. It also puts a SEAM in the val PAE "
                         "curve at whichever epoch it lands, and makes that arm's val PAE "
                         "incomparable to every arm measured before it -- so turn it on at the "
                         "start of a fresh comparison, never inside a running one.")
    ap.add_argument("--val-loss-emit-iptm", action="store_true",
                    help="have the held-out PAE pass also write chain_pair_iptm rows, for "
                         "run_logging/dockq_from_iptm.py to rebuild the DockQ metric from. Only "
                         "useful together with --val-loss-uncropped: dockq_from_iptm --compare "
                         "checks PER-POSE ipTM against the eval's own rows, and a cropped pass "
                         "sees different inputs, so every pose would differ for a legitimate "
                         "reason and the join error it exists to catch would be buried. With the "
                         "inputs identical, ANY ipTM difference is the join.")
    ap.add_argument("--emit-iptm-rows", default="",
                    help="ALSO write chain_pair_iptm per (interface, seed, sample) to this csv. "
                         "EXPERIMENTAL and off by default. The held-out PAE pass and the DockQ "
                         "eval forward the SAME 7,950 poses through the SAME head every epoch -- "
                         "verified identical as sets of (pdb, seed, sample) -- and ipTM is derived "
                         "from the PAE logits the loss pass already has, so the second forward is "
                         "redundant in principle. This emits the rows that would let the DockQ "
                         "metric be computed from the loss pass, WITHOUT removing the eval, so "
                         "the two can be compared on real data before anything is deleted. "
                         "THE JOIN IS THE RISK, not the arithmetic: phase_k_eval joins poses to "
                         "DockQ by COORDINATE IDENTITY against the _sample_<k>.cif files, and a "
                         "naive (iface, seed, sample) key join is only equivalent if the sample "
                         "ordering agrees. Measured on j583i8 epoch 1 it does, on all 13,800 rows "
                         "against validation/fold/results.csv -- but that is one pool on one day, "
                         "which is exactly why the eval keeps its own self-check and why nothing "
                         "here replaces it yet.")
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
    ap.add_argument("--pae-row-frac", type=float, default=0.0,
                    help="ROW-WEIGHTED PAE. 0 (default) is the unweighted term. Otherwise the top "
                         "this-fraction of ANCHOR ROWS by t_i -- the pre-max ipTM contribution of "
                         "that row -- carry --pae-row-mass of the loss and the rest share what is "
                         "left. WHY: ipTM is max_i t_i over ~332 rows, so selection reads ONE row "
                         "while the unweighted term grades all of them equally. This concentrates "
                         "the term on the rows selection can actually read WITHOUT changing what "
                         "the labels say is correct, so unlike a ranking loss it cannot be lowered "
                         "by distorting ipTM away from calibration. At 1.0 every row is in the top "
                         "set, the weights are uniform and the loss is the unweighted one exactly "
                         "-- that identity is the correctness gate, see test_pae_row_weight.py.")
    ap.add_argument("--pae-row-mass", type=float, default=0.8,
                    help="the share of the PAE loss given to the top --pae-row-frac of rows. With "
                         "frac 0.2 and mass 0.8 a top row counts 16x a bottom row and effective "
                         "sample size is 0.31n (about 102 of 332). NOT 1.0 deliberately: rows "
                         "outside the top set must keep SOME pull toward truth, or they are "
                         "unsupervised and free to drift over-confident -- and since ipTM is a "
                         "max, an over-confident row is exactly what gets selected.")
    ap.add_argument("--pae-row-floor", type=int, default=10,
                    help="never focus on fewer than this many rows. Our structures run 29-801 "
                         "anchor rows, so a bare fraction would leave the smallest ones training "
                         "on almost nothing.")
    ap.add_argument("--pae-row-log-every", type=int, default=50,
                    help="log the row diagnostic every Nth pose: t_i quantiles, the realised k and "
                         "effective sample size, and the rank correlation between t_i and that "
                         "row's own PAE loss. THAT CORRELATION IS THE POINT -- if the high-t_i "
                         "rows are already well fit, this whole weighting has no headroom. 0 is "
                         "off. The unweighted loss is logged alongside on the same poses so the "
                         "curve stays comparable to the unweighted arms.")
    ap.add_argument("--limit-units", type=int, default=0, help="debug: cap the number of units")
    ap.add_argument("--shuffle", default="structure",
                    choices=["structure", "none"],
                    help="`structure` shuffles the ORDER OF STRUCTURES and the units within each, "
                         "which is as much as can be shuffled while building each structure's "
                         "dataset item once. `none` is deterministic for debugging.")
    ap.add_argument("--expect-trainable", type=int, default=0,
                    help="REFUSE unless --train-only leaves exactly this many trainable tensors. "
                         "A regex is intent; this is the outcome. The bare negative lookahead of "
                         "the full-head pattern yields 19 tensors and 0.3M parameters against the "
                         "intended 102 and 5.3M -- launched truncated once, caught only because a "
                         "human read the startup line. The optimiser is non-empty either way, so "
                         "no existing guard fires. 0 (default) does not check.")
    ap.add_argument("--epoch-balance", action="store_true",
                    help="deal each step's units so the ranks get a similar TOTAL UNCROPPED SIZE, "
                         "not just a similar count. The crop equalises GPU work at 256 tokens but "
                         "not the read: a 1,650-token structure still has a ~100 MB z_trunk to "
                         "open before 97%% of it is discarded. The ranks barrier every step, so a "
                         "step costs its SLOWEST rank. Measured per-step spread in uncropped "
                         "tokens: 16.93%% median under plain dealing, 0.55%% balanced -- while the "
                         "per-EPOCH totals were already even at 1.77%%, which is why balancing "
                         "those changes nothing. Needs --epoch-cover-all.")
    ap.add_argument("--pose-batch-max-tokens", type=int, default=384,
                    help="do not batch poses for a unit above this token count. Activation memory "
                         "goes as nb x tokens^2, so 5 poses at 768 tokens is ~87 GB of a 98 GB "
                         "card while 256 is ~11 GB -- and the per-pose overhead batching removes "
                         "is 72%% of a 256-token pose against 17%% of a 768-token one, so the "
                         "large crops both need it least and afford it least.")
    ap.add_argument("--pose-batch", type=int, default=1,
                    help="POSES PER FORWARD. The confidence head is ALREADY batched over samples "
                         "-- x_pred_coords is [..., N_sample, N_atoms, 3] and every output carries "
                         "N_sample -- so calling it once per pose pays the Python and "
                         "host-to-device cost N times for one pair representation that does not "
                         "change between poses of a unit. 1 (default) is the historical "
                         "behaviour, bit-identical. The reason it was 1 is recorded on --accum: "
                         "token counts ran 263-3,400 so poses could not be stacked without "
                         "padding. Under a fixed-size crop that premise is gone -- poses of one "
                         "unit always share a token count by construction, whatever its size.")
    ap.add_argument("--epoch-cover-all", action="store_true",
                    help="EVERY UNIT ONCE PER EPOCH, dealt so the ranks hold DISJOINT structures "
                         "in each step. Without it, and without --rank-cross-seed, build_epoch "
                         "forms groups of `world` SEEDS OF ONE STRUCTURE, so a structure with "
                         "fewer than `world` seeds yields floor(1/3)=0 groups and NEVER TRAINS -- "
                         "412 of this population's 846 structures, silently, in every epoch. "
                         "Groups of same-structure seeds exist to make per-step work equal across "
                         "ranks for the RANKING loss, which needs several poses of one structure "
                         "in one step. A PAE-only arm has no such requirement, and a fixed-size "
                         "crop already equalises the work, so the units can simply be dealt.")
    ap.add_argument("--epoch-order", default="shuffle",
                    choices=["shuffle", "manifest", "tokens"],
                    help="HOW THE EPOCH'S STRUCTURES ARE ORDERED, independently of "
                         "--shuffle. `shuffle` (default) is the training order, unchanged. "
                         "`tokens` walks them ASCENDING BY TOKEN COUNT, which is what a "
                         "memory probe needs: it reaches the boundary having already "
                         "measured every size below it, so one run ending in an OOM still "
                         "yields the whole curve. `manifest` keeps the order the unit list "
                         "is already in -- which is SORTED BY PDB, not the selection csv's "
                         "row order. It was added believing otherwise; prefer `tokens`.")
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
    ap.add_argument("--ckpt-token-threshold", type=int, default=0,
                    help="SIZE-CONDITIONAL ACTIVATION CHECKPOINTING. Units at or below this many "
                         "tokens run with checkpointing OFF; larger units keep "
                         "--blocks-per-ckpt. 0 (default) disables the rule, so every unit uses "
                         "--blocks-per-ckpt and nothing changes.\n"
                         "WHY IT IS SAFE TO VARY PER UNIT. The peak is a PER-POSE property, not a "
                         "per-step one: _one_pose is a function, its frame is destroyed on "
                         "return, and every large tensor is freed before the next pose allocates "
                         "(the pose-lifetime fix). So a small unit running without checkpointing "
                         "cannot add to the peak that a large unit sets -- it only has to stay "
                         "UNDER it. The run's peak is 87.71 GB allocated on cards of 95.6 GB with "
                         "the largest unit at 1543 tokens, and that is the number a threshold "
                         "must not raise.\n"
                         "WHY IT PAYS. blocks_per_ckpt=1 recomputes every one of the head's four "
                         "pairformer blocks in backward, about 30%% more compute, and it is "
                         "applied to the 400-token units because the 1543-token ones need it. "
                         "gpu_backward is 70.3 min of a 143.9 min epoch.\n"
                         "SET IT FROM run_logging/bench_ckpt_by_size.py, NOT BY GUESSING. The "
                         "benchmark measures peak memory and CUDA-event time per unit at each "
                         "setting and reports the largest token count whose unchecked peak stays "
                         "under the checkpointed peak, with a margin. A threshold chosen by "
                         "arithmetic would be an OOM 40 minutes into an epoch.\n"
                         "IT CHANGES NO MATHEMATICS. Checkpointing trades compute for memory and "
                         "is gradient-exact; test_perblock_ckpt_equiv.py already gates "
                         "blocks_per_ckpt=1 against off at 0.000e+00 on every parameter, which is "
                         "exactly the equivalence this relies on.")
    ap.add_argument("--lean-readouts", action="store_true",
                    help="STOP COMPUTING THE THREE READOUTS --pae-only ALREADY THROWS AWAY. "
                         "--pae-only stubs plddt_loss, pde_loss and resolved_loss and skips the "
                         "all-atom lddt, so none of the three is trained -- but the HEAD still "
                         "builds pde_pred, plddt_pred and resolved_pred on every pose, and this "
                         "trainer puts all four into `pred`. Because nothing in the loss reads "
                         "them, backward never traverses those branches, and a branch backward "
                         "does not walk is not freed by backward either: their activations live "
                         "until _one_pose returns, through the window where the peak is set.\n"
                         "WHAT IT FREES, from the live checkpoint's shapes (c_z 256, c_s 384, "
                         "b_plddt 50), all fp32: the pde chain is N_tok^2 * 2304 B and the "
                         "plddt/resolved chain is N_atom * 84 KB, dominated by the "
                         "plddt_weight[atom_to_tokatom_idx] gather at 76.8 KB PER ATOM. That is "
                         "~1.3 GB at a median 617-token unit and ~6.5 GB at the 1543-token worst "
                         "case, against the 1.8 GB of headroom section 11 measured. ARITHMETIC, "
                         "not a measurement -- run_logging/bench_ckpt_by_size.py measures it.\n"
                         "WHY IT PAIRS WITH --ckpt-token-threshold: the threshold is limited by "
                         "headroom, so freeing headroom raises the threshold, and the two "
                         "compound.\n"
                         "IT CHANGES NO MATHEMATICS ON THE PAE PATH. phases/lean_readouts.py "
                         "mirrors ConfidenceHead.memory_efficient_forward with the pae lines "
                         "verbatim and is PINNED to that function's sha256, so a Protenix upgrade "
                         "hard-stops instead of silently running a stale copy. "
                         "run_logging/test_lean_readouts.py checks the pae logits are "
                         "BIT-identical, that the dropped work is not merely unused but never "
                         "called, and that the plddt placeholder still survives the indexing "
                         "loss.py:1772 does to it.\n"
                         "REQUIRES --pae-only, and is NOT inherited by the --loss-only val child, "
                         "which builds its own model in its own process.")
    ap.add_argument("--lean-single", action="store_true",
                    help="STOP COMPUTING THE PAIRFORMER'S SINGLE TRACK, which --lean-readouts left "
                         "running. PairformerBlock.forward updates z through five ops that are "
                         "functions of z ALONE, then updates s through attention_pair_bias and "
                         "single_transition, both of which read z -- so z NEVER DEPENDS ON s. "
                         "Under --pae-only the readout is linear_no_bias_pae(pae_ln(z_pair)), "
                         "which reads z only, and s feeds nothing but the pLDDT and resolved heads "
                         "--lean-readouts already stubs. The single track was therefore computed "
                         "in all 4 blocks for every pose of every unit of every epoch and thrown "
                         "away; lean_readouts.py's own comment says 's_single is now unused by "
                         "anything that is kept' and then keeps computing it.\n"
                         "HOW: PairformerBlock gates it on `if self.c_s > 0:` in both __init__ and "
                         "forward, so setting c_s=0 on a BUILT block stops the calls without "
                         "touching the module tree -- no renamed keys, no dropped state-dict "
                         "entries, and phase_k_eval's set-equality check on head tensors still "
                         "passes. Both that forward and PairformerStack.forward are PINNED to a "
                         "sha256 so a Protenix upgrade hard-stops rather than silently removing "
                         "the guard or the one-directional dependency this rests on.\n"
                         "WHAT IT IS WORTH, measured on CPU at the live configuration rather than "
                         "argued: 10,053,120 parameters (50.90%% of the head) stop being nominally "
                         "trainable, and with the readout parameters nothing reads either, "
                         "10,551,552 of 19,752,270 (53.42%%) are dropped from AdamW and from the "
                         "gradient all-reduce -- ~84 MB of moment state and ~42 MB of the 79 MB "
                         "flat collective buffer per step. BUT THE COMPUTE SAVING IS SMALL AND "
                         "THAT IS THE HONEST HEADLINE: the pair track is O(N^3) and the single "
                         "track's dominant term is O(N^2), so saved-activation bytes measured 4.0%% "
                         "of a block at 128 tokens, 2.9%% at 256 and 2.5%% at 384, and the "
                         "forward+backward time difference was BELOW CPU MEASUREMENT NOISE "
                         "(-4.7%% to +9.2%% across four token counts). Expect ~1-2%% of the epoch, "
                         "not a large win. It is here because it is free and provably dead, not "
                         "because it is big.\n"
                         "NUMERICS: the PAE logits are unchanged. "
                         "run_logging/test_lean_single.py gates z BIT-IDENTICAL with the single "
                         "track on and off, and gates that the single pathway's parameters "
                         "receive no gradient EITHER WAY -- which is the proof that they were "
                         "already dead rather than being killed by this flag.\n"
                         "REQUIRES --lean-readouts AND --pae-only. UNTESTED ON A GPU.")
    ap.add_argument("--pose-inblock", action="store_true",
                    help="INJECT THE POSE INSIDE THE LAST PAIRFORMER BLOCK, after tri_mul_in and "
                         "before tri_att_start, with BOTH the 39-bin one-hot distance term and the "
                         "raw-distance term -- the same pair of projections the native pre-stack "
                         "injection uses (confidence.py:297-305), shapes (c_z, num_bins) and "
                         "(c_z, 1).\n"
                         "THE PLACEMENT IS THE WHOLE POINT AND IS WHY THIS IS NOT A MODE OF "
                         "--pose-perblock. Autograd starts building at the first op whose "
                         "parameters or inputs require grad. With blocks 0-2 frozen and block 3's "
                         "two triangle multiplications frozen, injecting at the block's INPUT -- "
                         "which is where --pose-perblock injects -- makes the z entering "
                         "tri_mul_out require grad, brings BOTH tri-muls back into the backward, "
                         "and evaporates the saving SILENTLY: no error, no warning, just a slower "
                         "epoch. After tri_mul_in is the earliest point at which the frozen "
                         "tri-muls are already behind us.\n"
                         "ZERO-INIT AND VERIFIED EXACTLY ZERO, by probing the modules rather than "
                         "trusting the initialiser, so step 0 is BIT-IDENTICAL to the un-injected "
                         "head and any later divergence is attributable.\n"
                         "IT IS A MIRROR AND IT IS PINNED. Injecting between two ops inside "
                         "PairformerBlock.forward means reimplementing that function for one "
                         "block; the sha256 is pinned so a Protenix reorder is a hard stop rather "
                         "than an injection that silently lands somewhere else. Only the "
                         "inplace_safe=False branch is mirrored, which is the one this trainer "
                         "takes.\n"
                         "run_logging/test_pose_inblock_graph.py asserts the backward graph starts "
                         "where intended AND carries a negative control that moves the injection "
                         "to the block input and checks the tri-muls DO get gradient -- without "
                         "that control the assertion proves nothing. UNTESTED ON A GPU.")
    ap.add_argument("--pose-inblock-block", type=int, default=-1,
                    help="WHICH BLOCK --pose-inblock injects into; -1 means the LAST one, which is "
                         "the block the freeze leaves trainable. Any other value is for probing "
                         "and must be paired with a --train-only that leaves that block's "
                         "attention and transition trainable, or the injection trains against "
                         "frozen downstream weights.")
    ap.add_argument("--pose-inblock-pae", action="store_true",
                    help="ALSO INJECT THE POSE AT THE READOUT, after pae_ln: "
                         "linear_no_bias_pae(pae_ln(z) + pose_d(one_hot(d)) + pose_d_wo(d)). "
                         "AFTER the layer norm, which is a deliberate choice with a consequence: "
                         "an UNNORMALISED quantity is added to a NORMALISED one and nothing "
                         "downstream rescales it, since linear_no_bias_pae is the next and last "
                         "op. So the injection's effective influence is set by its initialisation "
                         "and by the optimiser, not by the norm. Injecting BEFORE the norm would "
                         "have bounded it. Implemented as a forward HOOK on pae_ln rather than a "
                         "wrapper module, because wrapping would rename pae_ln.weight to "
                         "pae_ln.ln.weight and break phase_k_eval's set-equality check on head "
                         "tensors. Zero-init and verified zero. Requires --pose-inblock.")
    ap.add_argument("--pair-dropout", type=float, default=-1.0,
                    help="ROW-WISE DROPOUT RATE ON THE PAIR REPRESENTATION in the confidence "
                         "head's pairformer blocks. ConfidenceHead builds its stack with "
                         "dropout=0.0, overriding Protenix's own 0.25 default for pair "
                         "stacks, so this arm has trained 19.8M parameters with NO dropout "
                         "and no masking of any kind. The plumbing is live -- all four "
                         "pair-rep sites in PairformerBlock call dropout_add_rowwise(z, "
                         "update, self.p_drop, self.training) and the head trains in train "
                         "mode -- only the rate is zero. Negative (default) touches nothing. "
                         "Structured noise over token pairs rather than per-element noise, "
                         "which is the AF3 design. MEASURED, not inferred from the name: "
                         "dropout_add_rowwise shares the mask along dim -3, so a dropped j is "
                         "dropped for EVERY i -- it zeroes whole COLUMNS of the [i,j] grid. "
                         "The 'rowwise' name refers to the mask being shared along rows, not "
                         "to rows being dropped. This matches openfold's DropoutRowwise.")
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
    ap.add_argument("--crop-allow", default="",
                    help="json of {pdb: [crop index, ...]} restricting which crops of a structure "
                         "the epoch rotation may use. A structure absent from the file keeps its "
                         "whole pool.\n"
                         "WHAT IT IS FOR. --rank-form rel resolves an interface's chains through "
                         "the PER-POSE OST mapping, because that is what its labels were built "
                         "against; the crop pool and the launcher's preflight use the STATIC roles "
                         "join. The two disagree about which copy of a duplicated chain an "
                         "interface uses -- on 9o4o the static join says asyms (0,3) and OST says "
                         "(0,2), the crop keeps 3 and drops 2 -- so the crop keeps the interface by "
                         "the static definition while rel's lookup finds nothing and pass A's "
                         "completeness guard stops the run. Measured on this pool: 5 structures "
                         "and 192 of 92,636 crops (0.21%%), none left with zero usable crops.\n"
                         "THE INDEX VALUES ARE THE ORIGINAL ONES, not a re-numbering, because the "
                         "rel labels are keyed by crop_index and a compacted index would read "
                         "another crop's target.")
    ap.add_argument("--grad-norm-every", type=int, default=0, metavar="K",
                    help="EVERY K OPTIMISER STEPS, measure the PAE and ranking gradients "
                         "SEPARATELY at identical weights and append |g_pae|, |g_rank|, their "
                         "ratio and cos(PAE,rank) to grad_split.csv. 0 (default) is off.\n"
                         "WHY IT EXISTS. --alpha-rank cannot be set from loss values: a rel loss "
                         "of 0.01-0.25 against a PAE loss of 2.3 says nothing about their "
                         "gradient magnitudes. --grad-probe already measures this, but it runs "
                         "ONE window and exits, so every alpha this project has used came from a "
                         "single measurement on one batch of 2-3 structures -- and the five such "
                         "measurements on record span 0.0041 to 0.4728, mostly because they froze "
                         "different amounts of the head. This logs the ratio along the whole run "
                         "instead, so the next arm's alpha comes from a distribution.\n"
                         "THE COST IS ONE EXTRA BACKWARD PER MEASURED POSE, which is why it is "
                         "periodic: at K=50 it is ~2%% of the step budget, at K=1 it roughly "
                         "doubles backward time. It reuses --grad-probe's machinery exactly, so "
                         "the numbers are the same quantity that tool reports.\n"
                         "READ THE ACCUMULATED RATIO, NOT THE PER-POSE ONE. The rel coefficients "
                         "sum to zero within an interface group, so per-pose contributions "
                         "largely cancel -- measured 0.0311 per-pose against 0.0041 accumulated "
                         "on 3hzrs8, a 7.6x difference. The accumulated figure is what the "
                         "optimiser sees.")
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="accept a label whose fingerprint differs ONLY in the `code` field, with "
                         "cif, bio, idx, trunk, loss and schema all matching. The data provenance "
                         "is what determines whether a label is CORRECT; the code hash additionally "
                         "says it is REPRODUCIBLE, and it covers whole files, so editing a docstring "
                         "in phase_j_build_cache.py invalidates all 5,000 labels. OFF by default: a "
                         "drift in any other field, or several fields, is still fatal, and every "
                         "accepted drift is printed with both hashes.")
    ap.add_argument("--allow-stale-labels", action="store_true",
                    help="--pae-entropy ONLY: accept a label whose fingerprint differs in the data "
                         "provenance fields (code, bio, cif, idx) instead of stopping. This is "
                         "sound for exactly one objective and no other. Entropy minimisation reads "
                         "nothing out of the label: the gradient is a function of pred['pae'] "
                         "alone, so a label built against an older bioassembly cannot change what "
                         "is learned -- it can only make the labelled pae_loss that is still "
                         "logged BESIDE it wrong. A `trunk` or `schema` difference is still fatal "
                         "under this flag, because those two say the label is paired with a "
                         "DIFFERENT trunk or laid out differently, which is a shape and ordering "
                         "question rather than a provenance one. Measured on validation/fold on "
                         "2026-11-12: 53 of 83 structures (1,590 labels) differ on bio+code "
                         "because all 83 bioassemblies were re-dumped at 7:34 PM on 2026-11-04, "
                         "about five hours AFTER those labels were written; the 30 built after the "
                         "re-dump match. Every accepted label is counted; the first one prints a "
                         "banner naming the differing fields and the end of the run prints the "
                         "total with the structures, so a run that used this flag cannot have its "
                         "labelled readouts mistaken for trustworthy ones.")
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
    ap.add_argument("--crop-pool", default="",
                    help="DIRECTORY of per-structure crop POOLS from "
                         "run_logging/select_crop_pool.py. Each <pdb>.npz holds N token masks and "
                         "the structure uses mask `epoch %% N`, so it sees a different view every "
                         "epoch instead of one fixed crop -- which is most of why Protenix crops "
                         "randomly at all. REQUIRES --rel-labels to point at an itm_labels_crop "
                         "directory built on the SAME pool: a cropped prediction against an "
                         "uncropped target trains toward something unreachable and nothing would "
                         "refuse. Mutually exclusive with --crop-tokens and --crop-margin.")
    ap.add_argument("--token-keep-union", type=float, default=-1.0,
                    help="STOCHASTIC TOKEN DROPOUT, biased toward the interface. Instead of "
                         "training on the union mask as a fixed crop, draw a fresh token "
                         "subset every epoch: keep each token INSIDE the union with this "
                         "probability and each token outside it with --token-keep-other. "
                         "Requires --crop-tokens, which supplies the union.\n"
                         "WHY. The fixed crop is memorisable and the model NEVER sees distal "
                         "context, which is why o2qczq scored 0.6305 cropped and 0.5387 "
                         "uncropped in-sample. A varying subset cannot be memorised, and "
                         "periphery tokens appearing some of the time means uncropped "
                         "inference is no longer a regime the head has never seen. It also "
                         "shrinks the tensors: a token dropped is its ROW AND COLUMN gone "
                         "from z, not masked to zero, so 0.5 of tokens is 0.25 of the pair "
                         "work.\n"
                         "ONE DRAW PER STRUCTURE PER EPOCH, not per pose: the relative loss "
                         "grades the GAPS between a structure's 50 poses, so poses on "
                         "different token sets would have their score differences measure "
                         "the draw. Per-epoch is exactly per-step here, because "
                         "build_epoch_cross_seed puts every seed of a structure in ONE step "
                         "-- and per-epoch is also what the prefetch worker can compute, "
                         "since it crops units before the step is known.\n"
                         "THE COST BEING ACCEPTED: ipTM pools over the scored pair's token "
                         "pairs and normalises by a d0 set from their count, so a varying "
                         "subset varies the achievable score while the label stays at its "
                         "full-structure value. d0 enters as 1/(1+(d/d0)^2) and so does not "
                         "cancel under the per-interface mean-centring. Negative (default) "
                         "leaves the crop fixed and every existing arm unchanged.")
    ap.add_argument("--token-keep-other", type=float, default=-1.0,
                    help="keep probability for tokens OUTSIDE the union; see "
                         "--token-keep-union. Lower than it, or the union is not a prior.")
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

    # THE ENTROPY TERM JOINS TERMS ONLY WHEN IT IS ON, and TERMS is what reaches trajectory.csv.
    #
    # TERMS is three things at once: the trajectory header, the keys the per-epoch all-reduce
    # packs into one collective, and the list line ~5917 asserts is present in `met` for every
    # pose. That assertion is the reason this is conditional rather than a fourth entry in the
    # literal: a run without --pae-entropy never puts the key in `met`, so an unconditional
    # TERMS would make every ordinary run stop on its first pose. Rebinding here, before any
    # reader runs, keeps one code path for both and makes the term's absence from the log a
    # crash rather than a silently missing column -- which is how a loss term that never
    # actually fired would otherwise look exactly like one that did.
    if a.pae_entropy > 0.0:
        globals()["TERMS"] = TERMS + ("pae_entropy",)
    _G_SIG = None
    if a.rel_g_sigmoid:
        try:
            _m, _w = (float(x) for x in str(a.rel_g_sigmoid).split(":"))
        except Exception:
            raise SystemExit(f"STOP: --rel-g-sigmoid {a.rel_g_sigmoid!r} is not MID:WIDTH, "
                             f"e.g. 0.23:0.1")
        if not _w > 0.0:
            raise SystemExit(f"STOP: --rel-g-sigmoid width {_w} must be positive; at 0 the "
                             f"weight is a hard step with no gradient either side.")
        if a.rel_g_weight:
            raise SystemExit("STOP: --rel-g-sigmoid and --rel-g-weight are two different "
                             "per-pose weights; pass one.")
        if not a.rel_standardize:
            raise SystemExit("STOP: --rel-g-sigmoid needs --rel-standardize, for the same reason "
                             "--rel-g-weight does: the weight is defined on the z-scored error "
                             "and the unstandardised path uses a closed-form gradient.")
        _G_SIG = (_m, _w)
    if a.rel_balance_accept:
        if not a.rel_standardize:
            raise SystemExit("STOP: --rel-balance-accept needs --rel-standardize, for the same "
                             "reason the other per-pose weights do: the weight is defined on the "
                             "z-scored error and the unstandardised path uses a closed-form "
                             "gradient a per-pose weight would invalidate.")
        if a.rel_g_weight or a.rel_g_sigmoid:
            raise SystemExit("STOP: --rel-balance-accept, --rel-g-weight and --rel-g-sigmoid are "
                             "three different per-pose weights; pass one.")
        if not 0.0 < a.rel_balance_accept < 1.0:
            raise SystemExit(f"STOP: --rel-balance-accept {a.rel_balance_accept} is not a DockQ "
                             f"threshold in (0, 1).")
    if a.pae_split_accept:
        # REFUSED EARLY, BEFORE ANY HARDWARE, like the other threshold flags. The split itself is
        # written so that nothing about it can raise mid-epoch -- a diagnostic that kills a
        # training run on pose 40,000 is worse than no diagnostic -- so every reason it could not
        # work has to surface here instead.
        if not 0.0 < a.pae_split_accept < 1.0:
            raise SystemExit(f"STOP: --pae-split-accept {a.pae_split_accept} is not a DockQ "
                             f"threshold in (0, 1).")
        if not a.pae_scored_pairs:
            raise SystemExit("STOP: --pae-split-accept needs --pae-scored-pairs. The split is by "
                             "the DockQ of the interfaces a pose SCORES, and with no scored pair "
                             "there is no interface to look a DockQ up for -- every pose would "
                             "land in the unclassified bucket and the two means would be empty.")
        # THE LABEL SOURCE IS STATTED HERE, BEFORE ANY HARDWARE, and not where it is loaded: the
        # load sits past init_process_group, so refusing there would already have taken a card
        # and, under torchrun, left the other ranks in a barrier until --dist-timeout-min. A
        # missing file is a stat, so it costs nothing to say so first.
        _ps_lab0 = a.rank_labels or os.path.join(a.fold, "results.csv")
        if not os.path.exists(_ps_lab0):
            raise SystemExit(
                f"STOP: --pae-split-accept {a.pae_split_accept:g} needs DockQ labels to put each "
                f"pose in a group, and there are none at {_ps_lab0}. Pass --rank-labels, or drop "
                f"the flag.")
    if a.rank_diag_only:
        if a.rank_loss:
            raise SystemExit("STOP: --rank-diag-only with --rank-loss is contradictory -- the "
                             "ranking loss already runs the scoring and its diagnostics. Pass "
                             "one.")
        if not a.rank_cross_seed:
            raise SystemExit("STOP: --rank-diag-only needs --rank-cross-seed; the per-pose "
                             "scoring is built per interface across its seeds and there is no "
                             "other path to it.")
        if not a.rel_labels:
            raise SystemExit("STOP: --rank-diag-only needs --rel-labels. The diagnostics compare "
                             "the predicted score against the TRUE iTM, and without the label "
                             "cache there is nothing to compare it to.")
    if a.rel_g_weight and not a.rel_standardize:
        raise SystemExit("STOP: --rel-g-weight needs --rel-standardize. The weight is defined on "
                         "the z-scored error, and the unstandardised path uses a closed-form "
                         "gradient that a per-pose weight would invalidate.")
    if a.rel_g_weight < 0.0:
        raise SystemExit(f"STOP: --rel-g-weight {a.rel_g_weight} is negative; a negative power "
                         f"makes the WORST poses dominate.")
    if a.allow_stale_labels and not a.pae_entropy > 0.0:
        raise SystemExit(
            "STOP: --allow-stale-labels is only sound with --pae-entropy, where the gradient is a "
            "function of pred['pae'] alone and the label is never read. Under any labelled "
            "objective a stale label IS the training signal, and accepting one would train "
            "against coordinates from a different bioassembly while reporting an ordinary loss.")

    # CROPPING IS ONLY COHERENT ON THE MINIMAL PATH. The dataset item carries a FULL-SIZE
    # label_dict, so cropping the prediction while the loss reads full-size masks would either
    # raise on a shape or line up by accident and score poses against the wrong residues. The
    # minimal path builds the loss inputs from the same cached label the crop slices, so both
    # sides are cropped together or neither is.
    if False:
        raise SystemExit(
            "unreachable: --crop-margin was deleted 2026-11-13. "
            "residues. Pass --minimal-items with its --alignment-preflight marker.")
    # EXTRA EVAL PASSES, parsed and validated HERE so a typo costs a second rather than surfacing
    # at the first epoch boundary hours in, on three ranks, after a full epoch of training.
    import re as _re_ee          # module scope has no `re`; the l2-sp block imports it locally too
    _EXTRA_EVAL = []
    _ee_names = set()
    for _spec in a.extra_eval:
        _d = {}
        for _kv in str(_spec).split(","):
            _kv = _kv.strip()
            if not _kv:
                continue
            if "=" not in _kv:
                raise SystemExit(f"STOP: --extra-eval {_spec!r}: {_kv!r} is not key=value.")
            _k, _v = _kv.split("=", 1)
            _d[_k.strip()] = _v.strip()
        _unknown = sorted(set(_d) - {"name", "pool", "fold", "seed_base", "budget", "every"})
        if _unknown:
            raise SystemExit(f"STOP: --extra-eval {_spec!r} has unknown key(s) {_unknown}. "
                             f"Known: name, pool, fold, seed_base, budget, every. The crop_tokens "
                             f"key went with --crop-tokens on 2026-11-13.")
        for _req in ("name", "pool", "every"):
            if not _d.get(_req):
                raise SystemExit(f"STOP: --extra-eval {_spec!r} is missing {_req}=.")
        _nm = _d["name"]
        if not _re_ee.fullmatch(r"[A-Za-z0-9_.-]+", _nm):
            raise SystemExit(f"STOP: --extra-eval name {_nm!r} must be filesystem-safe "
                             f"([A-Za-z0-9_.-]+); it names metric_<name>_ep<N>.json.")
        if _nm in _ee_names:
            raise SystemExit(f"STOP: --extra-eval name {_nm!r} is used twice, so the two passes "
                             f"would overwrite each other's metric file.")
        _ee_names.add(_nm)
        if not os.path.exists(_d["pool"]):
            raise SystemExit(f"STOP: --extra-eval {_nm}: pool {_d['pool']} does not exist.")
        _ef = _d.get("fold") or a.fold
        for _need in (_ef, os.path.join(_ef, "results.csv"),
                      os.path.join(_ef, "chain_roles.csv")):
            if not os.path.exists(_need):
                raise SystemExit(f"STOP: --extra-eval {_nm}: {_need} does not exist. A pool is "
                                 f"scored against its OWN fold's results and roles.")
        try:
            _ev = int(_d["every"])
            _bg = int(_d.get("budget") or 0)
            _sb = int(_d["seed_base"]) if _d.get("seed_base") else -1
        except ValueError as _e:
            raise SystemExit(f"STOP: --extra-eval {_nm}: every/budget/seed_base must be "
                             f"integers ({_e}).")
        if _ev <= 0:
            raise SystemExit(f"STOP: --extra-eval {_nm}: every={_ev} must be positive.")
        _EXTRA_EVAL.append(dict(name=_nm, pool=_d["pool"], fold=_ef, seed_base=_sb,
                                budget=_bg, every=_ev))
    _TOK_DROP = (a.token_keep_union >= 0.0) or (a.token_keep_other >= 0.0)
    if _TOK_DROP:
        if not (a.token_keep_union >= 0.0 and a.token_keep_other >= 0.0):
            raise SystemExit("STOP: --token-keep-union and --token-keep-other must BOTH be given; "
                             "one alone leaves the other at its -1 sentinel and the draw "
                             "undefined.")
        for _nm, _v in (("--token-keep-union", a.token_keep_union),
                        ("--token-keep-other", a.token_keep_other)):
            if not (0.0 < _v <= 1.0):
                raise SystemExit(f"STOP: {_nm}={_v} must be in (0, 1]. Zero would delete a whole "
                                 f"population outright, which is a crop and not a dropout.")
        raise SystemExit("STOP: --token-keep-union conditioned its probabilities on the "
                         "--crop-tokens union, which was deleted 2026-11-13 with its builder. "
                         "There is no 'inside' to bias toward any more.")
        if a.token_keep_other > a.token_keep_union:
            raise SystemExit(f"STOP: --token-keep-other {a.token_keep_other} exceeds "
                             f"--token-keep-union {a.token_keep_union}, so the periphery would "
                             f"survive MORE often than the interface. That is not the experiment.")
    if a.epoch_balance and not a.epoch_cover_all:
        raise SystemExit("STOP: --epoch-balance needs --epoch-cover-all; it reorders the deal "
                         "that builder performs and has no effect on the group-of-world builder.")
    if a.pose_batch > 1:
        # EVERY REASON THIS CAN BE WRONG, CHECKED AT STARTUP RATHER THAN AT POSE 0 OF EPOCH 1.
        if a.nsample % a.pose_batch != 0:
            raise SystemExit(
                f"STOP: --pose-batch {a.pose_batch} does not divide --nsample {a.nsample}. The "
                f"groups would be ragged ({a.nsample // a.pose_batch} full and one short), and "
                f"since each group contributes one sample-mean scaled by its own size, the poses "
                f"in the short group would carry a different weight from the rest.")
        if a.pose_batch > a.nsample:
            raise SystemExit(f"STOP: --pose-batch {a.pose_batch} exceeds --nsample {a.nsample}; "
                             f"only poses of ONE unit may share a forward, because the trunk is "
                             f"the thing being reused.")
        if a.rank_loss:
            raise SystemExit("STOP: --pose-batch > 1 is not supported with --rank-loss. A pose's "
                             "ranking coefficient is resolved per pose, and under --rank-form rel "
                             "the PAE mask is a per-pose chain pair -- batched, one mask would "
                             "stand for poses that need not share it.")
        if getattr(a, "emit_iptm_rows", ""):
            raise SystemExit("STOP: --pose-batch > 1 with --emit-iptm-rows would write ONE row "
                             "per group keyed with the first pose's sample index, for a value "
                             "pooled over the group -- the same wrong-sample defect the emit path "
                             "already documents.")
        if getattr(a, "pae_row_frac", 0) and float(a.pae_row_frac) > 0:
            raise SystemExit("STOP: --pose-batch > 1 with --pae-row-frac > 0 would count groups "
                             "where the row diagnostics count poses, and average the per-row "
                             "weighting over the group.")
        if a.loss_only:
            raise SystemExit("STOP: --pose-batch > 1 is not supported under --loss-only; the "
                             "measurement path reports per pose.")
        if not a.pae_only:
            # THE SHARED LABEL IS ONLY JUSTIFIED FOR PAE. _one_pose gives the whole group pose 0's
            # coordinates and mask, which is exact for the PAE target -- measured over all 8,981
            # label files, every pose against pose 0 on the PAE representative atoms, 0 moved. It
            # is NOT exact for pLDDT, PDE or resolved, which read atom-level labels that the
            # per-pose permutation DOES move (median 2 atoms, max 8). Without --pae-only those
            # three losses are live and would be supervised with the first pose's labels for all
            # nb poses, silently. Raised in external review of 03970def; the guard list had every
            # other exclusion and not this one.
            raise SystemExit(
                "STOP: --pose-batch > 1 requires --pae-only. The batched path gives the whole "
                "group pose 0's coordinates and mask, which is exact for the PAE target and NOT "
                "for pLDDT/PDE/resolved -- the per-pose label permutation moves atoms those "
                "losses read (median 2, max 8 of several thousand).")
        if not a.crop_pool:
            raise SystemExit("STOP: --pose-batch > 1 needs a crop. Poses of one unit share a "
                             "token count either way, but the cropped label path is the only one "
                             "that slices labels by RANGE; uncropped goes through "
                             "BC.minimal_loss_inputs, which takes a single pose index.")
    if a.crop_pool:
        if not os.path.isdir(a.crop_pool):
            raise SystemExit(f"STOP: --crop-pool {a.crop_pool} is not a directory. Build it with "
                             f"run_logging/select_crop_pool.py")
        if not a.minimal_items:
            raise SystemExit("STOP: --crop-pool needs --minimal-items, for the same reason "
                             "--crop-margin does: the full dataset item carries FULL-SIZE arrays "
                             "that the crop never reaches.")
        # THE LABELS MUST HAVE BEEN BUILT ON THIS POOL -- WHEN THERE ARE LABELS AT ALL.
        #
        # A cropped prediction against an uncropped target is the one failure in this feature that
        # produces no error at all: the residual absorbs it and the run looks healthy. So whenever
        # a relative-iTM target is in play, the label directory must carry crop_index, which only
        # build_itm_labels_cropped.py writes.
        #
        # BUT THAT ENTIRE HAZARD IS SPECIFIC TO THE RANKING LOSS. --rank-form rel compares a
        # predicted ipTM for a named chain pair against a true iTM for that pair, so target and
        # prediction must have been computed over the same token set. A PAE-only arm has no such
        # target: the PAE loss is a per token-pair cross-entropy taken from the cropped unit's own
        # coordinates, so it is self-consistent under any crop and there is nothing to mismatch.
        # Requiring rel labels there would demand an artefact the run never reads.
        _needs_rel = bool(a.rank_loss) and not a.loss_only
        if _needs_rel and not a.rel_labels:
            raise SystemExit("STOP: --crop-pool needs --rel-labels pointing at labels built on "
                             "the same pool (run_logging/build_itm_labels_cropped.py).")
        _probe = (sorted(f for f in os.listdir(a.rel_labels) if f.endswith(".npz"))
                  if a.rel_labels else [])
        if a.rel_labels and not _probe:
            raise SystemExit(f"STOP: --rel-labels {a.rel_labels} holds no npz files.")
        _pz = (np.load(os.path.join(a.rel_labels, _probe[0]), allow_pickle=True)
               if _probe else {"files": []})
        if _probe and "crop_index" not in _pz.files:
            raise SystemExit(
                f"STOP: --crop-pool is set but --rel-labels {a.rel_labels} has no crop_index -- "
                f"these are UNCROPPED labels. The head would predict a cropped ipTM against a "
                f"full-structure target and nothing downstream would notice. Build cropped labels "
                f"with run_logging/build_itm_labels_cropped.py --pool {a.crop_pool}")

    # ONE SWITCH FOR "IS THIS A CROPPED RUN". --crop-pool was added as a loader and a label
    # lookup but every execution gate still read `crop_margin > 0 or crop_tokens`, so a run given
    # --crop-pool alone trained on FULL structures while the relative loss read CROPPED targets --
    # no error anywhere, and the arm would have been named for a crop it never applied. Caught in
    # external review. Every such gate now reads _USE_CROP.
    # ONE CROP RULE, AND _USE_CROP IS STILL THE SWITCH EVERY GATE READS. --crop-tokens and
    # --crop-margin were deleted 2026-11-13: the first consumed a precomputed mask whose builder
    # no longer exists, the second defined its window from label_coordinate so nothing it
    # produced was a deployment number. The indirection stays because a crop flag that some
    # gates did not consult is the defect _USE_CROP was introduced for.
    _USE_CROP = bool(a.crop_pool)

    # _CROP_TOK_DIGEST IS GONE WITH --crop-tokens. It hashed the mask file's contents so a
    # rebuild under the same name could not resume as 'unchanged'. The crop pool's own
    # identity is covered by crop_pool in the resume guard plus the rel labels' manifest
    # digest, so nothing is left unverified by removing it.
    # A THRESHOLD WITH NOTHING TO TURN OFF IS A NO-OP THAT READS LIKE A SETTING. With
    # --blocks-per-ckpt 0 every unit is already uncheckpointed, so the rule would select between
    # "off" and "off" and the run would look configured when it is not.
    if a.ckpt_token_threshold > 0 and not a.blocks_per_ckpt:
        raise SystemExit(
            "STOP: --ckpt-token-threshold needs --blocks-per-ckpt > 0. With checkpointing already "
            "off for every unit the threshold selects between off and off, which changes nothing "
            "and hides that it changed nothing.")
    # PARSED AT STARTUP, NOT AT THE FIRST STEP. A malformed --lr-steps would otherwise surface
    # minutes in, on three ranks, after the model is built.
    try:
        # IMPORTED HERE, not relied on from later. `import lr_schedule as _LRS` happens ~850 lines
        # below, so using _LRS during startup validation raised UnboundLocalError on every rank
        # before pose 0. Caught by launching and reading epoch 1, which is what epoch 1 is for.
        import lr_schedule as _LRS0
        _LR_STEPS = tuple(_LRS0.parse_lr_steps(a.lr_steps)) if a.lr_steps else ()
    except ValueError as _e:
        raise SystemExit(f"STOP: {_e}")
    if _LR_STEPS and a.lr_final > 0:
        print(f"  NOTE: --lr-steps overrides --lr-final {a.lr_final:g} and the decay window; "
              f"the staircase is what runs.")
    if a.ckpt_token_threshold < 0:
        raise SystemExit("STOP: --ckpt-token-threshold cannot be negative.")
    if a.lean_readouts and not a.pae_only:
        # AT PARSE TIME, NOT AT THE INSTALL SITE. lean_readouts.install() refuses this too, but it
        # runs after the model is built -- minutes in, and on three ranks at once. The flag pair is
        # knowable from the argv alone, so it is checked from the argv alone.
        raise SystemExit(
            "STOP: --lean-readouts needs --pae-only. The three readouts it stops computing are "
            "the ones --pae-only stubs the loss terms for; without it, pLDDT, PDE and resolved "
            "ARE trained, and replacing them with placeholder zeros would train three terms "
            "against constant logits while reporting their losses as if they were real.")
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
    if a.crop_log and not a.crop_pool:
        raise SystemExit("STOP: --crop-log was given but --crop-pool is unset, so nothing is "
                         "cropped.")

    # PIN THE COMMIT NOW, before any work, so every bundle this process writes records the code it
    # is running rather than whatever HEAD has become hours later. Caching it lazily at the first
    # epoch boundary was not enough: that is hours in, and the tree can already have moved.
    _git_commit()

    # REL CONFIG IS CHECKED BEFORE ANY HARDWARE. These refusals were originally placed with the
    # rest of the ranking-loss setup, which runs after CUDA initialisation -- so a
    # misconfiguration cost a GPU allocation to discover, and the gate could not reach them
    # without a free card. A config error should never need hardware to be caught.
    if a.rank_loss and a.rank_form == "rel":
        if not a.rank_cross_seed:
            raise SystemExit(
                "STOP: --rank-form rel needs --rank-cross-seed. The loss centres on an "
                "INTERFACE's poses across all its seeds; within one seed it would centre on "
                f"{a.nsample} poses and remove most of the signal it exists to use.")
        if not a.rel_labels:
            raise SystemExit("STOP: --rank-form rel needs --rel-labels (see "
                             "run_logging/build_itm_labels.py).")
        if not os.path.isdir(a.rel_labels):
            raise SystemExit(f"STOP: --rel-labels {a.rel_labels} is not a directory.")
        _rm = os.path.join(a.rel_labels, "manifest.json")
        if not os.path.exists(_rm):
            raise SystemExit(
                f"STOP: {_rm} is missing, so the label set cannot be identified. Rebuild with "
                f"run_logging/build_itm_labels.py, which writes it.")

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
    # CAPTURE WHAT THE CUDA FAILURE ACTUALLY LOOKED LIKE, rather than inferring it afterwards.
    # tv6hue epoch 4 lost two of three held-out shards inside torch._C._cuda_init() and left only
    # the Python traceback, which does not say which card the child had, what its CPU affinity or
    # memory policy were, how much GPU memory was free at that instant, or how long it had been
    # running. Three hypotheses (NUMA inheritance, simultaneous context creation, memory
    # pressure) all survive that traceback, and a probe reproducing the conditions on an idle box
    # initialised 36 times out of 36 -- so the next occurrence has to be the thing that decides
    # it. This costs one line per --loss-only child and nothing in a training run.
    if a.loss_only:
        try:
            # LOCAL IMPORT: the module-level `subprocess` name does not exist -- the only other
            # import of it in this file is inside _git_commit(). Without this the probe would
            # raise NameError, be swallowed by the except below, and print nothing useful, which
            # is the exact failure mode this block exists to prevent.
            import subprocess as _sp
            with open("/proc/self/status") as _fh:
                _st = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in _fh if ":" in l}
            _free = _sp.run(
                ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
            print(f"  child env: CUDA_VISIBLE_DEVICES="
                  f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}  "
                  f"cpus={_st.get('Cpus_allowed_list')}  mems={_st.get('Mems_allowed_list')}  "
                  f"rss={_st.get('VmRSS')}  torch={torch.__version__}", flush=True)
            print(f"  child sees: {' | '.join(_free.strip().splitlines())}", flush=True)
        except Exception as _e:                   # noqa: BLE001
            print(f"  (child env probe failed: {type(_e).__name__}: {_e})", flush=True)
    _t_cuda = time.time()
    try:
        model, configs = RC.build_model(a.ckpt_dir, a.model_name, dev)
    except Exception as _e:                       # noqa: BLE001
        # The traceback alone cost us the diagnosis last time. Say what the device state was at
        # the moment it failed, then re-raise unchanged so the caller's handling is untouched.
        print(f"  CHILD BUILD FAILED after {time.time() - _t_cuda:.2f}s: "
              f"{type(_e).__name__}: {_e}", flush=True)
        try:
            import torch as _t
            print(f"  at failure: device_count={_t.cuda.device_count()} "
                  f"is_available={_t.cuda.is_available()} "
                  f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
        except Exception:                         # noqa: BLE001
            pass
        raise
    if a.loss_only:
        print(f"  child built model in {time.time() - _t_cuda:.2f}s", flush=True)
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
    # INSTALLED BEFORE BOTH POSE PATHWAYS, AND THAT ORDER IS LOAD-BEARING.
    #
    # pose_direction.py:234 and pose_perblock.py:232 each capture
    # `orig_mef = head.memory_efficient_forward` and REPLACE the attribute with a wrapper that
    # calls what it captured. They are a chain, not independent patches. Installing the lean
    # mirror afterwards would overwrite the outermost wrapper and silently drop the orientation
    # and per-block injections -- except it would not even be silent: the mirror still calls
    # self.pairformer_stack, whose _prep_blocks is also wrapped and raises _NoPose when the
    # one-hot cache is unset, so the run would die at pose 0 instead. Installing FIRST makes the
    # mirror the base of the chain, which is what it is a mirror of. Found in review.
    if a.pair_dropout >= 0.0:
        _blocks = list(getattr(model.confidence_head.pairformer_stack, "blocks", []))
        if not _blocks:
            raise SystemExit("STOP: --pair-dropout: the confidence head's pairformer_stack has "
                             "no .blocks, so there is nothing to set the rate on.")
        _was = sorted({float(getattr(b, "p_drop", -1)) for b in _blocks})
        for _b in _blocks:
            # p_drop IS THE ONE THAT MATTERS: all four pair-rep sites in PairformerBlock pass
            # self.p_drop to dropout_add_rowwise. dropout_row is the unused non-fused path; it is
            # set too so the two cannot disagree if a refactor ever switches paths.
            _b.p_drop = float(a.pair_dropout)
            _dr = getattr(_b, "dropout_row", None)
            if _dr is not None:
                for _obj in (_dr, getattr(_dr, "dropout", None)):
                    for _attr in ("p", "r"):
                        if _obj is not None and hasattr(_obj, _attr):
                            setattr(_obj, _attr, float(a.pair_dropout))
        _now = sorted({float(b.p_drop) for b in _blocks})
        if _now != [float(a.pair_dropout)]:
            raise SystemExit(f"STOP: --pair-dropout did not take: p_drop is {_now} after "
                             f"assignment, not [{a.pair_dropout}].")
        print(f"PAIR DROPOUT: row-wise rate on z set to {a.pair_dropout:g} on {len(_blocks)} "
              f"pairformer block(s), was {_was}. "
              f"confidence_head.training={model.confidence_head.training} "
              f"-- dropout is a NO-OP unless that is True")
    if a.lean_readouts:
        import lean_readouts
        _lr_sha = lean_readouts.install(model, pae_only=a.pae_only)
        print(f"LEAN READOUTS: pde, pLDDT and resolved are NOT COMPUTED. "
              f"memory_efficient_forward pinned at {_lr_sha[:16]}")
    # THE COMPUTE THAT FED THE DEAD READOUTS, which --lean-readouts did not touch. See
    # lean_readouts.drop_single_pathway for why z never depends on s and why this is an attribute
    # flip rather than a rebuild. MUST run after install(), because install() refuses to be the top
    # of the patch chain and this reads the same module tree.
    _dead_head = []
    if a.lean_single:
        import lean_readouts as _LS
        if not (a.lean_readouts and a.pae_only):
            raise SystemExit(
                "STOP: --lean-single requires --lean-readouts AND --pae-only. It turns off the "
                "pairformer's single track, which is what feeds pLDDT and resolved; without both "
                "of those the two heads ARE trained and would be trained against a representation "
                "that is no longer updated -- silently, since the shapes still line up.")
        _n_blk, _n_dead = _LS.drop_single_pathway(model)
        _dead_head = _LS.dead_head_params(model.confidence_head, single_dropped=True)
        print(f"LEAN SINGLE: attention_pair_bias and single_transition are NOT CALLED in "
              f"{_n_blk} pairformer block(s) (c_s set to 0; the modules stay in the tree, so the "
              f"checkpoint's key set is unchanged). {_n_dead/1e6:.3f}M head parameter(s) across "
              f"{len(_dead_head)} tensor(s) now provably receive no gradient; they are dropped "
              f"from the optimiser below. PairformerBlock.forward pinned at "
              f"{_LS.PINNED_BLOCK_SHA[:16]}, PairformerStack.forward at "
              f"{_LS.PINNED_STACK_SHA[:16]}")
    # AFTER lean_readouts AND --lean-single, BEFORE pose_direction/pose_perblock. It wraps
    # memory_efficient_forward and replaces the stack's _prep_blocks, so lean_readouts -- which
    # REPLACES that attribute rather than wrapping it -- must already be in place; its refusal list
    # carries this module's marker so the wrong order is a hard stop.
    if a.pose_inblock:
        # AND NOT ALONGSIDE --pose-perblock. Both replace the stack's _prep_blocks, so whichever
        # installs second wins and the other's injection silently disappears -- pose_perblock also
        # raises _NoPose from its own _prep_blocks if it is the loser, which is a symptom and not a
        # check. This combination has not been reasoned through and is refused rather than ordered.
        if a.pose_perblock:
            raise SystemExit(
                "STOP: --pose-inblock with --pose-perblock. Both replace "
                "pairformer_stack._prep_blocks, so one of the two injections is silently dropped "
                "depending on install order. If you want the pose at every depth AND inside the "
                "last block, that is a third design and needs its own module.")
        import pose_inblock
        _pib_blk = a.pose_inblock_block
        if _pib_blk < 0:
            _pib_blk = len(model.confidence_head.pairformer_stack.blocks) - 1
        _pib = pose_inblock.install(model, block_index=_pib_blk, pae_inject=a.pose_inblock_pae)
    elif a.pose_inblock_pae:
        raise SystemExit("STOP: --pose-inblock-pae without --pose-inblock. The readout injection "
                         "shares the pose geometry the in-block wrapper computes and stashes, so "
                         "on its own it would never fire and the flag would be a silent no-op.")
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
    if a.pose_readout:
        # AFTER both stack injections, because it refuses unless one of them is installed: it
        # reads their per-forward geometry rather than recomputing cdist and the one-hot.
        import pose_readout
        _prd = pose_readout.install(model, copy_distance=a.pose_readout_copy_distance)
    elif a.pose_readout_copy_distance:
        raise SystemExit("STOP: --pose-readout-copy-distance without --pose-readout. It selects "
                         "how the fifth site is initialised, not whether it exists.")
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
    # "m" is the scored-cross-chain pair mask; "w" the per-anchor-row multiplier --pae-row-frac
    # installs; "bins"/"pm" are stashed by _masked_calc on diagnostic poses only, so the per-row
    # loss can be read back AFTER the forward without recomputing the label.
    _pae_keep = {"m": None, "wm": None, "blocks": None, "bins": None, "pm": None,
                 "stash": False, "pdb": None, "unit": None, "replay": None, "unw_label": None}
    _scored_asym = {}
    _pae_mask_stats = {"masked": 0, "unmasked": 0}
    if a.emit_iptm_rows and not a.loss_only:
        raise SystemExit(
            "STOP: --emit-iptm-rows is only supported with --loss-only. rank_loss.iptm is the "
            "DIFFERENTIABLE scorer -- 'minus its torch.no_grad' -- so in a grad-enabled run it "
            "builds an fp32 [1,N,N,64] softmax that stays reachable from the loop variable across "
            "calculate_losses and backward, at the memory peak (2.96 GiB at --crop-size 3400). "
            "--loss-only already wraps the pose in torch.no_grad, which is why it is safe there.")
    _iface_of = {}
    # LAZILY OPENED, so a run without --emit-iptm-rows touches no file. Rows are flushed per unit
    # rather than held to the end: a measurement pass that dies on its last unit should still
    # leave the rows it earned, the same reason the subprocess trajectories are the durable record.
    _iptm_fh = {"fh": None, "w": None, "n": 0}
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
                # THE POOL'S OWN REFERENCE NAMING, kept so an emitted row can carry the `iface`
                # string results.csv uses. The asym pair is a MODEL-chain index pair and says
                # nothing about which reference chains it came from.
                _iface_of[(_r["pdb"], (min(_a1, _a2), max(_a1, _a2)))] = \
                    f"{_r['pdb']}_{_r['chain1']}{_r['chain2']}"
        if _miss:
            raise SystemExit(f"STOP: {len(_miss)} pool interface(s) have no role mapping, e.g. "
                             f"{_miss[:4]}. The PAE mask would silently drop their blocks.")
        _orig_calc = loss_fn.pae_loss.calculate_label

        def _masked_calc(*args, **kw):
            # REPLAY SERVES THE SECOND, UNWEIGHTED FORWARD of the same pose without recomputing
            # the label. calculate_label does the frame alignment; only the cross-entropy needs
            # redoing to read the unweighted number, and that number has to be per-pose rather
            # than sampled so trajectory.csv stays comparable with every unweighted arm.
            rp = _pae_keep["replay"]
            if rp is not None:
                return rp
            sq, bins, pm = _orig_calc(*args, **kw)
            m = _pae_keep["m"]
            if m is not None:
                pm = pm * m.to(pm.dtype)
            # THE WEIGHTED MASK MUST BE FLOAT, EXPLICITLY. PAELoss reduces as
            # sum(loss * pair_mask) / sum(pair_mask) -- already a weighted mean -- so a real-valued
            # mask is a weighted PAE term with no change to Protenix. But calculate_label returns a
            # BOOL pair_mask, so the earlier `w.to(pm.dtype)` cast every weight to True and the
            # whole term was a silent no-op that still logged itself as weighted. Caught in review;
            # test_pae_row_weight.py now asserts the weighted mask DIFFERS from the unweighted one.
            wm = _pae_keep["wm"]
            if wm is not None:
                pm = pm.to(torch.float32) * wm.to(torch.float32)
            _pae_keep["unw_label"] = (sq, bins, (pm > 0).to(torch.float32) if wm is not None
                                      else pm)
            if _pae_keep["stash"]:
                _pae_keep["bins"], _pae_keep["pm"] = bins, pm
            return sq, bins, pm

        loss_fn.pae_loss.calculate_label = _masked_calc
        print(f"PAE term MASKED to scored chain pairs: {len(_scored_asym)} structure(s), "
              f"{sum(len(v) for v in _scored_asym.values())} interface(s), from "
              f"{a.pae_scored_pairs}")

    # ------------------------------------------------------------------ ROW-WEIGHTED PAE TERM
    # WHY THIS EXISTS. ipTM is max_i t_i over the anchor rows, where t_i is row i's mean predicted
    # cross-chain TM weight. Selection therefore reads ONE row of ~332 while the PAE term grades
    # every row equally. This reweights the term toward the rows selection can read.
    #
    # WHAT IT DOES NOT DO, and this is the reason to prefer it to a ranking loss: it never changes
    # the LABEL. Every pair is still trained toward its own true error bin, so the term cannot be
    # lowered by distorting ipTM away from calibration, and it stays sensitive to the absolute
    # confidence level that a softmax ranking loss is blind to by construction.
    #
    # THE WEIGHTS ARE DETACHED. If they carried gradient the model could lower the loss by moving
    # weight onto rows it is already good at instead of getting better.
    if float(a.pae_row_frac) > 0:
        if not a.pae_scored_pairs:
            raise SystemExit("STOP: --pae-row-frac needs --pae-scored-pairs. t_i is a mean over a "
                             "row's SCORED cross-chain targets; with no pool there is no cross "
                             "block to average over and the ranking of rows would be meaningless.")
        if not 0.0 < float(a.pae_row_mass) < 1.0:
            raise SystemExit(
                f"STOP: --pae-row-mass must be in (0, 1), got {a.pae_row_mass}. At 1.0 every row "
                f"outside the top set gets weight 0, which is a hard top-k mask: those rows are "
                f"then unsupervised and free to drift over-confident, and since ipTM is a max an "
                f"over-confident row is exactly the one selection reads. Pass 0.8 for the "
                f"16x-focus-with-a-floor this flag is for.")
        if not 0.0 < float(a.pae_row_frac) <= 1.0:
            raise SystemExit(f"STOP: --pae-row-frac must be in (0, 1], got {a.pae_row_frac}.")
        if int(a.pae_row_floor) < 1:
            raise SystemExit(f"STOP: --pae-row-floor must be >= 1, got {a.pae_row_floor}.")
        # THE MATH LIVES IN phases/pae_row_weight.py so a test can import the same code this
        # run uses. run_logging/test_pae_row_weight.py asserts max(row_t(...)) equals Protenix's
        # own calculate_iptm, and that frac 1.0 reproduces the unweighted term bit for bit.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pae_row_weight as _rw
        _orig_pae_fwd = loss_fn.pae_loss.forward
        _pl = loss_fn.pae_loss
        _rf, _rm, _rfl = float(a.pae_row_frac), float(a.pae_row_mass), int(a.pae_row_floor)
        _row_seen = {"n": 0}
        _row_diag_p = os.path.join(a.out, f"pae_rows_rank{rank}.csv")
        _row_diag_fh = {"fh": None, "w": None}
        _pae_unw = {"v": float("nan")}

        def _unweighted(kw):
            """The m-masked but row-UNWEIGHTED loss for this pose, every pose.

            WHY EVERY POSE. trajectory.csv's pae_loss column is whatever the trained term returns,
            and that is now a reweighted mean. A mean-1 weight does NOT preserve the value unless
            the per-row loss is uncorrelated with the weight -- and the weight is chosen BY t_i,
            which the rho column exists to show is correlated. So the weighted number is not
            comparable with any unweighted arm, and a 1-in-50 sample is not a curve. The label is
            replayed rather than recomputed, so this costs one cross-entropy, not an alignment.
            """
            lab = _pae_keep["unw_label"]
            if lab is None:
                return float("nan")
            try:
                with torch.no_grad():
                    _pae_keep["replay"] = lab
                    return float(_orig_pae_fwd(**kw))
            except Exception as _e:
                print(f"  WARNING: unweighted PAE readback failed "
                      f"({type(_e).__name__}: {_e})", flush=True)
                return float("nan")
            finally:
                _pae_keep["replay"] = None

        def _row_log(pdb, unit, n, weighted, unw, kw):
            """Realised effective sample size, t_i quantiles, and the rank correlation between
            t_i and that row's own PAE loss.

            THE CORRELATION IS THE MEASUREMENT. If the high-t_i rows are already the best-fit ones
            then this weighting has no headroom, and that is answerable from the first epoch rather
            than from DockQ sixty epochs later.
            """
            try:
                with torch.no_grad():
                    bins, pm = _pae_keep["bins"], _pae_keep["pm"]
                    rho, rl_mean, ess, k_real = [float("nan")] * 3 + [0]
                    tq = [float("nan")] * 3
                    if bins is not None and pm is not None:
                        ess = _rw.effective_sample_size(pm)
                        hf = kw["has_frame"].reshape(-1).bool()
                        # CHUNKED for the same reason row_t is: a full fp32 log_softmax over the
                        # crop is ~555 MB at our largest, inside the live autograd graph.
                        lg = kw["logits"]
                        while lg.dim() > 4:
                            lg = lg[0]
                        lg = lg[:, hf] if lg.dim() == 4 else lg[hf].unsqueeze(0)
                        bq = bins if bins.dim() == lg.dim() else bins.unsqueeze(0)
                        nrow = int(hf.sum())
                        ce = torch.zeros(nrow, lg.shape[-2], device=lg.device,
                                         dtype=torch.float32)
                        for lo in range(0, nrow, _rw.ROW_CHUNK):
                            hi = min(lo + _rw.ROW_CHUNK, nrow)
                            c = lg[:, lo:hi].float()
                            ce[lo:hi] = -(bq[:, lo:hi].float()
                                          * torch.log_softmax(c, dim=-1)).sum(-1).mean(0)
                            del c
                        base = (pm > 0).to(ce.dtype)     # the unweighted support of the same mask
                        cnt = base.sum(-1)
                        rl = (ce * base).sum(-1) / cnt.clamp_min(1.0)
                        ok = cnt > 0
                        k_real = int(ok.sum())
                        if k_real > 2:
                            tt = pm.to(torch.float64).sum(-1)      # realised row weight
                            def _rk(x):
                                return torch.argsort(torch.argsort(x)).to(torch.float64)
                            x, y = _rk(tt[ok]), _rk(rl[ok].double())
                            x, y = x - x.mean(), y - y.mean()
                            den = float(x.norm() * y.norm())
                            if den > 0:
                                rho = float((x * y).sum() / den)
                            q = torch.tensor([0.5, 0.9, 1.0], dtype=torch.float64,
                                             device=tt.device)
                            tq = [float(v) for v in torch.quantile(tt[ok], q)]
                        rl_mean = float(rl[ok].mean()) if k_real else float("nan")
                row = {"pdb": pdb, "unit": unit, "n_rows": n, "n_scored_rows": k_real,
                       "ess": round(ess, 2), "pae_weighted": round(float(weighted), 6),
                       "pae_unweighted": round(float(unw), 6),
                       "row_loss_mean": round(rl_mean, 6),
                       "rho_rowweight_vs_rowloss": round(rho, 6),
                       "rw_p50": round(tq[0], 6), "rw_p90": round(tq[1], 6),
                       "rw_max": round(tq[2], 6)}
                if _row_diag_fh["fh"] is None:
                    _row_diag_fh["fh"] = open(_row_diag_p, "a", newline="")
                    _row_diag_fh["w"] = csv.DictWriter(_row_diag_fh["fh"], fieldnames=list(row))
                    if _row_diag_fh["fh"].tell() == 0:
                        _row_diag_fh["w"].writeheader()
                _row_diag_fh["w"].writerow(row)
                _row_diag_fh["fh"].flush()
            except Exception as _e:
                # NAMED, NOT SWALLOWED. An OOM here used to leave pae_rows_rank*.csv silently
                # empty while the launch note called it the diagnostic that answers the premise.
                print(f"  WARNING: row diagnostic failed for {pdb}/{unit} "
                      f"({type(_e).__name__}: {_e})", flush=True)

        def _row_weighted_fwd(**kw):
            hf = kw["has_frame"].reshape(-1).bool()
            n = int(hf.sum())
            _row_seen["n"] += 1
            diag = (int(a.pae_row_log_every) > 0
                    and _row_seen["n"] % int(a.pae_row_log_every) == 0)
            _pae_keep["unw_label"] = None
            blocks = _pae_keep["blocks"]
            if n == 0 or not blocks:
                _pae_keep["wm"] = None
                return _orig_pae_fwd(**kw)
            with torch.no_grad():
                _pae_keep["wm"] = _rw.weighted_mask(
                    kw["logits"], hf, blocks, _rf, _rm, _rfl, _pl.min_bin, _pl.max_bin)
            _pae_keep["stash"] = bool(diag)
            try:
                out = _orig_pae_fwd(**kw)
            finally:
                _pae_keep["wm"], _pae_keep["stash"] = None, False
            unw = _unweighted(kw)
            _pae_unw["v"] = unw
            if diag:
                _row_log(_pae_keep.get("pdb"), _pae_keep.get("unit"), n, out, unw, kw)
            _pae_keep["bins"] = _pae_keep["pm"] = _pae_keep["unw_label"] = None
            return out

        loss_fn.pae_loss.forward = _row_weighted_fwd
        _bk = min(332, max(_rfl, -(-int(_rf * 332) // 1)))
        _bw = _rw.row_weights(torch.arange(332, dtype=torch.float32), None, _rf, _rm, _rfl)[0]
        print(f"PAE term ROW-WEIGHTED, PER SCORED CHAIN PAIR: within each pair's block the top "
              f"{_rf:.0%} of its rows by t_i (floor {_rfl}) carry {_rm:.0%} of that block's loss "
              f"and the rest share {1 - _rm:.0%}. On a 332-row block that is k={_bk}, a top row "
              f"weighing {float(_bw.max()):.2f} against {float(_bw[_bw > 0].min()):.2f} "
              f"({float(_bw.max()) / float(_bw[_bw > 0].min()):.1f}x), effective sample size "
              f"{_rw.predicted_ess(332, _rf, _rm, _rfl):.0f} of 332. Weights DETACHED, mask FLOAT. "
              f"The unweighted value is read back EVERY pose into pae_loss_unweighted; the full "
              f"diagnostic runs every {a.pae_row_log_every} pose(s) -> "
              f"{os.path.basename(_row_diag_p)}")

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
    # RELATIVE-iTM LABEL CACHE, loaded lazily per structure and kept. One <pdb>.npz is a few KB
    # and a structure recurs every epoch, so reading it once per process is right; loading all 368
    # at startup would be wasted on a rank that never sees most of them.
    _rel_cache = {}
    _rel_stat = {"groups": 0, "poses": 0, "short_groups": 0, "csum": 0.0, "missing_files": 0,
                 # --pae-split-accept LIVES IN THIS DICT so it is reset at the same epoch
                 # boundary as every other per-epoch accumulator here. A module-level or
                 # per-unit accumulator would have carried epoch 1's poses into epoch 2's mean
                 # and the curve would have looked like a slowly flattening loss.
                 "ps_acc_sum": 0.0, "ps_acc_n": 0, "ps_un_sum": 0.0, "ps_un_n": 0,
                 "ps_straddle": 0, "ps_unclass": 0, "ps_batched": 0}

    # THE SAME MODULE phase_k_eval USES, so the per-pose chain join is literally the evaluator's
    # and cannot drift from it.
    import phase_k_eval as _PKE
    # THE SAME ROLES FILE THE PAE MASK USES. Defined here rather than reaching for _roles_p
    # because that name is bound inside `if a.pae_scored_pairs:` and this must be safe either way.
    _rel_roles_p = a.pae_scored_roles or os.path.join(a.fold, "chain_roles.csv")
    _rel_refs = (_PKE.pool_ref_pairs(a.pae_scored_pairs)
                 if (a.rank_form == "rel" and a.pae_scored_pairs) else {})
    # _roles_p, NOT <fold>/chain_roles.csv. Raised in review and correct: --pae-scored-roles
    # already steered the PAE mask and RKL.load_labels at line 1769, but these two calls still
    # hardcoded the default file. chain_roles.csv names 501 structures and interface_chain_pairs
    # SKIPS an interface whose roles it cannot find, so under --rank-form rel every wave-1
    # structure would have had no _rel_ifaces while pass A built _pose_pairs from the full pool --
    # surfacing as _exp != _got, or as silently incomplete rel diagnostics. chain_roles_all.csv
    # covers 853 of 853; the point of the flag is that both readers use it.
    _rel_ifaces = {_p: [t[0] for t in _l] for _p, _l in
                   ((_PKE.interface_chain_pairs(a.pae_scored_pairs, _rel_roles_p)[0]).items()
                    if (a.rank_form == "rel" and a.pae_scored_pairs) else {}.items())}
    # IFACE -> STATIC ASYM PAIR, for the DockQ lookup ONLY. _rank_lab is keyed by the asym pair
    # rank_loss.parse_iface derives from chain_roles, while rel keys everything by the interface
    # name. DockQ is a property of the INTERFACE -- results.csv is keyed by its `iface` column --
    # so which asym representation indexes it is an internal detail and the static one is
    # correct here. This is NOT the same question as which pair to SCORE, which is per pose.
    _rel_dqkey = {}
    if a.rank_form == "rel" and a.pae_scored_pairs:
        for _p2, _l2 in _PKE.interface_chain_pairs(
                a.pae_scored_pairs, _rel_roles_p)[0].items():
            for _ifc2, _x2, _y2 in _l2:
                _rel_dqkey[_ifc2] = (min(_x2, _y2), max(_x2, _y2))

    def _spearman(xs, ys):
        """Rank correlation without scipy: rank both, then Pearson. Ties get average ranks."""
        n = len(xs)
        if n < 3:
            return None

        def rank(v):
            order = sorted(range(n), key=lambda k: v[k])
            r = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r

        rx, ry = rank(xs), rank(ys)
        mx, my = sum(rx) / n, sum(ry) / n
        sx = sum((v - mx) ** 2 for v in rx) ** 0.5
        sy = sum((v - my) ** 2 for v in ry) ** 0.5
        if sx == 0 or sy == 0:
            return None
        return sum((a_ - mx) * (b_ - my) for a_, b_ in zip(rx, ry)) / (sx * sy)

    _ost_cache = {}

    def _pose_pairs(pdb, lab_p, samp):
        """[(iface, ax, ay)] for ONE pose, from OST -- the SAME join phase_k_eval scores with.

        WHY NOT THE STATIC PAIRS. --rank-form rel's labels are computed on the pose's OST-resolved
        pair; scoring the static pair and looking the label up by interface name would pair
        predicted ipTM for one chain pair with true iTM for another. Measured: the two mappings
        disagree on 1,095 of 18,400 (structure, seed, pose) checks across 52 of 368 structures,
        and 7stf resolves 7stf_BC to (3,1) statically against (3,0) per pose -- a different chain,
        not a reordering. The first version of this arm had exactly that mismatch and no coverage
        check could see it, because the keys still matched. Caught in review.

        REFUSES rather than falling back. A fallback to the static pair is the defect itself.
        """
        import phase_j_build_cache as _BC
        key = (pdb, os.path.basename(lab_p), int(samp))
        if key in _ost_cache:
            return _ost_cache[key]
        sd = _rank_key(pdb, os.path.basename(lab_p)[:-4], (0, 0))[1]
        try:
            ud = _BC.seed_dir(a.fold, pdb, int(sd))
        except Exception as _e:                              # noqa: BLE001
            raise SystemExit(
                f"STOP: --rank-form rel cannot resolve a unit directory for {pdb}/{sd} "
                f"({type(_e).__name__}), so this pose's chains cannot be mapped the way its "
                f"label was built.")
        got, why = _PKE.ost_chain_pairs(ud, pdb, int(samp), _rel_refs.get(pdb) or [])
        if not got:
            # `not got`, NOT `got is None`. ost_chain_pairs returns (None, why) when it cannot
            # read a mapping, but it returns an EMPTY LIST when it reads one and resolves no
            # interface from it -- and sorted([]) passed straight through here, emitting zero
            # keys for every pose of the structure while raising nothing. Pass A would then come
            # up short and the completeness guard would blame a missing score, several hundred
            # units after the real cause. An empty mapping is as fatal as no mapping.
            raise SystemExit(
                f"STOP: --rank-form rel has no OST chain mapping for {pdb}/{sd} sample {samp} "
                f"({why or 'the mapping resolved no scored interface'}). Falling back to the "
                f"static pair is the exact mismatch this resolution exists to prevent. "
                f"RECOVERY: if that pose's ost_json is a FAILURE record, "
                f"`python -I run_logging/rescue_ost_mapping.py --pdb {pdb} --verify` re-runs the "
                f"fixed scorer on CPU and writes OST's own mapping beside it.")
        _ost_cache[key] = sorted(got)
        return _ost_cache[key]

    def _rel_labels(pdb):
        """{(seed, sample, ax, ay): g} for one structure, from --rel-labels.

        A MISSING FILE IS FATAL, not empty. An empty dict would make every group "short", the loop
        would skip them all, and the run would train on PAE alone while its log and saved guard
        both said --rank-form rel -- the row-weight bool-cast failure in another costume.
        """
        # UNDER --crop-pool THE CACHE KEY CARRIES THE CROP. The label changes with the crop, so a
        # cache keyed by pdb alone would serve epoch 0's target for the whole run -- the silent
        # failure this path exists to prevent. n_crop comes from the POOL, which _crop_masks also
        # uses, so the two cannot drift apart.
        if a.crop_pool:
            _ent = _CROPPOOL.get(pdb)
            if _ent is None:
                raise SystemExit(f"STOP: --crop-pool has no entry for {pdb}, but its labels were "
                                 f"requested. Pool and manifest disagree.")
            _cidx = int(_crop_epoch[0]) % len(_ent[0])
            if (pdb, _cidx) in _rel_cache:
                return _rel_cache[(pdb, _cidx)]
        elif pdb in _rel_cache:
            return _rel_cache[pdb]
        if not a.rel_labels:
            raise SystemExit("STOP: --rank-form rel needs --rel-labels; build it with "
                             "run_logging/build_itm_labels.py")
        f = os.path.join(a.rel_labels, f"{pdb}.npz")
        if not os.path.exists(f):
            _rel_stat["missing_files"] += 1
            raise SystemExit(
                f"STOP: no relative-iTM label file for {pdb} at {f}. Every training structure "
                f"needs one; rebuild with run_logging/build_itm_labels.py and check its manifest "
                f"covers this manifest-selection.")
        z = np.load(f, allow_pickle=True)
        col = "g_max" if a.rank_pool == "max" else "g_mean"
        # THE LABEL MUST MATCH THE POOLING. Training the mean-pooled score toward a max-pooled
        # label would regress onto a different quantity than the one being produced, and both
        # columns exist precisely so this is a lookup rather than a conversion.
        if "iface" not in z.files:
            raise SystemExit(
                f"STOP: {f} is an OLD label cache keyed by an asym pair. Those caches paired a "
                f"value computed on the pose's OST pair with the STATIC pair as its key, which is "
                f"the mismatch review caught. Rebuild with run_logging/build_itm_labels.py.")
        if a.crop_pool:
            # KEYED BY CROP, AND THE KEY MUST MATCH THE MASK _crop_masks CHOSE. Both use
            # `epoch % n`, so a label built for crop 7 is only ever paired with the mask of crop 7.
            # If the two ever disagreed the head would be graded against a different crop's target
            # and the loss would still look perfectly healthy, which is why this is recomputed from
            # the same expression rather than passed in.
            if "crop_index" not in z.files:
                raise SystemExit(
                    f"STOP: --crop-pool is set but {f} has no crop_index. These are UNCROPPED "
                    f"labels; the head would be graded against a full-structure target. Rebuild "
                    f"with run_logging/build_itm_labels_cropped.py")
            _n = int(z["n_crop"])
            if _n != len(_CROPPOOL[pdb][0]):
                raise SystemExit(
                    f"STOP: {pdb} has {len(_CROPPOOL[pdb][0])} crop(s) in --crop-pool but its "
                    f"label cache reports {_n}. They were built from different pools, so mask k "
                    f"and label k are different crops and the head would be graded against the "
                    f"wrong target with no error anywhere.")
            _ci = _cidx
            _m = np.asarray(z["crop_index"]) == _ci
            if not _m.any():
                raise SystemExit(
                    f"STOP: {f} has no rows for crop {_ci} of {_n}. The label cache and the crop "
                    f"pool disagree about how many crops {pdb} has -- they were built from "
                    f"different pools.")
            d = {(int(sd), int(sp), str(ifc)): float(g)
                 for ifc, sd, sp, g in zip(z["iface"][_m], z["seed"][_m], z["sample"][_m],
                                           z[col][_m])}
            # NOT CACHED ACROSS EPOCHS. The label changes with the crop, so a cache keyed by pdb
            # alone would serve epoch 0's target forever -- the exact silent failure this whole
            # path exists to avoid.
            _rel_cache[(pdb, _ci)] = d
            return d
        d = {(int(sd), int(sp), str(ifc)): float(g)
             for ifc, sd, sp, g in zip(z["iface"], z["seed"], z["sample"], z[col])}
        _rel_cache[pdb] = d
        return d

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
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lr_schedule as _LRS
    _lr_horizon = [0]
    # STEPS PER EPOCH THE HORIZON WAS CALIBRATED FROM, 0 when cross-seed packing is not in use.
    # build_epoch_cross_seed compares its own packing against it every epoch.
    _xs_spe = [0]
    _probe = {"rows": [], "units": 0, "poses": 0, "sum": {}, "finite": {}, "nonzero": {}}
    # ONE-ELEMENT LIST, NOT A BARE BOOL, because the per-pose measurement site is inside a nested
    # function and rebinding a plain name there would create a local instead of flipping this.
    _gn_on = [False]
    _gn_fh = [None]

    def _gn_emit(step_i, ep_i):
        """Append this step's PAE-vs-ranking gradient split to grad_split.csv, then clear it.

        THE SAME QUANTITY --grad-probe REPORTS, by construction: both read _probe["sum"], which
        the per-pose site fills with two separate autograd.grad calls at identical weights. The
        difference is only that this one runs every K steps and does not exit.

        ACCUMULATED, AND SUMMED ACROSS RANKS, because that is what the optimiser sees. ||sum_i
        g_i|| is not sum_i ||g_i||, and the rel coefficients sum to zero within an interface
        group, so the per-pose figure overstates the surviving magnitude -- 0.0311 against 0.0041
        measured on 3hzrs8. The all-reduce is what makes this the real step gradient rather than
        this rank's share of it.

        EVERY RANK WRITES ITS OWN FILE and rank 0's is the one to read, matching how the
        trajectory rows are already split. A collective here means every rank must reach it, so
        it is called from the step boundary that every rank passes.
        """
        # NO EARLY RETURN BEFORE THE COLLECTIVE, AND THE VECTOR IS ALWAYS THE SAME LENGTH.
        #
        # The first version returned when this rank had no entries and built the vector from
        # `_n in _d`, so a rank whose units carried no rel coefficient would have skipped the
        # all_reduce -- or reached it with a shorter tensor -- while the others waited. Both are
        # hangs, not errors, and this file's own comments record two of them. Every rank walks
        # trainable_params in full and substitutes zeros, so the shape is a function of the
        # parameter list alone and participation is unconditional.
        _sm = _probe.get("sum") or {}
        _flat = {}
        for _nm in ("pae", "rank"):
            _d = _sm.get(_nm) or {}
            _v = torch.cat([(_d[_n].reshape(-1) if _n in _d
                             else torch.zeros(_pp.numel(), device=_pp.device,
                                              dtype=torch.float32))
                            for _n, _pp in trainable_params])
            if world > 1:
                torch.distributed.all_reduce(_v, op=torch.distributed.ReduceOp.SUM)
            _flat[_nm] = _v
        _gp, _gr = _flat["pae"], _flat["rank"]
        _np_, _nr = float(_gp.norm()), float(_gr.norm())
        _cos = (float(torch.dot(_gp, _gr)) / (_np_ * _nr)) if (_np_ > 0 and _nr > 0) else float("nan")
        if _gn_fh[0] is None:
            _gn_fh[0] = open(os.path.join(a.out, f"grad_split.r{rank}.csv"), "a", buffering=1)
            if _gn_fh[0].tell() == 0:
                _gn_fh[0].write("epoch,step,g_pae,g_rank,ratio,cos,alpha_rank,"
                                "alpha_for_parity,poses\n")
        # alpha_for_parity IS THE POINT OF THE FILE: the alpha that would make the ranking
        # gradient match the PAE one AT THIS STEP, given the alpha actually in force. The ratio
        # already includes a.alpha_rank, so dividing it out first is what makes the answer
        # independent of the arm's own setting.
        _r = (_nr / _np_) if _np_ > 0 else float("nan")
        _par = (float(a.alpha_rank) / _r) if (_r and _r == _r and _r > 0) else float("nan")
        _gn_fh[0].write(f"{ep_i},{step_i},{_np_:.6e},{_nr:.6e},{_r:.6e},{_cos:+.4f},"
                        f"{a.alpha_rank:g},{_par:.4g},{_probe.get('poses', 0)}\n")
        _probe["sum"] = {}
        _probe["poses"] = 0
    # THE IMPORT FOLLOWS THE MEASUREMENT, NOT JUST --grad-probe. This was gated on a.grad_probe
    # alone while the per-pose measurement site now also runs under --grad-norm-every, so the
    # first measured step died with "cannot access free variable 'RKL_probe'" on all three ranks.
    # Both flags reach the same code, so both must bring the module in.
    if a.grad_probe or a.grad_norm_every:
        import rank_loss as RKL_probe
        if not a.rank_loss:
            raise SystemExit(
                f"STOP: {'--grad-probe' if a.grad_probe else '--grad-norm-every'} needs "
                f"--rank-loss; there is nothing to compare the PAE gradient against.")
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
        if a.rank_form == "rel":
            # The refusals for this form run EARLY, before any hardware -- see the block beside
            # the LAYERNORM_TYPE check. This is only the report.
            _man = os.path.join(a.rel_labels, "manifest.json")
            _mj = json.load(open(_man)) if os.path.exists(_man) else {}
            print(f"RELATIVE iTM REGRESSION on chain_pair ipTM pooled by {a.rank_pool.upper()}: "
                  f"L = (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2 per interface over all "
                  f"its seeds, alpha {a.alpha_rank:g}"
                  + (", STANDARDISED by the within-interface std" if a.rel_standardize else "")
                  + (", WEIGHTED by the true within-interface std" if a.rel_sd_weight else "")
                  + (f", PER-POSE WEIGHTED by g^{a.rel_g_weight:g}" if a.rel_g_weight else "")
                  + (f", PER-POSE WEIGHTED by sigmoid((g-{_G_SIG[0]:g})/{_G_SIG[1]:g})"
                     if _G_SIG else "")
                  + (f", CLASS-BALANCED per list at DockQ {a.rel_balance_accept:g} "
                     f"(acceptable poses carry half the list's loss)"
                     if a.rel_balance_accept else "")
                  + f". Labels {a.rel_labels} ({_mj.get('structures', '?')} structure(s), "
                  f"{_mj.get('rows', '?')} row(s), column "
                  f"{'g_max' if a.rank_pool == 'max' else 'g_mean'}, digest "
                  f"{str(_mj.get('digest', '?'))[:12]}). The coefficients sum to EXACTLY zero, so "
                  f"this term cannot move an interface's level and its accumulated gradient "
                  f"cancels by construction -- alpha must come from --grad-probe norms, not from "
                  f"loss values.")
        print(f"RANKING LOSS on chain_pair_iptm: form {a.rank_form}, alpha {a.alpha_rank:g}, "
              f"{len(_rank_lab)} labelled (pdb, seed, interface) group(s) from {_rl_p}"
              + (f", {_rl_bad} unparsed row(s)" if _rl_bad else ""))
        # ONE FORM, AND EVERY POSE OF THE INTERFACE IS IN ITS GROUP.
        #
        # Six other --rank-form values used to live here: linear, softplus, expected, spearman,
        # distance, spearman_distance. Each carried its own pair family, tau, decay, straddle and
        # within/cross weighting, and the banner block this replaces was larger than the loss
        # itself -- its whole job was to say which of those knobs the chosen form silently
        # ignored. Deleted at the user's direction. A flag that is inert on the path actually
        # taken is how a run gets written up as the wrong experiment.
        #
        # WITHIN-SEED POSES ARE PART OF THE LOSS AND ALWAYS WERE UNDER rel. The group is every
        # pose of an interface across all its seeds -- 15 x 5 = 75 on a deep structure -- and the
        # mean-centring is over all of them, so the 5 poses sharing a seed are compared with each
        # other and with every pose of every other seed. The flag that DID drop them was
        # --rank-pairs, whose "cross" default excluded the within-seed family from the pairwise
        # forms; deleting it removes the only mechanism that ever filtered them out. A single-seed
        # structure gets a real 5-pose group on the same footing.
        #
        # --rank-cross-seed is a STEP GROUPING, not a pair filter: it puts all of a structure's
        # seeds in one optimiser step so pass A can score the whole interface before the first
        # backward. rel refuses to run without it, checked before any hardware is touched.
        if a.rank_cross_seed:
            print(f"  CROSS-SEED GROUPS: {a.rank_structs_per_step} structure(s) x all seeds per "
                  f"step; a pose's coefficient uses every other pose of its interface, so pass A "
                  f"scores the whole step before the first backward")
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

    # DockQ FOR --pae-split-accept, LOADED EVEN WITH THE RANKING LOSS OFF. This cannot ride on
    # the `elif a.rank_loss` block above, and it must not be gated on a.rank_cross_seed the way
    # the ranking diagnostics are: the arm this split exists for trains PAE alone, so either gate
    # would have made the flag inert on the only path that passes it -- a flag that does nothing
    # on the path actually taken is how a run gets written up as the wrong experiment.
    #
    # THE REFUSAL IS AT STARTUP. Everything below is written so the classification can never
    # raise, which means the absence of a label SOURCE has to be caught here or it would surface
    # as every pose silently unclassified.
    #
    # NO --loss-only INHERITANCE HAZARD, checked rather than assumed: val_loss_checkpoint builds
    # its child argv explicitly and does not forward the parent's flags, so this never reaches the
    # measurement subprocess -- which reads the VALIDATION fold and has no training-fold
    # results.csv to load. That is the class of bug this file has already paid for five times.
    _ps_on = float(a.pae_split_accept) > 0.0
    if _ps_on:
        _ps_src = "the ranking labels already loaded"
        if not _rank_lab:
            import rank_loss as RKL_ps
            _ps_p = a.rank_labels or os.path.join(a.fold, "results.csv")
            # SAID TWICE ON PURPOSE. The pre-hardware block already statted this path; repeating
            # it keeps the message right if the file goes away in between, and means the load
            # below never has to reason about a missing file.
            if not os.path.exists(_ps_p):
                raise SystemExit(
                    f"STOP: --pae-split-accept {a.pae_split_accept:g} needs DockQ labels to put "
                    f"each pose in a group, and there are none at {_ps_p}. Pass --rank-labels, "
                    f"or drop the flag.")
            _rank_lab, _ps_bad = RKL_ps.load_labels(_ps_p, _rel_roles_p)
            if not _rank_lab:
                raise SystemExit(
                    f"STOP: --pae-split-accept found no usable (pdb, seed, interface) -> DockQ "
                    f"group in {_ps_p}, so no pose could be classified.")
            _ps_src = f"{_ps_p}" + (f", {_ps_bad} unparsed row(s)" if _ps_bad else "")
        print(f"PAE LOSS SPLIT BY POSE GROUP at DockQ {a.pae_split_accept:g}: per-epoch plain "
              f"per-pose mean of the labelled pae_loss for the all-acceptable poses and for the "
              f"all-unacceptable ones, on the CROPPED training batches, from "
              f"{len(_rank_lab)} labelled (pdb, seed, interface) group(s) ({_ps_src}). "
              f"DIAGNOSTIC: it takes no gradient, changes no loss, and cannot fail the run.")

    def _pae_split_tally(pdb, lab_p, lab, arm_seed, i, samp_hint, nb, v):
        """Add ONE pose's labelled PAE loss to its DockQ group. Never raises.

        THE POSE INDEX IS NOT THE SAMPLE INDEX, and _rank_lab is indexed by the SAMPLE. This file
        has paid for that confusion twice already -- the rel target regressed pose 1 onto pose 4's
        true iTM, and --emit-iptm-rows wrote `i` while 73.8% of its rows carried another sample's
        DockQ and the aggregate moved by only 0.0009, which is exactly how little a wrong join
        shows. So the sample index is the label's own `sample_index`, and when the PAE mask has
        already resolved it for this pose (`_samp_i`) that one value is reused rather than read a
        second time: two resolutions of the same thing is the shape of the original bug.

        WHICH INTERFACES THE POSE SCORES comes from _pose_pairs -- the per-pose OST join -- not
        from _scored_asym's static pairs: the two disagree on 1,095 of 18,400 (structure, seed,
        pose) checks, so the static set can name an interface this pose does not score at all.
        DockQ is then looked up through _rel_dqkey, the STATIC pair that indexes _rank_lab, which
        is the same two-step the --rel-balance-accept site makes for the same reason.

        EXCLUSIONS ARE COUNTED, NEVER FOLDED INTO A GROUP. 82% of these structures carry more than
        one interface, so a pose can be acceptable on one and not on another; putting a straddler
        in either group would let a mean move because the membership moved rather than because the
        loss did, which is the one thing this read is for. A missing or non-finite DockQ, an
        unresolvable sample index and a batched row are unclassified: under --pose-batch > 1 the
        row's pae_loss is a MEAN over nb poses with different DockQ, so there is no per-pose value
        to attribute, and counting it under one pose's class would quietly average the groups
        together.

        NOTHING HERE RAISES -- not even the SystemExit _pose_pairs throws for an unreadable OST
        mapping, which is why the except names it. This is a diagnostic, and the opposite choice
        in --rel-balance-accept is right only because there the classification sets a WEIGHT and a
        wrong weight trains the head.
        """
        st = _rel_stat
        try:
            if int(nb) != 1:
                st["ps_batched"] += 1
                st["ps_unclass"] += 1
                return
            samp = samp_hint
            if samp is None:
                _sv = lab.get("sample_index") if hasattr(lab, "get") else None
                if _sv is None:
                    _sv = lab["sample_index"]
                samp = int(np.asarray(_sv).reshape(-1)[i])
            samp = int(samp)
            _cls, _bad = set(), False
            for _ifc, _x, _y in _pose_pairs(pdb, lab_p, samp):
                _dk = _rel_dqkey.get(_ifc)
                _m = _rank_lab.get(_rank_key(pdb, arm_seed, _dk)) if _dk is not None else None
                _dq = _m.get(samp) if _m else None
                if _dq is None or not math.isfinite(float(_dq)):
                    _bad = True
                    break
                _cls.add(float(_dq) >= float(a.pae_split_accept))
            if _bad or not _cls:
                st["ps_unclass"] += 1
            elif len(_cls) > 1:
                st["ps_straddle"] += 1
            elif True in _cls:
                st["ps_acc_sum"] += float(v)
                st["ps_acc_n"] += 1
            else:
                st["ps_un_sum"] += float(v)
                st["ps_un_n"] += 1
        except (Exception, SystemExit):                      # noqa: BLE001
            st["ps_unclass"] += 1

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

    def _set_pae_mask(pdb, asym, has_frame, unit=None, pairs_override=None):
        """[N_frame, N_token] over the CROPPED tokens: keep only scored cross-chain entries.

        Falls back to no mask -- never to an all-zero one. An empty mask would make PAELoss divide
        by eps and return a meaningless number rather than fail, which is the worst outcome.

        pairs_override EXISTS FOR --rank-form rel. _scored_asym is built once per structure from
        chain_roles, while rel resolves each interface's asym pair PER POSE through OST. On the 52
        structures where the two disagree, PAE would anchor a different interface from the one rel
        is ranking -- 7stf's (3,1) against (3,0) -- so the two terms would be supervising
        different objects on the same pose. That is objective misalignment, not gradient noise.
        Raised in review. The override is per pose because OST's answer is.
        """
        _pae_keep["pdb"], _pae_keep["unit"] = pdb, unit
        if not _scored_asym and pairs_override is None:
            return
        # COUNTED AND REPORTED, because "the override is wired" and "the override CHANGES
        # something" are different claims and only the second matters. Twice today a change to
        # this code path produced numbers bit-identical to the version it replaced -- once
        # because --ost-mapping was not forwarded to the workers, once here -- and in both cases
        # the identical numbers were the only evidence anything was wrong. So the run says how
        # many poses got an override and on how many it DIFFERED from the static pairs.
        if pairs_override is not None:
            _ov = _pae_keep.setdefault("ov", {"poses": 0, "differed": 0, "pdbs": set()})
            _ov["poses"] += 1
            if set(pairs_override) != set(_scored_asym.get(pdb) or ()):
                _ov["differed"] += 1
                _ov["pdbs"].add(pdb)
        pairs = pairs_override if pairs_override is not None else _scored_asym.get(pdb)
        if not pairs:
            _pae_keep["m"] = None
            _pae_keep["blocks"] = None
            _pae_mask_stats["unmasked"] += 1
            _pae_mask_check()
            return
        aa = asym.reshape(-1).long()
        fa = aa[has_frame.reshape(-1).bool()]
        keep = torch.zeros((fa.numel(), aa.numel()), dtype=torch.bool, device=aa.device)
        # PER-PAIR BLOCKS, KEPT SEPARATELY. Selection reads chain_pair_iptm[x, y] -- one max per
        # INTERFACE, each over just those two chains and with its own d0. 298 of the 368 training
        # structures score two or more pairs, so ranking rows on the union would rank a row in the
        # antigen by its heavy- and light-chain targets averaged together, which is a quantity no
        # ipTM reads. --pae-row-frac weights each block on its own t_i; the union is still what the
        # unweighted term uses, and the blocks sum back to it at uniform weights.
        blocks = []
        for x, y in pairs:
            blk = (((fa[:, None] == x) & (aa[None, :] == y))
                   | ((fa[:, None] == y) & (aa[None, :] == x)))
            keep |= blk
            if bool(blk.any()):
                blocks.append((blk, int(((aa == x) | (aa == y)).sum())))
        if not bool(keep.any()):
            _pae_keep["m"] = None
            _pae_keep["blocks"] = None
            _pae_mask_stats["unmasked"] += 1
            _pae_mask_check()
            return
        _pae_keep["m"] = keep
        _pae_keep["blocks"] = blocks
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
        # THE OUTCOME, NOT THE INTENT. --train-only matching SOMETHING is already checked above;
        # this checks it matched what was meant. The full-head pattern truncated to its bare
        # negative lookahead gives 19 tensors instead of 102 -- a non-empty optimiser, no guard
        # fired, and the run was stopped only because the startup line was read by eye.
        if a.expect_trainable and len(_kept) != a.expect_trainable:
            raise SystemExit(
                f"STOP: --expect-trainable {a.expect_trainable} but --train-only left "
                f"{len(_kept)} trainable tensor(s) ({_after:,} parameters). The pattern matched "
                f"something, so the optimiser is not empty and nothing else would have caught "
                f"this.\n  patterns: {_pats}\n  kept e.g. {[_rel[n] for n in _kept[:6]]}")
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
            if is_main:
                print(f"  --loss-only: inheriting the bundle's ranking configuration "
                      f"(alpha_rank {a.alpha_rank:g}) so the guard matches; the ranking term "
                      f"itself does not run -- a measurement takes no step.")

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
        # KEEP THE NAMED PARAMETERS DENSE, unioned in BEFORE the --lean-single dead-strip below so
        # a pose tensor nothing reads is still removed and --skip-dead-ops still refuses a module
        # it would stub. apply_lora has already frozen the whole head, so requires_grad is restored
        # for these at the `p.requires_grad_(id(p) in head_ids)` line that follows.
        if a.lora_keep_trainable:
            # LOCALLY, LIKE EVERY OTHER re USER IN THIS FILE. Module scope has no `re` -- see the
            # comment at the --pae-scored-pairs block -- so a bare re.search here would NameError
            # on the first LoRA run, after the model is built and the cards are already held.
            import re as _re_lk
            _keep_pats = [t.strip() for t in a.lora_keep_trainable.split(",") if t.strip()]
            _pfx_dot = prefixes[0] + "."
            _kept = [n for n, _p in model.named_parameters()
                     if n.startswith(_pfx_dot) and n not in set(trainable)
                     and any(_re_lk.search(_pt, n[len(_pfx_dot):]) for _pt in _keep_pats)]
            if not _kept:
                raise SystemExit(
                    f"STOP: --lora-keep-trainable {_keep_pats} matched NO head parameter. A "
                    f"pattern that matches nothing is indistinguishable from the flag being "
                    f"absent, and the tensors it was meant to protect would train frozen at "
                    f"their init while the run reported success.")
            _kept_n = sum(p.numel() for n, p in model.named_parameters() if n in set(_kept))
            trainable = trainable + _kept
            print(f"--lora-keep-trainable: {len(_kept)} head parameter(s) stay DENSE alongside "
                  f"the adapters ({_kept_n/1e6:.3f}M), e.g. "
                  f"{[n[len(_pfx_dot):] for n in _kept[:3]]}")

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
    # THE DEAD HEAD PARAMETERS COME OUT LAST, so --train-only and LoRA cannot put them back.
    #
    # WHY IT IS SAFE RATHER THAN MERELY CHEAP. Nothing reads these tensors' outputs, so autograd
    # produces no gradient for them -- which _apply_step then turns into an explicit zeros tensor
    # ("a rank with no gradient for a parameter still has to contribute a tensor, or the
    # collective's shapes do not line up"). So today they are all-reduced as zeros, AdamW allocates
    # exp_avg and exp_avg_sq for them, and with weight_decay=0.0 their update is exactly zero. They
    # are therefore bit-identical whether they are in the optimiser or not; removing them removes
    # ~84 MB of moment state and ~42 MB of the 79 MB flat collective buffer per step and NOTHING
    # else. Verified on CPU by run_logging/test_lean_single.py, which checks the single pathway's
    # parameters get .grad is None with the track ON as well as off.
    #
    # AND THE REPORTED NUMBER BECOMES HONEST. Every --pae-only arm in this project has printed
    # "258 trainable tensor(s), 19.8M of 464.5M parameters (4.3%)" while 158 of those tensors
    # received no gradient. The --grad-probe path already said so out loud and nothing acted on it.
    if _dead_head:
        _dead_abs = {f"{prefixes[0]}.{n}" for n in _dead_head}
        _before_n, _before_p = len(trainable), sum(
            p.numel() for n, p in model.named_parameters() if n in set(trainable))
        trainable = [n for n in trainable if n not in _dead_abs]
        if not trainable:
            raise SystemExit(
                "STOP: --lean-single left the trainable set EMPTY. Every tensor --train-only or "
                "LoRA selected is one nothing reads under --pae-only, so the run would report "
                "success while training nothing. Widen --train-only or drop --lean-single.")
        _after_p = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
        print(f"--lean-single: {_before_n - len(trainable)} of {_before_n} trainable tensor(s) "
              f"dropped from the optimiser as provably gradient-free "
              f"({(_before_p - _after_p)/1e6:.3f}M of {_before_p/1e6:.3f}M parameters, "
              f"{100.0 * (_before_p - _after_p) / max(1, _before_p):.2f}%). The remaining "
              f"{len(trainable)} tensor(s) / {_after_p/1e6:.3f}M parameters are the EFFECTIVE "
              f"capacity -- which is what the line below now prints.")
    head_ids = {id(p) for n, p in model.named_parameters() if n in set(trainable)}
    n_head = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
    n_all = sum(p.numel() for _n, p in model.named_parameters())
    print(f"trainable: {len(trainable)} tensor(s), {n_head/1e6:.1f}M of {n_all/1e6:.1f}M parameters "
          f"({n_head/n_all:.1%}), under {prefixes}")
    for n, p in model.named_parameters():
        p.requires_grad_(id(p) in head_ids)

    # ---- DEAD OPERATION ELIMINATION ------------------------------------------------------------
    # INSTALLED HERE, AND HERE FOR THE SAME REASON THE PREFIX CACHE IS: its central refusal is "a
    # module I would stub is TRAINABLE", read from requires_grad on the tensors, and requires_grad
    # is not final until the line above has applied --train-only, LoRA and --lean-single. Any
    # earlier and it would be checking a flag's intent instead of the model's state.
    #
    # BEFORE THE PREFIX CACHE, though either order is correct -- the husks leave named_parameters()
    # and named_modules() unchanged and move no value, so the cache's prefix set, its dropout scan
    # and its digest are identical either way. This order only means a dead_ops refusal happens
    # before the cache turns on, and that the cache's log line describes the tree that will run.
    # phases/dead_ops.py's docstring states the full order.
    _dops = None
    if a.skip_dead_ops:
        import dead_ops as DOPS
        # THE KERNEL NAMES COME FROM configs, THE SAME PLACE run_confidence_head GETS THEM.
        # Both default to "cuequivariance", and with hidden_scale_up the tri-mul cueq gate is
        # open -- so probing the "torch" reference implementation would certify a code path this
        # run never executes. Found in review after the first version hardcoded "torch".
        _dops = DOPS.install(model, autocast_dtype=torch.bfloat16,
                             triangle_multiplicative=configs.triangle_multiplicative,
                             triangle_attention=configs.triangle_attention)
    else:
        DOPS = None

    # ---- THE FROZEN-PREFIX CACHE ---------------------------------------------------------------
    # INSTALLED HERE AND NOT AT THE pose_inblock SITE, which is ~1,100 lines above. Its central
    # precondition is that every prefix tensor reads requires_grad=False, and requires_grad is not
    # set until the line above -- `trainable` is not final until --train-only, LoRA and
    # --lean-single have all had their say. Installing it earlier would check the flag's intent
    # instead of the model's state, which is the distinction the whole guard exists to make.
    _pfc = None
    if a.prefix_cache_gb > 0:
        import prefix_cache as PFC
        _pfc = PFC.install(model, a.prefix_cache_gb, lean_single=a.lean_single, device=dev,
                           crop_pool=a.crop_pool, crop_all_poses=a.crop_all_poses)
    else:
        PFC = None

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

    # ---- --save-init: THE HEAD BEFORE ANY STEP IS TAKEN ----------------------------------------
    # PLACED HERE, which is after every pathway has been installed (lean_readouts, --lean-single,
    # pose_inblock, pose_direction, pose_perblock all happen around :1776-1835) and after
    # _head_names is fixed, but far above the training loop. So the file holds exactly the
    # architecture the run was configured for, with the injections still at their zero init.
    if a.save_init:
        if world > 1:
            raise SystemExit(
                f"STOP: --save-init with WORLD_SIZE={world}. It writes the head and exits, and "
                f"exiting one rank of {world} leaves the others parked on the next collective "
                f"until the elastic agent times out. Run it on one card.")
        _si_d = os.path.dirname(os.path.abspath(a.save_init))
        if _si_d:
            os.makedirs(_si_d, exist_ok=True)
        _si_t = _head_ckpt_tensors()
        torch.save(_si_t, a.save_init)
        # THE ZERO-INIT CLAIM, ASSERTED AT THE MOMENT IT IS SAVED rather than trusted. If a future
        # change initialises either injection non-zero, step-zero parity against the shipped head
        # stops being the right test and this says so instead of the parity run failing later with
        # a number nobody can attribute.
        _si_inj = {k: v for k, v in _si_t.items() if "pose_inblock" in k or "pose_perblock" in k
                   or "frame_dir" in k}
        _si_nz = sorted(k for k, v in _si_inj.items() if float(v.abs().max()) != 0.0)
        print(f"--save-init: wrote {len(_si_t)} head tensor(s) to {a.save_init} before any step. "
              f"{len(_si_inj)} injected tensor(s) present; "
              + ("ALL EXACTLY ZERO, so this head is bit-identical to the un-injected one and its "
                 "PAE/ipTM must reproduce the shipped head."
                 if not _si_nz else
                 f"NOT ZERO: {_si_nz}. Step-zero parity against the shipped head is NOT the right "
                 f"check for this configuration."))
        sys.stdout.flush()
        raise SystemExit(0)
    # -------------------------------------------------------------------------------------------

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
        _sel_rows = list(csv.DictReader(fh))
    sel_pdbs = [r["pdb"].strip().lower() for r in _sel_rows]
    # THE TOKEN COUNT, KEPT THIS TIME. It was parsed and discarded, which left --epoch-order no
    # way to sort by size -- so "manifest" order silently meant whatever order `flat` happened to
    # be in, which is sorted by pdb. Absent or non-numeric `tokens` is simply missing here and
    # sorts first, named by the builder rather than guessed at.
    _sel_tokens = {r["pdb"].strip().lower(): int(r["tokens"])
                   for r in _sel_rows if str(r.get("tokens", "")).strip().isdigit()}
    if a.manifest_nshards and a.manifest_nshards > 1:
        if not (0 <= a.manifest_shard < a.manifest_nshards):
            raise SystemExit(f"STOP: --manifest-shard {a.manifest_shard} is not in "
                             f"[0, {a.manifest_nshards}).")
    # A PER-STRUCTURE SEED SET, because the population is no longer uniform in depth. 437
    # structures carry seeds 500000-500014 and 416 carry 500000 alone -- their other trunks were
    # reclaimed on 2026-11-04 -- so a cross product with ONE range is wrong for whichever group it
    # is not describing, and the incompleteness guard below then refuses the whole run for being
    # the shape it is meant to be. An optional `seeds` column in --manifest-selection declares each
    # structure's own depth; a row without one falls back to --seeds, so every existing arm's
    # selection csv keeps its exact present meaning.
    #
    # DECLARED, NOT DISCOVERED, and that is the whole point of this file. Globbing the trunks would
    # make a lost unit look like a smaller training set instead of an error, which is the defect
    # review round 8 removed. The seeds column is a contract: if it names a seed with no label or
    # no trunk, the run stops.
    def _parse_seed_field(txt, who):
        """'500000-500014' or '500000,500003' or a mix -> a set of ints."""
        out = set()
        for part in str(txt).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part.lstrip("-"):
                try:
                    lo, hi = (int(x) for x in part.split("-"))
                except ValueError:
                    raise SystemExit(f"STOP: {who} seed field {part!r} is not lo-hi")
                if hi < lo:
                    raise SystemExit(f"STOP: {who} seed range {part!r} runs backwards")
                out |= set(range(lo, hi + 1))
            else:
                try:
                    out.add(int(part))
                except ValueError:
                    raise SystemExit(f"STOP: {who} seed field {part!r} is not an integer")
        if not out:
            raise SystemExit(f"STOP: {who} has an empty `seeds` field; remove the column or name "
                             f"the seeds, but do not declare a structure with no depth")
        return out

    _seed_col = {}
    for _r in _sel_rows:
        _p = _r["pdb"].strip().lower()
        _txt = str(_r.get("seeds", "") or "").strip()
        if _txt:
            _seed_col[_p] = _parse_seed_field(_txt, f"{a.manifest_selection}:{_p}")
    want = [(p, sd) for p in sel_pdbs if p not in skip
            for sd in sorted(_seed_col.get(p) or range(lo_s, hi_s + 1))]
    if _seed_col:
        _depths = collections.Counter(len(_seed_col.get(p) or range(lo_s, hi_s + 1))
                                      for p in sel_pdbs if p not in skip)
        print(f"manifest: per-structure seed depths from {a.manifest_selection}: "
              f"{dict(sorted(_depths.items()))} "
              f"({len(_seed_col)} of {len(sel_pdbs)} row(s) declare `seeds`)")
    _rank_check_coverage(want)
    print(f"manifest: {len(sel_pdbs)} selected structure(s) minus {sorted(skip) or 'none'} "
          f"x seeds {lo_s}-{hi_s} (fallback for rows with no `seeds` column) = {len(want)} "
          f"intended unit(s), from {a.manifest_selection}")

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
                "STOP: --alignment-preflight is REQUIRED. It is not conditional on "
                "--minimal-items: the full dataset-item path is gone, so the minimal path is the "
                "only path and check_alignment can no longer run per structure. Omitting "
                "--minimal-items does NOT avoid this -- the message used to say it did, and that "
                "cost a launch attempt. check_alignment is what proves the trunk's features agree "
                "atom for atom with the dataset's, the one failure that would pair a pose with "
                "another molecule's label. Run:\n"
                "  phases/phase_k_alignment_preflight.py --out <marker>\n"
                "and pass that marker.")
        if not os.path.exists(a.alignment_preflight):
            raise SystemExit(f"STOP: no alignment preflight marker at {a.alignment_preflight}")
        _pf = json.load(open(a.alignment_preflight))
        # A SUPERSET MARKER AUTHORISES A SUBSET RUN, which exact digest equality cannot express.
        # Raised by the owner: a memory probe or a --limit-units benchmark builds a SHORT manifest,
        # so its digest differs from the full marker's and the run was refused -- even though a
        # marker proving alignment for 851 structures plainly vouches for the 2 a probe touches.
        # That made short runs impossible and blocked re-measuring --ckpt-token-threshold for a new
        # population, which is what let an OOM reach 2.6 h into an epoch.
        #
        # THE SUBSET TEST IS BY NAME AND STRICTER THAN THE DIGEST, not looser: every structure this
        # run will train must appear in the marker's `checked` list, and the marker must still
        # record ok and complete. A marker with no `checked` list cannot be used this way and falls
        # back to requiring digest equality, so an old marker never silently authorises more than
        # it did before.
        # THE MARKER ALREADY PROVES THIS WITHOUT A NAME LIST, which is the cheap version I missed
        # first. `complete: true` means the preflight checked EVERY structure in the selection it
        # names, after that selection's skips. So a run using the SAME --manifest-selection and the
        # SAME --manifest-skip is necessarily a subset of what was checked, whatever subset of UNITS
        # it walks. No per-structure list is needed and no rebuild of the marker is needed; I
        # regenerated one for 20 minutes of CPU, competing with a live trainer, before noticing.
        #
        # The `checked` list the preflight now also writes is a strictly tighter check and is used
        # when present -- it covers the case of a run whose selection differs but whose structures
        # happen to be covered.
        _pf_checked = set(_pf.get("checked") or [])
        _run_pdbs = {p for p, _lp, _c in flat}

        def _skipset(txt):
            return {x.strip().lower() for x in (txt or "").split(",") if x.strip()}

        # A WIDER SKIP LIST OVER THE SAME SELECTION IS STILL A SUBSET, which equality cannot see.
        # The marker is `complete` over (selection - its own skip). This run trains
        # (selection - this run's skip). So whenever the run's skip set CONTAINS the marker's,
        # the run's structures are a subset of what was checked -- set algebra, not a heuristic:
        #     run_skip >= pf_skip  =>  (sel - run_skip) <= (sel - pf_skip) = checked
        #
        # THE CASE THAT FORCED IT: ten structures were excluded for not fitting uncropped, so the
        # skip went from '9mx1,4zto,9q5m' to that plus ten names. Same selection, 841 of the 851
        # already-checked structures, and the digest changed -- so this refused a marker that
        # plainly vouched for every structure in the run. Equality of the skip STRING is also
        # order- and whitespace-sensitive, which would refuse a reordered list of identical names.
        #
        # STRICTLY NARROWER THAN THE DIGEST, never wider: a skip list that REMOVES a name the
        # marker skipped means the run trains a structure the preflight never saw, and that fails
        # this test and is refused as before.
        _pf_skip, _run_skip = _skipset(_pf.get("manifest_skip")), _skipset(a.manifest_skip)
        _same_population = (
            bool(_pf.get("complete"))
            and (_pf.get("manifest_selection") or "") == (a.manifest_selection or "")
            and _run_skip >= _pf_skip)
        _subset_ok = _same_population or (bool(_pf_checked) and _run_pdbs <= _pf_checked)
        if _pf.get("manifest_digest") != _manifest_digest and _subset_ok:
            _why = ((f"it is COMPLETE over the same selection and this run skips "
                     f"{len(_run_skip)} structure(s) against the marker's {len(_pf_skip)} "
                     f"(a superset: {sorted(_run_skip - _pf_skip) or 'identical'}), so every "
                     f"structure this run touches was checked") if _same_population else
                    f"this run's {len(_run_pdbs)} structure(s) are a SUBSET of the "
                    f"{len(_pf_checked)} named in its `checked` list")
            print(f"  alignment preflight: digest differs ({_pf.get('manifest_digest')} vs "
                  f"{_manifest_digest}) but {_why}, so it authorises this run.")
        elif _pf.get("manifest_digest") != _manifest_digest:
            raise SystemExit(
                f"STOP: the alignment preflight was run over manifest "
                f"{_pf.get('manifest_digest')}, this run is {_manifest_digest}. It says nothing "
                f"about the structures this run will train on; re-run it.\n"
                + (f"  Its selection/skip are {_pf.get('manifest_selection')!r} / "
                   f"{_pf.get('manifest_skip')!r} against this run's "
                   f"{a.manifest_selection!r} / {a.manifest_skip!r}, and it records no `checked` "
                   f"list, so neither subset path applies."
                   if not _pf_checked else
                   f"  {len(_run_pdbs - _pf_checked)} of this run's structure(s) are NOT in the "
                   f"marker's checked list, e.g. {sorted(_run_pdbs - _pf_checked)[:5]}."))
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
                 # THE EPOCH BUILDER IS PART OF THE TRAINING POPULATION, not a tuning knob.
                 # Toggling --epoch-cover-all on a resume changes WHICH STRUCTURES TRAIN -- the
                 # group-of-world builder drops every structure with fewer than `world` seeds,
                 # 412 of 846 here -- while AdamW state is restored as if nothing moved. Absent
                 # from a bundle written before the flag existed, so it reads as False, which is
                 # exactly what those runs did. Raised in external review of e4e39731.
                 "epoch_cover_all": bool(a.epoch_cover_all),
                 # IT CHANGES THE STEP SHAPE, so by this guard's own stated convention it belongs
                 # here: poses per forward alters the gradient scaling and, if the counter were
                 # ever wrong again, the effective batch. Absent from older bundles, so it reads
                 # as 1 via _PRE_CROP -- which is exactly what those runs did.
                 "pose_batch": int(a.pose_batch),
                 # IT CHANGES WHICH UNITS A RANK WALKS, so a resume that moved it would train a
                 # different per-rank stream while restoring AdamW as if nothing had.
                 "epoch_balance": bool(a.epoch_balance),
                 "manifest_digest": _manifest_digest, "crop_size": a.crop_size,
                 "alpha_confidence": a.alpha_confidence, "alpha_pae": a.alpha_pae,
                 "drop_pde": a.drop_pde, "clip_norm": a.clip_norm,
                 "rank_loss": a.rank_loss, "alpha_rank": a.alpha_rank,
                 "rank_pool": a.rank_pool, "rel_labels": a.rel_labels,
                 "rel_standardize": a.rel_standardize,
                 "rel_sd_weight": a.rel_sd_weight,
                 "rel_g_weight": a.rel_g_weight,
                 "rel_g_sigmoid": (a.rel_g_sigmoid or ""),
                 "rel_std_floor": (-1.0 if a.rel_std_floor is None else a.rel_std_floor),
                 # THE FOUR CROSS-SEED FIELDS. Every one of them changes the objective or the step
                 # shape, so a resume that moved any of them silently would be a different
                 # experiment wearing the same run id. FOURTH time a field has been added to this
                 # comparison without being added to the saved bundle below -- so both sides are
                 # edited in the same patch, and test_guard_fields.py asserts they agree.
                 "rank_form": a.rank_form,
                 "rank_cross_seed": a.rank_cross_seed,
                 "rank_structs_per_step": a.rank_structs_per_step,
                 # L2-SP CHANGES THE OBJECTIVE and was missing here: a resume could silently move
                 # lambda and the drift guard would not notice. Caught in review of cd8e6ab7.
                 "lora_keep_trainable": a.lora_keep_trainable,
                 "l2_sp": a.l2_sp,
                 "l2_sp_exclude": a.l2_sp_exclude,
                 # AND SO DOES THE SCHEDULE. lr_final changes the lr at every step after warmup,
                 # so a resume that added or dropped the cosine would be a different experiment.
                 "lr_final": a.lr_final,
                 "lr": a.lr, "warmup_steps": a.warmup_steps,
                 # ROW WEIGHTING CHANGES THE OBJECTIVE. The PAE term becomes a weighted mean
                 # over anchor rows, so moving frac, mass or the floor on resume blends two
                 # objectives into one curve -- the same failure the l2_sp and lr_final entries
                 # above exist to stop. Caught in review of 38a83283.
                 "pae_row_frac": a.pae_row_frac,
                 "pae_row_mass": a.pae_row_mass,
                 "pae_row_floor": a.pae_row_floor,
                 # The decay WINDOW and SHAPE change the lr at every step just as lr_final does,
                 # so they belong beside it rather than being silently re-derived on resume.
                 "lr_decay_start_epoch": a.lr_decay_start_epoch,
                 "lr_decay_end_epoch": a.lr_decay_end_epoch,
                 "lr_decay_shape": a.lr_decay_shape,
                 # THE STAIRCASE IS A SCHEDULE CHANGE like lr_final and the decay window, so a
                 # resume that silently changed it would splice two schedules into one curve.
                 "lr_steps": a.lr_steps,
                 # DROPOUT IS PART OF THE OBJECTIVE, not of the optimiser: flipping it on a
                 # resume changes the gradient's distribution at every step.
                 "pair_dropout": a.pair_dropout,
                 # THE AUGMENTATION DISTRIBUTION. Changing either probability changes what the
                 # head sees at every step, so a resume that silently changed them would splice
                 # two augmentations into one curve. Raised in review.
                 "token_keep_union": a.token_keep_union,
                 "token_keep_other": a.token_keep_other,
                 # CROPPING CHANGES THE OBJECTIVE -- the loss averages over a different set of
                 # tokens and pairs -- so resuming across a crop change would blend two experiments
                 # into one curve, which is exactly what this guard exists to stop.
                 "crop_contact": a.crop_contact,
                 # A CROP RULE CHANGE IS AN OBJECTIVE CHANGE: the loss averages over a different
                 # set of tokens and pairs, so resuming across it would blend two objectives.
                 # THE CROP DEFINITION ITSELF, not just its radii. a16911de changed the DEFAULT
                 # from the union over all nsample poses to pose 0 alone, so a resume across that
                 # change blends two crop definitions into one curve. f1268fe5 added the field to
                 # the saved bundle and stopped there -- which is the inert guard described two
                 # entries down, repeated one commit later. It is compared HERE and defaulted in
                 # _PRE_CROP, which are the two halves that make a field actually checked.
                 "crop_all_poses": a.crop_all_poses,
                 # SAME REASON AS CROPPING. --pae-scored-pairs changes which pairs the PAE term
                 # averages over and --pae-only changes which terms exist at all, so resuming
                 # across either blends two objectives into one curve, carrying the first one's
                 # AdamW moments. These were added to the saved bundle and to _PRE_CROP but NOT
                 # here, which left the guard inert: it compared neither, and the _PRE_CROP
                 # defaults for them were never consulted.
                 "pae_scored_pairs": a.pae_scored_pairs, "pae_only": a.pae_only,
                 # THE PARAMETER SET AGAIN, AND THIS ONE CORRUPTS SILENTLY IF IT IS NOT GUARDED.
                 # --lean-single drops 158 tensors from the optimiser's param group. AdamW's
                 # state_dict is keyed by the parameter's INDEX in that group, so resuming a
                 # --lean-single bundle without the flag (or the reverse) would load exp_avg and
                 # exp_avg_sq onto DIFFERENT tensors -- no shape error, because many of them are
                 # the same shape, and no way to notice afterwards. It changes no mathematics in
                 # the forward, which is exactly why nothing else would catch it.
                 "lean_single": a.lean_single,
                 # THE PARAMETER SET AGAIN: --pose-inblock ADDS trainable tensors (two, or four
                 # with --pose-inblock-pae), so a bundle written without it is missing them and
                 # the _missing check below refuses it anyway. Compared here so the refusal names
                 # the setting instead of reporting an absent tensor -- the same reason
                 # frame_direction is compared. The BLOCK matters too: the same two tensors at a
                 # different depth is a different operator.
                 "pose_inblock": a.pose_inblock, "pose_inblock_pae": a.pose_inblock_pae,
                 "pose_inblock_block": a.pose_inblock_block,
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
                     # ABSENCE IS EVIDENCE, same reading as --pae-row-frac below. Every bundle
                     # written before --epoch-cover-all existed ran the group-of-`world` builder,
                     # which is exactly what False means. Without this entry such a bundle would
                     # fall through to the unknown-key branch instead of migrating, and a resume
                     # that TURNS THE FLAG ON still shows as drift -- which is the point, since it
                     # changes which structures train while AdamW state is restored unchanged.
                     "epoch_cover_all": False, "pose_batch": 1, "epoch_balance": False,
                     # A bundle written before these flags existed ran the full-run schedule, which
                     # is exactly what the 0 defaults mean, and "cosine" is what it computed.
                     "lr_decay_start_epoch": 0.0, "lr_decay_end_epoch": 0.0,
                     "lr_decay_shape": "cosine", "lr_steps": "",
                     "pair_dropout": -1.0,
                     "token_keep_union": -1.0, "token_keep_other": -1.0,
                     # ABSENCE IS EVIDENCE: a bundle written before --pae-row-frac existed ran the
                     # unweighted term, which is exactly what frac 0 means. The mass and floor
                     # defaults are inert at frac 0 but are migrated so a resume that only turns
                     # weighting ON still shows up as drift on the frac field.
                     "pae_row_frac": 0.0, "pae_row_mass": 0.8, "pae_row_floor": 10,
                 "train_only": "",
                     "crop_contact": 5.0, "pae_scored_pairs": "",
                     # ABSENCE IS EVIDENCE, as for the two above: a bundle written before
                     # a16911de was written by code that had no such flag and always took the
                     # union, so it migrates to True. A resume that switches to pose 0 therefore
                     # reads as real drift and must be named with --resume-override crop_all_poses.
                     "crop_all_poses": True,
                     "pae_only": False,
                     # A BUNDLE PREDATING THE FLAG TRAINED THE FULL PARAMETER GROUP.
                     "lean_single": False,
                     # A BUNDLE PREDATING THE FLAG HAD NO IN-BLOCK INJECTION AT ALL.
                     "pose_inblock": False, "pose_inblock_pae": False,
                     "pose_inblock_block": -1,
                     "frame_direction": False,
                     "pose_perblock": False, "pose_perblock_tied": False,
                     "pose_perblock_share": False, "pose_perblock_copy": False,
                     "rank_loss": False, "alpha_rank": 1.0,
                     "rank_pool": "max", "rel_labels": "", "rel_standardize": False,
                     "rel_sd_weight": False, "rel_std_floor": -1.0,
                     "rel_g_weight": 0.0, "rel_g_sigmoid": "",
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
                     # A BUNDLE PREDATING --lr-final HELD lr FLAT AFTER WARMUP, which is exactly
                     # what lr_final 0.0 means -- the flag's own help says 0 "keeps the historical
                     # schedule ... hold --lr flat for the whole run, which is what every arm
                     # before this one did". So its absence is evidence of 0.0, not of an unknown
                     # setting, and it belongs here rather than in _unknown: that STOP fires
                     # BEFORE --resume-allow-drift is consulted and so cannot be overridden,
                     # which made ocqtyt's epoch-26 bundle unresumable (run 4av7s8, exit 1).
                     "lr_final": 0.0}
        _migrated = []

        _bad, _from_args, _unknown = {}, [], []
        # rank_form IS INERT WHEN NEITHER SIDE RUNS A RANKING TERM, and comparing it then refuses
        # a resume over a field that changed nothing.
        #
        # Every bundle written before the forms were cut down records rank_form 'linear', because
        # that was the argparse default and it was saved whether or not --rank-loss was passed.
        # 'rel' is now the only legal value, so EVERY pre-deletion PAE-only bundle became
        # unresumable -- vdmheg's epoch 27 among them, which is how this was found. The field
        # describes the SHAPE of a term that is not in the loss, so when rank_loss is false on
        # both sides a difference cannot blend two experiments into one curve, which is the thing
        # this guard exists to prevent.
        #
        # NARROW ON PURPOSE: if either side trains a ranking term, rank_form is compared as
        # before. The alternative -- --resume-override rank_form -- would work once and would
        # also disarm the guard for that field on the NEXT resume, which is exactly the
        # disarming the override's own warning describes.
        _rank_inert = (not bool(_g.get("rank_loss", False))) and (not bool(a.rank_loss))
        for k, v in _mine.items():
            if k == "rank_form" and _rank_inert:
                continue
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
            # THE ANCHOR BELONGS TO THE LINEAGE, NOT THE DIRECTORY. A resume writes to a FRESH
            # directory and launch.py refuses a non-empty one, so there is no window in which to
            # place the anchor there by hand -- which is what the old message asked for, making an
            # --l2-sp run effectively unresumable. So it is inherited from beside the resume
            # bundle and COPIED into this run's directory, which keeps the record of what was
            # actually used with the run that used it. Only if neither exists is this fatal.
            if not os.path.exists(_anchor_p):
                _parent_anchor = os.path.join(os.path.dirname(os.path.abspath(a.resume)),
                                              "l2sp_anchor.pt")
                if os.path.exists(_parent_anchor):
                    if is_main:
                        import shutil as _sh
                        # COPY VIA .tmp AND os.replace, NOT STRAIGHT TO _anchor_p. The non-main
                        # ranks wait on os.path.exists, which goes true the instant the file is
                        # CREATED -- so a direct copyfile lets them torch.load 79 MB of anchor
                        # while it is still being written. Measured 2026-11-03: egw5wq's resume
                        # died at launch with "PytorchStreamReader failed reading zip archive:
                        # failed finding central directory" on ranks 1 and 2, then rank 0 died on
                        # the gloo peer. os.replace is atomic within a filesystem, so existence
                        # now implies completeness and the existing wait loop becomes correct.
                        _tmp_anchor = _anchor_p + ".tmp"
                        _sh.copyfile(_parent_anchor, _tmp_anchor)
                        os.replace(_tmp_anchor, _anchor_p)
                        print(f"  --l2-sp: anchor INHERITED from {_parent_anchor} -> {_anchor_p}",
                              flush=True)
                    else:
                        # The non-main ranks wait for rank 0 to place it rather than racing to
                        # copy it themselves. Safe because rank 0 publishes atomically, above.
                        for _ in range(600):
                            if os.path.exists(_anchor_p):
                                break
                            time.sleep(0.1)
            if not os.path.exists(_anchor_p):
                raise SystemExit(
                    f"STOP: --l2-sp resumed but {_anchor_p} is missing, and there is no "
                    f"l2sp_anchor.pt beside {a.resume} to inherit. The anchor is the SHIPPED "
                    f"head; this run's weights are not it, so there is nothing to penalise "
                    f"distance from.")
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
    if _USE_CROP:
        _base = a.crop_log or os.path.join(a.out, "crop_sizes.csv")
        _stem, _ext = os.path.splitext(_base)
        crop_p = f"{_stem}_rank{rank}{_ext or '.csv'}"

    traj_p = os.path.join(a.out, "trajectory.csv")
    # AND A PER-RANK COPY, for the same reason crop_sizes.csv got one above. trajectory.csv is
    # written by rank 0 alone (write_traj=is_main), so its per-pose columns describe the third of
    # the units rank 0 walked while reading like the whole run -- which is how five plotters came
    # to report a rank-local mean. The per-EPOCH curve is safe either way, because batch_<term> is
    # already reduced across ranks; what is NOT recoverable from rank 0 alone is any per-pose
    # question that filters by pdb (an unwinnable-structure filter, a per-structure histogram),
    # because the other ranks' poses were never written down anywhere.
    #
    # trajectory.csv is left exactly as it is so nothing that reads it today changes. This is an
    # ADDITIONAL file per rank, rank 0 included, so the full set is self-describing rather than
    # "the main one plus the others".
    traj_rank_p = os.path.join(a.out, f"trajectory_rank{rank}.csv")
    # THE BATCH COLUMNS ARE THE CURVE; the bare TERMS are the single pose that happened to be last.
    # batch_<term> is the mean over ALL accum_eff poses in the step, reduced across every rank, and
    # batch_n is how many actually contributed -- accum_eff on a normal step, fewer on the partial
    # step an epoch ends with. Added after review 2026-10-16: reporting read the stepping pose's own
    # loss as the batch loss, which is 1 pose of 255, and no amount of reader-side work can
    # reconstruct the other ranks' poses because they were never written.
    traj_cols = ["wall_s", "step", "unit_i", "pdb", "unit", "pose", "lr"] + list(TERMS) + \
                ["batch_" + k for k in TERMS] + ["batch_n", "partial_step", "step_ok"] + \
                ["grad_norm", "outside_head", "epoch", "step_taken", "nonfinite_grad"] + \
                (["pae_loss_unweighted"] if float(a.pae_row_frac) > 0 else [])
    # pae_loss IS THE WEIGHTED TERM once --pae-row-frac is on, because it is what is trained, and
    # a mean-1 row weight does NOT preserve the value: the weighted mean equals the unweighted one
    # only if the per-row loss is uncorrelated with the weight, and the weight is chosen BY t_i.
    # So the comparable curve lives in pae_loss_unweighted, read back every pose from the replayed
    # label, and plot_train_pae.py --unweighted reads that column instead.
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
    # {(bucket_key, n_tok): [seconds, calls]} -- forward and backward GPU time against the pose's
    # OWN token count, so the cost-vs-length exponent is measured rather than fitted. See the
    # comment on _gpu_span for why the trajectory.csv-derived version was not good enough.
    _T_TOK = {}
    # [units run UNCHECKPOINTED, units run CHECKPOINTED] under --ckpt-token-threshold. Reported
    # per epoch because a threshold that silently matches nothing -- or everything -- looks
    # exactly like a threshold that works until the timings fail to move.
    # THE PRECOMPUTED CROP MASKS. __n and __meta are BOOKKEEPING, not structures: the evaluator's
    # loader filters only __n and so counts __meta as a structure, which is why it prints one more
    # than it has. Filter both.
    _CROPTOK = {}
    _CROPPOOL = {}
    # {pdb: bincount of the UNCROPPED asym_id}, so n_pair_tokens for a pair (i, j) is
    # counts[i] + counts[j] exactly as (asym == i) | (asym == j) would have summed it.
    _UNCROP_CHAIN_N = {}
    # VALIDATED AGAINST THE POOL AT LOAD, not trusted. An index past a structure's crop count, or
    # an empty list, would otherwise surface as an IndexError hundreds of units in or silently
    # pin every epoch to one crop.
    _CROP_ALLOW = {}
    if a.crop_allow:
        # THE FLAG, NOT THE DICT. _CROPPOOL is filled LAZILY, one structure at a time as units
        # arrive, so it is empty here even on a --crop-pool run -- the first version tested it and
        # refused every launch with "has nothing to restrict". The bound check against each
        # structure's crop count therefore cannot happen here either; it happens in _crop_masks,
        # where _rows is in hand, on the first unit of each restricted structure.
        if not a.crop_pool:
            raise SystemExit("STOP: --crop-allow without --crop-pool has nothing to restrict.")
        with open(a.crop_allow) as _fh:
            _raw = json.load(_fh)
        for _p, _v in _raw.items():
            _vv = sorted({int(x) for x in _v})
            if not _vv:
                raise SystemExit(f"STOP: --crop-allow leaves {_p} with no usable crop.")
            if _vv[0] < 0:
                raise SystemExit(f"STOP: --crop-allow gives {_p} a negative index {_vv[0]}.")
            _CROP_ALLOW[_p] = _vv
        if is_main:
            print(f"  --crop-allow: {len(_CROP_ALLOW)} structure(s) restricted from "
                  f"{os.path.basename(a.crop_allow)}: "
                  + ", ".join(f"{_p} {len(_v)} crop(s)"
                              for _p, _v in sorted(_CROP_ALLOW.items())))

    def _uncrop_npt(pdb_, pair_):
        """Uncropped scored chain-pair token count, or None when not cropping."""
        if not _CROPPOOL:
            return None
        c_ = _UNCROP_CHAIN_N.get(pdb_)
        if c_ is None:
            raise SystemExit(f"STOP: no uncropped chain counts cached for {pdb_}; the crop mask "
                             f"must be applied before the rel score is taken.")
        i_, j_ = int(pair_[0]), int(pair_[1])
        if i_ >= len(c_) or j_ >= len(c_):
            raise SystemExit(f"STOP: {pdb_} pair {pair_} is outside its uncropped asym range.")
        return int(c_[i_]) + int(c_[j_])
    if a.crop_pool:
        # ONE FILE PER STRUCTURE, each holding N masks of possibly DIFFERENT lengths -- a pool can
        # mix crop sizes, because 28 structures needed a larger rung to reach 40 passing crops.
        # So the masks are kept as a list of index arrays rather than one rectangular block.
        for _f in sorted(os.listdir(a.crop_pool)):
            if not _f.endswith(".npz"):
                continue
            _z = np.load(os.path.join(a.crop_pool, _f), allow_pickle=True)
            _ks = sorted((k for k in _z.files if k.startswith("c") and k[1:].isdigit()),
                         key=lambda k: int(k[1:]))
            _rows = [np.asarray(r, dtype=np.int64) for k in _ks for r in _z[k]]
            if _rows:
                _CROPPOOL[_f[:-4]] = (_rows, np.asarray(_z["sizes"]))
        if not _CROPPOOL:
            raise SystemExit(f"STOP: --crop-pool {a.crop_pool} produced no masks.")
        if is_main:
            _n = np.array([len(v[0]) for v in _CROPPOOL.values()])
            _mix = sum(1 for v in _CROPPOOL.values() if len(set(v[1].tolist())) > 1)
            print(f"CROP POOL: {len(_CROPPOOL)} structure(s) from {a.crop_pool} -- "
                  f"crops per structure min {_n.min()} median {int(np.median(_n))} "
                  f"max {_n.max()}; {_mix} structure(s) mix crop sizes; a structure uses "
                  f"mask `epoch % n` so it sees a different view each epoch", flush=True)

    # THE --crop-tokens LOADER WAS HERE, with _CROPTOK, _CROPPAIRS and their reporting.
    # Deleted 2026-11-13 with the flag: its masks came from a builder that no longer
    # exists, and --crop-pool is the one crop rule left.


    def _crop_masks(pdb_, zc_):
        """(keep_tok, keep_atom) from whichever crop rule is configured. One dispatch, two callers.

        A MISSING MASK IS FATAL, never a silent fall-through to the full structure: a run where
        some units were cropped and some were not would still write a complete trajectory and
        name itself after the crop.
        """
        _asym_full = np.asarray(zc_["ci_feat_asym_id"]).reshape(-1)
        ntok_ = int(_asym_full.shape[0])
        if _CROPPOOL:
            # THE UNCROPPED CHAIN-PAIR SIZE, recorded HERE because this is the last point at which
            # the full token set is in hand -- everything downstream sees the cropped features.
            # iptm_pooled needs it to pin d0 to the same value the rel target was built with.
            if pdb_ not in _UNCROP_CHAIN_N:
                _UNCROP_CHAIN_N[pdb_] = np.bincount(_asym_full.astype(np.int64))
            _ent = _CROPPOOL.get(pdb_)
            if _ent is None:
                raise SystemExit(f"STOP: --crop-pool has no entry for {pdb_}. The pool holds "
                                 f"{len(_CROPPOOL)} structure(s); this manifest needs {pdb_} too.")
            _rows, _ = _ent
            # epoch % n, NOT a draw. The n masks of a structure are independent enumerations of
            # its own eligible anchors, so a fixed rotation gives the same view distribution as a
            # random one and makes two runs on the same pool reproducible. The crop_index used
            # here MUST be the one the labels were keyed by -- see _rel_labels.
            # THE ROTATION RUNS OVER THE ALLOWED CROPS, which is the whole pool unless
            # --crop-allow names this structure. Still epoch % n rather than a draw, so two runs
            # on the same pool and allowlist see the same sequence.
            _alw = _CROP_ALLOW.get(pdb_) if _CROP_ALLOW else None
            if _alw and max(_alw) >= len(_rows):
                raise SystemExit(f"STOP: --crop-allow gives {pdb_} crop index {max(_alw)} but its "
                                 f"pool holds {len(_rows)}. The allowlist was built against a "
                                 f"different crop pool.")
            if _alw:
                _ci = int(_alw[int(_crop_epoch[0]) % len(_alw)])
            else:
                _ci = int(_crop_epoch[0]) % len(_rows)
            kt_ = np.zeros(ntok_, dtype=bool)
            _idx = _rows[_ci]
            if _idx.size and int(_idx.max()) >= ntok_:
                raise SystemExit(f"STOP: {pdb_} crop {_ci} indexes token {int(_idx.max())} but "
                                 f"its cached trunk has {ntok_}. The pool was built against a "
                                 f"different cache and slicing with it would scramble the "
                                 f"structure.")
            kt_[_idx] = True
        else:
            # UNREACHABLE: --crop-pool is the only crop rule, and _crop_masks is only called
            # when _USE_CROP, which is now bool(a.crop_pool). Kept as a refusal rather than
            # deleted so a future second rule cannot fall through to an undefined mask.
            raise SystemExit(f"STOP: _crop_masks reached with no --crop-pool entry for {pdb_}.")
        a2t_ = np.asarray(zc_["ci_feat_atom_to_token_idx"]).reshape(-1)
        if kt_.shape[0] != ntok_:
            raise SystemExit(f"STOP: {pdb_}'s mask has {kt_.shape[0]} token(s) but its cached "
                             f"trunk has {ntok_}. The mask was built against a different cache "
                             f"and slicing with it would scramble the structure.")
        if _TOK_DROP:
            # ONE DRAW PER (STRUCTURE, EPOCH), seeded so it is reproducible and so every rank and
            # the prefetch worker agree without communicating. Per-epoch IS per-step here:
            # build_epoch_cross_seed puts every seed of a structure in a single step.
            # hashlib, NOT hash(). Python salts str hashing per process (PYTHONHASHSEED), so
            # hash() would give each RANK a different draw for the same structure -- and under
            # --rank-cross-seed the ranks hold different SEEDS OF THE SAME STRUCTURE inside one
            # step, so they must agree exactly or those poses are scored on different token sets
            # and the relative loss grades the disagreement. It also has to survive a resume.
            _sd_bytes = f"{int(a.seed)}|{int(_crop_epoch[0])}|{pdb_}".encode()
            _rs = np.random.default_rng(
                int.from_bytes(hashlib.sha256(_sd_bytes).digest()[:8], "big"))
            _p = np.where(kt_, float(a.token_keep_union), float(a.token_keep_other))
            _keep = _rs.random(kt_.shape[0]) < _p
            # EVERY SUPERVISED PAIR MUST STAY SCOREABLE -- per pair, not "two chains somewhere".
            # ipTM for pair (x, y) pools over token pairs spanning x and y, so if either chain
            # loses its tokens that interface is scored on an EMPTY pool while its label stays in
            # the loss. A four-chain complex can keep two chains and still leave pair (3,1)
            # unscoreable, and 81% of this pool has 2+ scored pairs. This is NOT a requirement
            # that any ground-truth CONTACT survives -- the augmentation being tested is precisely
            # that contacts may vanish. Raised in review.
            _asym = np.asarray(zc_["ci_feat_asym_id"]).reshape(-1)
            _prs = _CROPPAIRS.get(pdb_)
            if _prs is None:
                raise SystemExit(
                    f"STOP: --token-keep-* needs the scored pairs for {pdb_}, which this mask "
                    f"does not carry (no {pdb_}__pairs entry). It was built before the builder "
                    f"wrote them; rebuild with run_logging/build_crop_pool.py (run_logging/build_crop_pool.py (build_union_crop_masks.py was removed 2026-11-13; the crop pool this project uses is --crop-pool, built by build_crop_pool*.py) was removed 2026-11-13; the crop pool this project uses is --crop-pool, built by build_crop_pool*.py) so the "
                    f"per-pair scoreability check has something to check.")
            # REPAIRED, NOT REFUSED. The first version raised SystemExit here and killed the
            # run 2,240 poses into vxwvzr's first epoch, on 9iut: chain 2 drew 3 survivors against
            # a floor of 4. That was not bad luck. 98 scored-chain instances in this pool have an
            # EXPECTED survivor count under 12 and some scored chains hold only 6 tokens in
            # total -- short peptide antigens -- so a 0.70 draw dips under the floor routinely.
            #
            # Escalating a whole run over one small chain is the wrong response to a constraint
            # that is local: the floor exists so each supervised pair has a non-degenerate ipTM
            # pool, and that is restored by adding back a few of THAT CHAIN's tokens. Dropping
            # context is the augmentation; a 6-token chain has no context to drop. The top-up uses
            # the same generator in a deterministic order, so every rank still agrees exactly.
            _MIN_CH = 4
            for _x, _y in sorted({(int(_p0), int(_p1)) for _p0, _p1 in _prs}):
                for _c in (_x, _y):
                    _idx_c = np.nonzero(_asym == _c)[0]
                    _floor = min(_MIN_CH, _idx_c.shape[0])
                    _have = int(_keep[_idx_c].sum())
                    if _have >= _floor:
                        continue
                    _missing = _idx_c[~_keep[_idx_c]]
                    _add = _rs.permutation(_missing)[:_floor - _have]
                    _keep[_add] = True
                    _tok_drop_stat["topped"] += 1
                    _tok_drop_stat["topped_tok"] += int(_add.shape[0])
            if int(_keep.sum()) < 16:
                raise SystemExit(
                    f"STOP: the token draw for {pdb_} at epoch {_crop_epoch[0]} left only "
                    f"{int(_keep.sum())} token(s) in total. Raise the keep probabilities.")
            _tok_drop_stat["kept"] += float(_keep.sum())
            _tok_drop_stat["tot"] += float(_keep.shape[0])
            _tok_drop_stat["n"] += 1
            kt_ = _keep
        inr_ = (a2t_ >= 0) & (a2t_ < ntok_)
        ka_ = np.zeros(a2t_.shape[0], dtype=bool)
        ka_[inr_] = kt_[a2t_[inr_]]
        return kt_, ka_

    # THE EPOCH THE TOKEN DRAW IS SEEDED FROM. Set once at each epoch boundary BEFORE the
    # prefetch workers start, then read-only for the epoch -- which is what makes it safe for
    # those threads to read without a lock. It has to be a closure cell rather than a parameter
    # because _cpu_crop runs in the worker, which crops units before the step is known.
    _crop_epoch = [0]
    _tok_drop_stat = {"kept": 0.0, "tot": 0.0, "n": 0, "topped": 0, "topped_tok": 0}
    _ckpt_hist = [0, 0]
    # THE EPOCH'S PEAK, KEPT BY HAND, because --mem-log resets the CUDA high-water mark before
    # every pose to make the per-pose number mean the pose. Without this the epoch line would
    # report the LAST pose's peak and call it the epoch's. Zeroed at each epoch boundary beside
    # reset_peak_memory_stats. Unused when --mem-log is off: the CUDA counter is then already the
    # epoch's, because nothing reset it mid-epoch.
    _mem_ep = [0]

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

    # PER-TOKEN-COUNT GPU TIME. The aggregate buckets answer "where did the epoch go" but not
    # "how does it scale", and the scaling is what decides whether cropping is worth the change.
    # It had to be inferred from trajectory.csv's wall_s differences instead, which absorb the
    # optimiser step, the all-reduce and the trunk load, and are logged at 0.1 s resolution -- so
    # a log-log fit on them read an exponent of 1.31 when fitting an additive constant alongside
    # the power gave 2.47. These are CUDA events around the forward and the backward only, keyed
    # by the pose's own token count, so the exponent comes out of measurement rather than a fit.
    #
    # COST: one dict update per pose per pass, against a ~0.45 s pose. The table is a few hundred
    # token values, printed once per epoch and reset with the other counters.
    def _gpu_span(key, n_tok=None):
        if not torch.cuda.is_available():
            return None, None
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        _ev_pending.append((key, e0, e1, n_tok))
        return e0, e1

    def _drain_gpu_spans():
        if not _ev_pending:
            return
        torch.cuda.synchronize()
        for key, e0, e1, n_tok in _ev_pending:
            _dt = e0.elapsed_time(e1) / 1000.0
            _T[key] += _dt
            _TN[key] += 1
            if n_tok is not None and n_tok > 0:
                _slot = _T_TOK.setdefault((key, int(n_tok)), [0.0, 0])
                _slot[0] += _dt
                _slot[1] += 1
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
    # AND THE CONTENT HALF ALONE, which is the half that costs anything and the only half that is
    # persisted. _fp_cache holds the COMPOSED string this process compares against labels;
    # _fp_content holds `cif|bio|idx|trunk` for the same key, so a new process (or a sibling rank)
    # can skip the four SHAs and recompose. Both are filled by _unit_fp and nothing else.
    _fp_content = {}
    # AND THE STAT SIGNATURE AS IT WAS WHEN THE CONTENT WAS HASHED, so what is persisted describes
    # the files that were actually read. See _unit_fp.
    _fp_sig_at_hash = {}
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
    # with a stat signature -- size and mtime_ns of every file the fingerprint covers. That is the
    # same bargain phase_k_eval's --join-cache already makes ("the cache key carries each trunk's
    # size and mtime, so a refold or rebuild invalidates it"), and it is a real reduction in
    # checking: a file rewritten with its size and mtime preserved would go unnoticed. Everything
    # that rewrites these files -- a refold, a rebuild, the 2026-10-14 geometry fix -- changes mtime.
    #
    # ONLY THE CONTENT HALF IS PERSISTED, AND THAT IS THE ~43 MIN FIX. Version 1 stored the WHOLE
    # fingerprint and refused a file whose stored code_fp or loss_fp differed from this process's,
    # dropping all ~11,040 entries at once -- so every edit to phase_j_build_cache or to the loss
    # config threw away the entire cache and the run re-SHA'd ~243 MB per unit from scratch, ~43 min
    # off the front of the launch. But `code` and `loss` are three short strings this process
    # already holds and they cannot change what sha256(trunk) is. So the file now stores only
    # BC.unit_content_fingerprint's output -- cif|bio|idx|trunk -- and the schema/code/loss prefix
    # is recomposed at every use by BC.unit_fingerprint(..., content=...). The composed string is
    # byte-identical to version 1's, gated by phases/test_unit_fingerprint_split.py against a
    # literal of the pre-split format, so the provenance comparison is unchanged: a stale code_fp
    # still mismatches the label's stored fingerprint and still raises, exactly as before. code_fp
    # and loss_fp are still written into the file, as a record of what wrote it, and are no longer
    # read for invalidation.
    #
    # THE STAT SIGNATURE IS STILL THE WHOLE INVALIDATION, and it covers every file the content half
    # hashes -- trunk, cif, bioassembly, indices. Nothing is adopted from disk without it matching.
    #
    # PER RANK, AND ALL RANKS' FILES ARE READ. Each rank sees a third of the units and the shuffle
    # moves that third every epoch, so one rank's file covers only part of the manifest. Writing
    # one file per rank and loading every one of them accumulates coverage across epochs and across
    # runs without any rank having to write what another measured.
    _FP_CACHE_VER = 2
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

    def _unit_fp(pdb, trunk_p, memo=None):
        """The COMPOSED provenance string for (pdb, trunk_p), memoised through its content half.

        THE ONE PLACE unit_fingerprint IS CALLED FROM. Both callers -- resolve_trunk's candidate
        loop and run_unit's per-unit re-assertion -- used to call BC.unit_fingerprint directly and
        each kept its own `if key not in cache` dance around it. Two call sites meant two chances
        for the memoisation and the timing to drift apart; this is one.

        WHAT IS MEMOISED AND WHAT IS NOT. The four SHAs (cif, bioassembly, indices rows, trunk) are
        memoised, in `_fp_content`, and that memo is what survives to disk. schema, code_fp and
        loss_fp are re-composed on every call from THIS process's values, so a cache written by a
        different checkout cannot smuggle its code_fp into a comparison -- the composed string is
        always this process's claim about this process's code. `_fp_cache` then holds the composed
        result so a repeat call is one dict lookup.

        `memo` OVERRIDES THE COMPOSED CACHE ONLY, deliberately: resolve_trunk takes a caller-
        supplied dict for the composed strings, and the content memo stays global because it is
        keyed on file identity and nothing else.
        """
        memo = _fp_cache if memo is None else memo
        key = (pdb, trunk_p)
        hit = memo.get(key)
        if hit is not None:
            return hit
        content = _fp_content.get(key)
        if content is None:
            _t0 = time.perf_counter()
            content = BC.unit_content_fingerprint(a, pdb, trunk_p)
            # A WORKER'S HASH GOES IN ITS OWN BUCKET, because `fingerprint_hash` is in _EXCL -- the
            # sum that is subtracted from the epoch's wall clock to get the residual. Several
            # warmer threads hashing concurrently with the main thread can put MORE than the
            # epoch's wall clock into that bucket and drive the printed residual negative. That is
            # precisely the double-count the `rank_pass_a` comment on _EXCL was written about; it
            # already applied to the cropped arm's _pf_worker and this makes it explicit for both.
            # fingerprint_hash_warm is reported and is in neither _EXCL nor _leaves.
            _tick("fingerprint_hash" if threading.current_thread() is threading.main_thread()
                  else "fingerprint_hash_warm", _t0)
            # THE SIGNATURE IS TAKEN NOW, NOT AT SAVE TIME. _fp_cache_save used to re-stat at the
            # boundary, so a file rewritten between the hash and the save was persisted with a
            # signature describing the NEW file and content describing the OLD -- the one direction
            # a provenance cache must never be wrong in. Taking it beside the hash makes the pair
            # consistent by construction. (Version 1 had the same pattern; this is a strengthening,
            # not a regression fix.)
            _fp_sig_at_hash[key] = _fp_sig(pdb, trunk_p)
            _fp_content[key] = content
        fp = BC.unit_fingerprint(a, pdb, trunk_p, code_fp, loss_fp, content=content)
        memo[key] = fp
        return fp

    def _fp_cache_load(label="startup"):
        """Adopt every rank's CONTENT fingerprints whose stat signature still holds.

        CALLED AT EVERY EPOCH BOUNDARY, NOT ONLY AT STARTUP, so the three ranks pool their work.
        Each rank hashes only the units it was dealt and the three jointly cover all 6,903 units in
        epoch 1 -- but a rank that never loaded its siblings' files re-hashed, from epoch 2 onward,
        every unit the shuffle moved onto it. Reloading after _fp_cache_save at the boundary makes
        epoch 2 free for all three instead of one.

        IT CANNOT UNDO A HASH THIS PROCESS COMPUTED. Entries already in `_fp_content` are skipped
        before the signature is even looked at, so an adopted entry only ever fills a hole.
        """
        kept = dropped = 0
        for fp_path in sorted(glob.glob(os.path.join(roots[0], "unit_fingerprints.r*.json"))):
            try:
                blob = json.load(open(fp_path))
            except (OSError, ValueError):
                continue
            # A STALE SCHEMA INVALIDATES THE WHOLE FILE: version 1 stored composed fingerprints
            # under the key "fp" and there is no way to recover the content half from one, since
            # the prefix it was composed with is whatever that process held. code_fp and loss_fp
            # are NOT checked -- that was the bug, see the block comment above.
            if blob.get("ver") != _FP_CACHE_VER:
                dropped += len(blob.get("entries", {}))
                continue
            for key, ent in blob.get("entries", {}).items():
                pdb, _, tp = key.partition("\t")
                if not tp or (pdb, tp) in _fp_content:
                    continue
                if ent.get("content") and ent.get("sig") == _fp_sig(pdb, tp):
                    _fp_content[(pdb, tp)] = ent["content"]
                    # THE SIGNATURE JUST VERIFIED, carried forward so a later save re-writes what
                    # was actually checked rather than re-statting at the boundary.
                    _fp_sig_at_hash[(pdb, tp)] = ent["sig"]
                    kept += 1
                else:
                    dropped += 1
        # AT A BOUNDARY, ONLY ADOPTIONS ARE WORTH A LINE. `dropped` is sticky -- a save only ever
        # adds, so an entry whose inputs really did change is re-reported at every boundary for the
        # rest of the run -- and one such line per epoch per rank is noise, not information.
        if kept or (dropped and label == "startup"):
            print(f"fingerprint cache ({label}): {kept} content entry(ies) reused from "
                  f"{roots[0]}, {dropped} invalidated by a changed input or an older cache "
                  f"version", flush=True)
        return kept

    def _fp_cache_save():
        """Atomic, and only ever ADDS to what this rank already had on disk."""
        try:
            prev = json.load(open(_fp_disk))
            entries = prev.get("entries", {}) if prev.get("ver") == _FP_CACHE_VER else {}
        except (OSError, ValueError):
            entries = {}
        # list(), BECAUSE A WARMER THREAD CAN STILL BE WRITING IT. _tr_worker calls resolve_trunk,
        # which fills _fp_content, and a warmer that outlived _tr_end's join would make this
        # "RuntimeError: dictionary changed size during iteration" -- a crash at an epoch boundary,
        # after the epoch's work, which is the worst possible moment. A snapshot costs one list of
        # ~11,040 tuples. The signature comes from the hash, not from a fresh stat; see _unit_fp.
        for (pdb, tp), content in list(_fp_content.items()):
            entries[f"{pdb}\t{tp}"] = {"content": content,
                                       "sig": _fp_sig_at_hash.get((pdb, tp)) or _fp_sig(pdb, tp)}
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
        kt, ka = _crop_masks(pdb, zc)
        _tick("cpucrop_tokens", _t)
        if kt is None:
            return ("nocrop", None, None)
        _t = time.perf_counter()
        # MAP z_trunk WHEN ASKED. The uncropped path has had this since --trunk-memmap landed
        # (phase_k_train.py:5408); the cropped path never took it and read the whole member.
        _zt_src = None
        if a.trunk_memmap:
            try:
                _zt_src = RC.npy_memmap(trunk_p, "z_trunk")
            except Exception:                                       # noqa: BLE001
                _zt_src = None
            # COUNTED, NOT INFERRED. npy_memmap REJECTS a member it cannot map -- deflated, bad
            # header, non-C-order, wrong span -- by RETURNING None rather than raising, so the
            # except clause above never fires for the ordinary failure and a run could report
            # --trunk-memmap while mapping nothing at all. Two counters, so the epoch line says
            # what fraction was actually mapped. Raised in external review of 03970def.
            _TN["trunk_mapped" if _zt_src is not None else "trunk_memmap_fallback"] += 1
        sl = BC.slice_cached_unit(zc, lab_l, kt, ka, torch.device("cpu"), torch, zt_src=_zt_src)
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
    # UNITS THAT WERE DELIVERED AND THEN EVICTED. A worker emits each unit exactly ONCE, so an
    # evicted result can never return to the queue -- and _pf_take's loop only exits on a match or
    # on the budget expiring, so without this it waits the FULL 120 s for something that no longer
    # exists, then recrops inline. The eviction pops in insertion order, i.e. the lowest-numbered
    # stashed unit, which is precisely the one the main loop asks for next: so one eviction becomes
    # a cascade of 120 s waits. Reproduced on CPU in run_logging/prefetch_deadlock_repro.py --
    # W=1 gives 400 hits and 0 stalls, W=4 gives 40 consecutive full-budget stalls and does not
    # recover. That is ggfhj5's epoch 12: every rank at 0% GPU in futex_wait, nine minutes with no
    # trajectory write, and --dist-timeout-min 240 keeping anything from erroring out. Recording
    # the eviction turns each of those waits into an immediate inline recrop.
    _pf_dropped = set()
    # HIT/MISS IS REPORTED, because "crop_slice did not move" is how the first version's total
    # failure went unnoticed -- the inline fallback ticks the same counter as the fast path.
    _pf_stat = {"hit": 0, "miss": 0, "wait_s": 0.0, "dropped": 0, "exhausted": 0}

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
        # ALREADY DELIVERED AND EVICTED: it will never arrive again, so do not wait for it.
        if lab_p in _pf_dropped:
            _pf_dropped.discard(lab_p)
            _pf_stat["dropped"] += 1
            _pf_stat["miss"] += 1
            return None
        _t_w = time.perf_counter()
        deadline = _t_w + budget_s
        while True:
            # EVERY WORKER HAS FINISHED ITS STRIDE AND THE QUEUE IS EMPTY, so nothing more can
            # ever arrive. Observed in the reproduction: with W=4 the workers exit while the main
            # loop is still asking for units, after which EVERY remaining take cost the whole
            # budget. Waiting here is waiting on no one.
            if (_pf_thread is not None
                    and not any(_t.is_alive() for _t in _pf_thread)
                    and _pf_q.empty()):
                _pf_stat["exhausted"] += 1
                _pf_stat["miss"] += 1
                _pf_stat["wait_s"] += time.perf_counter() - _t_w
                return None
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
                # EVICT THE NEWEST, NOT THE OLDEST.
                #
                # `next(iter(...))` is insertion order, so this dropped the LOWEST-numbered unit
                # -- which, because the main loop consumes in that same order, is precisely the
                # one it is about to ask for. Every eviction therefore bought a guaranteed miss
                # and an inline recrop of the full trunk on the main thread.
                #
                # It got worse when the crop got faster: making _cpu_crop ~4.6x cheaper let the
                # workers run hundreds of units ahead instead of the "~W" this comment used to
                # claim, so on rdi25k epoch 5 the hit rate fell 68-80% -> 34-58% and evictions
                # doubled to 980-1,524 per rank. The feature partly cannibalised itself.
                #
                # The newest entry is the one the main loop will want LAST, so dropping it costs
                # the least and the entry it wants next survives. Found in adversarial review.
                _dk = next(reversed(_pf_pending))
                _pf_pending.pop(_dk)
                _pf_dropped.add(_dk)

    # DID THE CROP PREFETCH ACTUALLY START THIS EPOCH? Read by the epoch report, which cannot use
    # _pf_thread (it is cleared by _pf_end, which runs first) and must not use a.prefetch_units
    # (set on uncropped arms, where _pf_start now declines).
    _pf_started = [False]

    def _pf_start(units):
        nonlocal _pf_q, _pf_thread
        _pf_started[0] = False
        if not a.prefetch_units:
            return
        # NOT ON AN UNCROPPED ARM, where every result is thrown away. _pf_take is reached only
        # inside `if a.crop_margin > 0 or a.crop_tokens`, so with the crop off these workers ran a
        # full np.load plus interface_crop_tokens plus slice_cached_unit per unit and nobody ever
        # consumed one -- then blocked forever on a full put() once --prefetch-units x
        # --prefetch-workers results had piled up, holding those slices in RSS for the whole epoch.
        # The default arm passes --prefetch-units 2 --prefetch-workers 4, so that is 8 wasted
        # crops and 8 retained slices on every uncropped run to date. The uncropped path has its
        # own warmer (_tr_step_start) and does not use this queue.
        if not (_USE_CROP):
            return
        _pf_stop.clear(); _pf_pending.clear(); _pf_dropped.clear()
        _pf_stat.update(hit=0, miss=0, wait_s=0.0, dropped=0, exhausted=0)
        _w = max(1, a.prefetch_workers)
        # DEEP ENOUGH FOR EVERY WORKER TO BE IN FLIGHT. At maxsize 1 with W workers, W-1 of them
        # block on put() and the extra threads buy nothing.
        _pf_q = queue.Queue(maxsize=max(1, a.prefetch_units) * _w)
        _pf_thread = [threading.Thread(target=_pf_worker, args=(units[i::_w],), daemon=True)
                      for i in range(_w)]
        for _t in _pf_thread:
            _t.start()
        _pf_started[0] = True

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
        _pf_thread = None; _pf_q = None; _pf_pending.clear(); _pf_dropped.clear()

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

    # ---- THE STEP-SCOPED TRUNK CACHE, for the UNCROPPED arm -------------------------------------
    #
    # WHY. On an uncropped arm the only per-unit I/O is RC.load_unit, and the cross-seed objective
    # calls run_unit TWICE on every unit of a step -- a no_grad scoring pass A, then the backward
    # pass B -- so each unit's ~243 MB was materialised and copied to the device twice. Rank 0,
    # epoch 3 of mqird6: 4,600 load_unit calls for 2,300 units, 37.0 min of `unit_setup` with
    # 0.2 min in named leaves and 31.0 of the epoch's 35.9 min residual INSIDE units. Holding the
    # HOST arrays for the duration of ONE step halves the reads outright, and warming them on a
    # worker thread puts the surviving half behind the GPU.
    #
    # KEYED BY TRUNK PATH AND NEVER POPPED, which is the whole design and is not negotiable. The
    # cropped prefetch (_pf_take) is a POP QUEUE keyed by label with a 120 s blocking wait, and
    # pass A walking the same units as pass B consumed each unit's result and left pass B waiting
    # out the full budget per unit -- ~3 hours an epoch, observed on probe dsswrj, looking like a
    # hang with flat CPU and 0% GPU. A read-only dict cannot do that: both passes LOOK UP the same
    # key, a lookup never blocks, and a miss just reads inline.
    #
    # PURELY ADVISORY, SO NOTHING CAN DEADLOCK ON IT. Every path that consults it falls back to the
    # ordinary read, so a worker that is slow, failed, capped out or never started costs time and
    # nothing else. There is no wait, no budget and no eviction cascade to get wrong.
    #
    # ITS OWN CAP, IN BYTES, SEPARATE FROM THE CROP PREFETCH'S. _pf_pending's max(16, 8*workers)
    # would be 32 entries, and at 243 MB per uncropped trunk that is ~7.8 GB of RSS per rank
    # against --max-rss-gb. This caps on BYTES, defaults off, and REFUSES rather than evicts when
    # full: every entry in a step is wanted twice, so dropping the oldest would drop one pass B is
    # about to ask for, while refusing the newest only forgoes a saving. A deep step of the largest
    # remaining structures is 10 units x 1,650 tokens ~= 8.5 GB, so a 4 GB cap covers a typical
    # step and partially covers the worst, which is the intended behaviour.
    #
    # CLEARED AT EVERY STEP BOUNDARY. Scope is one step because that is the span over which a unit
    # is read twice; holding longer would grow without bound, since the epoch's units are distinct.
    _tr_cap = int(float(a.trunk_cache_gb) * (1 << 30))
    # CUDA IS PART OF THE PRECONDITION, not an accident of this box. unit_from_host aliases the
    # host arrays when the device is CPU, so two passes sharing one dict would share any in-place
    # write; `.to(cuda)` always copies. dev is cuda at line 1528, and this says so out loud.
    #
    # AND ONLY WHERE THERE IS A PASS A. The cache is scoped to a step and warmed at a step
    # boundary, both of which come from the cross-seed grouping; without it `_xs_at` is empty,
    # _tr_step_start is never called, every unit is read exactly once and there is nothing to hit.
    # The cache would then fill to its cap, refuse everything after that and hold the RSS for no
    # benefit -- so the same condition that gates _xs_prepare gates this.
    _tr_on = (_tr_cap > 0 and not (_USE_CROP)
              and getattr(dev, "type", "") == "cuda"
              and a.rank_cross_seed and a.rank_loss and not a.loss_only)
    _tr_cache = {}
    # PATH -> RESERVED BYTES, not a bare set of paths. A set records THAT a read is in flight; the
    # cap needs to know HOW MUCH. Review found that _tr_bytes counted only ADMITTED entries, so the
    # pre-read check `_tr_bytes[0] + _sz > _tr_cap` was evaluated against a number that four
    # concurrent warmers all saw as the same low value: each passed, each allocated its whole
    # ~243 MB, and TRUNK_CACHE_GB bounded the cache while the PROCESS went ~4x over it. The
    # reservation is taken under the lock at the same instant the check passes and released in the
    # `finally`, so the cap now bounds admitted + in-flight, which is what host RSS actually is.
    _tr_inflight = {}
    _tr_lock = threading.Lock()
    _tr_bytes = [0]
    _tr_resv = [0]
    # SHARED WITH THE WARMERS, AND `+=` ON A DICT ENTRY IS NOT ATOMIC, so a simultaneous put can
    # lose one sample out of thousands. Acceptable for a diagnostic and NOT acceptable for anything
    # the run branches on; nothing branches on these. Same bargain as _cpu_crop's counters.
    _tr_stat = {"hit": 0, "miss": 0, "put": 0, "full": 0, "err": 0, "dup": 0}
    _tr_threads = [None]
    # EVERY warmer ever started and not yet reaped. _tr_threads[0] holds only the CURRENT step's,
    # which is what made an older generation's thread unjoinable.
    _tr_live = []
    # A GENERATION COUNTER, NOT AN EVENT, AND THAT IS A BUG FIX. The first version used one shared
    # threading.Event: _tr_step_start did set() -> join(timeout=30) -> clear(). A warmer still
    # inside a 243 MB read when the join expired then resumed its loop, found the flag CLEARED, and
    # went on warming the PREVIOUS step's remaining units -- spending the new step's cap on trunks
    # nobody would ask for, so the step's real units were refused and read twice anyway. It was
    # never joined again either, because _tr_threads[0] had been replaced, so up to
    # --prefetch-workers orphans could leak per step boundary and live for the rest of the run, all
    # of them writing _fp_cache / _fp_content / _T without a lock while the main thread iterated
    # them at the epoch boundary. A generation number cannot be "un-cancelled": a worker compares
    # its own `gen` against this cell and a stale worker can only exit.
    #
    # AND IT LETS THE JOIN GO, which is the other half. Joining W warmers on the MAIN thread at
    # every one of 230 step boundaries put the main thread behind whatever read was in flight --
    # inside the very critical path this exists to shorten. Nothing is joined per step now; the
    # epoch boundary still joins, once, so no warmer races _fp_cache_save.
    _tr_gen = [0]

    def _tr_get(trunk_p):
        """The cached host arrays for this trunk, or None. LOOKS UP, NEVER POPS, NEVER BLOCKS."""
        if not _tr_on:
            return None
        with _tr_lock:
            hit = _tr_cache.get(trunk_p)
            _tr_stat["hit" if hit is not None else "miss"] += 1
        return hit

    def _tr_put(trunk_p, host, gen=None):
        """Offer host arrays to the cache. Refused when full, which costs only the saving.

        `gen` is the warmer's generation; a put from a step that is over is dropped rather than
        charged to the current step's cap.
        """
        if not _tr_on or host is None:
            return
        nb = sum(int(v.nbytes) for v in host.values() if hasattr(v, "nbytes"))
        with _tr_lock:
            if gen is not None and gen != _tr_gen[0]:
                return
            if trunk_p in _tr_cache:
                return
            # THIS PATH'S OWN RESERVATION IS DISCOUNTED, because the bytes it reserved before the
            # read are the very bytes it is now offering -- charging both would refuse a put that
            # the reservation had already made room for.
            _mine = _tr_inflight.get(trunk_p, 0)
            if _tr_bytes[0] + (_tr_resv[0] - _mine) + nb > _tr_cap:
                _tr_stat["full"] += 1
                return
            _tr_cache[trunk_p] = host
            _tr_bytes[0] += nb
            _tr_stat["put"] += 1

    def _tr_worker(units, gen):
        """Materialise a step's trunks into the cache. NO TORCH, NO CUDA, NO EXCEPTION ESCAPES.

        RESOLVES THE TRUNK ITSELF, in the thread, exactly as _pf_worker does -- resolve_trunk
        raises SystemExit on an unresolvable fingerprint and that must surface on the main thread
        when the unit is really reached, with its own diagnostics, not here where it would kill a
        warmer and nothing else. A swallowed SystemExit here leaves _trunk_choice unwritten, so the
        main thread re-resolves and refuses properly.

        NEVER memmap. A mapping is lazy, so reading one here would transfer the work it is supposed
        to absorb straight back to the main thread's device copy. The warmer exists to spend host
        time, in a heap buffer, while the GPU is busy.

        `gen` IS CHECKED THREE TIMES: at the top of each unit, under the lock before reserving, and
        again inside _tr_put. A warmer whose step is over does no read and caches nothing.
        """
        for _t in units:
            if gen != _tr_gen[0]:
                return
            tp = None
            try:
                tp = resolve_trunk(*_t)
                # THE SIZE IS CHECKED BEFORE THE READ, not after. Reserving _tr_inflight and then
                # refusing in _tr_put means the whole ~243 MB is already on the heap when the
                # refusal happens, so the cap bounded the CACHE and not the PROCESS: with four
                # warmers that is ~1 GB over the cap at any instant. The npz's file size is a
                # faithful proxy for the host dict's bytes, since every member is stored
                # uncompressed at the dtype it is read back as.
                _sz = os.path.getsize(tp)
                with _tr_lock:
                    if gen != _tr_gen[0]:
                        return
                    # ADMITTED PLUS ALREADY-RESERVED, which is the whole point of the fix. With
                    # only _tr_bytes[0] here, every concurrent warmer saw the same pre-read total.
                    if (tp in _tr_cache or tp in _tr_inflight
                            or _tr_bytes[0] + _tr_resv[0] + _sz > _tr_cap):
                        _tr_stat["dup"] += 1
                        continue
                    _tr_inflight[tp] = _sz
                    _tr_resv[0] += _sz
                try:
                    _tr_put(tp, RC.read_unit_host(tp), gen=gen)
                finally:
                    with _tr_lock:
                        _tr_resv[0] -= _tr_inflight.pop(tp, 0)
            except BaseException:          # noqa: BLE001 -- see the docstring
                # INSIDE THE LOCK, because `+=` on a dict entry is not atomic and several warmers
                # reach this. The same reason `dup` is counted under the lock above.
                with _tr_lock:
                    _tr_stat["err"] += 1
                    if tp is not None:
                        # BOTH, OR THE CAP LEAKS. An exception between the reservation and the
                        # `finally` -- resolve_trunk raising, a read error -- would otherwise leave
                        # the bytes reserved forever and the cache would refuse everything after a
                        # few failures.
                        _tr_resv[0] -= _tr_inflight.pop(tp, 0)

    def _tr_reset():
        """Retire the current generation and drop its arrays. One lock, no joins."""
        with _tr_lock:
            _tr_gen[0] += 1
            _tr_cache.clear()
            # THE RESERVATION IS NOT CLEARED HERE, only the admitted bytes. A warmer from the
            # retired generation may still be inside its read; its `finally` will pop its own entry
            # and decrement, and clearing the dict underneath it would make that pop a no-op and
            # leave _tr_resv permanently above zero. The generation check stops NEW reservations,
            # which is what retiring a step needs to do.
            _tr_bytes[0] = 0
            return _tr_gen[0]

    def _tr_step_start(step_units):
        """Drop the previous step's arrays and start warming this step's. Called per step."""
        if not _tr_on:
            return
        gen = _tr_reset()
        _w = max(1, min(a.prefetch_workers, len(step_units)))
        # EVERY GENERATION'S THREADS ARE TRACKED, not just the latest. _tr_threads[0] was
        # OVERWRITTEN here each step, so a warmer from an earlier generation became unreachable and
        # _tr_end could never join it -- it could outlive the epoch boundary and run while
        # _fp_cache_save iterated shared state. Retired threads are reaped first so the list cannot
        # grow across 231 steps.
        _tr_live[:] = [t for t in _tr_live if t.is_alive()]
        _tr_threads[0] = [threading.Thread(target=_tr_worker, args=(step_units[i::_w], gen),
                                           daemon=True) for i in range(_w)]
        for _t in _tr_threads[0]:
            _t.start()
        # REGISTERED AFTER START, so _tr_end can join this generation even once _tr_threads[0]
        # has been replaced by the next step's.
        _tr_live.extend(_tr_threads[0])

    def _tr_end():
        """At the EPOCH boundary: retire, then join, so nothing races _fp_cache_save.

        THE RESET IS UNCONDITIONAL, NOT GATED ON THERE BEING THREADS. Returning early when
        _tr_threads[0] is falsy left the cache holding up to --trunk-cache-gb across the evaluator
        subprocesses in the case where the main thread had filled it but no warmer ever started.
        """
        _tr_reset()
        # JOIN EVERY GENERATION STILL ALIVE, not only the newest. The generation counter already
        # guarantees a stale warmer cannot reach the cache; this guarantees it is not still RUNNING
        # when _fp_cache_save walks shared state at the boundary.
        # DEDUPED. The current generation is registered in BOTH _tr_live and _tr_threads[0], so
        # joining the concatenation gave every current thread two 30 s windows and doubled the
        # worst-case boundary stall to 30 s x 2 x prefetch_workers. Threads are hashable, and
        # dict.fromkeys keeps the order.
        for _t in dict.fromkeys(list(_tr_live) + list(_tr_threads[0] or ())):
            _t.join(timeout=30)
        _still = [t for t in _tr_live if t.is_alive()]
        if _still:
            print(f"  trunk warmer: {len(_still)} thread(s) still alive after a 30s join; their "
                  f"generation is retired so they cannot reach the cache, but they are reading.",
                  flush=True)
        # A STRAGGLER STAYS REGISTERED. Clearing the list outright dropped the only reference to
        # a thread that outlived its join, so it was never joined again and the next boundary
        # could not wait for it either. _tr_start already reaps the ones that have since died.
        _tr_live[:] = _still
        _tr_threads[0] = None
        # A WARMER THAT OUTLIVES THE JOIN CAN STILL BE HOLDING ONE READ, and that is all it can do:
        # its generation is retired, so _tr_put drops the result and the loop exits at its next
        # top. Nothing it does after this line reaches the cache or the byte count.

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
            if _unit_fp(pdb, tp, memo=want_fp_cache) == stored:
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
            want_f = fields(_unit_fp(pdb, cands[0], memo=want_fp_cache))
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
            _STALE_OK = {"code", "bio", "cif", "idx"}
            if a.allow_stale_labels and diff and set(diff) <= _STALE_OK:
                if not code_drift.get("stale_warned"):
                    print(f"  ACCEPTING STALE LABELS (--allow-stale-labels, --pae-entropy "
                          f"{a.pae_entropy:g}): {lab_p} differs on {diff}. THE LABELLED LOSSES IN "
                          f"THIS RUN'S trajectory.csv ARE NOT TRUSTWORTHY -- pae_loss, pde_loss, "
                          f"plddt_loss and resolved_loss are all computed against these labels. "
                          f"The entropy objective does not read them, so the GRADIENT is "
                          f"unaffected; nothing else here is.")
                    code_drift["stale_warned"] = True
                code_drift["stale_n"] = code_drift.get("stale_n", 0) + 1
                code_drift.setdefault("stale_pdbs", set()).add(pdb)
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
        # WHERE THE RESIDUAL IS, SPLIT ONCE. Epoch 17 left 14.9 min of 143.9 unaccounted -- the
        # second-largest single item in the table and the only one nobody can act on, because
        # "residual" does not say whether it is inside a unit or between units. unit_setup closes
        # the first half of that question: it covers everything from entering a unit to the moment
        # its first pose starts, which is the trunk read, the fingerprint, the dataset item and
        # the loss-feature prep that the existing per-step buckets step over. Whatever is left
        # after this is OUTSIDE the units, and that is a different fix.
        #
        # NOT A _gpu_span. This region launches no kernels of its own; a CUDA event here would
        # time the queue drain of the previous unit's poses and attribute it to setup.
        _t_unit_setup = time.perf_counter()
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
        _want = _unit_fp(pdb, trunk_p)
        _got = str(np.asarray(lab["fingerprint"]))
        if _got != _want:
            _dw = dict(x.split("=", 1) for x in _want.split("|") if "=" in x)
            _dg = dict(x.split("=", 1) for x in _got.split("|") if "=" in x)
            _df = sorted(k for k in set(_dw) | set(_dg) if _dw.get(k) != _dg.get(k))
            # THE SAME ALLOWANCE AS resolve_trunk, OR THE FLAG DOES NOTHING. This re-assertion
            # runs per unit after resolve_trunk has already accepted the label, so leaving it
            # strict would stop the run one call later with a message that names no flag --
            # exactly the half-wired shape the --lora-keep-trainable guard had.
            _ok_drift = (_df == ["code"] and a.allow_code_drift) or (
                a.allow_stale_labels and _df and set(_df) <= {"code", "bio", "cif", "idx"})
            if not _ok_drift:
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
        if _USE_CROP:
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
                _kt, _ka = _crop_masks(pdb, _zc)
                _tick("trunk_read_plus_crop_tokens", _t0)
                if _kt is None:
                    crop_stats["undefined"] += 1
                    crop_rows.append({"pdb": pdb, "unit": arm_seed, "n_tok": "", "kept_tok": "",
                                      "frames_cleared": "", "status": "no-crop-definable"})
                else:
                    _t1 = time.perf_counter()
                    # THE INLINE RECROP NEEDS THE MAPPING TOO, and this is the path that matters
                    # MOST because it is on the MAIN THREAD and therefore on the critical path.
                    # When --trunk-memmap landed on the cropped path it was wired only into
                    # _cpu_crop, which runs on the prefetch workers; every prefetch MISS fell
                    # through to here and still materialised the whole ~100 MB z_trunk. Measured
                    # on rdi25k epoch 5: 980 / 1,115 / 1,524 units per rank took this path, 42-66%
                    # of 2,308, which is most of why the feature returned 10% rather than the 2.4x
                    # its single-card smoke suggested. Raised in adversarial review of 03970def.
                    _zts = None
                    if a.trunk_memmap:
                        try:
                            _zts = RC.npy_memmap(trunk_p, "z_trunk")
                        except Exception:                           # noqa: BLE001
                            _zts = None
                        _TN["trunk_mapped" if _zts is not None
                            else "trunk_memmap_fallback"] += 1
                    crop_slice = BC.slice_cached_unit(_zc, lab, _kt, _ka, dev, torch, zt_src=_zts)
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
            # TIMED, AT LAST. This is the UNCROPPED path's only trunk read and it carried no timer
            # of any kind, which is why 31.0 of epoch 3's 35.9 min residual could be placed INSIDE
            # units and no further. load_unit now ticks `trunk_npz_open`, `trunk_npz_read` and
            # `trunk_h2d` through this closure's own accumulator, so the read and the device copy
            # land in the epoch table separately -- the first answers whether prefetching it would
            # pay (item 4 of docs/training_speedups.md), the second whether pinned staging would.
            # ~243 MB per unit and 4,600 units per rank per epoch, i.e. ~1.1 TB each way.
            #
            # THE STEP'S CACHED HOST ARRAYS, IF THEY ARE THERE. A hit skips the npz entirely and
            # leaves only the device copy; a miss reads inline AND OFFERS THE RESULT, so pass B
            # hits on whatever pass A had to read itself. Both are lookups -- see the cache's own
            # comment for why this is a dict and not the cropped path's pop queue.
            _host = _tr_get(trunk_p)
            if _tr_on and _host is None:
                _host = RC.read_unit_host(trunk_p, tick=_tick, memmap=a.trunk_memmap)
                _tr_put(trunk_p, _host)
            feat_head, s_inputs, s_trunk, zt, poses, _nc = RC.load_unit(
                trunk_p, dev, tick=_tick, memmap=a.trunk_memmap, host=_host)

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
        # SIZE-CONDITIONAL CHECKPOINTING, decided here because this is the first point in the unit
        # where the token count is known and no pose has run yet. Assigned on EVERY unit rather
        # than only when it changes, so the stack can never carry a previous unit's setting into
        # this one -- a silent carry-over would be an OOM on the next large unit.
        if a.ckpt_token_threshold > 0:
            _bpc = (None if 0 <= n_tok <= a.ckpt_token_threshold
                    else (a.blocks_per_ckpt or None))
            model.confidence_head.pairformer_stack.blocks_per_ckpt = _bpc
            _ckpt_hist[0 if _bpc is None else 1] += 1
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

        _tick("unit_setup", _t_unit_setup)

        def _one_pose(i, nb=1):
            # nb POSES IN ONE FORWARD. The slice is the only difference: x already carried a
            # sample dimension of 1, so widening it is what the head wanted all along.
            x = poses[i:i + nb].to(dev).float() if torch.is_tensor(poses) \
                else torch.as_tensor(poses[i:i + nb], dtype=torch.float32, device=dev)
            _fe0, _fe1 = _gpu_span("gpu_forward", n_tok)
            # WHICH POSE THE FROZEN-PREFIX CACHE IS LOOKING AT, and whether it may fill.
            #
            # (pdb, arm_seed, pose) IS ENOUGH OF AN IDENTITY because the cache is dropped at every
            # step boundary and a step holds each pose exactly once; n_tok goes in too, so a unit
            # re-entered at a different token count is a MISS rather than a shape that lines up.
            # fill=score_only is the pass: pass A computes the prefix and offers it, pass B takes
            # it. A fill in pass B would pay a D2H for a tensor nothing reads again.
            #
            # CLEARED IN A `finally`, so a forward that happens anywhere else in this process
            # cannot be keyed as this pose and hit on it. The plan is decided once, inside
            # _prep_blocks, so clearing it here does not disturb the backward that re-runs the
            # block under blocks_per_ckpt=1.
            if _pfc is not None:
                if nb != 1:
                    raise SystemExit(
                        "STOP: --pose-batch > 1 is incompatible with the frozen-prefix cache. "
                        "PFC keys a prefix by (pdb, seed, POSE) and a batched forward covers nb "
                        "poses at once, so one key would stand for several and a later pose could "
                        "hit on another's prefix.")
                PFC.set_pose((pdb, arm_seed, int(i)), fill=bool(score_only), n_tok=n_tok)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    plddt, pae, pde, resolved = model.run_confidence_head(
                        input_feature_dict=feat_head, s_inputs=s_inputs, s_trunk=s_trunk,
                        z_trunk=zt, pair_mask=None, x_pred_coords=x,
                        triangle_multiplicative=configs.triangle_multiplicative,
                        triangle_attention=configs.triangle_attention, inplace_safe=False,
                        # CHUNKING WAS TRIED AND REMOVED: --chunk-size 64 cost 5.8x in speed for
                        # no measurable memory, and Protenix's own config says chunk_size "should
                        # set to null for normal training".
                        chunk_size=None)
            finally:
                if _pfc is not None:
                    PFC.clear_pose()
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
                # THE SAME RANGE AS x, so label row k belongs to pose i+k. Slicing with [i] gave a
                # (N_atom, 3) array and [i:i+nb] gives (nb, N_atom, 3), which is the shape the loss
                # already expects for N_sample -- the single-pose path was the special case.
                ld = {"coordinate": torch.as_tensor(_cc[i:i + nb], dtype=torch.float32, device=dev),
                      "coordinate_mask": torch.as_tensor(_cm[i:i + nb].astype(np.float32),
                                                         device=dev)}
                # ONE LABEL FOR THE GROUP, WHICH IS WHAT PROTENIX DOES AND WHAT ITS LOSS
                # SIGNATURE SAYS. protenix/model/protenix.py:1713 permutes the ground truth ONCE,
                # against a single mini-rollout, then scores every sample against it; loss.py's
                # docstrings accordingly take `true_coordinate [..., N_atom, 3]` and
                # `coordinate_mask [N_atom]`. Our cache instead permutes PER POSE
                # (phase_j_build_cache.py:1052), which is why five labels exist at all.
                #
                # FOR A PAE-ONLY ARM THAT COSTS EXACTLY NOTHING, measured over the whole training
                # set rather than argued: run_logging/check_pae_label_shared.py compared every
                # pose against pose 0 on the PAE representative atoms across all 8,981 label
                # files -- 8,981 identical, 0 moved, worst displacement 0.000000 A. The per-pose
                # permutation only ever swaps symmetric SIDE-CHAIN atoms, and PAE is built from
                # the per-token representative atoms, which it never touches. The losses that
                # would have noticed -- pLDDT, PDE, resolved -- are stubbed out under --pae-only,
                # and the startup guard refuses --pose-batch > 1 without it.
                ld = {k: v[0] for k, v in ld.items()}
            else:
                if nb != 1:
                    raise SystemExit(
                        "STOP: --pose-batch > 1 needs the cropped label path. "
                        "BC.minimal_loss_inputs takes ONE pose index and there is no range form, "
                        "so batching here would silently train every pose against pose i's label.")
                _, ld = BC.minimal_loss_inputs(lab, i, dev, torch)
            # THE POSE'S SAMPLE INDEX, resolved ONCE here and reused by the mask, pass A and
            # pass B. Three readers previously resolved it separately, and the first version of
            # the rel wiring had one of them use the pose index instead.
            _samp_i = None
            if a.rank_form == "rel" and a.rank_loss:
                _sv_i = lab.get("sample_index") if hasattr(lab, "get") else None
                if _sv_i is None:
                    try:
                        _sv_i = lab["sample_index"]
                    except Exception:                 # noqa: BLE001
                        _sv_i = None
                if _sv_i is None:
                    raise SystemExit(
                        f"STOP: --rank-form rel needs `sample_index` in {lab_p} to resolve this "
                        f"pose's chain pair; without it the PAE mask and the rel target could "
                        f"describe different interfaces.")
                _samp_i = int(np.asarray(_sv_i).reshape(-1)[i])
            # PAE IS MASKED TO THE SAME PAIRS rel RANKS, per pose, under --rank-form rel.
            _po = None
            if _samp_i is not None:
                _po = {(min(_x, _y), max(_x, _y))
                       for _ifc, _x, _y in _pose_pairs(pdb, lab_p, _samp_i)}
            _set_pae_mask(pdb, feat_head["asym_id"], feat_cache["has_frame"], arm_seed,
                          pairs_override=_po)
            # PASS A STOPS HERE. The scoring sweep needs the pose's ipTM and nothing else: no loss,
            # no metrics, no backward. Returning before calculate_losses is not an optimisation, it
            # is the point -- pass A must not touch the loss state that pass B reports, and it runs
            # inside torch.no_grad() so no graph is built for the forward it just did.
            #
            # THE SAME ipTM FUNCTION AS PASS B, deliberately. If the two disagreed, the coefficient
            # would have been computed for a different function than the one being differentiated.
            if score_only:
                import rank_loss as RKL
                # THE POSE INDEX IS NOT THE SAMPLE INDEX, and the external labels are keyed by
                # the SAMPLE. The label npz carries the permutation in `sample_index`; measured on
                # the TRAINING fold it is non-identity in 3971 of 4000 files (99.3%) over 120
                # distinct patterns, e.g. 10fj/free_500000 -> [0, 4, 3, 2, 1]. This file already
                # documents the same trap for --emit-iptm-rows, where writing `i` moved the
                # aggregate by only 0.0009 while 73.8% of rows carried another sample's DockQ.
                # The rel target repeated it until external review caught it: pose 1 was being
                # regressed onto pose 4's true iTM.
                # ONE RESOLUTION, SHARED. _samp_i was resolved above for the PAE mask; under rel
                # it is always set, and for the other forms pass A still needs it for its key.
                if _samp_i is not None:
                    _samp_a = _samp_i
                else:
                    _si_a = lab.get("sample_index") if hasattr(lab, "get") else None
                    if _si_a is None:
                        try:
                            _si_a = lab["sample_index"]
                        except Exception:            # noqa: BLE001
                            _si_a = None
                    if _si_a is None:
                        raise SystemExit(
                            f"STOP: pass A needs `sample_index` in {lab_p} to key the group's "
                            f"labels the way results.csv and the rel label cache do. Without it a "
                            f"pose would be trained against another pose's label.")
                    _si_a = np.asarray(_si_a).reshape(-1)
                    if i >= _si_a.shape[0]:
                        raise SystemExit(f"STOP: pose {i} has no sample_index entry in {lab_p} "
                                         f"({_si_a.shape[0]} entries).")
                    _samp_a = int(_si_a[i])
                _sc = {}
                # KEYED BY WHAT IS STABLE ACROSS POSES. For rel the unit of comparison is the
                # reference interface; its asym pair is resolved PER POSE, so the key is the
                # iface and the pair is a per-pose detail of how the score is computed.
                if a.rank_form == "rel":
                    _items = [(_ifc, (_x, _y)) for _ifc, _x, _y
                              in _pose_pairs(pdb, lab_p, _samp_a)]
                else:
                    _items = [(_p, _p) for _p in sorted(_scored_asym.get(pdb) or ())]
                for _k_, _p in _items:
                    # POOLED THE SAME WAY AS PASS B. If these differed, the coefficients would
                    # describe one quantity and the backward would move another.
                    _v = RKL.iptm_pooled(pred["pae"], feat_head, configs, _p,
                                         pool=a.rank_pool,
                                         n_pair_tokens=_uncrop_npt(pdb, _p))
                    if _v is not None:
                        _sc[_k_] = float(_v.detach())
                del pred, x, plddt, pae, pde, resolved
                return {"pose": i, "sample": _samp_a, "iptm": _sc}
            if a.emit_iptm_rows and _scored_asym.get(pdb):
                # THE SAME FUNCTION PASS A USES. If this computed ipTM differently from the
                # ranking loss's own scorer, the emitted rows would describe a statistic the run
                # does not otherwise use, which is the failure the pass A comment warns about.
                import rank_loss as RKL
                if _iptm_fh["fh"] is None:
                    _iptm_fh["fh"] = open(a.emit_iptm_rows, "w", newline="")
                    _iptm_fh["w"] = csv.writer(_iptm_fh["fh"])
                    _iptm_fh["w"].writerow(["iface", "seed", "sample", "iptm"])
                _unit = os.path.basename(lab_p).rsplit(".", 1)[0]
                # THREE ARGUMENTS. _rank_key(pdb, arm_seed_or_int, pair) returns a key whose
                # second element is the parsed seed; calling it with one argument raised
                # TypeError on the FIRST emitted row, so --emit-iptm-rows could never have
                # produced a single line. Every other call site in this file passes three and
                # takes [1] for the seed -- see the pass A key at the --rank-cross-seed site.
                # Found in review of 511519e7.
                _seed = _rank_key(pdb, _unit, (0, 0))[1]
                # THE POSE INDEX IS NOT THE SAMPLE INDEX. The label npz carries an explicit
                # pose -> sample permutation in `sample_index`, written by
                # phase_j_build_cache.sample_indices from COORDINATE IDENTITY against the
                # <pdb>_sample_<k>.cif files, and it is almost never the identity: measured
                # non-identity on 396 of 400 validation label files. phase_k_eval writes the
                # MAPPED k, and results.csv is keyed by it. Writing `i` here looked right -- the
                # aggregate moved by 0.0009 -- while 73.8% of rows carried another sample's DockQ
                # and 56 of 92 interfaces selected a different pose. Caught in review.
                _si = lab.get("sample_index") if hasattr(lab, "get") else None
                if _si is None:
                    try:
                        _si = lab["sample_index"]
                    except Exception:            # noqa: BLE001
                        _si = None
                if _si is None:
                    raise SystemExit(
                        f"STOP: --emit-iptm-rows needs `sample_index` in {lab_p} to key rows the "
                        f"way results.csv does, and this label does not carry it. Without it the "
                        f"rows would join to another sample's DockQ.")
                _si = np.asarray(_si).reshape(-1)
                if i >= _si.shape[0]:
                    raise SystemExit(f"STOP: pose {i} has no sample_index entry in {lab_p} "
                                     f"({_si.shape[0]} entries).")
                _samp = int(_si[i])
                for _p in sorted(_scored_asym.get(pdb) or ()):
                    # POOLED THE SAME WAY AS PASS B. If these differed, the coefficients would
                    # describe one quantity and the backward would move another.
                    _v = RKL.iptm_pooled(pred["pae"], feat_head, configs, _p,
                                         pool=a.rank_pool,
                                         n_pair_tokens=_uncrop_npt(pdb, _p))
                    if _v is None:
                        continue
                    _ifc = _iface_of.get((pdb, _p))
                    if _ifc is None:
                        continue
                    # FULL PRECISION, NOT SIX DECIMALS. dockq_from_iptm.py already records that
                    # the emitter's 6 dp makes ties MORE likely than the eval's and that 1 of 92
                    # interfaces on j583i8 epoch 1 tied at that precision with two different
                    # DockQ values behind it -- so rounding here can change the SELECTED POSE and
                    # therefore the metric. repr() round-trips exactly. Found in review.
                    _iptm_fh["w"].writerow([_ifc, _seed, _samp, repr(float(_v.detach()))])
                    _iptm_fh["n"] += 1
                _iptm_fh["fh"].flush()
            try:
                cum, met = loss_fn.calculate_losses(feat_cache, pred, ld, mode="train")
                if a.pae_entropy > 0.0:
                    # REPLACE, NOT ADD. `cum` above is the labelled objective; keeping any of it
                    # would make this a semi-supervised arm and the comparison would no longer be
                    # "what can be learned without labels". met is left as computed so
                    # trajectory.csv still carries the labelled pae_loss as a READOUT -- it is
                    # the only way to see whether sharpening is also fitting, and it costs a
                    # forward that has already happened.
                    _pl = pred["pae"]
                    _lp = torch.log_softmax(_pl.float(), dim=-1)
                    _ent = -(_lp.exp() * _lp).sum(-1)          # [..., n_tok, n_tok]
                    # NameError, NOT dir(). `_pae_keep` is a closure variable from the
                    # `if a.pae_scored_pairs:` block, and dir() with no argument lists only the
                    # LOCAL names -- so `"_pae_keep" in dir()` is False even when the mask exists,
                    # and the scored-pair mask would have been silently discarded in favour of a
                    # mean over every token pair. Exactly the silent-fallback class this file's
                    # other guards exist to prevent.
                    try:
                        _m_ent = _pae_keep.get("m")
                    except NameError:
                        _m_ent = None
                    # AND THE FALLBACK IS A STOP, NOT A MEAN, WHENEVER A MASK WAS ASKED FOR.
                    #
                    # Falling back to _ent.mean() when --pae-scored-pairs is set would train a
                    # DIFFERENT objective -- entropy over every token pair, antigen interior
                    # included -- while the log, the args and the run name all still say the
                    # scored pairs. That is the same shape of bug as the dir() one above, one
                    # level up: not a discarded mask but a quietly substituted loss, and
                    # unlabelled training has no held-out number that would catch it early.
                    # A mean is only legitimate when no mask was requested at all.
                    _want_mask = bool(getattr(a, "pae_scored_pairs", None))
                    if _want_mask:
                        if _m_ent is None or not torch.is_tensor(_m_ent):
                            raise SystemExit(
                                f"STOP: --pae-entropy with --pae-scored-pairs but no scored-pair "
                                f"mask reached the loss for {pdb}/{arm_seed} pose {i} "
                                f"(_pae_keep={'absent' if _m_ent is None else type(_m_ent)}). "
                                f"Refusing to average entropy over every token pair.")
                        if tuple(_m_ent.shape[-2:]) != tuple(_ent.shape[-2:]):
                            raise SystemExit(
                                f"STOP: scored-pair mask {tuple(_m_ent.shape)} does not match the "
                                f"PAE grid {tuple(_ent.shape)} for {pdb}/{arm_seed} pose {i}.")
                        _w = _m_ent.to(_ent.dtype)
                        _den = _w.sum()
                        if float(_den) <= 0.0:
                            raise SystemExit(
                                f"STOP: scored-pair mask is all zero for {pdb}/{arm_seed} pose "
                                f"{i}; the entropy term would be 0.0 with no gradient and the "
                                f"pose would look trained.")
                        _ent_v = (_ent * _w).sum() / _den
                    else:
                        _ent_v = _ent.mean()
                    cum = a.pae_entropy * _ent_v
                    met = dict(met)
                    met["pae_entropy"] = float(_ent_v.detach())
                if nb > 1:
                    # PROTENIX'S LOSSES MEAN OVER THE SAMPLE AXIS -- PAELoss loss.py:995, PDELoss
                    # :771, PLDDTLoss :1372, ExperimentallyResolvedLoss :1058 all end in a mean.
                    # Downstream, _obj = cum / _step_poses_g[0] divides by a POSE count, so with
                    # nb = 1 each pose contributed L_i / accum_eff. Batched, the group would
                    # contribute mean(L) / accum_eff -- nb times too little gradient -- and the
                    # run would train perfectly happily with a quietly smaller effective lr.
                    # Scaling by nb restores sum(L_i) / accum_eff, which is what the unbatched
                    # path accumulated. Raised in adversarial review.
                    cum = cum * float(nb)
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
            # THE PAE SPLIT IS TALLIED HERE, from the SAME number that is reported and
            # differentiated. Reading pae_loss anywhere else -- recomputing it, or taking it from
            # the batch means below -- would make the split describe a quantity the run does not
            # train on, which is the failure the pass A comment warns about. _samp_i is passed in
            # because the PAE mask has already resolved this pose's sample index under rel; the
            # tally re-reads it only when nothing else has.
            if _ps_on:
                _pae_split_tally(pdb, lab_p, lab, arm_seed, i, _samp_i, nb, out["pae_loss"])
            out["pose"] = i
            # HOW MANY POSES THIS ROW STANDS FOR. One at --pose-batch 1; nb above it. The
            # accumulator, the batch means and the step boundary all count POSES, and without
            # this a batched row would be indistinguishable from a single-pose one.
            out["n_pose_in_group"] = int(nb)
            if float(a.pae_row_frac) > 0:
                out["pae_loss_unweighted"] = _pae_unw["v"]
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
                _be0, _be1 = _gpu_span("gpu_backward", n_tok)
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
                # ONE RESOLUTION, ABOVE BOTH READERS. The first version resolved the per-pose
                # chain pair only inside the backward block, so the --grad-probe block below --
                # which has its own scoring loop -- passed the IFACE NAME straight to
                # iptm_pooled and died on `i, j = pair` with "too many values to unpack". Found by
                # actually running the probe; the static gate had passed. This file has the same
                # lesson written at line 3707 about `_cs_div`: one binding, above both readers.
                _si_b = _samp_i
                # NAMED _rel_pp, NOT _pp. The --grad-probe block below iterates
                # `for (_n, _pp), _g in zip(trainable_params, _gs)`, so a dict called _pp in this
                # scope is shadowed by a torch Parameter by the time _asym_for reads it --
                # "AttributeError: 'Parameter' object has no attribute 'get'". Found by running
                # it; the static gate could not see a name collision.
                _rel_pp = None
                if a.rank_form == "rel" and _cs_src:
                    if _si_b is None:
                        raise SystemExit(
                            f"STOP: pass B has no sample index for {pdb} pose {i}; it is resolved "
                            f"once beside the PAE mask and must be set under --rank-form rel.")
                    _rel_pp = {_ifc: (_x, _y) for _ifc, _x, _y
                               in _pose_pairs(pdb, lab_p, _si_b)}

                def _asym_for(_k):
                    """The asym pair to score for a coefficient key.

                    Under rel the key is a reference interface name and the pair is per pose;
                    under every other form the key IS the pair.
                    """
                    if _rel_pp is None:
                        return _k
                    _v = _rel_pp.get(_k)
                    if _v is None:
                        raise SystemExit(
                            f"STOP: no per-pose chain pair for interface {_k} on {pdb} pose {i}; "
                            f"its coefficient cannot be applied to the score it describes.")
                    return _v

                if (a.grad_probe or _gn_on[0]) and _cs_src:
                    _rt = None
                    for _p, _cv in _cs_src:
                        if not _cv:
                            continue
                        # SAME POOLING AS THE BACKWARD. This read RKL_probe.iptm, which is
                        # MAX-pooled, while training's backward uses iptm_pooled(pool=rank_pool).
                        # So the first rel probe's headline numbers -- parity alpha 121.7 and
                        # cos(PAE,rel) +0.2043 -- described the max-pooled score and NOT the
                        # objective being trained. Caught in external review; the parity must be
                        # re-measured.
                        _sp_ = RKL_probe.iptm_pooled(pred["pae"], feat_head, configs,
                                                     _asym_for(_p), pool=a.rank_pool)
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
                        _sr = RKL.iptm_pooled(pred["pae"], feat_head, configs,
                                              _asym_for(_p), pool=a.rank_pool)
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
        # THE WITHIN-SEED PER-UNIT PATH IS GONE WITH THE PAIRWISE FORMS.
        #
        # It read `if a.rank_loss and not a.loss_only and not a.rank_cross_seed`, built linear
        # coefficients from RKL.coeffs over the 5 poses of ONE seed, and was the fallback for
        # every form that did not need pass A. rel is the only form left and it refuses to run
        # without --rank-cross-seed, so this branch was unreachable; keeping it would leave a
        # second, differently-shaped ranking objective in the file for no caller.
        #
        # THIS IS NOT THE REMOVAL OF WITHIN-SEED COMPARISONS. Under rel every pose of an
        # interface is in one group across all its seeds, so poses sharing a seed are compared
        # with each other there -- see the banner block beside the RANKING LOSS print.


        # HOW MANY POSES PER FORWARD, decided once per unit and refused wherever it would change
        # the objective rather than just the speed.
        #   score_only  -- pass A keys ipTM per pose and returns one row each; batching it would
        #                  need the per-pose unpacking that pass B has and pass A does not.
        #   rank_loss   -- a pose's coefficient is resolved per pose, and under --rank-form rel the
        #                  PAE mask is a PER-POSE chain pair (_samp_i). Batched, one mask would
        #                  stand for five poses that may not share it.
        #   nsample mismatch -- a short final group is fine, but only whole poses of ONE unit may
        #                  share a forward; the trunk is the thing being reused.
        _pb = max(1, int(getattr(a, "pose_batch", 1) or 1))
        if score_only or a.rank_loss:
            _pb = 1
        # NOT ON A LARGE CROP. Activation memory goes as nb x tokens^2 (measured
        # peak_GB = 2.02 + 2.896e-05 * tokens^2 per sample), so 5 poses at 768 tokens is ~87 GB of
        # a 98 GB card -- no margin for fragmentation. At 256 it is ~11 GB. The batching is also
        # worth least exactly there: the fixed per-pose cost is 72% of a 256-token pose and 17% of
        # a 768-token one. 818 of this pool's 846 structures crop to 256, so the threshold costs
        # almost nothing. At the user's direction.
        if _pb > 1 and n_tok > int(getattr(a, "pose_batch_max_tokens", 384) or 384):
            _pb = 1
        for i in range(0, n_pose, _pb):
            _nb = min(_pb, n_pose - i)
            _t_pose = time.perf_counter()
            # NO GRAPH UNDER --loss-only. Skipping only the backward would still build the
            # autograd graph for every pose and free it unused -- correct, but it pays the
            # activation memory of training for a measurement that needs none.
            # RESET BEFORE EVERY MEASURED POSE. max_memory_allocated is a HIGH-WATER MARK and the
            # only reset was once per epoch, so every pose after the first large one reported that
            # large one's number: the peak-versus-tokens curve was monotone in WALK ORDER, not a
            # function of the pose at all. The epoch's own peak is kept by hand below so the epoch
            # line still reports the epoch. Only under --mem-log, so the live run is untouched.
            #
            # OUTSIDE THE PASS BRANCH, NOT INSIDE THE else. It used to sit in the else, so poses
            # taken under no_grad -- every pass A pose, and all of them under --loss-only -- were
            # never reset while the MEM line below printed for them anyway. A whole step of pass A
            # lines then carried one stale number, identical on all of them, and a reader taking
            # max(peak) over the log was mixing a no-grad forward with a pose that backwards.
            _pass = "A" if (a.loss_only or score_only) else "B"
            if a.mem_log and torch.cuda.is_available():
                _mem_ep[0] = max(_mem_ep[0], torch.cuda.max_memory_allocated(dev))
                torch.cuda.reset_peak_memory_stats(dev)
            if a.loss_only or score_only:
                with torch.no_grad():
                    m = _one_pose(i, _nb)
            else:
                m = _one_pose(i, _nb)
            # PASS A'S POSES GET THEIR OWN BUCKET. Ticking them into pose_total would double-count
            # every pose and make the epoch accounting -- which is checked against the wall clock
            # and reported as a residual -- silently wrong.
            _tick("pose_a_total" if score_only else "pose_total", _t_pose)
            if a.mem_log:
                if torch.cuda.is_available():
                    _mem_ep[0] = max(_mem_ep[0], torch.cuda.max_memory_allocated(dev))
                # AT A CONSISTENT BOUNDARY: after backward, and after the pose's tensors are gone.
                # allocated = live tensors. reserved = what the allocator holds, which is what
                # nvidia-smi shows -- which is why a high nvidia-smi number proves nothing by itself.
                # THE PASS IS PART OF THE MEASUREMENT. A pass A pose runs under no_grad and
                # stores no activations, so its peak is not comparable to a pass B pose's and is
                # not what any checkpointing threshold is decided from.
                print(f"MEM {pdb}/{arm_seed} pose {i}{'' if _nb == 1 else f'-{i + _nb - 1}'} "
                      f"pass {_pass} tokens {n_tok}: "
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
        if not (a.rank_cross_seed and (a.rank_loss or a.rank_diag_only)) or a.loss_only:
            return
        import rank_loss as RKL
        _rng_cpu = torch.get_rng_state()
        _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _t_a = time.perf_counter()
        local = {}
        # pose -> TRUE sample index, carried separately so _xs_coef can stay keyed by the POSE
        # index pass B looks up while the LABEL lookup uses the sample index.
        local_samp = {}
        try:
            for _pdb, _lab, _cands in step_units:
                _tp = resolve_trunk(_pdb, _lab, _cands)
                _sd = _rank_key(_pdb, os.path.basename(_lab)[:-4], (0, 0))[1]
                for _m in run_unit(_pdb, _lab, _tp, score_only=True):
                    local_samp[(_pdb, _sd, int(_m["pose"]))] = int(_m["sample"])
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
        if a.rank_form == "rel":
            # EXPECTED KEYS ARE (pdb, seed, pose, IFACE) for rel, because that is what pass A
            # emits. Built from the pool's interface names, which every pose must score.
            _exp = {(_pdb, _rank_key(_pdb, os.path.basename(_lab)[:-4], (0, 0))[1], _i, _ifc)
                    for _pdb, _lab, _cands in step_units
                    for _ifc in (_rel_ifaces.get(_pdb) or ())
                    for _i in range(a.nsample)}
        else:
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
            # THE PERMUTATION IS GATHERED TOO. Without it a rank would hold scores for another
            # rank's poses and no way to key their labels, and the group would silently shrink.
            _bag_s = [None] * world
            dist.all_gather_object(_bag_s, local_samp)
            _tick("rank_gather", _t_g)
            allsc, allsamp = {}, {}
            for _d in _bag:
                allsc.update(_d or {})
            for _d in _bag_s:
                allsamp.update(_d or {})
        else:
            allsc = dict(local)
            allsamp = dict(local_samp)

        # GROUPED BY (pdb, interface), which is what an interface IS. Keyed by (seed, sample)
        # inside the group so group_coeffs can tell cross-seed pairs from within-seed ones.
        groups = {}
        for (_pdb, _sd, _i, _pair), _v in allsc.items():
            groups.setdefault((_pdb, _pair), {})[(_sd, _i)] = _v
        n_groups = 0
        for (_pdb, _pair), s_by in groups.items():
            dq_by = {}
            if a.rank_form == "rel":
                # THE LABEL IS THE TRUE iTM, NOT DockQ. Keyed (seed, sample) to match s_by.
                _gl = _rel_labels(_pdb)
                for (_sd, _i) in s_by:
                    # SAMPLE INDEX FOR THE LABEL, POSE INDEX FOR THE KEY. The cache is keyed by
                    # the sample; s_by and _xs_coef are keyed by the pose. Using _i on both sides
                    # trained pose 1 against pose 4's label on 99.3% of units.
                    _sx = allsamp.get((_pdb, _sd, int(_i)))
                    if _sx is None:
                        raise SystemExit(
                            f"STOP: no sample_index recorded for {_pdb}/{_sd} pose {_i}, so its "
                            f"true-iTM label cannot be keyed. Pass A must emit one per pose.")
                    # _pair IS THE IFACE NAME under rel, not an asym tuple -- see pass A.
                    _g = _gl.get((int(_sd), int(_sx), str(_pair)))
                    if _g is not None:
                        dq_by[(_sd, _i)] = float(_g)
            else:
                for (_sd, _i) in s_by:
                    _lab = _rank_lab.get(_rank_key(_pdb, _sd, _pair))
                    if _lab is not None and _i in _lab:
                        dq_by[(_sd, _i)] = float(_lab[_i])
            if a.rank_form == "rel" and len(dq_by) != len(s_by):
                # EXACT COVERAGE REQUIRED. The completeness guard elsewhere validates DockQ
                # labels, not this cache, so a partially covered group would quietly train on
                # fewer poses than it scored -- and a group centred on a SUBSET has a different
                # mean, so the surviving poses' targets would all be wrong, not merely fewer.
                raise SystemExit(
                    f"STOP: {_pdb} interface {_pair} scored {len(s_by)} pose(s) but only "
                    f"{len(dq_by)} have a true-iTM label. Centring on a subset changes the mean "
                    f"for every surviving pose, so this is not a partial-data case. Rebuild the "
                    f"cache with run_logging/build_itm_labels.py and check its manifest covers "
                    f"this manifest-selection.")
            if len(dq_by) < 2:
                _rel_stat["short_groups"] += 1
                continue
            # THE FORM IS PASSED THROUGH, NOT RE-DERIVED. This read
            #     form=("softplus" if a.rank_form == "softplus" else "linear")
            # which sent any third form to the LINEAR loss while every log line and the saved
            # guard still said otherwise -- a run that looks right and is the wrong experiment.
            # rank_loss.group_coeffs raises on an unknown form, so a typo fails loudly instead.
            _acc_by = None
            if a.rank_form == "rel" and a.rel_balance_accept:
                # DockQ PER POSE, keyed (seed, pose) exactly like s_by. _rank_lab is keyed by the
                # STATIC asym pair (_rel_dqkey) and indexed by the SAMPLE, while s_by is keyed by
                # the pose -- the same two-step lookup the Spearman diagnostic does below, which
                # is why this is hoisted rather than duplicated.
                _acc_by = {}
                _dk0 = _rel_dqkey.get(_pair)
                for (_sd0, _i0) in s_by:
                    _l0 = _rank_lab.get(_rank_key(_pdb, _sd0, _dk0)) if _dk0 is not None else None
                    _sx0 = allsamp.get((_pdb, _sd0, int(_i0)))
                    if _l0 is None or _sx0 is None or _sx0 not in _l0:
                        # EXACT COVERAGE, FOR THE SAME REASON THE iTM CACHE DEMANDS IT: a pose
                        # whose DockQ is unknown cannot be put in either class, and guessing it
                        # into the majority class would quietly change every weight in the list.
                        raise SystemExit(
                            f"STOP: --rel-balance-accept needs a DockQ for every pose, but "
                            f"{_pdb} interface {_pair} pose {_i0} (seed {_sd0}) has none. The "
                            f"classification sets every weight in the list, so a missing label "
                            f"is not a partial-data case. Check --rank-labels covers this "
                            f"manifest-selection.")
                    _acc_by[(_sd0, _i0)] = float(_l0[_sx0]) >= a.rel_balance_accept
                _na0 = sum(1 for _v0 in _acc_by.values() if _v0)
                if _na0 == 0 or _na0 == len(_acc_by):
                    # THE FALLBACK IS A SYMPTOM, NOT A FEATURE. rel_coeffs falls back to a
                    # uniform mean when a list has only one class, silently and correctly -- 0.5/0
                    # is not a weight. But on a pool selected to have BOTH classes in every list,
                    # a one-sided list cannot be real: it means the DockQ lookup returned the
                    # wrong rows, or the threshold is not the one the pool was built with. Counted
                    # and named per epoch so that shows up instead of hiding inside a loss value.
                    _rel_stat["bal_onesided"] = _rel_stat.get("bal_onesided", 0) + 1
                    _rel_stat.setdefault("bal_onesided_ifaces", []).append(
                        f"{_pdb}_{_pair}({'all-acc' if _na0 else 'none-acc'},n={len(_acc_by)})")
                _rel_stat["bal_lists"] = _rel_stat.get("bal_lists", 0) + 1
                _rel_stat.setdefault("bal_nacc", []).append(_na0)
            if a.rank_form == "rel":
                # NOT group_coeffs. rel is a regression, not a pairwise family, so none of the
                # pairwise knobs apply and passing them would make flags look honoured.
                c, npair, lval = RKL.rel_coeffs(s_by, dq_by,
                                                standardize=a.rel_standardize,
                                                sd_weight=a.rel_sd_weight,
                                                g_weight=a.rel_g_weight,
                                                g_sigmoid=_G_SIG,
                                                accept_by=_acc_by,
                                                **({} if a.rel_std_floor is None
                                                   else {"std_floor": a.rel_std_floor}))
                _rel_stat["groups"] += 1
                _rel_stat["poses"] += npair
                # DIAGNOSTIC 1: WITHIN-INTERFACE SPEARMAN of the predicted mean-pooled ipTM
                # against DockQ -- the arm's actual prediction. The head measured +0.317 at
                # d32kqp epoch 66 against +0.690 for the TRUE iTM, and this is the quantity that
                # should move between them. It is far more sensitive than selected DockQ (t=7.6
                # against t=2.7) because it uses all 50 poses rather than just the argmax.
                #
                # DIAGNOSTIC 2: SCORE SPREAD, std of the predicted ipTM over the interface's
                # poses, against std of the TRUE iTM on the same poses. Below 1 the model is
                # collapsing its scores, above 1 it is exaggerating pose-to-pose differences, and
                # ~1 means it has the magnitude right. Every ranking arm in this repo lowered its
                # loss by margin collapse (x5frz6: within-seed std 0.00593 -> 0.00147 while the
                # loss fell), so this is the falsifier for that failure mode in one number.
                _ks = sorted(s_by)
                _sv = [float(s_by[_k2]) for _k2 in _ks]
                _gv = [float(dq_by[_k2]) for _k2 in _ks if _k2 in dq_by]
                _dk = _rel_dqkey.get(_pair)
                _dv = []
                if _dk is not None:
                    for (_sd2, _i2) in _ks:
                        _lab2 = _rank_lab.get(_rank_key(_pdb, _sd2, _dk))
                        _sx2 = allsamp.get((_pdb, _sd2, int(_i2)))
                        if _lab2 is not None and _sx2 is not None and _sx2 in _lab2:
                            _dv.append(float(_lab2[_sx2]))
                if len(_dv) == len(_sv) and len(_sv) >= 3:
                    _rho = _spearman(_sv, _dv)
                    if _rho is not None:
                        _rel_stat.setdefault("rho", []).append(_rho)
                    # DIAGNOSTIC 3: THE SELECTION METRIC ITSELF, in sample and for free.
                    # _sv is this epoch's predicted ipTM for the interface's poses and _dv their
                    # DockQ, already aligned -- so the pose the head would pick is _sv's argmax
                    # and its DockQ is what top-1 selection would score. rho says whether the
                    # ordering is right; this says whether being right there is worth anything,
                    # which is the quantity the held-out eval reports and the one we are paid on.
                    #
                    # IT IS NOT THE HELD-OUT NUMBER AND MUST NOT BE READ AS ONE. These are
                    # training interfaces, scored on the CROPPED poses this epoch happened to
                    # forward, under whatever dropout was active. It moves with the held-out
                    # metric only to the extent the arm generalises -- which is exactly the gap
                    # worth watching, and why the oracle goes out beside it as the ceiling.
                    _am = max(range(len(_sv)), key=lambda _j: _sv[_j])
                    _rel_stat.setdefault("sel_dq", []).append(_dv[_am])
                    _rel_stat.setdefault("orc_dq", []).append(max(_dv))
                    # DIAGNOSTIC 4: THE THREE READS THAT MOVE BEFORE TOP-1 DOES.
                    #
                    # RANK OF THE FIRST ACCEPTABLE POSE. Top-1 is a step function -- it only
                    # changes when the argmax crosses the threshold -- so an arm can improve for
                    # several epochs with top-1 flat. The rank of the first acceptable pose is
                    # the same ordering read continuously: 1 means top-1 succeeded, 40 of 75
                    # means the acceptable poses are sitting in the middle of the list.
                    #
                    # SCORE GAP. Mean predicted score of the acceptable poses minus the mean of
                    # the unacceptable ones, per list. It is the quantity the balanced loss acts
                    # on directly, and it is signed: at or below 0 the head prefers the bad poses.
                    #
                    # SELECTED DockQ AS A FRACTION OF HEADROOM, with the floor being the mean
                    # DockQ over ALL of the list's poses -- the expected score of a RANDOM pick --
                    # and the ceiling the best pose. 0 means "no better than picking blind", 1
                    # means the oracle. The floor is the random pick and NOT the shipped head
                    # because shipped's per-list number is not available inside a training run,
                    # and a floor that is not actually computed here would be a different
                    # quantity from the one the ceiling table reports.
                    _ACC = 0.23
                    _ord = sorted(range(len(_sv)), key=lambda _j: -_sv[_j])
                    _fr = next((_r + 1 for _r, _j in enumerate(_ord) if _dv[_j] >= _ACC), None)
                    if _fr is not None:
                        _rel_stat.setdefault("first_acc_rank", []).append(_fr)
                    _hi = [_sv[_j] for _j in range(len(_sv)) if _dv[_j] >= _ACC]
                    _lo = [_sv[_j] for _j in range(len(_sv)) if _dv[_j] < _ACC]
                    if _hi and _lo:
                        _rel_stat.setdefault("score_gap", []).append(
                            sum(_hi) / len(_hi) - sum(_lo) / len(_lo))
                    _base = sum(_dv) / len(_dv)
                    _orc = max(_dv)
                    if _orc - _base > 1e-9:
                        _rel_stat.setdefault("headroom", []).append(
                            (_dv[_am] - _base) / (_orc - _base))
                    _rel_stat.setdefault("acc_top1", []).append(1.0 if _dv[_am] >= _ACC else 0.0)
                if len(_gv) == len(_sv) and len(_sv) >= 3:
                    _ms = sum(_sv) / len(_sv)
                    _mg = sum(_gv) / len(_gv)
                    _ss = (sum((v - _ms) ** 2 for v in _sv) / len(_sv)) ** 0.5
                    _sg = (sum((v - _mg) ** 2 for v in _gv) / len(_gv)) ** 0.5
                    _rel_stat.setdefault("sd_s", []).append(_ss)
                    _rel_stat.setdefault("sd_g", []).append(_sg)
                    if _sg > 1e-9:
                        _rel_stat.setdefault("sd_ratio", []).append(_ss / _sg)
                if npair:
                    _rel_stat["csum"] = max(_rel_stat["csum"],
                                            abs(sum(c.values())))
            else:
                # UNREACHABLE, AND A STOP RATHER THAN A FALLBACK. This was the group_coeffs call
                # for the six deleted forms, with tau, weight_by_gap, pairs, within_weight,
                # rank_decay_k, straddle, spearman_tau, dist_weight and rel_scale all threaded
                # through it. --rank-form now admits only rel, so nothing reaches here; if a
                # future form is added and forgets to wire itself up, this says so instead of
                # training rel's coefficients under another form's name.
                raise SystemExit(f"STOP: --rank-form {a.rank_form!r} has no coefficient path. "
                                 f"rel is the only form; see --rank-form's help.")
            if npair == 0:
                continue
            n_groups += 1
            _xs_stat["pairs"] += npair
            _xs_stat["loss"] += lval
            for (_sd, _i), _cv in c.items():
                if not a.rank_diag_only:
                    # THE ONLY ROUTE from a coefficient to the loss: the pose loop reads
                    # _xs_coef and multiplies by alpha. Leaving it empty under --rank-diag-only
                    # means the term cannot be added by construction, rather than being
                    # multiplied by a zero someone could later change.
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
        # BOTH TERMS' COVERAGE, AND THE FAILURE IS "NEITHER", NOT "NOT THE RANKING TERM".
        #
        # The original check failed whenever a trainable tensor got no RANKING gradient, which
        # assumes the ranking term should reach everything PAE reaches. That is false for any term
        # that is a function of the PAE logits alone: on the first --rank-form rel probe it
        # reported 158 of 258 tensors with no ranking gradient and refused to launch, naming
        # confidence_head.plddt_weight, lower_bins, upper_bins and resolved -- parameters of the
        # pLDDT and resolved heads, which a PAE-derived score cannot touch and which --pae-only
        # switches off anyway.
        #
        # So the question the gate should ask is whether a tensor gets gradient from NEITHER term,
        # which means it is being optimised by nothing while counted as trainable. The per-term
        # counts are printed either way, because "158 of 258 tensors receive no PAE gradient
        # under --pae-only" is a fact about every arm in this project and should not stay hidden
        # behind a check that only looks at the other term.
        nz_r = pr["nonzero"].get("rank") or set()
        nz_p = pr["nonzero"].get("pae") or set()
        absent_r = [n_ for n_ in names if n_ not in nz_r]
        absent_p = [n_ for n_ in names if n_ not in nz_p]
        dead = [n_ for n_ in names if n_ not in nz_r and n_ not in nz_p]
        _ov = _pae_keep.get("ov")
        if _ov:
            print(f"\n  PAE MASK OVERRIDE (--rank-form rel): applied on {_ov['poses']} pose(s), "
                  f"DIFFERED from the static chain_roles pairs on {_ov['differed']} "
                  f"({100 * _ov['differed'] / max(1, _ov['poses']):.1f}%)"
                  + (f" across {len(_ov['pdbs'])} structure(s), e.g. "
                     f"{sorted(_ov['pdbs'])[:4]}" if _ov["pdbs"] else
                     " -- NO structure in this window disagrees, so the fix is UNEXERCISED here "
                     "and the numbers are EXPECTED to match the pre-fix run"))
        elif a.rank_form == "rel":
            print("\n  PAE MASK OVERRIDE: NOT APPLIED on any pose, which under --rank-form rel "
                  "means the per-pose pairs never reached _set_pae_mask -- the fix is not live.")
        print(f"\n  GRADIENT COVERAGE over {len(names)} trainable tensor(s): "
              f"PAE reaches {len(names) - len(absent_p)}, the {a.rank_form} term reaches "
              f"{len(names) - len(absent_r)}, NEITHER reaches {len(dead)}")
        if absent_r and not dead:
            print(f"    ({len(absent_r)} get no {a.rank_form} gradient but DO get PAE -- expected "
                  f"for a term that is a function of the PAE logits alone, e.g. "
                  f"{absent_r[:3]})")
        if dead:
            # EXPECTED UNDER --pae-only, AND SAID OUT LOUD RATHER THAN SWALLOWED.
            #
            # Measured on the first rel probe: of 258 tensors marked trainable, PAE reaches 100,
            # the rel term reaches 100, and NEITHER reaches 158. The 158 are pLDDT, PDE and
            # resolved head parameters, whose losses --pae-only switches off, so they sit at their
            # pretrained values for the whole run. That is true of EVERY --pae-only arm in this
            # project -- nskyne, d32kqp and the 45q96x lineage all print "258 trainable tensor(s),
            # 19.8M of 464.5M parameters (4.3%)" while only 100 tensors are optimised, and AdamW
            # carries moment state for the other 158 regardless.
            #
            # So it is not a reason to refuse a --pae-only launch, but it IS a reason the
            # trainable-count line should not be read as the effective capacity. Without
            # --pae-only a dead tensor has no such explanation and still fails.
            msg = (f"{len(dead)} of {len(names)} trainable tensor(s) get NO gradient from either "
                   f"term, so they stay at their pretrained values: {dead[:6]}"
                   + (f" (+{len(dead)-6} more)" if len(dead) > 6 else ""))
            if a.pae_only:
                print(f"    NOTE, expected under --pae-only: {msg}")
                print(f"    EFFECTIVE trainable set is {len(names) - len(dead)} tensor(s), not "
                      f"{len(names)}. The startup trainable line counts what is MARKED "
                      f"trainable, not what receives gradient.")
            else:
                fail.append(msg)
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
        # REPORTS WHAT WAS ACTUALLY COVERED, not "all of them". The previous wording claimed
        # every trainable tensor got a ranking gradient, which directly contradicted the coverage
        # line printed two lines above it once that line existed: on a --pae-only rel probe, 100
        # of 258 tensors are reached and the old string still said 258. A success message that
        # restates the check's premise rather than its result is how a known limitation gets read
        # as a guarantee.
        print(f"  PASS: the {len(names) - len(absent_r)} tensor(s) the {a.rank_form} term can "
              f"reach all received a finite, non-zero gradient, the PAE gradient is finite and "
              f"non-zero, the cosine is finite, and an optimiser step completed."
              + (f" {len(dead)} tensor(s) are reached by neither term (expected under "
                 f"--pae-only)." if dead else ""))
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
                # THE TASK GRADIENT NORM IS MEASURED FIRST, BEFORE THE PENALTY IS ADDED, and this
                # ordering is the whole point. paper_results.md section 17: reporting the penalty
                # as a share of the LOSS VALUE is the trap that made 1e-5 look like a meaningful
                # setting -- it was 0.5% of the objective and 0.08% of the GRADIENT, i.e. off.
                # The loss value does not act on the weights; the gradient does. This is the only
                # point in the step where the task gradient still exists uncontaminated: after
                # the all-reduce, before the L2 term, before the clip. `gn` from clip_grad_norm_
                # below is the TOTAL after the penalty and cannot be decomposed back.
                _tsq = 0.0
                for _p_t in params:
                    if _p_t.grad is not None:
                        _tsq += float(_p_t.grad.detach().pow(2).sum())
                _sq = 0.0
                for _n_sp, _p_sp in _l2sp_params:
                    _d_sp = _p_sp.detach() - _l2sp_anchor[_n_sp]
                    # .grad CAN BE None HERE and the penalty must still apply. zero_grad is
                    # set_to_none=True, and under --pae-only some anchored tensors are reached by
                    # neither loss term -- the --grad-probe PASS message says so explicitly. The
                    # world>1 path happens to zero-fill before its collective, which masked this;
                    # a single-GPU --l2-sp run would have hit
                    # AttributeError: 'NoneType' object has no attribute 'add_'.
                    # L2-SP prices distance from the anchor whether or not the task touched the
                    # tensor, so the right fix is to CREATE the gradient, not to skip the tensor.
                    if _p_sp.grad is None:
                        _p_sp.grad = torch.zeros_like(_p_sp)
                    _p_sp.grad.add_(_d_sp, alpha=2.0 * a.l2_sp)
                    _sq += float((_d_sp * _d_sp).sum())
                _l2sp_drift["sq"] = _sq
                _l2sp_drift["n"] += 1
                # PER-STEP SHARES, NOT A RUNNING MEAN, and NON-FINITE STEPS ARE EXCLUDED.
                #
                # Excluded because clip_grad_norm_(error_if_nonfinite=True) below DROPS the
                # gradients of a non-finite step and takes no step, so its share describes nothing
                # -- and nan is absorbing, so folding one in would have destroyed the whole
                # epoch's number with no signal but the NON-FINITE GRADIENT line far above.
                # --max-nonfinite defaults to 10, so up to ten such steps are tolerated per run.
                #
                # Per-step rather than a mean because the mean answers the wrong question. The
                # danger case for an anchor is the step where the TASK gradient nearly vanishes:
                # this head has had 16 of 69 taken steps under grad_norm 1e-3, the late ones
                # clustered near 2.5e-4, and on those the penalty is in sole charge of the
                # update. A ratio-of-means is dominated by the big-gradient steps and shows a few
                # percent while that is happening. The median, the max and a count of steps where
                # the penalty EXCEEDS the task gradient are what show it.
                _tgn_step = _tsq ** 0.5
                if _tgn_step == _tgn_step and _tgn_step > 0.0:
                    _l2sp_drift.setdefault("shares", []).append(
                        (2.0 * a.l2_sp * _sq ** 0.5) / _tgn_step)
                    _l2sp_drift.setdefault("tgns", []).append(_tgn_step)
                # ||2*lambda*(W-W0)|| = 2*lambda*||W-W0|| exactly, because the penalty gradient is
                # a scalar multiple of the displacement over THIS SAME parameter set (_l2sp_params,
                # which is the anchored 92 and not the full 102 -- the 10 zero-init tensors are
                # excluded from the anchor, so they contribute to _tsq but not to _sq, which is
                # correct: they receive no penalty gradient).
                _l2sp_drift["task_gn_sum"] = _l2sp_drift.get("task_gn_sum", 0.0) + _tsq ** 0.5
                _l2sp_drift["l2_gn_sum"] = (_l2sp_drift.get("l2_gn_sum", 0.0)
                                            + 2.0 * a.l2_sp * _sq ** 0.5)
                _l2sp_drift["share_n"] = _l2sp_drift.get("share_n", 0) + 1
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
        # THE DEAD-OP RE-CHECK, HERE AND NOT AT THE CROSS-SEED STEP BOUNDARY. The obvious home for
        # it was beside PFC.step_reset() in the unit loop -- but that site is inside
        # `if ui in _xs_at:`, and _xs_at is only populated under --rank-cross-seed, so on any other
        # arm the check would have run ZERO times for the whole run while the install line claimed
        # "re-checked at every step boundary". The prefix cache survives that because it has a
        # second, unconditional call at the epoch boundary; this has no equivalent. Immediately
        # after opt.step() is also the strictly better moment: an optimiser step is the only thing
        # that could have written a stubbed tensor, and this is the first instruction after it.
        if _dops is not None:
            DOPS.step_check()
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
               # --crop-margin 0 IS WHAT "UNCROPPED" MEANS here; the argument's own help says 0
               # is the whole structure. crop_contact and crop_size are still passed because they
               # are inert at margin 0 and passing them keeps the recorded config complete.
               "--crop-contact", str(a.crop_contact),
               "--crop-size", str(a.crop_size), "--accum", str(a.accum),
               # THE PATH IS PER SHARD, and the sharded launcher below rewrites it the same way
               # it rewrites --out. Handing every child this one path meant three processes
               # opening it with "w" and truncating each other: rows overwritten, interleaved or
               # lost, and the existing coverage checks look at trajectory.csv, not at this file,
               # so the corruption would not have been detected. Found in review of 511519e7.
               *(["--emit-iptm-rows", os.path.join(out, "iptm_rows.csv")]
                 if a.val_loss_emit_iptm else []),
               # EIGHTH INSTANCE OF THE --loss-only INHERITANCE CLASS. The three crop args above
               # are passed because "a different crop would make the held-out number incomparable
               # to the training curve it is read against"; --crop-all-poses changes the crop by
               # the same argument and was not passed, so a parent using the union would have been
               # monitored on a pose-0 crop. Caught in review.
               *(["--crop-all-poses"]
                 if (a.crop_all_poses and not a.val_loss_uncropped) else []),
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
        # THE IN-BLOCK PATHWAY AND THE LEAN TRIO -- the NINTH AND TENTH instances of the
        # --loss-only inheritance class, both found by review before an arm was launched.
        # (Ninth and tenth, not eighth and ninth: :6165 already claims the eighth for
        # --crop-all-poses. The running count in these comments is off by one further up --
        # :6177 says "sixth" and :6200 "now seven instances" for what are the seventh and
        # eighth -- so the ordinals are a narrative, not an index. Worth one more than nothing
        # because the POINT of the running count is that this keeps happening.)
        #
        # --pose-inblock ADDS tensors, so a child without it fails the resume guard's parameter-set
        # comparison (phase_k_train.py:3373-3381) and the held-out point is simply absent.
        #
        # --lean-single is worse, because it would NOT fail loudly here. It changes no mathematics
        # on the PAE path, so the child would compute the right number -- but the guard compares
        # it, so the child refuses the bundle and the monitor reports nothing. Forwarded with the
        # two flags it hard-requires (phase_k_train.py:1788), since the child parses them itself.
        if a.pose_inblock:
            cmd += ["--pose-inblock", "--pose-inblock-block", str(a.pose_inblock_block)]
            if a.pose_inblock_pae:
                cmd.append("--pose-inblock-pae")
        # FORWARDED WHENEVER THE PARENT HAS IT, not only alongside --lean-single. lean_readouts
        # replaces memory_efficient_forward rather than adding or removing tensors, so its absence
        # in the child was invisible to both the key-set guard and the resume guard -- the child
        # was simply computing readouts the parent does not, slower and on a different code path.
        if a.lean_readouts:
            cmd.append("--lean-readouts")
        if a.lean_single:
            if not a.lean_readouts or not a.pae_only:
                raise SystemExit(
                    "STOP: --lean-single without --lean-readouts and --pae-only reached "
                    "val_loss_checkpoint. The parse-time check should have refused this run.")
            cmd.append("--lean-single")
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
            os.makedirs(os.path.join(a.out, f"val_loss_ep{epoch}"), exist_ok=True)

            def _spawn(_i):
                _env = dict(env)
                _env["CUDA_VISIBLE_DEVICES"] = _cards[_i]
                _o = os.path.join(a.out, f"val_loss_ep{epoch}", f"shard{_i}")
                os.makedirs(_o, exist_ok=True)
                _lg = os.path.join(_o, "shard.log")
                # ROTATE EVERY OUTPUT OF A PREVIOUS ATTEMPT, NOT JUST THE LOG.
                # _flush() opens trajectory.csv with "a" and writes the header only when the file
                # is absent, so a shard that crashed AFTER emitting rows would have the retry
                # append a second copy of the poses it already wrote. The pooling then sees a
                # duplicate pose key and voids the epoch anyway -- meaning the retry would have
                # fixed CUDA-init failures (which happen before any row exists) and silently
                # failed to fix every later one. Caught in review; the five-case test does not
                # touch a real filesystem and could not have seen it.
                # They are rotated, not deleted: a partial trajectory is the evidence of where
                # the attempt died, and "never delete" applies to measurements too.
                _n = 0
                while os.path.exists(os.path.join(_o, f"shard.log.failed{_n or ''}")):
                    _n += 1
                _sfx = f".failed{_n or ''}"
                for _old in ([os.path.join(_o, _f) for _f in
                              ("shard.log", "trajectory.csv", "iptm_rows.csv")]
                             + sorted(glob.glob(os.path.join(_o, "crop_sizes_rank*.csv")))):
                    if os.path.exists(_old):
                        os.rename(_old, _old + _sfx)
                _cmd = [x for x in cmd]
                _cmd[_cmd.index("--out") + 1] = _o
                # AND THE EMIT PATH, for the same reason --out is rewritten: the emitter opens it
                # with "w", so a shared path means three children truncating one file.
                if "--emit-iptm-rows" in _cmd:
                    _cmd[_cmd.index("--emit-iptm-rows") + 1] = \
                        os.path.join(_o, "iptm_rows.csv")
                _cmd += ["--manifest-shard", str(_i), "--manifest-nshards", str(_nsh)]
                _fh = open(_lg, "w")
                # IF Popen RAISES, CLOSE THE HANDLE. Otherwise the fd leaks and, worse, the
                # exception escapes heldout_pae() into the training loop -- a measurement is not
                # allowed to end a run. Caught in review.
                try:
                    _p = subprocess.Popen(_cmd, stdout=_fh, stderr=subprocess.STDOUT,
                                          text=True, env=_env)
                except Exception:                 # noqa: BLE001
                    try:
                        _fh.close()
                    except Exception:             # noqa: BLE001
                        pass
                    raise
                return (_i, _o, _lg, _fh, _p)

            def _await(_ps):
                _failed = []
                for _i, _o, _lg, _fh, _p in _ps:
                    _p.wait()
                    try:
                        _fh.close()
                    except Exception:             # noqa: BLE001
                        pass
                    if _p.returncode != 0:
                        try:
                            _tail = open(_lg).read().strip().splitlines()[-6:]
                        except OSError:
                            _tail = [f"(no readable log at {_lg})"]
                        _failed.append((_i, _tail))
                return _failed

            # A 2 s STAGGER, AND IT IS MEASURED TO BE NEARLY POINTLESS -- kept only because it
            # costs 4 seconds and cannot hurt. Say so plainly rather than let it read as a fix.
            # On tv6hue epoch 4 two of three children died inside torch._C._cuda_init() -- "CUDA
            # driver initialization failed" and "No CUDA GPUs are available" -- while the third
            # completed normally. Not OOM (that says "CUDA out of memory") and not compute-mode
            # exclusivity (all three cards report Default).
            # WHY THE STAGGER DOES ALMOST NOTHING, measured on a real child: the model build,
            # which is where CUDA is first touched, takes 67.6 s. A child reaches context
            # creation about a minute after it is launched, so a 2 s offset at launch is well
            # inside the jitter and all three can still initialise at the same instant. Raised in
            # review and confirmed by instrumenting the child.
            # The RETRY is what the guarantee rests on, and on the retry path the children are
            # run strictly ONE AT A TIME, which is the only thing that actually removes the
            # concurrency. The first attempt stays parallel because that is the 30 min/epoch path.
            # THE ROOT CAUSE IS STILL NOT ESTABLISHED. A probe reproducing the suspected
            # conditions -- inherited --membind=0 against node-1 GPUs, three contexts created at
            # once -- initialised 36 times out of 36 on this box, so neither NUMA inheritance nor
            # simple concurrency is sufficient on its own. The child now records its affinity,
            # memory policy, RSS, torch version and every card's free memory, so the next
            # occurrence produces evidence instead of another guess.
            _handles = {}
            for _i in range(_nsh):
                if _i:
                    time.sleep(2.0)
                _handles[_i] = _spawn(_i)
            _bad = _await(list(_handles.values()))

            # RETRY THE FAILED SHARDS BEFORE VOIDING THE EPOCH. Losing the whole epoch's held-out
            # point to a transient that cost one child 40 seconds is a bad trade: the measurement
            # is 2.1 min of a 30 min cycle, it reads a bundle already written to disk, and the
            # weights cannot move while it runs, so re-running a failed shard measures exactly the
            # same model as its siblings. Before this, ANY shard failure voided the number, and
            # tv6hue epoch 4 is the first time in the project that actually happened.
            # ONLY THE FAILED INDICES ARE RE-RUN. Re-running a shard that already succeeded would
            # throw away good work and give the pooling a second chance to see a duplicate.
            for _try in range(1, int(getattr(a, "val_loss_shard_retries", 2) or 0) + 1):
                if not _bad:
                    break
                _idx = [_i for _i, _ in _bad]
                print(f"  held-out shard(s) {_idx} of epoch {epoch} failed; retry {_try} of "
                      f"{int(getattr(a, 'val_loss_shard_retries', 2) or 0)}", flush=True)
                for _i, _tail in _bad:
                    print(f"    shard {_i} said:\n      " + "\n      ".join(_tail), flush=True)
                time.sleep(15.0 * _try)
                # ONE AT A TIME ON A RETRY, NOT TWO SECONDS APART. A 2 s launch gap does not put
                # 2 s between two CUDA context creations: each child imports torch and builds the
                # model first, and those take long enough to swallow the offset entirely -- which
                # is exactly how three children launched in a loop ended up initialising at the
                # same instant in the first place. Waiting for each retry to EXIT before starting
                # the next is the only way to guarantee no concurrency, and on the retry path the
                # cost is real but small: failures are rare, a shard is ~2.1 min, and 4 extra
                # minutes beats losing the epoch's held-out point. The FIRST attempt stays
                # parallel, because that is the 30 min/epoch path and it almost always works.
                # Raised in review.
                _still = []
                for _i in _idx:
                    _handles[_i] = _spawn(_i)
                    _still += _await([_handles[_i]])
                _bad = _still
            procs = [_handles[_i] for _i in range(_nsh)]
            if _bad:
                # ANY shard failing voids the epoch's number. Pooling the survivors would report a
                # mean over a BIASED subset of the held-out set -- a different population than
                # every other epoch -- which is worse than no point at all, because it is silently
                # comparable-looking. The next epoch tries again and the gap shows in val_loss.csv.
                for _i, _tail in _bad:
                    print(f"  WARNING: held-out shard {_i} of epoch {epoch} failed:\n    "
                          + "\n    ".join(_tail), flush=True)
                print(f"  WARNING: epoch {epoch} has no held-out point; {len(_bad)} of {_nsh} "
                      f"shard(s) failed after "
                      f"{int(getattr(a, 'val_loss_shard_retries', 2) or 0)} retry/retries and a "
                      f"partial pool would be a different population.", flush=True)
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
            # MERGE THE PER-SHARD ipTM FILES, with the same disjointness standard the pose
            # population above is held to. Each shard now writes its own iptm_rows.csv (they used
            # to share one path and truncate each other), so the merged file is the only thing
            # dockq_from_iptm.py can read. A duplicate or an unreadable shard file leaves NO
            # merged file rather than a partial one: dockq_from_iptm refuses duplicates anyway,
            # and a silently short file is exactly the "metric over whatever showed up" it warns
            # against.
            if a.val_loss_emit_iptm:
                _merged = os.path.join(a.out, f"val_loss_ep{epoch}", "iptm_rows.csv")
                _rows_i, _keys_i, _bad_i = [], set(), []
                for _i, _o, _lg, _fh, _p in procs:
                    _f = os.path.join(_o, "iptm_rows.csv")
                    try:
                        with open(_f) as fh:
                            _rd = csv.DictReader(fh)
                            for _r in _rd:
                                _k = (_r["iface"], _r["seed"], _r["sample"])
                                if _k in _keys_i:
                                    _bad_i.append(f"duplicate {_k} in shard {_i}")
                                    break
                                _keys_i.add(_k)
                                _rows_i.append(_r)
                    except OSError as _e:
                        _bad_i.append(f"shard {_i} unreadable ({_e})")
                # COMPLETENESS, NOT JUST UNIQUENESS. Rejecting duplicates says no row appears
                # twice; it says nothing about a row that never appeared. RKL.iptm can return
                # None for a scored interface and the emitter skips that row silently, the child
                # still exits 0, and the merged file comes out SHORT -- which the trajectory
                # coverage check above cannot see, because it counts poses and these are
                # per-interface rows. So the merged key set is checked against the population the
                # pool and the seed range define: every interface in --val-loss-pool must carry
                # exactly (seeds x nsample) rows. Caught in review of 8bcf8202; it is not a
                # blocker while the separate eval still runs, and it is one the moment it does
                # not.
                if not _bad_i:
                    try:
                        _pool_if = set()
                        with open(a.val_loss_pool) as fh:
                            for _r in csv.DictReader(fh):
                                _pool_if.add(f"{_r['pdb']}_{_r['chain1']}{_r['chain2']}")
                        _lo_s, _, _hi_s = a.val_loss_seeds.partition("-")
                        _nseed = (int(_hi_s) - int(_lo_s) + 1) if _hi_s else 1
                        _exp_depth = _nseed * int(a.nsample)
                        _depth = {}
                        for _r in _rows_i:
                            _depth[_r["iface"]] = _depth.get(_r["iface"], 0) + 1
                        _miss_if = sorted(_pool_if - set(_depth))
                        _extra_if = sorted(set(_depth) - _pool_if)
                        _short = sorted(k for k, v in _depth.items() if v != _exp_depth)
                        if _miss_if:
                            _bad_i.append(f"{len(_miss_if)} pool interface(s) emitted no rows, "
                                          f"e.g. {_miss_if[:3]}")
                        if _extra_if:
                            _bad_i.append(f"{len(_extra_if)} emitted interface(s) are not in the "
                                          f"pool, e.g. {_extra_if[:3]}")
                        if _short:
                            _bad_i.append(f"{len(_short)} interface(s) do not have the expected "
                                          f"{_exp_depth} row(s) ({_nseed} seed(s) x "
                                          f"{a.nsample} sample(s)), e.g. "
                                          + ", ".join(f"{k}={_depth[k]}" for k in _short[:3]))
                    except (OSError, KeyError, ValueError) as _e:
                        _bad_i.append(f"could not establish the expected ipTM population ({_e})")
                if _bad_i:
                    print(f"  WARNING: epoch {epoch} ipTM rows NOT merged: {_bad_i[0]}"
                          + (f" (+{len(_bad_i)-1} more)" if len(_bad_i) > 1 else "")
                          + f"; {_merged} not written", flush=True)
                else:
                    with open(_merged, "w", newline="") as fh:
                        _w = csv.writer(fh)
                        _w.writerow(["iface", "seed", "sample", "iptm"])
                        for _r in _rows_i:
                            _w.writerow([_r["iface"], _r["seed"], _r["sample"], _r["iptm"]])
                    print(f"  merged {len(_rows_i)} ipTM row(s) from {_nsh} shard(s) "
                          f"-> {_merged}", flush=True)
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

    def eval_checkpoint(step, epoch, train_pool="", tag="", fold="",
                        seed_base=-1, budget=0):
        """One sharded evaluation. tag names its outputs. Nothing here crops.

        tag EXISTS BECAUSE TWO PASSES CAN SHARE A POOL. A cropped and an uncropped reading of
        valfull are different numbers, and writing both under metric_valfull_ep<N>.json would
        silently overwrite one with the other -- the same class of mistake as the five
        metric_train files that held the only held-out numbers a run produced. The caller names
        each pass and the name reaches the metric file and the shard directory.
        """
        """Save the head, score it, log it. Returns the metric dict.

        With `train_pool`, scores that pool of TRAINING interfaces using the run's own --fold and
        --seeds instead of the validation set, and writes metric_<pool>_ep<N>.json. Everything else
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
        # NAMED AFTER THE POOL, like the files it writes. "pool@fold" named the FOLD, which is
        # "fold" for every pool in this project and so said nothing; and the flags that drive this
        # are spelled --eval-pool-* precisely so no line anywhere calls an evaluation "train".
        kind = (os.path.basename(os.path.dirname(os.path.abspath(train_pool)).rstrip("/")) or "pool"
                if train_pool else "validation")
        # NAMED AFTER THE POOL, NOT THE MECHANISM. These used to be metric_train_ep<N>.json
        # whatever --train-eval-pool pointed at, so on this arm -- which points it at
        # training/valfull/pool.csv -- five files called "metric_train" held the only HELD-OUT
        # numbers the run produced. A pool is in-sample or it is not and only the pool knows, so
        # the pool's directory names the file: metric_valfull_ep25.json. "heldout" would be the
        # wrong word to standardise on, since --train-eval-pool aimed at the training pool really
        # is in-sample. run_logging/eval_dirs.pool_tag is the reader-side counterpart of this one
        # line and test_eval_dir_names.py asserts the two cannot drift apart.
        _ptag = (os.path.basename(os.path.dirname(os.path.abspath(train_pool)).rstrip("/"))
                 or "pool") if train_pool else ""
        # AN EXPLICIT TAG WINS over the pool's directory name, and a cropped primary pass is
        # renamed so it cannot be read as the uncropped one. plot_val_2x3.py reads <pool>, and it
        # must keep getting the deployment-shaped number.
        if tag:
            _ptag = tag
        mj = os.path.join(a.out, (f"metric_{_ptag}_ep%d.json" if train_pool else "metric_ep%d.json")
                          % epoch)
        extra = []
        if train_pool:
            # THE POOL'S OWN FOLD, which is not always this run's. Scoring a validation subset
            # here means --fold, --results, --roles and --seed-base all have to come from that
            # fold, not from the one being trained on: a results.csv from the wrong fold would
            # look up DockQ for interfaces it does not contain and the coverage check would fail,
            # and a wrong --seed-base would ask for seeds the fold never wrote.
            _tef = fold or a.train_eval_fold or a.fold
            _seed_base = (seed_base if seed_base >= 0
                          else (a.train_eval_seed_base if a.train_eval_seed_base >= 0
                                else _train_eval_seed_base))
            # THE ROLES TABLE CAN ALSO LIVE BESIDE THE POOL, same rule as the seed manifest
            # below and for the same reason. The fold's own chain_roles.csv names only the 53
            # structures of the original valfull wave, so an 83-structure pool spanning BOTH
            # waves leaves 49 of its 141 interfaces unmappable and phase_k_eval stops on them.
            # A pool directory describes its own pool: pool.csv, chain_roles.csv, seed_manifest
            # .json. Falling back to the fold keeps every existing pool working untouched.
            _prd = os.path.dirname(os.path.abspath(train_pool))
            _srol = os.path.join(_prd, "chain_roles.csv")
            _roles = _srol if os.path.exists(_srol) else os.path.join(_tef, "chain_roles.csv")
            extra = ["--fold", _tef,
                     "--results", os.path.join(_tef, "results.csv"),
                     "--pool", train_pool,
                     "--roles", _roles,
                     "--seed-base", str(_seed_base)]
            # AND WHICH SEEDS THAT POOL KEPT. A trimmed fold keeps 30 of a 100-wide sweep, chosen
            # per structure, so the evaluator's contiguous window asks for seeds the fold never
            # wrote and its coverage guard stops on them -- which is exactly what killed the
            # epoch-5 valfull eval. The manifest sits beside the POOL, not under the fold, because
            # it describes what this pool's metric claims. Forwarded only when it exists, so an
            # untrimmed pool keeps the contiguous behaviour.
            _sman = os.path.join(_prd, "seed_manifest.json")
            if os.path.exists(_sman):
                extra += ["--seed-manifest", _sman]
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
        # AND THE IN-BLOCK PATHWAY, which had no evaluator counterpart at all until review found
        # it. The block index goes too: pose_inblock_d* carry no depth in their names, so an
        # evaluator that installed at a DIFFERENT block would pass the key-set guard and score a
        # different operator. --lean-single is deliberately NOT forwarded: it sets c_s = 0 on
        # built blocks, leaves every module in the tree, and so changes neither the checkpoint's
        # key set nor the PAE path's mathematics -- the evaluator has no such flag and needs none.
        if a.pose_inblock:
            extra = extra + ["--pose-inblock",
                             "--pose-inblock-block", str(a.pose_inblock_block)]
            if a.pose_inblock_pae:
                extra = extra + ["--pose-inblock-pae"]
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
        # THE EVALUATOR MUST SELECT BY WHAT THE RUN IS TRAINING. --rank-pool mean optimises the
        # mean-pooled ipTM, and this pass selected by the MAX -- so the automatic validation curve
        # measured a different rule from the objective and neither number meant what it said.
        # Caught in review. --pool-mean also makes the evaluator report the shipped MAX selection
        # from the same pass, so the baseline is preserved rather than traded away: both appear as
        # mean_dockq / mean_dockq_select_max in its json.
        if a.rank_pool == "mean" and a.rank_loss:
            extra = extra + ["--pool-mean", "--select-by", "mean"]
        else:
            # BOTH POOLINGS REPORTED EVEN WHEN SELECTION IS THE MAX. --pool-mean makes the
            # evaluator write mean_dockq_select_mean alongside the selected number, from the SAME
            # pass, so a PAE-only arm can be read against both the shipped max baseline and the
            # mean-pooled one without a second evaluation and without assuming two runs were
            # otherwise identical. Selection is untouched: --select-by stays at the default max,
            # which is the shipped convention and what this arm's json already records.
            #
            # WHY IT WAS MISSING: the flag was gated on `rank_pool == mean AND rank_loss`, which
            # is false for an arm with no ranking loss, so epoch 5 of axrorr reported only the max
            # column -- and I compared it against the shipped MEAN baseline by mistake.
            extra = extra + ["--pool-mean"]
        # THE CROP, FORWARDED. phase_k_eval's --crop-tokens takes the same npz --crop-tokens does
        # here, so a cropped reading of a pool is the evaluator's own code path and not a
        # reimplementation. Absent, the pass is uncropped, which is what every pass has been.
        # SHARDED OVER EVERY CARD THE RUN WAS GIVEN. The other ranks are parked at a barrier for
        # the whole evaluation, so their cards are idle: at 73 min a pass, evaluating on one card
        # wasted ~47% of the run's wall clock with two cards doing nothing. The driver puts one
        # shard on each card -- ~6 GB beside a parked rank's ~6 GB on a 98 GB card -- and merges.
        cards = a.eval_cards or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        shard_work = os.path.join(a.out, (f"{_ptag}_eval_shards_ep%d" if train_pool
                                           else "eval_shards_ep%d") % epoch)
        budget = (budget if budget else
                  (_train_eval_budget if train_pool else a.eval_budget))
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
        # ASCENDING BY TOKENS when asked. The pdb tie-break keeps it deterministic across ranks
        # and epochs. A structure the manifest gave no numeric `tokens` for sorts first and is
        # named below rather than silently placed.
        _manifest_order = (a.epoch_order in ("manifest", "tokens"))
        if a.epoch_order == "tokens":
            order = sorted(by, key=lambda _q: (_sel_tokens.get(_q, 0), _q))
        elif _manifest_order:
            order = list(by)
        else:
            order = sorted(by)
            rng_e.shuffle(order)
        # A GROUP IS `world` SEEDS OF ONE STRUCTURE -- the unit of equal work.
        groups = []
        for _p in order:
            u = by[_p][:]
            if not _manifest_order:
                rng_e.shuffle(u)
            for g in range(len(u) // world):
                groups.append([(_p, _lp, _c) for _lp, _c in u[g * world:(g + 1) * world]])
        if not _manifest_order:
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

    # ONE-SHOT, so the coverage check runs on the first epoch this PROCESS sees rather than only
    # on global epoch 0, which a resume never reaches.
    _cov_checked = [False]

    def build_epoch_cover_all(ep):
        """Every unit exactly once per epoch, dealt round-robin so ranks hold disjoint structures.

        WHY THIS EXISTS. build_epoch's unit of work is `world` SEEDS OF ONE STRUCTURE, which makes
        per-step work equal across ranks by construction -- the right answer when the RANKING loss
        needs several poses of one structure in a single step. Its cost is that a structure with
        fewer than `world` seeds produces NO group: `for g in range(len(u) // world)` is an empty
        range at len(u)=1, so the structure never appears in any epoch and nothing says so. This
        population is 412 structures at 1 seed and 434 at 15, so 412 of 846 would never train.
        Found by adversarial review and independently by external review of 216b0630.

        A PAE-ONLY ARM HAS NO SUCH REQUIREMENT. Each pose contributes an independent cross-entropy;
        nothing is compared within a structure, so a step does not need several of its seeds.
        Work is equalised a different way: the 256-token crop makes nearly every unit the same
        size, and measured per-step rank imbalance at this batch size is ~0.5%.

        EQUAL COUNTS PER RANK, OR THE FIRST ALL-REDUCE HANGS RATHER THAN FAILS. len(flat) need not
        divide `world`, so the tail is truncated to a multiple of it -- at most world-1 units, a
        different few each epoch because the shuffle is reseeded as seed+ep.

        THE DEAL IS units[rank::world], so one step's units are a contiguous global block split
        across the ranks: rank r takes global indices r, r+world, r+2*world ... Within a step of
        `structs_per_step` units per rank that is `structs_per_step * world` DISTINCT units,
        spanning that many distinct structures except where a deep structure's seeds collide by
        chance -- which costs an extra trunk read and nothing else.
        """
        import epoch_plan as _EP
        _mine = _EP.cover_all(flat, world, rank, a.seed, ep, a.epoch_order, _sel_tokens,
                              balance=bool(a.epoch_balance), per_step=structs_per_step)
        # THE FIRST EPOCH THIS PROCESS SEES, NOT GLOBAL EPOCH 0. Gating on `ep == 0` made the
        # guard inert on every RESUMED run -- which is most of them -- so the defect it exists to
        # catch (412 structures silently absent) would have gone unreported on exactly the runs
        # least likely to be watched closely. Confirmed on rdi25k, which resumed at ep=4 and
        # printed no coverage line at all. Raised in adversarial review.
        if is_main and not _cov_checked[0]:
            _cov_checked[0] = True
            # SAY THE COVERAGE OUT LOUD, because the failure this builder exists to fix was
            # silent: 412 structures absent from every epoch with nothing in the log.
            # THE DEAL ACTUALLY WALKED, not the round-robin one. Without balance/per_step this
            # validated a different assignment from the one the ranks use.
            _cov = _EP.coverage(flat, world, a.seed, ep, a.epoch_order, _sel_tokens,
                                balance=bool(a.epoch_balance), per_step=structs_per_step)
            _all = {_t[0] for _t in flat}
            print(f"  epoch plan (cover-all): {len(flat)} unit(s) over {len(_all)} structure(s); "
                  f"every rank walks {len(_mine)}; structures covered {len(_cov)} of {len(_all)}"
                  + ("" if len(_cov) == len(_all)
                     else f"  *** {len(_all) - len(_cov)} MISSING ***"), flush=True)
            if len(_cov) != len(_all):
                raise SystemExit(
                    f"STOP: --epoch-cover-all left {len(_all) - len(_cov)} structure(s) out of "
                    f"epoch {ep}. That is the defect this builder exists to prevent.")
        return _mine

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
        # ASCENDING BY TOKENS when asked. The pdb tie-break keeps it deterministic across ranks
        # and epochs. A structure the manifest gave no numeric `tokens` for sorts first and is
        # named below rather than silently placed.
        _manifest_order = (a.epoch_order in ("manifest", "tokens"))
        if a.epoch_order == "tokens":
            order = sorted(by, key=lambda _q: (_sel_tokens.get(_q, 0), _q))
        elif _manifest_order:
            order = list(by)
        else:
            order = sorted(by)
            rng_e.shuffle(order)
        S = max(1, int(a.rank_structs_per_step))
        # THE STEP IS A CONSTANT NUMBER OF UNITS, NOT A CONSTANT NUMBER OF STRUCTURES. With a
        # uniform manifest the two are the same thing and this packs S structures per step exactly
        # as before. With mixed depth they are not: 437 structures carry 15 seeds and 416 carry 1,
        # so S=2 structures per step would mean 30 units for one pair and 2 for another.
        #
        # THAT WOULD HAVE BEEN SILENT, which is why the packing changed rather than the quotas.
        # Three sites downstream read the step's size as a per-EPOCH constant:
        #   n_units = len(steps[0])        quotas for EVERY step come from the FIRST step's length
        #   _step_poses_g[0]               the divisor of the PAE term's per-step mean
        #   the _xs_steps walk            slices ep_units by those quotas and demands they sum
        # A longer step would be sliced to the first step's length and the rest of its units simply
        # never trained; a shorter one would hand a rank fewer units than its quota -- python slices
        # do not raise -- and the ranks would take different step counts and HANG on an all-reduce.
        # Keeping every step the same unit count leaves all three correct as written.
        #
        # THE BUDGET IS S x THE DEEPEST STRUCTURE, so a step is the same size in poses as it was.
        # Structures are taken in `order` and any whose depth does not fit the remaining budget is
        # passed over for a later one that does -- 15 ones fill a 15-unit remainder exactly, which
        # is why the two populations pack with nothing left over.
        # THE PACKING IS _LRS.pack_cross_seed_steps, WHICH IS ALSO WHAT SETS THE LR HORIZON.
        # It used to be an inline loop here and a formula there, and they drifted: the horizon
        # claimed len(order)//S = 426 steps while this built 232, a 1.84x error on the decay
        # window. One implementation cannot disagree with itself. The sort inside it is by
        # descending depth and STABLE, so shuffling `order` first keeps per-epoch randomness in
        # which structures share a step while making the COUNT a function of the depth histogram
        # alone -- plain first-fit over a shuffled order gave 220..229 steps on this population,
        # so no constant steps-per-epoch existed to calibrate against.
        _depth = {_p: len(by[_p]) for _p in order}
        _chunks, _dropped, U = _LRS.pack_cross_seed_steps(_depth, order, S,
                                                           preserve_order=_manifest_order)
        # THE DEEPEST STRUCTURE'S UNIT COUNT. Both diagnostics below interpolate `_maxd` and
        # nothing ever bound it: `NameError: name '_maxd' is not defined`, raised from the
        # _manifest_order branch, which the arm never takes and a size-ordered memory probe always
        # does. So the one configuration that exists to find the memory ceiling died before
        # reaching a single pose.
        #
        # Derived from _depth rather than from U, and then checked against U, so the message
        # cannot drift from the packing it describes -- lr_schedule.pack_cross_seed_steps:154
        # computes U = S * max(depth), and if that ever changes this stops rather than printing a
        # number that is no longer true.
        _maxd = max(_depth.values()) if _depth else 0
        if _chunks and U != S * _maxd:
            raise SystemExit(
                f"STOP: a cross-seed step is {U} unit(s) but S={S} times the deepest structure "
                f"({_maxd}) is {S * _maxd}. build_epoch_cross_seed's diagnostics describe a step "
                f"as 'S x deepest', and pack_cross_seed_steps no longer agrees.")
        steps = []
        for chunk in _chunks:
            # WHICH SEEDS, AND IN WHAT ORDER, STAYS RANDOM -- that is this shuffle, and it is the
            # one that makes a structure's 15 units arrive in a different order each epoch.
            _lists = []
            for _p in chunk:
                u = by[_p][:]
                if not _manifest_order:
                    rng_e.shuffle(u)
                _lists.append([(_p, _lp, _c) for _lp, _c in u])
            if _manifest_order:
                units = [x for _l in _lists for x in _l]
            else:
                # DEAL THE STEP ROUND-ROBIN ACROSS ITS STRUCTURES, DO NOT SHUFFLE IT GLOBALLY.
                # The split below takes CONTIGUOUS slices, so the order of this list IS the rank
                # assignment. A global shuffle gave each rank a random 10 of the step's 30 units,
                # which for a step of 2 deep structures means a random split of 15+15 -- so one
                # rank could hold 8 units of the larger structure and another 2. Memory per unit
                # goes as tokens^2 (measured: peak_GB = 2.02 + 2.896e-05 * tokens^2, residual
                # under 0.3 GB from 268 to 1364 tokens), so an uneven split is an uneven WAIT:
                # every rank sits at the step's all-reduce until the slowest finishes.
                #
                # MEASURED ON THE 841-STRUCTURE POPULATION, per-step tokens^2 work, max-min over
                # min: global shuffle median 17.2% / mean 18.5% / p90 40.1%, 90 of 230 steps worse
                # than 20%. Round-robin: median 0.0% / mean 2.3% / p90 0.0%, 13 of 230. The 13 are
                # the all-shallow steps -- 30 distinct 1-seed structures, where no assignment can
                # give three ranks the same structures and the imbalance is irreducible here.
                #
                # IT IS EXACT WHENEVER EVERY STRUCTURE'S DEPTH DIVIDES `world`: 216 of the 230
                # steps are 2 structures x 15 seeds, 15 = 3 x 5, so each rank gets exactly 5 units
                # of each -- the balance is structural, not probabilistic.
                #
                # RANDOMNESS IS PRESERVED WHERE IT MATTERS: the rotation's starting structure is
                # shuffled, so it is not always the alphabetically-first one that leads, and the
                # per-structure seed order above is untouched. What is removed is only the random
                # ASSIGNMENT of units to ranks, which carried no training signal -- the ranking
                # coefficients are gathered across all ranks before any backward, so which rank
                # holds which seed cannot change the objective.
                rng_e.shuffle(_lists)
                units = _LRS.deal_round_robin(_lists)
            steps.append(units)
        # THE HORIZON AND THE BUILDER AGREE, ASSERTED EVERY EPOCH. The LR schedule was
        # calibrated once at startup from _LRS.steps_per_epoch; if this epoch packs a different
        # number of steps then lr_at_step is reading the wrong horizon and the arm is not running
        # the schedule it reports. Cheap, and it is the check whose absence let 426-vs-232 ship.
        if steps and _lr_horizon[0] and _xs_spe[0] and len(steps) != _xs_spe[0]:
            raise SystemExit(
                f"STOP: epoch {ep} packed {len(steps)} cross-seed step(s) but the lr horizon was "
                f"built from {_xs_spe[0]} step(s)/epoch. The decay window would be wrong by "
                f"{_xs_spe[0] / max(1, len(steps)):.3f}x. Both come from "
                f"_LRS.pack_cross_seed_steps, so this means the population changed mid-run.")
        if steps and is_main and len({len(x) for x in steps}) != 1:
            # NOT REACHABLE BY CONSTRUCTION, asserted anyway: everything downstream treats the
            # step size as a constant, and a packing bug here is the one that deadlocks instead of
            # failing.
            raise SystemExit(f"STOP: cross-seed steps are not all the same unit count: "
                             f"{sorted({len(x) for x in steps})}. The quota bookkeeping, the PAE "
                             f"per-step mean and the _xs_steps walk all assume one size.")
        if not steps:
            raise SystemExit(
                f"STOP: --rank-cross-seed with --rank-structs-per-step {S} needs at least {S} "
                f"structure(s); the manifest resolved {len(order)}.")
        if _manifest_order and is_main:
            # SAY WHAT WILL ACTUALLY RUN, AND WHAT WILL NOT. A short final step is dropped rather
            # than padded (see above), and in a size-sorted manifest the dropped chunk is the
            # LARGEST structures -- precisely the ones a memory probe exists to reach. A probe
            # that silently never ran them would recommend a threshold from the wrong structure.
            print(f"  EPOCH ORDER manifest: {len(order)} structure(s), "
                  f"{len(steps)} step(s) of {U} unit(s) (S={S} x deepest {_maxd}); "
                  f"first {order[:3]} last {order[-3:]}")
            if _dropped:
                print(f"  EPOCH ORDER manifest: {len(_dropped)} structure(s) DROPPED as a short "
                      f"final step: {_dropped}. A step is {U} unit(s) "
                      f"(--rank-structs-per-step {S} x deepest structure {_maxd}); these could not "
                      f"fill one exactly. Trim the FRONT of the manifest or adjust S, or these are "
                      f"never run.", flush=True)
        if _dropped and _manifest_order:
            # REFUSE RATHER THAN WARN. Raised in review: in a size-sorted order the dropped chunk
            # is the LARGEST structures, which is precisely what a memory probe exists to reach, so
            # a probe that silently never ran them would report a threshold from the wrong
            # structure. A warning in a log is not a gate. The shuffled modes keep the old
            # behaviour because there the omitted set rotates with the epoch seed.
            raise SystemExit(
                f"STOP: --epoch-order {a.epoch_order} dropped {len(_dropped)} structure(s) as a "
                f"short final step: {_dropped[:6]}. In a deterministic order the dropped chunk is "
                f"the largest structures and they would never train, which silently changes what a "
                f"size-ordered epoch means. Trim the manifest to a multiple of the "
                f"{U}-unit budget, or adjust --rank-structs-per-step.")
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
                                  "epoch_cover_all": bool(a.epoch_cover_all),
                                  "nsample": a.nsample, "lr": a.lr,
                                  "lr_final": a.lr_final,
                                  # ALSO HALF-WIRED, and this one decides the lr at every step of
                                  # the cosine arm: the decay window and shape were compared but
                                  # never written, so a bundle moved without its args.json fell
                                  # through to the pre-crop defaults (0, 0, "cosine") and a resume
                                  # would silently run a WHOLE-RUN cosine instead of the 20-50
                                  # window it was configured for. Found by test_guard_fields.py
                                  # the first time it ran, not by reading the diff.
                                  "lr_decay_start_epoch": a.lr_decay_start_epoch,
                                  "lr_decay_end_epoch": a.lr_decay_end_epoch,
                                  "lr_decay_shape": a.lr_decay_shape,
                                  "lr_steps": a.lr_steps,
                                  "pair_dropout": a.pair_dropout,
                                  "token_keep_union": a.token_keep_union,
                                  "token_keep_other": a.token_keep_other,
                                  "warmup_steps": a.warmup_steps, "epochs": a.epochs,
                                  "manifest_skip": a.manifest_skip,
                                  "manifest_digest": _manifest_digest,
                                  "crop_size": a.crop_size,
                                  "alpha_confidence": a.alpha_confidence,
                                  "pae_scored_pairs": a.pae_scored_pairs,
                                  "pae_only": a.pae_only,
                                  "lean_single": a.lean_single,
                                  "pose_inblock": a.pose_inblock,
                                  "pose_inblock_pae": a.pose_inblock_pae,
                                  "pose_inblock_block": a.pose_inblock_block,
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
                                  "l2_sp": a.l2_sp,
                                  "l2_sp_exclude": a.l2_sp_exclude,
                                  # SAVED AS WELL AS COMPARED. Added to the compared dict only,
                                  # which test_guard_fields.py catches as "half-wired": the bundle
                                  # then records no value, and a bundle moved away from its
                                  # args.json lands in _unknown and hits the hard stop that runs
                                  # BEFORE --resume-override and --resume-allow-drift are
                                  # consulted -- unresumable, by a check no override can bypass.
                                  # That is verbatim the crop_margin/crop_contact failure the
                                  # comment above describes as "I turned a safety check into a
                                  # wall", re-introduced.
                                  "lora_keep_trainable": a.lora_keep_trainable,
                                  # FIFTH TIME. The three row-weight fields went into the
                                  # comparison side and not here, so a bundle moved without its
                                  # args.json would carry no record of the PAE objective and read
                                  # as UNWEIGHTED. Found in review of c431fb93. The pattern is now
                                  # tested rather than re-noticed: test_guard_fields.py asserts
                                  # every compared field is also saved, so a sixth is a failing
                                  # gate instead of a comment.
                                  "pae_row_frac": a.pae_row_frac,
                                  "pae_row_mass": a.pae_row_mass,
                                  "pae_row_floor": a.pae_row_floor,
                                  "rank_loss": a.rank_loss, "alpha_rank": a.alpha_rank,
                 "rank_pool": a.rank_pool, "rel_labels": a.rel_labels,
                 "rel_standardize": a.rel_standardize,
                 "rel_sd_weight": a.rel_sd_weight,
                 "rel_g_weight": a.rel_g_weight,
                 "rel_g_sigmoid": (a.rel_g_sigmoid or ""),
                 "rel_std_floor": (-1.0 if a.rel_std_floor is None else a.rel_std_floor),
                                  "rank_form": a.rank_form,
                                  "rank_cross_seed": a.rank_cross_seed,
                                  "rank_structs_per_step": a.rank_structs_per_step,
                                  # BOTH SIDES IN ONE PATCH, which test_guard_fields.py asserts.
                                  # Four separate fields once went into the comparison side with
                                  # a comment claiming otherwise. rank_pairs, rank_within_weight,
                                  # rank_topk_decay, rank_straddle, rank_tau, rank_weight_by_gap,
                                  # rank_spearman_tau, rank_dist_weight and rank_rel_scale have
                                  # since been deleted outright with the forms that read them, so
                                  # a bundle recording them resumes only under a run that still
                                  # has them -- which is none.
                                  "alpha_pae": a.alpha_pae, "drop_pde": a.drop_pde,
                                  "clip_norm": a.clip_norm,
                                  "crop_contact": a.crop_contact,
                 # THE CROP DEFINITION, for the same reason as the two above: a16911de changed
                 # the DEFAULT, so resuming a pre-change bundle under post-change code blends two
                 # crops into one curve unless the guard compares it. Benign only while pose 0
                 # equals the union; the guard's standard is that the setting is compared, not
                 # that it currently happens to be a no-op.
                 "crop_all_poses": a.crop_all_poses,
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
            elif a.epoch_cover_all:
                ep_units = build_epoch_cover_all(ep)
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
            # BEFORE _pf_start, NOT AFTER. The token draw is seeded from this cell and the
            # prefetch workers crop units in their own threads from the moment they start, so
            # setting it afterwards left those units drawn against the PREVIOUS epoch's value
            # while the inline path used the current one -- different token sets for poses of the
            # same structure, which is exactly the invariant the draw exists to hold. Caught by
            # test_token_dropout.py, which pins this ordering.
            _crop_epoch[0] = ep
            _tok_drop_stat["kept"] = _tok_drop_stat["tot"] = 0.0
            _tok_drop_stat["n"] = _tok_drop_stat["topped"] = 0
            _tok_drop_stat["topped_tok"] = 0
            # STARTED PER EPOCH, over THIS epoch's shuffled order, so the worker walks the units
            # in the order the main loop will consume them.
            _pf_start(ep_units)
            # PER EPOCH, like _pf_stat and for the same reason: the printed line reports an epoch,
            # and a run-cumulative counter printed as one would be a different number every time.
            for _k in _tr_stat:
                _tr_stat[_k] = 0
            _T0, _TN0 = dict(_T), dict(_TN)
            _T_TOK0 = {k: list(v) for k, v in _T_TOK.items()}
            # PEAK MEMORY PER EPOCH, reset here so each epoch reports its own rather than the
            # high-water mark of the whole run. This is the number that decides whether activation
            # checkpointing can stay off: --mem-log answers it too but emits one line per pose,
            # 8,100 per rank per epoch.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _mem_ep[0] = 0
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
                    # BEFORE PASS A, so the warmer is already reading unit 1 while pass A is on
                    # the GPU with unit 0. It also drops the PREVIOUS step's arrays, which is what
                    # bounds the cache to one step.
                    _tr_step_start(_xs_steps[_xs_at[ui]])
                    # BEFORE PASS A AND AFTER THE OPTIMISER STEP THAT ENDED THE LAST ONE. Both
                    # halves matter: the entries of the step that just finished are dead (its poses
                    # never recur), and this is also where the cache re-verifies that nothing moved
                    # the frozen prefix -- an optimiser step is the only thing that could have.
                    if _pfc is not None:
                        PFC.step_reset()
                    # THE DEAD-OP RE-CHECK IS DELIBERATELY *NOT* HERE. This site is inside
                    # `if ui in _xs_at:`, which is only reached under --rank-cross-seed, so a
                    # check placed here runs zero times on every other arm -- silently. It lives
                    # immediately after opt.step() in _apply_step instead, which is unconditional
                    # and is the actual instant a stubbed tensor could have been written.
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
                    # THE SCHEDULE LIVES IN phases/lr_schedule.py so it can be asserted before
                    # a sixty-epoch run rather than inferred after one. Driven by `step`, which
                    # resume restores, so a resumed run picks the schedule up where it left off.
                    # _lr_horizon[0] is steps-per-epoch x epochs, derived from the EPOCH BUILDER's
                    # grouping -- see the comment at its assignment for the 205-vs-184 bug.
                    cur_lr = _LRS.lr_at_step(
                        step, lr, a.lr_final, a.warmup_steps,
                        _lr_horizon[0] / max(1.0, float(a.epochs)), a.epochs,
                        a.lr_decay_start_epoch, a.lr_decay_end_epoch, a.lr_decay_shape,
                        _LR_STEPS)
                    for g in opt.param_groups:
                        g["lr"] = cur_lr
                    # POSES, NOT YIELDS. One yield is one pose at --pose-batch 1 and nb poses
                    # above it, and the step boundary below compares this against accum_local,
                    # which is a POSE count. Counting yields would make the effective batch nb
                    # times the requested --accum, drop the steps per epoch by nb, and stretch the
                    # lr horizon by nb -- all silently, since every log line would still say
                    # "poses". Raised in adversarial review.
                    _nb_row = int(_m.get("n_pose_in_group", 1) or 1)
                    n_in_accum += _nb_row
                    # EVERY POSE CONTRIBUTES TO THE WINDOW SUM, which is what the batch columns
                    # average. The per-pose TERMS stay in the row too, for provenance. A batched
                    # row carries the group's MEAN, so it is weighted by the poses behind it.
                    for _k in TERMS:
                        batch_sum[_k] += float(met[_k]) * _nb_row
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
                                         **{k: met[k] for k in TERMS},
                                         **({"pae_loss_unweighted":
                                             met.get("pae_loss_unweighted", float("nan"))}
                                            if float(a.pae_row_frac) > 0 else {}),
                                         **means})
                            continue
                        step += 1
                        # THE GRADIENT SPLIT FOR THIS STEP, written before the next one arms.
                        if _gn_on[0]:
                            _gn_emit(step, ep + 1)
                        # ARM THE NEXT MEASURED STEP. Keyed on the step counter AFTER the
                        # increment so step K, 2K ... are measured; the poses of the step are
                        # accumulated as they are seen, so this has to be set before they are.
                        _gn_on[0] = bool(a.grad_norm_every
                                         and (step + 1) % a.grad_norm_every == 0)
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
                                 **{k: met[k] for k in TERMS},
                                 **({"pae_loss_unweighted":
                                     met.get("pae_loss_unweighted", float("nan"))}
                                    if float(a.pae_row_frac) > 0 else {}),
                                 **means})
                    # EVERY RANK FLUSHES ITS OWN FILE; only rank 0 writes trajectory.csv.
                    # The condition used to be `write_traj and ...`, which is is_main, so ranks 1+
                    # never flushed AND never cleared `rows`: they accumulated the whole epoch in
                    # memory and wrote nothing. That is the same defect the crop log was fixed for
                    # in review round 6, and it is why every per-pose figure described a third of
                    # the run.
                    if len(rows) % 20 == 0:
                        _flush(traj_rank_p, traj_cols, list(rows))
                        if write_traj:
                            _flush(traj_p, traj_cols, rows)
                        rows = []
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
            # AND THE TRUNK WARMER, so its arrays are not still held while the evaluator
            # subprocesses have to fit beside this rank -- the same reason empty_cache() runs
            # below for the device side.
            _tr_end()
            # SO THE EVALUATOR SUBPROCESSES DO NOT INHERIT UP TO --prefix-cache-gb OF HELD HOST
            # MEMORY. Same reason _tr_end is unconditional.
            if _pfc is not None:
                PFC.step_reset()
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
            # ACCUMULATED, NOT ASSIGNED. _parts reports _T[k] - _T0[k], which is right only for
            # the run-cumulative counters _tick produces. _pf_stat is RESET per epoch by
            # _pf_start, so assigning it here printed w_N - w_{N-1} -- negative exactly when the
            # wait shrank, which is the phenomenon this bucket was added to track -- and a count
            # delta of 0 whenever two epochs had the same unit count, which filtered the bucket
            # out of both the printed line and _acc from epoch 2 onward. Only epoch 1 was right.
            _T["prefetch_wait"] += _pf_stat["wait_s"]
            _TN["prefetch_wait"] += _pf_stat["hit"] + _pf_stat["miss"]
            # sorted(list(_T)), BECAUSE A WARMER CAN ADD A KEY MID-ITERATION. _tr_worker's
            # resolve_trunk ticks fingerprint_hash_warm, and a defaultdict grows on the FIRST tick
            # of a key -- which is enough for "dictionary changed size during iteration" at an
            # epoch boundary. The snapshot is 20-odd strings.
            _parts = [(k, _T[k] - _T0.get(k, 0.0), _TN[k] - _TN0.get(k, 0))
                      for k in sorted(list(_T))]
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
            # THE THREE trunk_* BUCKETS ARE EXCLUSIVE LEAVES TOO. They are the UNCROPPED path's
            # load_unit, so they never co-occur with trunk_read_plus_crop_tokens or crop_slice
            # (the crop branch returns its own tensors and load_unit is not called), and they sit
            # in unit_setup before any pose, so they do not overlap pose_total either. Under
            # --rank-cross-seed pass A ticks them into the same _T, which is already why
            # rank_pass_a is kept out of this list.
            _EXCL = ("pose_total", "crop_slice", "fingerprint_hash", "prefetch_issue",
                     "trunk_npz_open", "trunk_npz_read", "trunk_h2d",
                     "trunk_read_plus_crop_tokens", "label_open", "loss_feature_prep",
                     "loss_stats_collective_plus_wait", "apply_step_total",
                     "prefetch_wait", "grad_collective_plus_wait",
                     # rank_gather and pose_a_total are leaves. rank_pass_a is NOT: it wraps the
                     # score_only run_unit calls, which tick pose_a_total, label_open,
                     # fingerprint_hash, trunk_read_plus_crop_tokens and loss_feature_prep into
                     # the same _T. Including it counted pass A two or three times and could push
                     # the residual negative on a --rank-cross-seed arm -- the same double-count
                     # reasoning that correctly kept cpucrop_* out.
                     "rank_gather", "pose_a_total")
            _acc = sum(v for k, v, c in _parts if k in _EXCL and c)
            print(f"  [rank {rank}] epoch {ep + 1} took {_ep_s/60:.1f} min -- {_shown}", flush=True)
            # COST VERSUS LENGTH, measured. Buckets are fixed so the lines are comparable epoch to
            # epoch and arm to arm; a token value with fewer than 8 poses behind it in this epoch
            # is still counted, since the bucket is the unit of report, not the token value.
            _BK = (0, 350, 450, 550, 650, 800, 1000, 1300, 10 ** 9)
            _tb = {}
            for (_k, _nt), _v in _T_TOK.items():
                _p = _T_TOK0.get((_k, _nt), [0.0, 0])
                _ds, _dc = _v[0] - _p[0], _v[1] - _p[1]
                if _dc <= 0:
                    continue
                _b = max(_i for _i in range(len(_BK) - 1) if _nt >= _BK[_i])
                _slot = _tb.setdefault(_b, {"s": 0.0, "n": 0, "tok": 0.0})
                _slot["s"] += _ds
                _slot["n"] += _dc
                _slot["tok"] += _nt * _dc
            if _tb:
                _line = "  ".join(
                    f"[{_BK[_b]}-{'inf' if _BK[_b+1] >= 10**9 else _BK[_b+1]}] "
                    f"{1000*_d['s']/_d['n']:.0f}ms/{_d['n']}@{_d['tok']/_d['n']:.0f}tok"
                    for _b, _d in sorted(_tb.items()))
                print(f"  [rank {rank}] epoch {ep + 1} gpu time by token count "
                      f"(forward+backward, mean per call): {_line}", flush=True)
            # DID THE PREFETCH ACTUALLY FIRE? crop_slice alone cannot answer it: a miss falls back
            # to the inline crop and ticks the same counter, which is exactly how the first
            # version's 100% miss rate looked identical to no prefetch at all.
            # ON _pf_thread, NOT ON a.prefetch_units. _pf_start now declines to start on an
            # uncropped arm, where nothing ever consumed its results -- but the default arm still
            # passes --prefetch-units 2, so keying the print on the flag made every epoch of every
            # uncropped run report "prefetch (4 worker(s)): 0/0 hit (0.0%)", which reads as a
            # regression to 0% rather than as "there is no prefetch on this path".
            if _pf_started[0]:
                _pf_n = _pf_stat["hit"] + _pf_stat["miss"]
                print(f"  [rank {rank}] epoch {ep + 1} prefetch "
                      f"({max(1, a.prefetch_workers)} worker(s)): {_pf_stat['hit']}/{_pf_n} hit "
                      f"({100.0 * _pf_stat['hit'] / max(1, _pf_n):.1f}%), "
                      f"{_pf_stat['wait_s']/60:.1f} min waiting on the worker"
                      # REPORTED, because these two are the multi-worker failure made visible. A
                      # non-zero `dropped` means results were delivered and evicted -- harmless
                      # now (an immediate inline recrop) but it is the condition that used to
                      # cost 120 s each and cascade. `exhausted` means the workers finished while
                      # the main loop was still asking. Both were invisible before and both are
                      # the thing to watch when --prefetch-workers goes above 1.
                      + (f", {_pf_stat['dropped']} evicted-then-wanted, "
                         f"{_pf_stat['exhausted']} asked after the workers finished"
                         if (_pf_stat['dropped'] or _pf_stat['exhausted']) else ""), flush=True)
            # DID THE TRUNK CACHE ACTUALLY FIRE? The same question and the same reason: a miss
            # reads inline and ticks the same trunk_* buckets, so without a hit rate a cache that
            # never fires is indistinguishable from one that always does.
            #
            # HOW TO READ THE RATE, stated because the first version printed "ceiling 50%" and that
            # was wrong in the direction that stops you tuning. 50% is what the cache alone gives:
            # pass A misses, fills, and pass B hits. A unit the WARMER got to before pass A is a
            # hit in BOTH passes, so the range is 50% (the warmers contributed nothing) to 100%
            # (every unit was warmed in time). Printing 50% as the ceiling would read a healthy 80%
            # as impossible and a useless 50% as perfect. Note the cap also bounds it: at 4 GiB a
            # 30-unit step of ~7.3 GB fits about half its units, so ~75% is the practical target
            # there and `full` says how many were turned away.
            if _tr_on:
                _tr_n = _tr_stat["hit"] + _tr_stat["miss"]
                print(f"  [rank {rank}] epoch {ep + 1} trunk cache "
                      f"({a.trunk_cache_gb:g} GiB, {max(1, a.prefetch_workers)} warmer(s)): "
                      f"{_tr_stat['hit']}/{_tr_n} hit "
                      f"({100.0 * _tr_stat['hit'] / max(1, _tr_n):.1f}%; 50% = the warmers "
                      f"contributed nothing, 100% = every unit was warmed before pass A), "
                      f"{_tr_stat['put']} cached, {_tr_stat['full']} refused for cap, "
                      f"{_tr_stat['dup']} skipped as present/full, {_tr_stat['err']} "
                      f"warmer error(s)", flush=True)
            # DID THE PREFIX CACHE ACTUALLY FIRE? Same question, same reason: a miss recomputes
            # the prefix and ticks the same gpu_forward bucket, so without a hit rate a cache that
            # never fires looks exactly like one that always does.
            #
            # HOW TO READ THE RATE. The ceiling here really IS 50%: every pose is forwarded twice
            # per step, pass A always misses (it is the fill) and pass B hits, so 50% means the
            # cache worked on every pose and anything below it is poses that were refused for cap.
            # That is the opposite of the trunk cache's range and the reason both lines say so.
            if _pfc is not None:
                _ps = PFC.stats()
                _pn = _ps["hit"] + _ps["miss"]
                print(f"  [rank {rank}] epoch {ep + 1} prefix cache "
                      f"({a.prefix_cache_gb:g} GiB, cut in block {_pfc['block_index']} after "
                      f"tri_mul_in, sig {_pfc['sig']}): "
                      f"{_ps['hit']}/{_pn} hit ({100.0 * _ps['hit'] / max(1, _pn):.1f}%; 50% is "
                      f"the ceiling -- pass A fills and pass B hits), {_ps['put']} cached, "
                      f"{_ps['full']} refused for cap, {_ps['shape']} dropped on a shape "
                      f"mismatch, {_ps['nokey']} forward(s) with no pose key, {_ps['err']} "
                      f"copy error(s); {_ps['d2h_bytes'] / (1 << 30):.1f} GiB staged to host, "
                      f"{_ps['h2d_bytes'] / (1 << 30):.1f} GiB read back", flush=True)
                PFC.reset_stats()
            # DID THE DEAD-OP SKIP ACTUALLY FIRE? Same question as the two caches above, and the
            # same reason: a stub that is never reached costs nothing and saves nothing, and the
            # epoch time is the only other signal. The count is per stub invocation, so under
            # --prefix-cache-gb expect FEWER calls than stubs x poses x passes -- on a cache hit
            # the blocks before the cut are identities and their stubs are not reached at all.
            if _dops is not None:
                _ds = DOPS.stats()
                print(f"  [rank {rank}] epoch {ep + 1} dead ops: {_ds['calls']:,} stub call(s) "
                      f"across {_ds['stubbed']} skipped module(s) "
                      f"({_ds['tensors']} tensor(s) held at the floor); each one is an O(N^3) or "
                      f"O(N^2) pair operation NOT computed, and each returned a zeros tensor the "
                      f"block added to z -- bit-identical to running it", flush=True)
                DOPS.reset_stats()
            # THE DRIFT IS THE EXPERIMENT under --l2-sp, so it is reported, not inferred. The
            # full head reached ||dW||^2 = 67,052 unconstrained over 30 epochs; this says where
            # the penalised run actually sits.
            # THE REALISED DRAW, reported rather than assumed. The configured probabilities say
            # what was asked for; this says what the epoch actually trained on, which is the
            # number to compare against the fixed crop's 0.598.
            if _TOK_DROP and _tok_drop_stat["n"]:
                _tk = _tok_drop_stat["kept"] / max(_tok_drop_stat["tot"], 1.0)
                print(f"  [rank {rank}] epoch {ep + 1} token draw: kept {_tk:.4f} of tokens over "
                      f"{_tok_drop_stat['n']} unit(s) "
                      f"(union p={a.token_keep_union:g}, other p={a.token_keep_other:g}; the "
                      f"fixed crop was 0.598), so ~{_tk * _tk / 0.598 ** 2:.2f}x the fixed "
                      f"crop's pair work; {_tok_drop_stat['topped']} chain(s) topped up to the "
                      f"4-token floor, {_tok_drop_stat['topped_tok']} token(s) added", flush=True)
            if a.l2_sp and _l2sp_drift["n"]:
                # AS A SHARE OF THE OBJECTIVE, not just in absolute terms. A coefficient that
                # LOOKS weak is not weak if the squared displacement is large: at the 67,052 the
                # unconstrained head reached, lambda 1e-4 is a penalty of 6.7 against a loss near
                # 2.9 -- more than twice the objective. So "is 1e-4 weak here" is only answerable
                # against the epoch's own loss, and this prints the ratio rather than leaving a
                # reader to do it. Raised in review.
                _l2p = a.l2_sp * _l2sp_drift["sq"]
                _l2_ref = (ep_loss["sum"] / ep_loss["n"]) if ep_loss["n"] else float("nan")
                _l2_share = (_l2p / _l2_ref) if (_l2_ref == _l2_ref and _l2_ref > 0) else \
                    float("nan")
                print(f"  [rank {rank}] epoch {ep + 1} l2-sp: ||W-W0||^2 = "
                      f"{_l2sp_drift['sq']:.1f}, penalty = {_l2p:.4f} "
                      f"= {100.0 * _l2_share:.1f}% of the epoch's mean loss {_l2_ref:.4f} "
                      f"(unconstrained full head reached 67,052 by epoch 30, which at "
                      f"lambda {a.l2_sp:g} would be {a.l2_sp * 67052.0:.2f})", flush=True)
                # THE NUMBER THAT DECIDES WHETHER THIS LAMBDA IS ON. The line above is kept for
                # continuity with every earlier arm's logs, but it is the misleading one: section
                # 17 measured 1e-5 at 0.5% of the objective and 0.08% of the gradient. Read THIS
                # line instead. The reference points, measured on egw5wq at epoch 6 over 804 steps
                # against a median task grad_norm of 1.080 with ||W-W0|| = 43.2: 1e-5 -> 0.08%,
                # 1e-4 -> 0.80%, 1e-3 -> 7.99%. Below ~1% is "no L2", not "a little L2".
                _sh = _l2sp_drift.get("shares") or []
                if _sh:
                    # MEDIAN, TO MATCH THE ESTIMATOR THE THRESHOLDS CAME FROM. Section 17's
                    # reference points are ratios against the MEDIAN grad_norm (1.080, p10 0.475,
                    # p90 2.421). That distribution is right-skewed, so a mean sits well above the
                    # median and a mean-based share would read systematically LOW against the very
                    # numbers printed beside it -- comparing two different estimators and calling
                    # it a comparison.
                    def _med(_v):
                        _q = sorted(_v)
                        _m = len(_q) // 2
                        return _q[_m] if len(_q) % 2 else 0.5 * (_q[_m - 1] + _q[_m])
                    _gshare = _med(_sh)
                    _tgn = _med(_l2sp_drift.get("tgns") or [float("nan")])
                    _over = sum(1 for _x in _sh if _x >= 1.0)
                    print(f"  [rank {rank}] epoch {ep + 1} l2-sp GRADIENT share: median "
                          f"{100.0 * _gshare:.2f}%, max {100.0 * max(_sh):.2f}%, "
                          f"{_over} of {len(_sh)} step(s) where the penalty EXCEEDS the task "
                          f"gradient; median task gradient {_tgn:.4f}, penalty gradient at the "
                          f"epoch's end 2*lambda*||W-W0|| = {2.0 * a.l2_sp * _l2sp_drift['sq'] ** 0.5:.4e} "
                          f"(>=1% acts, <1% is off; measured on egw5wq: 1e-5 0.08%, 1e-4 0.80%, "
                          f"1e-3 7.99%)", flush=True)
                _l2sp_drift["shares"] = []
                _l2sp_drift["tgns"] = []
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
            # DEFINED ONCE, ABOVE BOTH READERS. The ranking prints and the spread.r<rank>.csv
            # writer used to be one block, so the median helper lived inside the ranking gate;
            # the csv writer now sits outside it (--pae-split-accept must write a row on a
            # PAE-only arm, where that gate never opens), and leaving the def behind would have
            # raised NameError there on the first epoch of exactly that arm.
            def _med(v):
                w = sorted(v)
                return w[len(w) // 2] if w else float("nan")
            # WHETHER THE RANKING COLUMNS HAVE ANYTHING IN THEM, carried out of the gate as a
            # flag rather than re-tested below: the gate reads _xs_stat["groups"], and the block
            # it guards RESETS that counter to 0, so re-testing it after the hoist would have
            # written a row of nan ranking columns on every ranking epoch.
            _sp_rank_due = False
            if (a.rank_loss or a.rank_diag_only) and a.rank_cross_seed and _xs_stat["groups"]:
                # THE LOSS VALUE IS ACCUMULATED IN PASS A, not reconstructed from the
                # coefficients: rel's value is a sum of squares, so sum_i c_i s_i is not it.
                # 0 is the floor, not "no alignment".
                if a.rank_form == "rel" and _rel_stat.get("rho"):
                    _rh = _rel_stat["rho"]
                    print(f"  [rank {rank}] epoch {ep + 1} WITHIN-INTERFACE SPEARMAN "
                          f"(predicted mean ipTM vs DockQ): median {_med(_rh):+.4f} over "
                          f"{len(_rh)} interface-group(s)   [the head was +0.3170 at d32kqp ep66; "
                          f"the TRUE iTM reaches +0.6900 -- this is the arm's prediction]")
                    if _rel_stat.get("sd_ratio"):
                        print(f"  [rank {rank}] epoch {ep + 1} SCORE SPREAD: median std(predicted) "
                              f"{_med(_rel_stat['sd_s']):.4f} against std(true iTM) "
                              f"{_med(_rel_stat['sd_g']):.4f}, median ratio "
                              f"{_med(_rel_stat['sd_ratio']):.3f}   "
                              f"[<1 collapsing, >1 exaggerating, ~1 right magnitude]")
                    if _rel_stat.get("bal_lists"):
                        _nb = _rel_stat.get("bal_nacc") or [0]
                        _one = _rel_stat.get("bal_onesided", 0)
                        print(f"  [rank {rank}] epoch {ep + 1} CLASS BALANCE: "
                              f"{_rel_stat['bal_lists']} list(s) weighted at DockQ "
                              f"{a.rel_balance_accept:g}, median {_med(_nb)} acceptable; "
                              f"{_one} fell back to a uniform mean for having only ONE class"
                              + ("   [EXPECTED 0 on a pool where every list has both -- a "
                                 "non-zero count means the DockQ lookup is wrong, not that the "
                                 "data is odd]" if _one else ""))
                        if _one:
                            print(f"    one-sided: "
                                  f"{', '.join(_rel_stat['bal_onesided_ifaces'][:8])}"
                                  + (" ..." if _one > 8 else ""))
                    if _rel_stat.get("first_acc_rank") or _rel_stat.get("score_gap"):
                        _fa = _rel_stat.get("first_acc_rank") or []
                        _sg = _rel_stat.get("score_gap") or []
                        _hd = _rel_stat.get("headroom") or []
                        _t1 = _rel_stat.get("acc_top1") or []
                        print(f"  [rank {rank}] epoch {ep + 1} RANKING READS ({a.rank_pool}"
                              f"-pooled, the readout this arm uses): top-1 acceptable "
                              f"{(sum(_t1) / len(_t1) if _t1 else float('nan')):.1%} over "
                              f"{len(_t1)} list(s); median rank of the first acceptable pose "
                              f"{_med(_fa) if _fa else float('nan')}; score gap "
                              f"{(sum(_sg) / len(_sg) if _sg else float('nan')):+.5f} over "
                              f"{len(_sg)} two-class list(s); selected DockQ at "
                              f"{(sum(_hd) / len(_hd) if _hd else float('nan')):.1%} of headroom "
                              f"above a random pick")
                    if _rel_stat.get("sel_dq"):
                        _sq = _rel_stat["sel_dq"]
                        _oq = _rel_stat["orc_dq"]
                        _mean = sum(_sq) / len(_sq)
                        _fa = sum(1 for _v in _sq if _v >= 0.23) / len(_sq)
                        print(f"  [rank {rank}] epoch {ep + 1} IN-SAMPLE SELECTED DockQ: mean "
                              f"{_mean:.4f}, frac acceptable {_fa:.4f} over {len(_sq)} "
                              f"interface-group(s)   [oracle mean "
                              f"{sum(_oq) / len(_oq):.4f}; CROPPED training poses, top-1 of this "
                              f"epoch's own forward -- not the held-out number]")
                    _sp_rank_due = True
                # THE LINE NAMES THE OBJECTIVE THAT RAN. It once printed
                # "({rank_form}, {rank_pairs}-seed pairs)" for every form, so a rel run -- which
                # forms no pairs at all -- was logged as "(rel, cross-seed pairs)". This is the
                # one line most likely to be read back months later to establish what an arm
                # trained on. There is now one objective and it is named in full.
                _obj = "relative regression on true iTM, mean-centred per interface"
                print(f"  [rank {rank}] epoch {ep + 1} rank-loss ({_obj}): "
                      f"{_xs_stat['loss'] / _xs_stat['groups']:+.5f} over "
                      f"{_xs_stat['groups']} interface-group(s) in {_xs_stat['steps']} step(s)"
                      # THE REASON MUST MATCH THE FORM. Raised in review: this said "no
                      # straddling pair" whatever the objective, which is false under rel -- rel
                      # forms no pairs, and a skipped rel step means the step held no interface
                      # with at least two labelled poses to centre over.
                      + ((f" [{_xs_stat['skipped']} step(s) SKIPPED: "
                          + ("no interface with two or more labelled poses to centre over"
                             if a.rank_form == "rel" else
                             "no straddling pair in either structure")
                          + ", so no task gradient -- not handed to AdamW]")
                         if _xs_stat.get("skipped") else "") + ", "
                      # THE COUNT IS POSES FOR THE EXPECTED FORM, NOT PAIRS, and its floor is
                      # -max(DockQ) rather than 0. Calling 150 poses "150 pair(s)" and claiming a
                      # zero floor would misreport the objective in the one line most likely to
                      # be read back later.
                      f"{_xs_stat['pairs']} "
                      # POSES FOR rel TOO, not just for expected. rank_loss.rel_coeffs returns
                      # (coeffs, n_poses, loss), so this counter holds a POSE count under rel and
                      # calling it "pair(s)" overstated the group size by the pair-count factor.
                      + ("pose(s)" if a.rank_form in ("expected", "rel") else "pair(s)")
                      + ("  (0 is the FLOOR -- every pair ordered)"
                         if a.rank_form == "softplus" else "")
                      + ("  (0 is the FLOOR -- centred prediction equals centred true iTM)"
                         if a.rank_form == "rel" else "")
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
            # THE PAE LOSS, SPLIT BY POSE GROUP. OUTSIDE the ranking gate above, which is the
            # whole point: that gate needs --rank-cross-seed and a non-empty pass A, and this
            # read exists for an arm that runs neither. Per rank and not reduced, like the
            # spread medians below -- each rank sees its own structures and an all-reduce in a
            # print block is a collective nothing else here takes.
            #
            # WHAT TO READ: the two means separately, over epochs. The aggregate pae_loss falls
            # whichever group is being fit, so only the DIFFERENCE says whether one is being
            # sacrificed for the other. The counts are printed beside them because the means are
            # over different, unequal populations and a count that moves between epochs changes
            # a mean without any pose's loss changing.
            if _ps_on:
                _pm_a = ((_rel_stat["ps_acc_sum"] / _rel_stat["ps_acc_n"])
                         if _rel_stat["ps_acc_n"] else float("nan"))
                _pm_u = ((_rel_stat["ps_un_sum"] / _rel_stat["ps_un_n"])
                         if _rel_stat["ps_un_n"] else float("nan"))
                print(f"  [rank {rank}] epoch {ep + 1} PAE BY POSE GROUP (DockQ "
                      f"{a.pae_split_accept:g}, CROPPED training batches, the loss this arm "
                      f"trains on): acceptable {_pm_a:.4f} over {_rel_stat['ps_acc_n']} pose(s), "
                      f"unacceptable {_pm_u:.4f} over {_rel_stat['ps_un_n']} pose(s), "
                      f"acceptable minus unacceptable {_pm_a - _pm_u:+.4f}; "
                      f"{_rel_stat['ps_straddle']} pose(s) STRADDLED the threshold across their "
                      f"interfaces and are in neither, {_rel_stat['ps_unclass']} unclassified"
                      + (f" ({_rel_stat['ps_batched']} of them --pose-batch rows, whose pae_loss "
                         f"is a mean over the batch)" if _rel_stat["ps_batched"] else "")
                      + "   [one group falling while the other does not is the signal]",
                      flush=True)
            # THE SAME MEDIANS, TO A FILE. Until now the print above was the only record
            # of the within-interface spread, so reading a 20-epoch curve out of it meant
            # grepping a log format that is free to change -- and the quantity the rel loss
            # exists to control was the one quantity not in any csv. One row per epoch per
            # rank, beside trajectory.csv. Per rank, not reduced: each rank sees its own
            # structures, an all-reduce here would be a collective in a print block, and
            # three near-identical medians are a cheap consistency check.
            #
            # ONE WRITER, NOT TWO. The pae-split columns are appended to THIS row rather than
            # written to a file of their own, the way insample_dq was appended: a second writer
            # would mean two header lines and two rows per epoch in the same file whenever both
            # reads were on, and nothing downstream could tell which row carried which. So the
            # ranking columns go out as nan when the ranking block did not run, which is what a
            # PAE-only arm produces, and the header stays fixed for the life of the file.
            #
            # A RESUMED RUN GETS A RAGGED TAIL -- the header was written when the file was
            # created and is not rewritten, so rows appended after this change carry seven more
            # fields than that header names. That is what appending insample_dq already did
            # (training/runs/nad5rc/spread.r0.csv still holds 6-field rows under a 9-field
            # header), and it is the deliberate trade: rewriting a header in place means
            # rewriting the file, and nothing reads these columns positionally.
            if _sp_rank_due or _ps_on:
                _rh = _rel_stat.get("rho") or []
                _sp = os.path.join(a.out, f"spread.r{rank}.csv")
                _new = not os.path.exists(_sp)
                with open(_sp, "a") as _fh:
                    if _new:
                        _fh.write("epoch,n_ifaces,rho,sd_pred,sd_true,sd_ratio,"
                                  "insample_dq,insample_frac,oracle_dq,"
                                  "pae_acc,pae_unacc,n_pae_acc,n_pae_unacc,"
                                  "n_pae_straddle,n_pae_unclassified,pae_acc_minus_unacc\n")
                    _fh.write("%d,%d,%.6f,%.6f,%.6f,%.6f" % (
                        ep + 1, len(_rh), _med(_rh or [float("nan")]),
                        _med(_rel_stat.get("sd_s") or [float("nan")]),
                        _med(_rel_stat.get("sd_g") or [float("nan")]),
                        _med(_rel_stat.get("sd_ratio") or [float("nan")])))
                    _sq = _rel_stat.get("sel_dq") or []
                    _oq = _rel_stat.get("orc_dq") or []
                    _fh.write(",%.6f,%.6f,%.6f" % (
                        (sum(_sq) / len(_sq)) if _sq else float("nan"),
                        (sum(1 for _v in _sq if _v >= 0.23) / len(_sq)) if _sq
                        else float("nan"),
                        (sum(_oq) / len(_oq)) if _oq else float("nan")))
                    _na = _rel_stat["ps_acc_n"]
                    _nu = _rel_stat["ps_un_n"]
                    _ma = (_rel_stat["ps_acc_sum"] / _na) if _na else float("nan")
                    _mu = (_rel_stat["ps_un_sum"] / _nu) if _nu else float("nan")
                    _fh.write(",%.6f,%.6f,%d,%d,%d,%d,%.6f\n" % (
                        _ma, _mu, _na, _nu,
                        _rel_stat["ps_straddle"], _rel_stat["ps_unclass"], _ma - _mu))
                for _k3 in ("rho", "sd_s", "sd_g", "sd_ratio", "sel_dq", "orc_dq",
                            "bal_nacc", "bal_onesided_ifaces",
                            "first_acc_rank", "score_gap", "headroom", "acc_top1"):
                    _rel_stat[_k3] = []
                _rel_stat["bal_lists"] = 0
                _rel_stat["bal_onesided"] = 0
                # RESET WITH THE REST, at the same boundary, or epoch 2's mean would be epochs
                # 1 and 2 pooled -- a curve that cannot help but look like it is flattening.
                _rel_stat["ps_acc_sum"] = 0.0
                _rel_stat["ps_un_sum"] = 0.0
                for _k4 in ("ps_acc_n", "ps_un_n", "ps_straddle", "ps_unclass", "ps_batched"):
                    _rel_stat[_k4] = 0
            print(f"  [rank {rank}] epoch {ep + 1} accounted {_acc/60:.1f}m of {_ep_s/60:.1f}m, "
                  f"residual {(_ep_s - _acc)/60:.1f}m "
                  f"(exclusive buckets only; gpu_* overlap pose_total, "
                  f"unit_setup overlaps the per-unit leaves)", flush=True)
            # THE RESIDUAL, SPLIT. unit_setup is a SUPERSET of label_open, fingerprint_hash,
            # trunk_read_plus_crop_tokens, crop_slice and loss_feature_prep, so it is deliberately
            # NOT in _EXCL -- including it would double-count the same minutes, which is the trap
            # the comment beside rank_pass_a describes. What it buys is the part of the residual
            # it EXPLAINS: setup time not already claimed by those leaves is inside-unit work, and
            # whatever is left after that is between-unit work, which is a different fix.
            _us = dict((k, v) for k, v, c in _parts if c)
            if _us.get("unit_setup"):
                # THE THREE trunk_* BUCKETS ARE LEAVES TOO, and leaving them out of this sum after
                # instrumenting load_unit would report the very time they now name as still
                # unaccounted -- the measurement would look like it had changed nothing.
                _leaves = sum(_us.get(k, 0.0) for k in
                              ("label_open", "fingerprint_hash", "trunk_read_plus_crop_tokens",
                               "crop_slice", "cpucrop_label", "cpucrop_slice", "cpucrop_tokens",
                               "cpucrop_trunk", "loss_feature_prep", "prefetch_issue",
                               "prefetch_wait", "trunk_npz_open", "trunk_npz_read", "trunk_h2d"))
                print(f"  [rank {rank}] epoch {ep + 1} unit_setup {_us['unit_setup']/60:.1f}m, "
                      f"of which {_leaves/60:.1f}m is already in named leaves -> "
                      f"{(_us['unit_setup'] - _leaves)/60:.1f}m of the residual is INSIDE units, "
                      f"the rest is between them", flush=True)
            if torch.cuda.is_available():
                _gb = 1 << 30
                # UNDER --mem-log THE CUDA COUNTER IS NOT THE EPOCH'S, see _mem_ep.
                _ep_peak = max(_mem_ep[0], torch.cuda.max_memory_allocated()) if a.mem_log \
                    else torch.cuda.max_memory_allocated()
                print(f"  [rank {rank}] epoch {ep + 1} memory: peak allocated "
                      f"{_ep_peak/_gb:.2f} GB, peak reserved "
                      f"{torch.cuda.max_memory_reserved()/_gb:.2f} GB, "
                      f"blocks_per_ckpt={model.confidence_head.pairformer_stack.blocks_per_ckpt}"
                      + (f", ckpt_threshold={a.ckpt_token_threshold} tok -> "
                         f"{_ckpt_hist[0]} unit(s) UNCHECKPOINTED, {_ckpt_hist[1]} checkpointed"
                         if a.ckpt_token_threshold > 0 else ""),
                      flush=True)
            _ckpt_hist[0] = _ckpt_hist[1] = 0
            # EVERY RANK SAVES ITS OWN, at the boundary rather than at exit: x3k2m4 died after
            # epoch 1 and would have discarded all 4,880 hashes it had just computed.
            _fp_cache_save()
            # AND READ BACK WHAT THE OTHER RANKS JUST WROTE. The three ranks jointly cover all
            # 6,903 units in epoch 1, but the shuffle moves a third of them onto a different rank
            # every epoch -- so without this a rank re-hashes, from epoch 2 onward, every unit it
            # has not personally seen, while the hash it needs is already on disk beside its own.
            # Entries this process computed are never replaced; see _fp_cache_load.
            #
            # NOT SYNCHRONISED, AND IT DOES NOT NEED TO BE. Every rank writes its own file
            # atomically via os.replace, so a sibling either sees the previous version or the new
            # one and never a torn one; a miss here costs one re-hash and the next boundary picks
            # it up. os.replace is also why this is safe to do while the siblings are writing.
            _fp_cache_load(label=f"epoch {ep + 1} boundary")
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
                # THE EPOCH'S HEAD, WRITTEN HERE AND NOT BY THE EVALUATOR.
                #
                # IT USED TO BE WRITTEN ONLY WHEN AN EVAL RAN. The one `torch.save` of
                # head_ep<N>_step<S>.pt sits INSIDE the evaluation function, before it spawns
                # phase_k_eval -- so under NOEVAL=1 the function is never called and NO
                # epoch-boundary head was ever written. The live arm mqird6 has only
                # head_step250/500/750/1000.pt from the periodic --save-every path, and at 230
                # steps per epoch those land at arbitrary points MID-epoch: there is no head
                # checkpoint corresponding to any completed epoch of that run.
                #
                # THE HEAD IS THE ARTIFACT AND THE EVAL IS ONE CONSUMER OF IT. Coupling them meant
                # turning off the consumer silently cost the artifact. Nothing was lost on the
                # current run -- resume_ep<N>.pt is written unconditionally just above and contains
                # the head -- but it has to be extracted, which is exactly the friction this
                # removes.
                #
                # SAME NAME, so every downstream scorer, plot and eval_dirs reader keeps working,
                # and the evaluator's own save becomes a no-op rewrite of an identical file rather
                # than the only copy. ~79 MB per epoch against a 3,072 GiB quota.
                _ep_cp = os.path.join(a.out, f"head_ep{ep + 1}_step{step}.pt")
                try:
                    torch.save(_head_ckpt_tensors(), _ep_cp)
                    print(f"  saved {_ep_cp} "
                          f"({os.path.getsize(_ep_cp)/2**20:.0f} MiB: the epoch's head)",
                          flush=True)
                except OSError as _e:
                    # A CHECKPOINT IS WORTH REPORTING AND NOT WORTH THE RUN, the same bargain
                    # _fp_cache_save makes. The resume bundle above already carries the head.
                    print(f"  WARNING: could not write {_ep_cp} ({_e}); the epoch's head is still "
                          f"inside resume_ep{ep + 1}.pt", flush=True)
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
                _flush(traj_rank_p, traj_cols, list(rows))
                if is_main:
                    _flush(traj_p, traj_cols, rows)
                rows.clear()
                if is_main:
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
                _flush(traj_rank_p, traj_cols, list(rows))
                if is_main:
                    _flush(traj_p, traj_cols, rows)
                rows.clear()
                if is_main:
                    eval_checkpoint(step, ep + 1)
                if world > 1:
                    dist.barrier()
            if (a.train_eval_pool and a.train_eval_every
                    and (ep + 1) % a.train_eval_every == 0 and step > ep_start_step):
                # SAME BARRIER DISCIPLINE as the validation path above: one evaluator, sharded
                # over every card, the other ranks parked for its duration.
                if world > 1:
                    dist.barrier()
                _flush(traj_rank_p, traj_cols, list(rows))
                if is_main:
                    _flush(traj_p, traj_cols, rows)
                rows.clear()
                if is_main:
                    eval_checkpoint(step, ep + 1, train_pool=a.train_eval_pool,
                                    )
                if world > 1:
                    dist.barrier()
            # EVERY EXTRA PASS, each on its own cadence. SAME BARRIER DISCIPLINE: one evaluator
            # sharded over every card while the other ranks park, so these must not interleave.
            for _ee in _EXTRA_EVAL:
                if not ((ep + 1) % _ee["every"] == 0 and step > ep_start_step):
                    continue
                if world > 1:
                    dist.barrier()
                _flush(traj_rank_p, traj_cols, list(rows))
                if is_main:
                    _flush(traj_p, traj_cols, rows)
                rows.clear()
                if is_main:
                    print(f"  extra eval '{_ee['name']}': pool {_ee['pool']}, budget "
                          f"{_ee['budget'] or 'default'}, every {_ee['every']} epoch(s), "
                          f"uncropped", flush=True)
                    eval_checkpoint(step, ep + 1, train_pool=_ee["pool"],
                                    tag=_ee["name"],
                                    fold=_ee["fold"], seed_base=_ee["seed_base"],
                                    budget=_ee["budget"])
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
        # EVERY RANK, including rank 0. _flush clears nothing, so the same rows go to both files;
        # the final clear below is what retires them.
        if rows:
            _flush(traj_rank_p, traj_cols, list(rows))
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
    # THE HORIZON MUST COME FROM THE EPOCH BUILDER, NOT FROM len(flat) // world. build_epoch
    # takes floor(n_seeds / world) WHOLE seed-groups of each structure, so at ten seeds and world 3
    # one seed per structure sits out and the rank walks 3 groups, not 10/3. len(flat) // world
    # counts that sitting-out seed, and the error lands on the LEARNING RATE: at 368 structures it
    # claimed 205 steps per epoch against an actual 184, which stretches the decay window by 11%.
    # The configured "1e-2 flat to epoch 20, 1e-4 by epoch 50" came out as flat to 22.3 and 1e-4 at
    # 55.7, leaving lr ~7.95e-4 at epoch 50 -- eight times the intended value -- so the schedule
    # the arm is testing was not the schedule it ran. Caught in review of 38a83283.
    _units_by_pdb = {}
    for _p, _lp, _c in flat:
        _units_by_pdb[_p] = _units_by_pdb.get(_p, 0) + 1
    _units_per_rank = _LRS.units_per_rank(_units_by_pdb, world,
                                          bool(a.rank_cross_seed), len(flat),
                                          cover_all=bool(a.epoch_cover_all))
    _spe_ep = _LRS.steps_per_epoch(_units_by_pdb, world, a.nsample, accum_local,
                                   cover_all=bool(a.epoch_cover_all),
                                   cross_seed=bool(a.rank_cross_seed), n_units=len(flat),
                                   structs_per_step=(a.rank_structs_per_step
                                                     if a.rank_cross_seed else 0),
                                   preserve_order=(a.epoch_order in ("manifest", "tokens")))
    # THE SPLIT, REPORTED HONESTLY. _full is whole accumulations and the partial exists only when
    # the poses do not divide accum_local -- at 368 structures they do (1,104 x 5 / 30 = 184), so
    # there is no partial step and the old unconditional "+ 1 partial" overstated the horizon.
    _poses_pr = _units_per_rank * a.nsample
    _full = _poses_pr // max(1, accum_local)
    _part = 1 if _poses_pr % max(1, accum_local) else 0
    _expect = _spe_ep * max(1, math.ceil(a.epochs))
    _naive = ((len(flat) // max(1, world)) * a.nsample // accum_local + 1)
    if _naive != _spe_ep:
        print(f"  steps/epoch {_spe_ep} from the epoch builder, NOT {_naive} from "
              f"len(flat)//world: {len(flat)} unit(s) over {len(_units_by_pdb)} structure(s), of "
              f"which this rank walks {_units_per_rank} "
              f"(floor(seeds/{world}) whole group(s) per structure; the remainder sits out and is "
              f"reseeded each epoch). The lr schedule uses the former.")
    steps_cap = 0
    _lr_horizon[0] = _expect
    # WHAT THE HORIZON WAS BUILT FROM, so build_epoch_cross_seed can refuse if it packs a
    # different number of steps than the schedule was calibrated for.
    _xs_spe[0] = _spe_ep if a.rank_cross_seed and a.rank_structs_per_step else 0
    # --lr-steps IS CHECKED FIRST, because it takes precedence over --lr-final and the decay
    # window (see its help). This line derived the description from lr_final alone, so a run with
    # a staircase printed "flat after warmup (no decay)" while the staircase was in fact applied
    # -- parse_lr_steps' own docstring warns that "the run trains at the wrong lr for twenty
    # epochs and nothing in the log says so", and the startup line was saying exactly nothing.
    if a.lr_steps:
        _sched = "STAIRCASE: " + f"{a.lr:g}" + "".join(
            f" -> {_v:g} from epoch {_e:g}" for _e, _v in _LRS0.parse_lr_steps(a.lr_steps))
    elif a.lr_final <= 0.0:
        _sched = "flat after warmup (no decay)"
    else:
        _sched = f"COSINE {a.lr:g} -> {a.lr_final:g} over steps {a.warmup_steps}..{_expect}"
    print(f"\ntraining: lr {a.lr:g}, warmup {a.warmup_steps}, accum {a.accum}, "
          f"~{_expect} optimiser step(s) for {a.epochs} epoch(s) "
          f"({_full} full + {_part} partial per epoch); no step cap, the epoch loop decides")
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
    if _USE_CROP:
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
                print(f"\ncrop: pool {a.crop_pool} over "
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

    if code_drift.get("stale_n"):
        _sp = sorted(code_drift.get("stale_pdbs", ()))
        print(f"\nSTALE LABELS ACCEPTED on {code_drift['stale_n']} unit(s) across {len(_sp)} "
              f"structure(s) via --allow-stale-labels: {_sp[:8]}"
              f"{' ...' if len(_sp) > 8 else ''}\n"
              f"  The entropy gradient never read them, so the TRAINING is unaffected. The "
              f"labelled columns of trajectory.csv -- pae_loss, pde_loss, plddt_loss, "
              f"resolved_loss, loss -- were computed against labels whose bioassembly provenance "
              f"no longer matches, and are NOT a diagnostic for this run. pae_entropy is.")
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
