"""`trainer/constants.py` -- the module-level block of `phase_k_train.py`.

EXTRACTED VERBATIM from lines 1-107 (source `7fc6870c`): the imports every stage shares,
ROOT, TERMS, and the two module-level helpers `_git_commit` and `head_param_names`.

These were the last 69 unwritten context attributes. They were invisible to me because I
was reading `main()` -- and they are not in it. `head_param_names` in particular is the
function that derives the trainable set from the module tree, which is the reason every
injection must be installed before it runs.
"""
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




# THE DEFERRED IMPORTS main() DID INSIDE ITS OWN BODY (phase_k_train.py:1690-1694, 2455).
# They are imported late there so a --help or a refusal never pays for loading Protenix. Every
# stage reads them off the context, so they are bound here and exported with the rest. These
# were the last names with no writer, and they were invisible to a search for assignments
# because an `import x as Y` binds Y without ever looking like `Y = `.
import phase_j_dataset as DS                                        # noqa: E402
import phase_j_build_cache as BC                                    # noqa: E402
import phase_j_rescore_cache as RC                                  # noqa: E402
import rank_loss as RKL_probe                                       # noqa: E402
from protenix.model.loss import ProtenixLoss                        # noqa: E402
from protenix.utils.permutation.permutation import SymmetricPermutation  # noqa: E402
