"""`trainer/optimiser.py` -- Build AdamW over exactly the trainable set.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 3439-3450 (source `d34baebc`).
Statements and comments are the original; only the names this closed over inside
main() became parameters. test_optimiser_equivalence.py compares it line for line,
so a change to the live trainer fails the test instead of drifting silently.
"""
import os
import math
import time

import torch
import torch.distributed as dist


def build_optimiser(ctx, a, model, trainable):
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
    return opt, _drift_history
