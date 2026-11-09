#!/usr/bin/env python
"""`trainer/main.py` -- the entry point: wire the stages, in the order the invariants require.

WHAT THIS REPLACES. `phases/phase_k_train.py` is 8,513 lines of which `main()` is 8,402. The
stages below are that function, extracted region by region, each one compared line for line
against its source by `test_region_equivalence.py`. The order here is not a style choice: it is
the order the source enforces, and several steps are load-bearing for reasons that cost GPU time
to learn (see each module's header).

NOT YET A CUTOVER. `phases/phase_k_train.py` remains the arm that runs experiments and the
oracle this copy is checked against. Per the review on issue #9 the gate before any GPU switch is
a paired original-vs-copy run: Stage A compares the first optimiser step -- pose order, per-pose
logits, every loss term, per-parameter gradients and AdamW moments; Stage B a 3-rank
checkpoint-resume; Stage C a short epoch on a bounded fixture. None of those have run.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # phases/, for pose_* and the model modules
sys.path.insert(0, "/home/ebrahim/Protenix")

from context import Ctx                         # noqa: E402
import cli                                      # noqa: E402
import config as config_mod                     # noqa: E402
import ddp                                      # noqa: E402
import bootstrap                                # noqa: E402
import dataset                                  # noqa: E402
import trainable as trainable_mod               # noqa: E402
import optimiser                                # noqa: E402
import resume as resume_mod                     # noqa: E402
import loop                                     # noqa: E402


def main(argv=None):
    a = cli.parse(argv)
    config_mod.validate(a) if hasattr(config_mod, "validate") else None

    rank, world, local_rank, is_main, dist, dev = ddp.setup_dist(a)
    ctx = Ctx(a=a, rank=rank, world=world, local_rank=local_rank,
              is_main=is_main, dist=dist, dev=dev)

    # THE HEAD, AND EVERY MUTATION OF IT, BEFORE THE TRAINABLE SET IS DERIVED.
    # head_param_names() reads the module tree rather than a hand-written list, so an injection
    # installed after it would be created and never trained -- silently, since the shapes line up.
    ctx.model, ctx.configs, ctx._dead_head = bootstrap.build_head(a, dev)

    ctx.dataset = dataset.resolve_training_dataset(ctx, a, is_main=is_main)

    ctx.prefixes, ctx.trainable, ctx._dops = trainable_mod.select_trainable(ctx, a, ctx.model)
    ctx.opt, ctx._drift_history = optimiser.build_optimiser(ctx, a, ctx.model, ctx.trainable)

    ctx.start_epoch, ctx.start_step = resume_mod.load_resume(ctx, a, ctx.model, ctx.opt)

    return loop.train_loop(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
