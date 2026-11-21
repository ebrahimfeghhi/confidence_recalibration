#!/usr/bin/env python
"""`trainer/main.py` -- the entry point: wire the stages, in the order the invariants require.

WHAT THIS REPLACES. `phases/phase_k_train.py` is 8,513 lines of which `main()` is 8,402. The
stages below are that function, extracted region by region, each one compared line for line
against its source by `test_region_equivalence.py`. The order here is not a style choice: it is
the order the source enforces, and several steps are load-bearing for reasons that cost GPU time
to learn (see each module's header).

NO FORWARD PASS HAS EVER EXECUTED FROM THIS PACKAGE.

Every stage is wired and all 8 CPU tests pass, including `test_context_contract.py`: every
`rt.<attr>` the package reads is written by some earlier stage. That is a necessary condition
for running, not a sufficient one. The gate is still a real execution -- a CPU smoke through
dataset resolution, one unit, a PAE and rel score and a forward -- and then the paired
original-vs-copy ladder.

Twice before this, a green suite was reported as readiness. It was not, because parsing, name
resolution and extraction fidelity are all true of a skeleton.

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
import constants                                # noqa: E402
import cli                                      # noqa: E402
import config as config_mod                     # noqa: E402
import ddp                                      # noqa: E402
import bootstrap                                # noqa: E402
import dataset                                  # noqa: E402
import trainable as trainable_mod               # noqa: E402
import optimiser                                # noqa: E402
import resume as resume_mod                     # noqa: E402
import loop                                     # noqa: E402
import runtime as runtime_mod                   # noqa: E402
import derive as derive_mod                     # noqa: E402
import unit as unit_mod                         # noqa: E402
import evaluate as evaluate_mod                 # noqa: E402
import step_plan                                # noqa: E402


def main(argv=None):
    a = cli.parse(argv)
    config_mod.validate(a) if hasattr(config_mod, "validate") else None

    # setup_dist runs BEFORE the real context exists -- it is what decides this rank's device --
    # so it fills a scratch Ctx whose attributes are folded into the real one below.
    rt0 = Ctx(a=a)
    rank, world, local_rank, is_main, dist, dev = ddp.setup_dist(rt0, a)
    rt = Ctx(a=a, rank=rank, world=world, local_rank=local_rank,
             is_main=is_main, dist=dist, dev=dev)

    # THE MODULE-LEVEL BLOCK FIRST. ROOT, TERMS, head_param_names and the shared imports were
    # never inside main(), which is why they were the last names with no writer -- I was reading
    # the function and they are above it.
    for _n in [k for k in vars(rt0) if not k.startswith("__")]:
        setattr(rt, _n, getattr(rt0, _n))
    for _n in dir(constants):
        if not _n.startswith("__"):
            setattr(rt, _n, getattr(constants, _n))

    # THE DERIVED VALUES BEFORE ANYTHING READS THEM: the lr staircase, the token-drop schedule,
    # the extra-eval spec, the trajectory paths. config.py holds this region's REFUSALS; these
    # are its DERIVATIONS, and they were left in main() when config.py was culled.
    derive_mod.derive(rt, a)

    # THE HEAD, AND EVERY MUTATION OF IT, BEFORE THE TRAINABLE SET IS DERIVED.
    # head_param_names() reads the module tree rather than a hand-written list, so an injection
    # installed after it would be created and never trained -- silently, since the shapes line up.
    # What the head build and everything after it read. _LRS is NOT here: it is set by
    # the dataset stage further down, and demanding it at this point was my error, not
    # a missing export -- the gate is a claim about order and has to be accurate.
    rt.require("TERMS", "ROOT", "head_param_names", "_LR_STEPS")

    rt.model, rt.configs, rt._dead_head = bootstrap.build_head(rt, a, dev)

    # THE RESOLVED DATASET GOES ONTO THE CONTEXT. A sed of mine removed this call while deleting
    # a stale line beside it, leaving `dataset` imported and never used -- caught in review, not
    # by any test here, which is why test_context_contract.py now exists.
    for _k, _v in dataset.resolve_training_dataset(rt, a, is_main=is_main).items():
        setattr(rt, _k, _v)

    # ORDER, NOT MERE PRESENCE. The contract test proves a writer exists SOMEWHERE; these prove
    # it ran BEFORE its reader on this configuration. Review on issue #9 made that distinction,
    # and it is the difference between a static scan and an initialisation failure.
    rt.require("model", "configs", "loss_fn")

    rt.prefixes, rt.trainable, rt._dops = trainable_mod.select_trainable(rt, a, rt.model)
    rt.require("prefixes", "trainable")
    rt.opt, rt._drift_history = optimiser.build_optimiser(rt, a, rt.model, rt.trainable)

    rt.start_epoch, rt.start_step = resume_mod.load_resume(rt, a, rt.model, rt.opt)

    # THE GLUE BETWEEN THE STAGES, LAST, because it reads the model, the trainable set and the
    # optimiser that the stages above build. Inside main() this was simply the rest of the
    # function body; here it is explicit, and everything it binds is exported onto the context.
    rt.require("opt", "start_epoch", "start_step")
    runtime_mod.build_runtime(rt, a)

    # THE CALLBACKS loop.py REACHES THROUGH THE CONTEXT. Inside main() these were nested defs
    # closing over its scope; here each is bound explicitly so the loop's dependencies on them
    # are visible rather than lexical.
    rt.run_unit = lambda *args, **kw: unit_mod.run_unit(rt, *args, **kw)
    rt.eval_checkpoint = lambda *args, **kw: evaluate_mod.eval_checkpoint(rt, *args, **kw)
    rt.val_loss_checkpoint = lambda *args, **kw: evaluate_mod.val_loss_checkpoint(rt, *args, **kw)
    rt.build_epoch_cross_seed = lambda *args, **kw: step_plan.build_epoch_cross_seed(*args, **kw)

    # THE SIGNATURE IS THE LIVE ONE. train_loop takes the step cap, the base lr, a tag, whether
    # this rank writes the shared trajectory, and where to resume from. Calling it with the
    # context alone was a wiring bug the bloat audit caught -- a TypeError on the first run.
    # THE LOOP'S OWN INPUTS, checked at the boundary rather than at first read 900 steps in.
    rt.require("run_unit", "eval_checkpoint", "build_epoch_cross_seed",
               "traj_p", "traj_cols", "accum_eff", "accum_local", "_T", "_TN")

    return loop.train_loop(rt,
                           steps_cap=0,
                           lr=a.lr,
                           tag="train",
                           write_traj=is_main,
                           from_epoch=rt.start_epoch,
                           from_step=rt.start_step)


if __name__ == "__main__":
    raise SystemExit(main())
