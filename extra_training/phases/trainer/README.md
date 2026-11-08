# `trainer/` — a minimal copy of the trainer, built piece by piece

Nothing here runs the experiment. `phases/phase_k_train.py` does. This is a separate copy being
assembled with a verification step per piece, so cutover is a later, deliberate decision.

## The vocabulary, because nothing below parses without it

| term | what it is |
|---|---|
| **structure** | one antibody–antigen complex, e.g. `9uax`. 849 of them in training. |
| **pose** | one predicted 3D structure for it. Scored against truth by **DockQ** (0–1, ≥0.23 is "acceptable"). |
| **seed** | one folding run. Each produces **5 samples**, so a seed is 5 poses. |
| **unit** | one (structure, seed) pair — the cached trunk the head reads. `9uax/free_500012`. |
| **interface** | one chain pair inside a structure, e.g. `9uax_AB`. What DockQ is measured on. |
| **budget 30** | 30 seeds × 5 samples = **150 poses** per interface to choose from. |
| **ipTM** | the model's own confidence. **Selection is an argmax of ipTM within one interface.** |
| **oracle** | the DockQ of the best pose available. The ceiling. |
| **pass A / pass B** | every pose is forwarded twice: A under `no_grad` to get the scores the relative loss needs, B to rebuild and backward it. |
| **cross-seed step** | all seeds of a structure in ONE optimiser step, because a pose's loss coefficient depends on every other pose in its interface. |
| **rank** | one of 3 GPU processes. Each walks a third of the units; gradients are all-reduced. |

## What the thing actually does, start to finish

1. A **unit** is a cached trunk: the frozen 464M-parameter model already ran, and its pair
   representation `z` is on disk.
2. The **confidence head** reads `z`, plus the pose's geometry injected inside pairformer block 3,
   and predicts a **PAE** distribution.
3. **ipTM** is computed from that PAE — one number per pose per interface.
4. Two losses: **PAE** against the true error, and **rel**, which regresses mean-centred ipTM onto
   mean-centred true iTM *within* an interface. Only `rel` can teach ranking; PAE dominates the
   gradient.
5. 30 structures' worth of poses accumulate into one optimiser step. Only **20 tensors** move.
6. Every 5 epochs the head is scored on 83 held-out structures: pick the argmax-ipTM pose per
   interface, report its DockQ.

The whole experiment is step 4 — whether a ranking term makes step 6's argmax pick better poses.

## Files

| file | what it owns | verified against |
|---|---|---|
| `cli.py` | 92 flags + `parse()`, including `--labels` defaulting to `<fold>/labels` | 91/91 fields of a real run's `args.json` |
| `config.py` | cross-flag validation, 26 refusals | the same 91 after `validate()`, plus 6 refusal messages |
| `checkpoints.py` | the resume-guard decision | 12 real resume bundles reading as continuations |
| `step_plan.py` | which units each rank walks in which step | the unit order three ranks actually walked |
| `inventory.py` | AST map of every closure in `main()` | — (a tool) |
| `import_graph.py` | reachability from the live entry points | 12 known-live files all reached |

## How each piece is verified, and why it is done this way

**Never by diffing extracted code against a copy of itself.** That proves the copy is a copy.

Every run writes `args.json` from the values the *original* parser produced, `trajectory_rank<N>.csv`
from the units the *original* planner dealt out, and `resume_ep<N>.pt` from the guard the *original*
assembled. Those are ground truth the new code did not write, and each test checks against one.

It catches things reading does not. `--rank-tau` defaults to `None` while the guard records the
derived sentinel `-1.0`, so `args.json` is not a valid oracle for derived fields — found by a test,
not by inspection.

## Scope

**PAE + `rel` only.** 56 flags are gone: LoRA, all cropping, the tau-based ranking forms, PAE
row-weighting, L2-SP, per-block pose injection, frame direction, the probes, cosine/exponential
decay. `--val-loss-*` stays.

**No backwards compatibility.** The copy's own bundles resume; bundles written by the original do
not. That collapses the guard from 64 fields to 32 and removes the migration machinery entirely.
Cutover therefore means starting a fresh arm, not switching one mid-flight.

## What is NOT done

`main()` is still 8,380 lines. Ported so far: the config surface and the step plan. The bootstrap
(manifest resolution, model and optimiser construction), the step (`run_unit`, `_one_pose`,
`_apply_step`) and `train_loop` are not.

Everything remaining touches gradients or the optimiser, and the only thing that would prove a port
did not change the maths is a **paired GPU run** — original and copy from the same bundle and seed,
comparing per-step losses and head tensors. The config tests say nothing about that, and no CPU
test can.
