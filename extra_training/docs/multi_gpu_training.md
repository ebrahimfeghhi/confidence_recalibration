# Multi-GPU FT3 training

**Notation.** A *rank* is one training process holding one GPU. *accum* is the number of poses whose
gradients are summed before one optimiser step (the "batch"). *accum_local* is a rank's share of it;
*accum_eff* is `accum_local x world_size`, the poses that actually contribute to a step. The *head*
is `confidence_head`, 19.8M of the model's 464.4M parameters; everything else is frozen and in
`.eval()`, which is what makes this FT3.

## Why not DistributedDataParallel

DDP wraps a module's `forward`. Ours is `RC.confidence(model, configs, feat, s_in, s_tr, z, x, nc)`
over a frozen trunk, so DDP would both fight the custom call and expect gradients for the 444M
parameters that deliberately have none. Instead each rank holds an identical replica — same
checkpoint, so identical initial parameters — and only the head gradients are summed per step.

After the all-reduce every rank holds bit-identical gradients, so the clip, the non-finite decision,
the AdamW state and therefore the parameters stay in lockstep with no parameter broadcast and no
further communication.

## The three things that make it equivalent to one GPU

| requirement | what goes wrong otherwise |
|---|---|
| each pose scaled by `accum_eff`, not by `accum_local` | the gradient is `world_size` too large, which reads as a learning rate that is merely too high |
| the all-reduce happens **before** `clip_grad_norm_` | clipping partial gradients then summing is a different function from clipping the sum |
| every rank takes the same number of steps | the first mismatched collective hangs forever instead of failing |

The third is enforced twice: shards are **whole structures truncated to an equal count**, and every
unit is asserted to hold `--nsample` poses. 500 structures over 3 ranks is 167/167/166, so 2
structures are dropped per epoch; the per-epoch reshuffle makes it a different 2 each time rather
than excluding the same data every epoch.

Sharding is by structure, not by unit, because `--shuffle=structure` keeps a structure's units
consecutive so one built dataset item serves all of them. Sharding by unit would make every rank
build every structure and multiply the item cache by the world size for nothing.

## gloo, not NCCL

| | |
|---|---|
| NCCL on this box | **broken**: a 20-line two-rank `all_reduce` on cards 1 and 3 fails with `CUDA error: an illegal memory access was encountered`, with P2P enabled and disabled alike |
| versions | NCCL 2.26.2+cuda12.2, torch 2.7.1+cu128, driver 595.99.02 |
| reproduces without the model | yes — it is not this code |
| gloo cost, measured | 60 ms per 79 MB all-reduce, ~6 s per epoch over ~98 steps |

So `--dist-backend` defaults to **gloo**. It is a CPU backend, so the head's 248 gradient tensors are
flattened into one buffer, staged through host memory, reduced, and copied back — one collective per
step rather than 248. Switch to `nccl` if the driver/NCCL pairing is ever fixed.

## Memory: the cap is per process

| | |
|---|---|
| a dataset item | 331–886 MB, avg 518 (measured) |
| a rank's shard | ~167 structures, ~86 GB held |
| per-rank cap used | `--preload-items 170 --max-rss-gb 95` |
| three ranks together | under the 300 GB the user asked us not to exceed |

An earlier note in `queue_ft3_pilot.sh` claimed a "~110 GB baseline" per process. That was an
estimate with no measurement behind it and it is **wrong**: the evaluator, which builds the same
464M-parameter model, holds 4.1 GB. What costs memory is the item cache, not the model. Holding 200
items per rank would have been ~312 GB, and a 300 GB cap applied *per rank* would have permitted
900 GB.

## The collective timeout must exceed the evaluation

`torch`'s default process-group timeout is **30 minutes**. The per-epoch evaluation at
`--eval-budget 100` covers all 5,300 validation units and takes **~73 minutes**, measured on GATE 1
(06:52 to 08:05 PT on 2026-10-16). Rank 0 evaluates while the other ranks wait at a `dist.barrier()`,
so on the default the barrier times out and kills the run at the FIRST epoch boundary -- after hours
of correct training, and the failure reads as a network fault rather than a timeout somebody chose.

`--dist-timeout-min` defaults to 240, which leaves room for an evaluation that runs long under
contention. Found before the first multi-GPU run started, not after it died.

## Evidence

### The first version of the selftest was not evidence

It compared the two runs' final parameters against a tolerance relative to the PARAMETER magnitude:
`1e-4 x 1.04e+02 = 1.04e-02` absolute. One Adam step at lr 1e-3 moves a parameter by about 1e-3
(measured: 9.999e-04). So a multi-GPU run that took **no step at all** differed from the single-GPU
run by ~1e-3 and PASSED, by a factor of ten.

Worse for requirement 1: Adam's update is `m_hat/sqrt(v_hat)`, which is invariant to a constant
factor on the gradient, so after one step a gradient scaled by `world_size` produces very nearly the
same parameters. The thing the test most needed to catch was the thing it was least able to see.
Raised in review on 2026-10-16.

### What it checks now

| check | measured | what a bug scores |
|---|---|---|
| 0. the lr-0 run is a fixed point | 9.999e-04 displacement | 0, and nothing below means anything |
| 1. pre-clip gradient norm, 1 GPU vs 2 | 0.000736 both, **2.4e-07** relative (tol 1e-4) | **1.0** for a per-rank loss scale |
| 2. update from the starting head | **1.04e-04** relative (tol 1e-3) | **1.0** for a run that took no step |
| 3. the ranks after 5 steps | **0.000e+00** — bit-identical (tol 1e-9) | >0 for a missed all-reduce |
| writers of `head_final.pt` | 1 | 2 before the rank-0 guard |

Check 1 is the one that covers requirement 1, and no parameter comparison can substitute for it:
`clip_grad_norm_` returns the norm BEFORE clipping, which the trainer records per step as
`grad_norm`, and that number is linear in the gradient, so Adam's normalisation cannot hide a scale
error in it.

Check 2 compares DISPLACEMENTS from the starting head, with the tolerance relative to the
displacement, so a do-nothing run scores 1.0 instead of passing. The test computes that negative
control explicitly and fails if it would have passed.

Note the margins: check 1 passes with 400x of room, check 2 with only 10x. That is Adam amplifying a
small gradient difference — with a gradient norm of 7.4e-04 spread over 19.8M parameters, `v_hat` is
small enough that `eps` matters, so the step is nearly `lr * sign` and a 2.4e-07 gradient difference
becomes ~1e-04 in the parameters. It is another reason the gradient is the better place to measure.

Check 3 exists because exact agreement with one GPU is only meaningful for a single full-batch step:
with more steps the two runs partition the poses differently -- one GPU takes consecutive poses, the
ranks take one from each structure -- so their trajectories diverge legitimately. What must hold over
many steps is that the RANKS agree with EACH OTHER, and a missed all-reduce shows up there and
nowhere else.

### The saved head must be the evaluated head

The partial accumulation left at the end of an epoch used to be flushed AFTER the epoch loop, which
is after the last evaluation. At production settings that means: evaluate step 292, then take step
293, then write `head_final.pt` -- so the shipped head was one optimiser step away from the only head
whose validation metric existed, and the number reported for the run did not belong to the checkpoint
the run produced. Raised in review on 2026-10-16.

The remainder fires in the multi-GPU case and not the single-GPU one, because `steps_cap` is computed
from the global pose count while each rank gets a TRUNCATED shard: 166 structures rather than 166.67,
so 294 steps are budgeted and 292 are reached, leaving 80 poses accumulated. On one GPU `steps_cap`
equals the complete steps the data supports, so the loop breaks with an empty accumulator.

Three fixes, all in `_apply_partial`:

| | |
|---|---|
| flushed inside the epoch loop, before the evaluation | the evaluated head is the saved head |
| gradient scaled by `accum_eff / n_contributing`, the count all-reduced rather than assumed | the last step of an epoch is the same size as every other, not a fraction of one |
| `_apply_step`'s success flag respected | a non-finite gradient no longer advances the step counter |

Verified on 3 structures over 2 ranks, where the shard truncation bites: `flushing a partial
accumulation of 2/4 pose(s), gradient scaled by 2.0000`, the count reduced across ranks, and no
warning after the loop.

### Other bugs the selftest caught

**Both ranks wrote `head_final.pt`.** The content was identical because the replicas agree, but two
processes writing one path can interleave and leave a truncated checkpoint. Checkpoint writes are now
rank 0 only, with `--debug-save-rank-heads` for the selftest's per-rank comparison.

**The sweep was missing `--allow-code-drift`** and would have stopped on its first unit.

## Expected speed

One epoch measured 7.9 h on a single card (25,000 poses at ~1.0 s plus 500 structure switches at
~7.1 s). On three ranks the step count is unchanged and the poses are split three ways, so ~2.6 h is
expected. The sweep runs `--epochs 3` at one learning rate, per the user's direction on 2026-10-16.
