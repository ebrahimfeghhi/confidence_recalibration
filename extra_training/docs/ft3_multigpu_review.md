# FT3 multi-GPU training: review brief

**Review round 1 is addressed.** Both findings were correct and both are fixed; see "Round 1" below
for what changed and what the fixes measure. Commit `c186e4b5` on `main`, fixes on top of
`fd27f238`.

**For review.** Originally commit `c186e4b5` on `main`. `docs/multi_gpu_training.md` is canonical for the
measurements and the design; this file is the review-facing summary and says what to attack. Where a
number appears in both, that file is the one to trust.

**Notation.** A *rank* is one training process holding one GPU. *accum* is the number of poses whose
gradients are summed before one optimiser step. *accum_local* is a rank's share; *accum_eff* is
`accum_local x world_size`, the poses that actually contribute to a step. The *head* is
`confidence_head`, 19.8M of the model's 464.4M parameters; everything else is frozen and in
`.eval()`, which is what makes this FT3.

## What changed

FT3 confidence-head training now runs one learning rate across three GPUs instead of one, taking the
same optimiser steps in a third of the wall clock: 7.9 h per epoch measured on one card, ~2.6 h
expected on three.

| file | change |
|---|---|
| `phases/phase_k_train.py` | multi-GPU: rank-aware structure sharding, summed head gradients, rank-0-only writes, `--dist-backend`, `--nsample` |
| `phases/phase_k_train_ddp_selftest.py` | new. Proves the multi-GPU gradient equals the single-GPU one |
| `shell_scripts/queue_ft3_pilot.sh` | sweep launches via `torchrun --nproc_per_node=3`; `N_LR=1`; per-rank RAM caps; `--allow-code-drift` |
| `docs/multi_gpu_training.md` | new. The measurements |

`N_LR=1` and three GPUs are both at the user's direction on 2026-10-16: *"Just run 1 lr for 3 epochs
using 3 GPUs I want to make sure that works."*

## The design, in one paragraph

Not `DistributedDataParallel`. The forward is `RC.confidence(model, configs, feat, s_in, s_tr, z, x,
nc)` over a frozen trunk, so DDP would both fight that custom call and expect gradients for the 444M
parameters that deliberately have none. Instead every rank holds an identical replica — same
checkpoint, so identical initial parameters — and only the head gradients are summed per step. After
the all-reduce every rank holds bit-identical gradients, so the clip, the non-finite decision, the
AdamW state and therefore the parameters stay in lockstep with no parameter broadcast. The sum is a
sum rather than a mean because each pose is already divided by `accum_eff` at `backward()`.

## What makes it equal to one GPU

Three requirements, each of which fails silently if wrong.

| requirement | what goes wrong otherwise | enforced by |
|---|---|---|
| each pose scaled by `accum_eff`, not `accum_local` | the gradient is `world_size` too large, which reads as a learning rate merely too high | `(cum / accum_eff).backward()` |
| the all-reduce happens **before** `clip_grad_norm_` | clipping partial gradients then summing is a different function from clipping the sum | order inside `_apply_step` |
| every rank takes the same number of steps | the first mismatched collective hangs forever instead of failing | shards are whole structures truncated to an equal count; every unit is asserted to hold `--nsample` poses |

## Evidence

`phases/phase_k_train_ddp_selftest.py` is constructed so the answer is known in advance. Two
structures with one seed each are 2 units of 5 poses = 10 poses:

- one GPU, `--accum 10` → one step over all 10 poses
- two GPUs, one structure per rank → `accum_local` 5, `accum_eff` 10, one step over the same 10

Same poses, same single step, so the final head must match.

| check | measured | what a bug scores |
|---|---|---|
| 0. the lr-0 run is a fixed point | 9.999e-04 displacement | 0, and nothing below means anything |
| 1. pre-clip gradient norm, 1 GPU vs 2 | 0.000736 both, **2.4e-07** relative (tol 1e-4) | **1.0** for a per-rank loss scale |
| 2. update from the starting head | **1.04e-04** relative (tol 1e-3) | **1.0** for a run that took no step |
| 3. the ranks after 5 steps | **0.000e+00** — bit-identical (tol 1e-9) | >0 for a missed all-reduce |
| writers of `head_final.pt` | 1 | 2 before the rank-0 guard |

Run it with `python -I phases/phase_k_train_ddp_selftest.py --cards 1,3`; it needs two idle cards and
leaves GPU 0 alone by default. The four runs are INIT (lr 0, the starting head), ONE (1 GPU, full
batch), TWO (2 GPUs, same full batch) and MANY (2 GPUs, 5 steps, every rank writing its own head).

Two real bugs it caught before the sweep could, each of which would have cost a run:

1. **Both ranks wrote `head_final.pt`.** Identical content, since the replicas agree, but two
   processes writing one path can interleave and leave a truncated checkpoint. Writes are now rank 0
   only.
2. **The sweep was missing `--allow-code-drift`** and would have stopped on its first unit. The label
   fingerprint covers whole files, and `phase_j_build_cache.py` gained `resolve_seed_dir` in
   `2f6d2f92`, so all 5,000 labels now differ in the `code` field and in nothing else — cif, bio,
   idx, trunk, loss and schema all match.

## A retraction

An earlier comment in `queue_ft3_pilot.sh` claimed a "~110 GB baseline" per training process, and I
used that figure to argue three GPUs were infeasible. It was an estimate with no measurement behind
it and it is **wrong**: the evaluator, which builds the same 464M-parameter model, holds 4.1 GB RSS.
What costs memory is the dataset-item cache (331–886 MB per structure, avg 518), not the model. The
stale comment is replaced in this commit.

The correct constraint is that `--max-rss-gb` is **per process** while the user's 300 GB is a
whole-box figure, so each rank gets `--preload-items 170 --max-rss-gb 95`. Holding 200 items per rank
would have been ~312 GB, and the old 300 GB cap applied per rank would have permitted 900 GB.

## Round 1: the two findings, and what the fixes measure

**1. The equivalence test could pass even if distributed training did nothing.** Correct, and the
arithmetic was as stated: the tolerance was 1e-4 relative to a mean parameter magnitude of 1.04e+02,
so 1.04e-02 absolute, while one Adam step at lr 1e-3 moves a parameter by ~1e-3 — since measured at
9.999e-04. A no-op passed by a factor of ten. The deeper problem was that Adam's `m_hat/sqrt(v_hat)`
is invariant to a constant factor on the gradient, so a `world_size` scale error — the exact failure
requirement 1 exists to prevent — is nearly invisible in a one-step parameter comparison.

Rewritten as four checks (table above): the pre-clip **gradient norm**, which is linear in the
gradient and so immune to Adam's normalisation; the **displacement** from the starting head with the
tolerance relative to the displacement; an explicit **negative control** that fails the test if a
do-nothing run would have passed; and **rank-to-rank agreement over 5 steps**, which is the only
check that can see replicas drifting.

Exact agreement with one GPU over MANY steps is not achievable and is not claimed: the two runs
partition the poses differently, so their trajectories diverge legitimately. That is why the
multi-step check is between ranks.

**2. The final head differed from the last evaluated head.** Correct. The partial accumulation was
flushed after the epoch loop, i.e. after the last evaluation: evaluate step 292, take step 293, save
the final head. Fixed on all three counts — flushed inside the epoch loop before the evaluation,
scaled by `accum_eff / n_contributing` with the count all-reduced rather than assumed, and
`_apply_step`'s success flag respected so a non-finite gradient no longer advances the counter.
Verified on 3 structures over 2 ranks, where the shard truncation makes the remainder appear:
`flushing a partial accumulation of 2/4 pose(s), gradient scaled by 2.0000`, and no warning after the
loop.

**3. Found while fixing the above, not raised in review.** `torch`'s default collective timeout is 30
minutes. The per-epoch evaluation covers 5,300 validation units and takes **~73 minutes**, measured
on GATE 1. Rank 0 evaluates while the other ranks wait at a barrier, so on the default the barrier
would have timed out and killed the run at the FIRST epoch boundary, after hours of correct training,
with the failure reading as a network fault. `--dist-timeout-min` now defaults to 240.

Point 3 is the one worth drawing a lesson from: the two review findings were both about the final
checkpoint, and chasing them surfaced a third defect that would have destroyed the run outright.

## What to attack first

The weakest points, stated so they get looked at rather than skipped.

1. **The multi-step check is between ranks, not against one GPU.** Rank agreement after 5 steps is
   exact (0.000e+00), but it cannot show that the 3-rank trajectory is a good optimisation, only that
   the replicas stay together. Nothing here establishes that 3 GPUs reach the same validation metric
   as 1 GPU would over 3 epochs, and the pose partitioning differs, so they will not take identical
   paths. If that matters, the comparison has to be the metric, not the parameters.
2. **2 structures of 500 are dropped per epoch** to keep the ranks equal (167/167/166 truncated to
   166). Reshuffling makes it a different 2 each epoch, but it is a real 0.4%, and padding rather
   than truncating is the alternative.
3. **`accum_eff` is 255, not 256**, since 256 does not divide by 3. The loss is scaled by 255 so the
   gradient is a true mean over the poses that contributed, but the effective batch is not the number
   anyone wrote down.
4. **The per-epoch evaluation serialises, and it is expensive.** Rank 0 evaluates for **~73
   minutes** (measured: 73 units/min over 5,300 units) while the other two ranks wait at a barrier,
   so two cards idle for 73 min three times per run. Against a ~2.6 h epoch that is ~47% overhead,
   and ~3.7 h of the run is evaluation. A smaller `--eval-budget` for the per-epoch passes, with the
   full 100 only at the end, would cut most of it -- the metric reports every budget, so a budget-25
   pass compares against the N=25 baseline (0.5855 / 0.7391) rather than the N=100 one.
5. **`--allow-code-drift` is being used in anger.** Its safety rests on training having had no
   ambiguous pose directories, so the new resolver is a no-op there. GATE 2 tests that independently
   by rebuilding a label live and comparing; a GATE 2 pass justifies the flag, and a failure means the
   flag is hiding something.
6. **gloo was chosen because NCCL failed, not on merit.** NCCL is broken on this box — a 20-line
   two-rank `all_reduce` on cards 1 and 3 fails with `CUDA error: an illegal memory access was
   encountered`, with P2P enabled and disabled alike, on NCCL 2.26.2+cuda12.2 under torch 2.7.1+cu128
   and driver 595.99.02, and it reproduces with no model loaded. The 60 ms per 79 MB measurement says
   the cost is irrelevant for a 19.8M-parameter head, but nobody has established the cause, and it
   would matter for anything larger.

## Related

| doc | what it covers |
|---|---|
| `docs/multi_gpu_training.md` | the design and all measurements for this change |
| `docs/pose_provenance.md` | the pose-directory and score-row repairs that preceded it, and why GATE 1 failed overnight |
| `docs/cached_training.md` | why cached training is FT3 at all, which GATE 2 enforces |
