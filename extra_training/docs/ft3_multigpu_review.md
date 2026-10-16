# FT3 multi-GPU training: review brief

**For review.** Commit `c186e4b5` on `main`. `docs/multi_gpu_training.md` is canonical for the
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

| | |
|---|---|
| tensors compared | 248, same keys |
| worst absolute difference | 1.8e-06 – 8.2e-06 across runs |
| mean parameter magnitude | 1.04e+02 |
| relative | **1.8e-08 – 8e-08** |
| tolerance | 1e-04 |
| writers of `head_final.pt` | 1 |

Run it with `python -I phases/phase_k_train_ddp_selftest.py --cards 1,3`; it needs two idle cards and
leaves GPU 0 alone by default.

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

## What to attack first

The weakest points, stated so they get looked at rather than skipped.

1. **The equivalence test covers one step, not a run.** It proves the gradient and the step, not
   three epochs. The argument that nothing can drift — bit-identical gradients and identical AdamW
   state after every all-reduce, so no mechanism for the replicas to separate — is worth checking
   rather than trusting.
2. **2 structures of 500 are dropped per epoch** to keep the ranks equal (167/167/166 truncated to
   166). Reshuffling makes it a different 2 each epoch, but it is a real 0.4%, and padding rather
   than truncating is the alternative.
3. **`accum_eff` is 255, not 256**, since 256 does not divide by 3. The loss is scaled by 255 so the
   gradient is a true mean over the poses that contributed, but the effective batch is not the number
   anyone wrote down.
4. **The per-epoch evaluation serialises.** Rank 0 evaluates for ~16.5 min while the other two ranks
   wait at a barrier, so two cards idle three times per run.
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
