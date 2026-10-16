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

## Evidence

`phases/phase_k_train_ddp_selftest.py` runs two structures with one seed each — 10 poses — once on
one GPU with `--accum 10`, then on two GPUs where each rank takes one structure and accumulates 5.
Both perform exactly one optimiser step over the same 10 poses, so the final head must match.

| run | result |
|---|---|
| tensors compared | 248, same keys |
| worst absolute difference | 1.8e-06 – 8.2e-06 across runs |
| mean parameter magnitude | 1.04e+02 |
| relative | **1.8e-08 – 8e-08**, tolerance 1e-04 |
| writers of `head_final.pt` | 1 |

The residual is float32 non-associativity: one GPU sums 10 poses in sequence, two GPUs sum 5+5 then
add. It is four to six orders of magnitude below the tolerance, which is itself far below anything a
real gradient error could hide under.

The selftest also caught a genuine bug: **both ranks wrote `head_final.pt`**. The content was
identical because the replicas agree, but two processes writing one path can interleave and leave a
truncated checkpoint. Checkpoint writes are now rank 0 only.

## Expected speed

One epoch measured 7.9 h on a single card (25,000 poses at ~1.0 s plus 500 structure switches at
~7.1 s). On three ranks the step count is unchanged and the poses are split three ways, so ~2.6 h is
expected. The sweep runs `--epochs 3` at one learning rate, per the user's direction on 2026-10-16.
