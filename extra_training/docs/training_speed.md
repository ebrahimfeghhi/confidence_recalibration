# Training speed on the full set: what epoch 1 measured, and what to do about it

Written 2026-10-20 for review. Every number here is measured on the live run, not estimated.
Nothing in the "remaining" section has been implemented.

## Why this exists

The 30-structure probe ran at 0.300 s/pose. The full set came in at roughly twice that, and my
first explanation -- I/O, because 60 GB of trunks fit in page cache and ~1 TB does not -- was a
guess. The last time I attributed a cost without measuring it, 0.67 s of a 3.10 s step that I had
called "the collective" turned out to be straggler wait (`b829879d`). So the trainer now times
itself.

## How it is instrumented

`phases/phase_k_train.py`, always on (a `perf_counter` pair is ~100 ns against a 0.58 s pose).
Per-rank counters printed as per-epoch deltas at every epoch boundary.

- **CPU spans** by `perf_counter`: the per-unit fingerprint hash, the trunk read plus crop-token
  selection, the crop slice, the prefetch issue.
- **GPU spans by CUDA events**, not wall clock. A kernel launch returns before the kernel runs, so
  `perf_counter` around `.backward()` times the launch. Events are drained **once per step** at the
  gradient reduction, where a sync happens anyway -- draining per pose would impose ~84 syncs per
  step and distort what it measures. Any events left by an epoch's tail are drained at the epoch
  boundary.
- **One `torch.cuda.synchronize()` per step before the collective's timer starts.** Without it that
  timer absorbs every kernel the rank has not finished and "collective" becomes mostly local
  compute. One sync per 290-step epoch is not per-operation synchronisation.
- `pose_total` is wall clock per pose and is GPU-inclusive, because `float(met["loss"])` inside it
  forces a device sync.
- `cropped_tokens` is a workload counter, printed as a per-rank mean.

**`grad_collective_plus_wait` deliberately measures both together.** One rank cannot separate them:
a rank that arrives early blocks until the slowest arrives. The spread of the per-rank figures is
what separates them.

## Epoch 1 of x3k2m4, verbatim

Run `x3k2m4`, commit `947fb1fb`. 488 structures, 4880 units, 1620 units/rank, 8100 poses/rank,
290 optimiser steps, accum 85 (28/rank), lr 1e-2, masked PAE-only, crop 10/5 at cap 3400.

```
[rank 0] epoch 1 took 77.9 min -- crop_slice 12.9m/1620  cropped_tokens mean 334  fingerprint_hash 8.1m/1620  gpu_backward 20.7m/8100  gpu_forward 6.7m/8100  grad_collective_plus_wait 0.6m/290  pose_total 25.0m/8100  prefetch_issue 8.1m/1619  trunk_read_plus_crop_tokens 2.5m/1620
[rank 1] epoch 1 took 77.9 min -- crop_slice  7.8m/1620  cropped_tokens mean 324  fingerprint_hash 7.6m/1620  gpu_backward 19.0m/8100  gpu_forward 6.1m/8100  grad_collective_plus_wait 0.7m/290  pose_total 23.0m/8100  prefetch_issue 7.6m/1619  trunk_read_plus_crop_tokens 2.5m/1620
[rank 2] epoch 1 took 77.9 min -- crop_slice 15.0m/1620  cropped_tokens mean 319  fingerprint_hash 8.5m/1620  gpu_backward 19.1m/8100  gpu_forward 6.1m/8100  grad_collective_plus_wait 0.7m/290  pose_total 23.1m/8100  prefetch_issue 8.6m/1619  trunk_read_plus_crop_tokens 2.6m/1620
```

Rank 0, as an **exclusive** budget of 77.9 min. The first version of this table was not
exclusive and double-counted ~8 min; see *Review round 1* below.

| bucket | min | per | note |
|---|---|---|---|
| `pose_total` | 25.0 | 8100 poses | wall clock in the pose loop |
| `crop_slice` | 12.9 | 1620 units | **recurs every epoch** |
| `prefetch_issue` | 8.1 | 1619 units | **contains `fingerprint_hash` entirely** |
| ├ `fingerprint_hash` | 8.1 | 1620 units | nested, not additional |
| └ fadvise + resolve | ~0.0 | | the syscall itself is free |
| `trunk_read_plus_crop_tokens` | 2.5 | 1620 units | `np.load` + `interface_crop_tokens` |
| `grad_collective_plus_wait` | 0.6 | 290 steps | **not the only collective — see below** |
| **accounted** | **49.1** | | |
| **residual** | **28.8** | | **the largest single bucket, and unknown** |

Not in the budget, because they are CUDA-event spans that overlap `pose_total` rather than adding
to it: `gpu_forward` 6.7 min and `gpu_backward` 20.7 min over 8100 poses each. Their **ratio**
(3.1x) is the usable signal. Their sum is **not** a validated GPU-busy figure -- see *Review
round 1*.

## The cropped-token distribution, all 4860 units

```
p50 332   p90 523   p99 678   max 801   mean 326
units > 1000 tokens: 0        crop cap: 3400
status: 4860 "cropped", 0 "no-crop-definable"
```

Two consequences. First, `--crop-size 3400` is doing nothing -- no crop comes within 4x of it, so
any memory budgeting that assumed 3400 was ~18x too conservative. Second, there is no large-crop
case for activation checkpointing to protect.

## Already done, and what it should buy

Both landed in `947fb1fb`; the third in `9db84d64`, after epoch 1 exposed it.

1. **One shared fingerprint cache.** `BC.unit_fingerprint` SHAs the trunk (~205 MB), the job cif
   and the bioassembly pickle. `resolve_trunk` memoised it in a mutable default argument -- which
   is why I first reported this item as already done -- but `run_unit` re-asserts the same
   fingerprint per unit and did not share that memo, so all three files were hashed on every unit
   of every epoch. Both callers now share one cache. **Epoch 1 still shows 8.1 min because epoch 1
   is first touch; epoch 2 onward should show ~0.** The safety property is unchanged: a bypass of
   `resolve_trunk` leaves the cache without the key, so it still computes and still asserts.
2. **One-unit-ahead prefetch**, `posix_fadvise(WILLNEED)`. One syscall, no thread, no queue; bytes
   land in reclaimable page cache, so it does not raise RSS against `--max-rss-gb` or lower
   MemAvailable in the way that matters to others on the box.
3. **`resolve_trunk` caches its choice**, because the prefetch called it on the next unit and cost
   8.1 min of 77.9 -- more than the trunk read it exists to hide.

**No epoch-2 projection.** The first version of this document predicted ~63 min by subtracting
8.1 (hash) and ~7 (prefetch) from 77.9. Those are the same work, so it subtracted it twice; the
arithmetic a non-exclusive table invites.

Worse, the projection cannot hold at all for a **resumed** run. `_fp_cache` is in-memory and is not
persisted in the resume bundle, so a new process starts it empty and re-pays every first-touch
hash. `x3k2m4` died after epoch 1 and `jhgfxy` resumed from its bundle in a fresh process, so
**epoch 2 pays the hashing again** and only epochs 3+ within that process see the benefit. The
`resolve_trunk` choice cache has the same lifetime.

That reframes the fix: the shared fingerprint cache saves ~8 min per epoch *after the first of each
process*, and this project has restarted four times in one day. A cache persisted beside the fold
-- keyed on each input's size and mtime plus `code_fp`/`loss_fp`, the pattern the evaluator's
`--join-cache` already uses -- would make restarts free. Not implemented; it trades away the
re-verification a fresh process currently performs, so it needs its own invalidation story.

Epoch 2 should be treated as a **new baseline**, not as a test of a prediction.

## Remaining, ranked by measured size

### 1. Fix the accounting and find the hidden synchronisation — do this first

Two things are measured but wrong, and one is unmeasured:

- **The residual is 28.8 min, not 20.7**, once the hash is counted once. It is the largest bucket
  by far and it is not all optimisable CPU work -- it also contains GPU execution that completes
  outside `pose_total` and synchronisation that no timer covers.
- **`BC.minimal_loss_inputs` is NOT the suspect.** The first version of this document named it,
  reasoning that it runs per pose. It does not run at all here: with all 4,860 units cropping
  successfully, `crop_feat_loss` is always set, so both the per-unit (line 1624) and per-pose
  (line 1707) `minimal_loss_inputs` branches are dead code for this workload.
- **`_batch_means()` was entirely untimed, and it is the step's FIRST collective.** It all-reduces
  the loss statistics before `_apply_step` touches gradients, so an early-arriving rank blocks
  *there*, and none of that wait appeared in `grad_collective_plus_wait`. The 0.6 min figure
  therefore cannot be read as "synchronisation is cheap" -- the collective that was timed is not
  the one that absorbs the straggler.

Now instrumented, pending the next epoch: `label_open`, `loss_feature_prep` (everything between
the loss-feature construction and the per-pose loop), `loss_stats_collective_plus_wait`,
`apply_step_total`, an exclusive `prefetch_issue`, and a printed residual so no reader has to sum
the buckets by hand.

### 2. Disable activation checkpointing entirely — ~6.7 min/epoch

`blocks_per_ckpt=1` recomputes all four blocks' forwards during backward. That is consistent with
the measurement: `gpu_backward` 20.7 ≈ recompute (6.7, matching `gpu_forward`) + true backward (~14).

**No size threshold is needed**, which is simpler than the threshold this was first proposed with:
at a 801-token maximum, `z` is 801² x 128 x 2 bytes ≈ 164 MB per copy in bf16, so a few GB across
four blocks against ~76 GB free per card.

Verification before adopting: loss and per-parameter gradient agreement against checkpointing on,
on one unit, plus peak-memory headroom at the p100 crop rather than the mean.

### 3. On-disk cropped-input cache — `crop_slice` 12.9 min, every epoch forever

The largest *recurring* known cost. Needs a cache key over crop margin, crop contact, crop size,
schema and the source fingerprints, and a disk bound: `/home` is at 80% and the run already writes
~110 GB of resume bundles. This is a project with its own invalidation story, not a small change.

### 4. Rank imbalance — ~7 min, but NOT what it looks like

`crop_slice` spreads 7.8 / 12.9 / 15.0 min, a 92% spread, and the fastest rank burns the gap as
barrier wait. **But the quadratic workload is already balanced**: sum of (cropped tokens)² per rank
is 2.113e8 / 1.973e8 / 1.956e8, an 8% spread, and the mean token counts are 334 / 324 / 319.

So balancing by structure size -- the proposed fix -- would not address this. A 92% spread in time
against an 8% spread in work points at I/O rather than compute: which trunks each rank happened to
read, and whose bytes were still in page cache.

**But the destination of the wasted time is not established.** The first version of this document
said the fastest rank burns the gap at the gradient collective; that cannot be right when the
gradient collective is 0.6 min. With `_batch_means` now timed, the next epoch will show whether the
wait lands there instead. Test the page-cache hypothesis before reassigning any structures.

## Closed

- **Per-epoch trunk fingerprint hashing**, within one process. Fixed; across restarts it is still
  paid once per process.

## Reopened

- **The gradient collective is cheap; total synchronisation is not established.** 0.6 min of 77.9
  covers only the gradient reduction. `_batch_means`'s collective ran before that timer started and
  was invisible. The `--accum 256` recommendation stays withdrawn -- it rested on an unsplit
  3.10 s/step number -- but "synchronisation costs 0.6 min" was overstated and the question is open
  until both collectives are measured.

## Trust boundaries

- `gpu_forward` / `gpu_backward` are CUDA events and their **ratio** is trustworthy. Their **sum is
  not a GPU-busy total**, and the earlier claim that the cards were idle ~65% of the epoch is
  withdrawn. `_one_pose` can return before its backward's kernels finish, so `pose_total` need not
  contain all of a pose's GPU work, and an event interval is elapsed stream time rather than kernel
  occupancy -- which is why 6.7 + 20.7 exceeds a 25.0 min pose wall clock. Sampling `nvidia-smi`
  ten times during epoch 2 gave mostly 0% with bursts to 100%, which is suggestive of substantial
  idle but is not a measurement of it. A profiler is needed for that number.
- `grad_collective_plus_wait` bundles collective and straggler by construction, and covers only
  the gradient reduction.
- `pose_total` is wall clock and includes an implicit sync, but not necessarily all trailing GPU
  work.
- The residual is a subtraction, not a measurement.
- Everything here is **one epoch of one run**, and epoch 1 pays first-touch costs.

## Review round 1 (2026-10-20)

An external review of this document at `ef2c2af1` found five errors. All five were verified against
the code and all five were real.

1. **~8 min double-counted.** `prefetch_issue` nests `fingerprint_hash`: the prefetch resolves unit
   *i+1* before the main loop reaches it, so it pays every first-touch hash. The per-rank pairs --
   8.1/8.1, 7.6/7.6, 8.5/8.6 -- are what full nesting predicts. Residual corrected 20.7 -> 28.8.
   `prefetch_issue` is now exclusive.
2. **The 63-minute projection was unsound**, both from that double subtraction and because a
   resumed process re-pays the hashing. Withdrawn.
3. **`minimal_loss_inputs` is bypassed** on the cropped path, so it cannot be the residual.
   Corrected, and the actual cropped path is now instrumented.
4. **`_batch_means`'s collective was untimed** and is the step's first, so the straggler's wait was
   invisible. Now timed.
5. **GPU-busy and 65%-idle were overclaims** from CUDA-event sums. Withdrawn.

Revised order, which this document adopts: fix the accounting and expose the hidden
synchronisation; instrument the cropped data path; run a controlled checkpointing on/off
comparison; only then decide the cache representation; investigate the imbalance with the finer
numbers rather than rebalancing by size.
