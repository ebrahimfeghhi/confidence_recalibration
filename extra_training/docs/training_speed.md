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

Rank 0, as a budget of 77.9 min:

| bucket | min | per | note |
|---|---|---|---|
| `pose_total` | 25.0 | 8100 poses | wall clock in the pose loop, GPU-inclusive |
| ├ `gpu_forward` | 6.7 | 8100 | CUDA events; overlaps `pose_total`, not additive |
| └ `gpu_backward` | 20.7 | 8100 | CUDA events; GPU busy totals 27.4 min |
| `crop_slice` | 12.9 | 1620 units | **recurs every epoch** |
| `fingerprint_hash` | 8.1 | 1620 units | **first touch only**, see below |
| `prefetch_issue` | 8.1 | 1619 units | **my own overhead**, since fixed |
| `trunk_read_plus_crop_tokens` | 2.5 | 1620 units | `np.load` + `interface_crop_tokens` |
| `grad_collective_plus_wait` | 0.6 | 290 steps | |
| **unaccounted** | **~20.7** | | **the largest single bucket, and unknown** |

GPU busy is 27.4 min of 77.9, so the cards are idle ~65% of the epoch.

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

Projected epoch 2: ~78 - 8.1 - ~7 = **~63 min**. If it comes in materially above that, the
unaccounted bucket is bigger than this table implies. **Unverified at the time of writing.**

## Remaining, ranked by measured size

### 1. Instrument the unaccounted ~21 min — do this first

It is the largest bucket and I cannot name it. Prime suspect: `BC.minimal_loss_inputs` is called
**per pose**, 8100 times per epoch, not per unit. Also unaccounted: the label `np.load`, device
transfers, and Python overhead in the pose loop. A two-line change, and a 21-minute bucket could
dwarf items 2 and 3 — so optimising them first risks a careful fix to a small cost.

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

So balancing by structure size — the proposed fix — would not address this. A 92% spread in time
against an 8% spread in work points at I/O rather than compute: which trunks each rank happened to
read, and whose bytes were still in page cache. Worth confirming before acting, by timing
`slice_cached_unit` separately from the `np.load` inside it.

## Closed

- **The gradient collective.** 0.6 min of 77.9. My earlier `--accum 256` recommendation for the
  full fold rested on an unsplit 3.10 s/step number and is withdrawn.
- **Per-epoch trunk fingerprint hashing.** Fixed, pending epoch-2 confirmation.

## Trust boundaries

- `gpu_forward` / `gpu_backward` are CUDA-event based and trustworthy. The earlier coarse version
  in run `sa9y4m` timed `.backward()` with wall clock and measured the launch; those numbers were
  never quoted.
- `grad_collective_plus_wait` bundles collective and straggler by construction.
- `pose_total` is honest because of the implicit sync inside it.
- The ~21 min is a residual, not a measurement.
- Everything here is **one epoch, one run**. Epoch 1 pays first-touch costs the other 499 do not.
