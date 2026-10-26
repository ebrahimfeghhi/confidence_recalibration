# Review request: three throughput changes, 2026-10-26

All four commits are on `main` at `github.com/ebrahimfeghhi/confidence_recalibration`, under
`extra_training/`. A training run (`dntxmn`) is live on cards 4,5,7 — **please do not launch,
resume or stop anything, and do not touch those cards.** Review by reading; if you want something
executed, say so and I'll run it.

```
6e417086  pin each rank's cores AND memory to one socket, splitting the ranks across both
08957d1c  find and fix the multi-worker prefetch stall, with the CPU-only repro
e165d0ad  turn on --prefetch-workers 4 for the queued arm
511519e7  uncropped held-out PAE behind a flag, and a check of the join it unblocks
```

## Context: what the time is being spent on

Measured on `dntxmn` epoch 15, a 40.3 min cycle = 29.9 min training + 10.4 min eval. Inside
training, per-rank timers from `run.log`:

```
GPU work (pose_total)      14.6 min   identical on all 3 ranks, flat across all 15 epochs
CPU slicing (cpucrop_slice) 16.5 / 16.7 / 27.5 min   by rank
barrier idle                 9.7 /  9.6 /  0.0 min
```

All three ranks report the same 22,400 units and byte-identical `kept_tok` sums of 14,912,212, so
the work split is equal by construction (`15fb1809`). The CPU is the critical path for every rank.

---

## 1. Per-rank NUMA pinning — `6e417086`

**Files:** `shell_scripts/numa_rank_wrap.sh` (new), `run_logging/launch.py`,
`shell_scripts/launch_fullhead_l2sp_posefree.sh`

**Claim.** The ranks were unpinned (`taskset` reported `0-287`), the kernel scattered them, and
memory followed first-touch:

| rank | cores | pages node0/node1 | `cpucrop_slice` |
|---|---|---|---|
| 0 | socket 0 | 74.6% / 25.4% | 16.5 min |
| 1 | socket 0 | 88.0% / 12.0% | 16.7 min |
| 2 | socket 1 | 41.8% / 58.2% | 27.5 min |

Box is 2× AMD EPYC 9565, 72 cores each, node distance 32 vs 10 local. Rank 2 is the only one
reaching across for a large share of its data, and slicing is memory-bandwidth-bound. Nothing was
pinned, so which rank suffered changed over the run (rank 0 slowest at epoch 1, ranks 0+2 tied
through epoch 9, rank 2 alone after epoch 10; cost 1.7–9.7 min, mean 5.7).

**Mechanism:** `torchrun --no-python` execs the wrapper per rank, which reads `LOCAL_RANK` from
`NUMA_RANK_MAP` and `exec`s `numactl --cpunodebind=N --membind=N <interpreter> ...`. Map is
`0:0,1:0,2:1`.

**What I verified.** `numactl --show` under the wrapper reports cpubind/nodebind/membind 0, 0, 1
for ranks 0/1/2; the wrapper refuses `LOCAL_RANK=3` against that map; `launch.py --dry-run` emits
the expected command.

**What I did NOT verify, and where I'd attack it:**
- **The causal claim is unproven.** `numa_maps` only covers each rank's ~2–3 GB of mapped pages;
  the slicing reads trunk embeddings out of an 814 GB page cache I cannot attribute per rank. I
  ruled out the obvious alternative (nodes symmetric at 512.6 vs 515.3 GB used, two external jobs
  on each, box load 14 of 288) but that is not the same as establishing NUMA as the cause.
- **I chose to split 2/1 rather than bind all three to socket 1**, where all three GPUs are
  (`nvidia-smi topo -m` says NUMA affinity 1 for GPUs 4,5,7). Reasoning: slicing is
  bandwidth-bound and 16.5 min even on the fast ranks, so two sockets of bandwidth beat one, and
  the GPU locality given up looks free because `gpu_forward` was 5.6 min on all three ranks
  regardless of socket. **Is that reasoning wrong?** H2D transfer cost may simply not be
  separately timed anywhere, in which case my "appears free" is an absence of measurement, not a
  measurement of absence.
- `--membind` is hard, not `--preferred`. Each socket has 515 GB vs a `--max-rss-gb 80` cap, but
  I have not tested behaviour under a failed allocation.
- `-u` is passed explicitly because `--no-python` stops torchrun prepending it, and unbuffered
  stdout is load-bearing (`run.log` is tailed live by two watchers). **Please check I haven't
  changed anything else about how the interpreter is invoked** — `sys.executable` is used for the
  interpreter path, which is the venv python only because `launch.py` itself runs under it.

---

## 2. Multi-worker prefetch stall — `08957d1c`, enabled in `e165d0ad`

**Files:** `phases/phase_k_train.py` (the `_pf_*` block),
`run_logging/prefetch_deadlock_repro.py` (new)

**Background.** `1083ed10` set `--prefetch-workers 3` everywhere; `7033430b` reverted it because
`ggfhj5` lost epoch 12 to an apparent deadlock (all ranks 0% GPU, every thread in `futex_wait`, no
trajectory write for 9 min). `f39c83ea` asked for a CPU-only reproduction and said the path stays
off until there is one. The mechanism was never established.

**Claim.** It is not a deadlock. `_pf_take`'s docstring justifies its 120 s budget with *"the
worker emits exactly one result per unit, in the main loop's own order"* — false for W>1, since
`_pf_start` hands worker i the stride `units[i::W]`. `_pf_pending` absorbs out-of-order results
but **evicted** above `max(16, 8*W)`, popping in insertion order, i.e. the lowest-numbered stashed
unit — exactly what the main loop asks for next. A worker emits each unit once, so an evicted
result never returns to the queue, and the take burns the full budget before recropping inline.
Then the next unit is also gone. Cascade.

**Second failure found by the repro and not predicted:** at W=4 the workers finish their strides
and exit while the main loop is still asking, after which every remaining take costs the full
budget.

**Fix:** record evictions in `_pf_dropped` so a take for a vanished unit returns immediately; and
return immediately when every worker thread is dead and the queue is empty. New `dropped` /
`exhausted` counters are printed in the per-epoch prefetch line when non-zero.

**Measured** (`run_logging/prefetch_deadlock_repro.py`, sleeps for crop and GPU, budget scaled
120 s → 0.5 s):

```
W=1, shipped            400 takes, 400 hits,   0 stalls,   2.58 s
W=4, before the fix     400 takes,  35 hits, 365 stalls, 732.47 s   (uncapped run)
W=4, with the fix       400 takes, 135 hits,   0 stalls,   2.56 s
W=1/2/4, no slow unit   400 takes, 400 hits,   0 stalls
```

**Where I'd attack it:**
- **The repro is a transcription, not the real code.** `_pf_take`/`_pf_worker`/`_pf_start` are
  closures inside `phase_k_train.py`'s main function and cannot be imported.
  `assert_source_matches()` reads the real file and refuses to run if the transcribed lines
  changed (it fired on this commit when the fix changed the eviction line). **Is the
  transcription actually faithful?** That is the single biggest risk in this commit. The honest
  alternative is extracting the block into an importable module — I did not, because the queued
  arm loads this file.
- **The repro cannot show the throughput gain**, because its sleeps release the GIL, so one worker
  already keeps up with the stubbed GPU and all worker counts finish in the same wall clock. The
  gain is predicted from live timers (16.5 min slice vs 14.6 min GPU, 3.7–13.7 min
  `prefetch_wait`), not measured. **The real crop holds the GIL** — the `--prefetch-workers` help
  itself says "a single thread needs ~0.7 s per unit against ~0.84 s of GPU and the GIL eats that
  margin". Does 4 Python threads doing GIL-bound tensor slicing actually parallelise? If the
  slicing is mostly in C with the GIL released, yes; if not, this buys nothing and `e165d0ad`
  should be reverted. **I consider this the weakest link in the whole set.**
- **`e165d0ad` turns W=4 on for the next run**, which also carries the NUMA pin — two speed
  changes at once. I argue epoch 1 separates them (`cpucrop_slice` absolute = workers, spread
  across ranks = NUMA). Check that reasoning.
- Is `_pf_dropped` unbounded? It holds unit keys, not slices, and is cleared in `_pf_start` /
  `_pf_end`. Worst case is one entry per unit per epoch (~1,461 strings).
- `exhausted` is checked at the top of the wait loop with `any(_t.is_alive() ...)`. Is there a
  race where a worker is alive but about to exit without emitting, or where the queue is
  momentarily empty mid-flight and the take gives up early? I believe a false "exhausted" only
  costs an inline recrop, never a wrong result — **please confirm that, since it is the one way
  this fix could silently degrade throughput rather than fail loudly.**

---

## 3. Uncropped held-out PAE + join check — `511519e7`

**Files:** `phases/phase_k_train.py` (two flags, spawned monitor command),
`shell_scripts/verify_iptm_join.sh` (new)

**Goal.** The held-out PAE pass and the DockQ eval forward the same 7,950 poses through the same
head every epoch; the only reason they cannot share one forward is that the eval is uncropped and
the PAE pass inherits the training crop. Merging them is worth ~4–5 min of a ~10 min eval phase.

**Change.** `--val-loss-uncropped` passes `--crop-margin 0` to the spawned `--loss-only` monitor
(that argument's help defines 0 as the whole structure); `--val-loss-emit-iptm` has it write ipTM
rows. `--crop-all-poses` is no longer passed when uncropped. **Both default off and the queued arm
does not set them** — turning uncropped on makes that run's val PAE incomparable to every arm
measured before it and puts a seam in the curve, so it belongs at the start of a fresh comparison.

**Stated cost:** the pass inherits the run's crop precisely so the held-out number is comparable
to the training curve. Uncropped, train-vs-val PAE stops being apples to apples and the
generalisation-gap readings we have been using (copy's gap wider than share's by +0.0067 ± 0.0017
over 12 matched epochs) stop being computable the same way.

**The part that most needs review.** `dockq_from_iptm.py` says *"THE ARITHMETIC IS PROVEN, THE JOIN
IS NOT"*: `phase_k_eval` maps poses to DockQ by **coordinate identity** against the
`_sample_<k>.cif` files because sample ordering is not guaranteed, while the rebuild uses a naive
`(iface, seed, sample)` key. Its docstring claims agreement on all 13,800 rows of `j583i8` epoch 1
— **that is the circular evidence your earlier review caught as P1 in `f1268fe5`**: the emitted
`sample` column held the pose index rather than `sample_index`, and `sample_index` is non-identity
on 396 of 400 label files, so 73.8% of rows would have carried another sample's DockQ. `f1268fe5`
fixed the column; the join has not been re-checked since.

`verify_iptm_join.sh` is the re-check: an uncropped `--loss-only` pass on an existing checkpoint,
emitting rows, then `dockq_from_iptm.py --compare` against that epoch's own eval shards. It must
be uncropped to mean anything — `--compare` tests per-pose ipTM against the eval's rows, so with
inputs, head and weights identical, ipTM must agree to float precision and **any** difference is
the join.

**Where I'd attack it:**
- **Is "any ipTM difference is the join" actually true?** Non-determinism in the forward
  (TF32/cuDNN kernel selection, different batch shapes without the crop, different sharding
  across cards) could produce small ipTM differences with a perfectly correct join. If so, the
  check needs a tolerance and a stated rationale for it, and I have specified neither.
- **Memory.** Uncropped means the full pair matrix. The crop keeps ~51% of tokens (mean 326 of
  666) and the pair term goes as n², so the uncropped pass is several times the pair area.
  `dntxmn` peaks at 54.83 GB allocated with cropping. The DockQ eval already runs uncropped, so it
  is feasible in principle — but the PAE pass loads a second 464M-parameter model and I have not
  estimated its uncropped peak. **This could OOM.**
- The script reads pool/seeds/fold/accum from the run's `args.json` to avoid drift, and refuses on
  a missing bundle, missing shards, an existing output dir, or occupied cards. Is anything else it
  should refuse on?
- `--crop-contact` and `--crop-size` are still passed when uncropped, on the grounds that they are
  inert at margin 0 and keep the recorded config complete. **Confirm they are actually inert** —
  if `--crop-size` still truncates at 3400 independent of margin, "uncropped" is a misnomer.

---

## What I am asking for

Ranked findings, most severe first, with a concrete failure scenario for each — inputs or state →
wrong output. I am specifically **not** asking you to confirm these; the three soft spots I would
most like attacked are:

1. whether GIL-bound tensor slicing parallelises across 4 Python threads at all (if not, `e165d0ad`
   is wrong);
2. whether the repro's transcription is faithful enough to establish the fix;
3. whether "any ipTM difference is the join" survives forward non-determinism.

Nothing here is deleted and no arm's objective changed; all three are speed-only. If you think any
of them should be reverted before the queued arm launches (~6:15 PM Eastern), say so first.
