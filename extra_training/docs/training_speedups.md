# Epoch time on the confidence-head arms: freeze the stack, cache it, and stop reading the trunk twice

Every figure below is **measured** on `mqird6` (rel + PAE at alpha-rank 75, uncropped, 841
structures, `--ckpt-token-threshold 855`) on 2026-11-07, rank 0, epoch 3. The speedup
numbers derived from them are **ANALYTICAL AND UNVALIDATED** — nothing in this document has
been run on a GPU, because the cards are occupied by the live run the measurements come from.

**Nothing here changes what any arm learns except item 3**, which is a capacity cut and says so.
Items 1, 2, 4, 5 and 6 are I/O and bookkeeping: they change epoch time and no number the run
produces.

| # | item | state | est. saving | changes numerics? | confidence |
|---|---|---|---|---|---|
| 1 | time the trunk read (`trunk_npz_open/read/h2d`) | **implemented**, always on | 0 (it is the measurement) | no | high |
| 2 | fingerprint disk cache survives a code edit; ranks pool it | **implemented**, always on | ~43 min per launch, plus ~5.5 min/epoch from epoch 2 | no | high |
| 3 | freeze pairformer blocks 0-2 (`FREEZE_BLOCKS=1`) | **implemented**, opt-in | 110 → ~28 min backward; 1.5-1.7x | **YES — capacity cut** | medium |
| 4 | step-scoped host trunk cache + warmer (`--trunk-cache-gb`) | **implemented**, opt-in | 10-20 min/epoch | no | medium |
| 5 | `np.memmap` the `z_trunk` member (`--trunk-memmap`) | **implemented**, opt-in | ~6 min/epoch | no | medium |
| 6 | cache blocks 0-2's output between pass A and pass B | **documented only** | ~26 min/epoch | no | low |
| 7 | the rank-2 straggler | **documented only, do not change `--numa-map`** | ~13 min/epoch | no | low |

Also implemented and worth a line of its own: the crop prefetch workers no longer start on an
uncropped arm, where every result they produced was discarded. See *The prefetch that was running
for nothing*.

## The baseline we are cutting into

Epoch 3, rank 0, 232.2 min total. Epoch wall clock came in at 3.83 / 3.87 / 3.75 h for epochs
2, 3 and 4; call it **3.87 h**.

| bucket | min | per | share |
|---|---|---|---|
| `gpu_backward` | 110.0 | 11,500 calls @ 0.574 s | **47.4%** |
| `gpu_forward` | 69.2 | 23,000 calls @ 0.181 s | 23,000 = 11,500 pass A + 11,500 pass B |
| `unit_setup` | 37.0 | 4,600 units | |
| `loss_stats_collective_plus_wait` | 13.5 | | |
| `rank_gather` | 8.2 | | |
| `fingerprint_hash` | 5.5 | | |
| `apply_step_total` | 1.8 | | |
| `grad_collective` | 1.1 | | |
| **accounted** | **196.2** | | |
| **residual** | **35.9** | | |

Population: 841 structures, 6,903 units, 230 steps, 5 poses per unit (verified 2300/2300 units in
epoch 1), 34,515 poses per epoch across all ranks. Tokens per unit: min 128, median 582, p90 1004,
max 1650. Peak allocated **81.30 GB**, identical across all 3 ranks and all epochs, against a
~90 GB card budget.

The head trains 19.8M of 464.4M params (4.3%), all under `confidence_head`. **The 4 pairformer
blocks are UNTIED** — `self.blocks = nn.ModuleList()` with a distinct block appended per layer in
`/home/ebrahim/Protenix/protenix/model/modules/pairformer.py` — at 4,749,184 params each, 24.0% of
the head each. Everything outside the stack is 755,534 params (3.8%). `c_z = 256` in this model,
**NOT** the 128 in the `confidence.py` docstring; verified from
`confidence_head.linear_no_bias_pae.weight` of shape (64, 256). `pairformer_dropout = 0.0` — the
run's `config.json` has no dropout key, so it takes the `ConfidenceHead` default of 0.0.

**77% of the epoch is the model's own forward and backward on real poses.** That is the number that
decides where to look: items 3 and 6 attack forward/backward, and the ruled-out list is mostly
things that attack the other 23%. The other 23% is still worth 53 min, and items 1, 2, 4 and 5 are
what is actually recoverable in it — the bulk of it is one untimed trunk read, done twice per unit
per step.

## Idea 1 — freeze blocks 0-2, train only block 3

**ONE FLAG, NO CODE CHANGE:** `--train-only 'pairformer_stack\.blocks\.3\.'`.

Verified that `--train-only` genuinely sets `p.requires_grad_(...)` at
`extra_training/phases/phase_k_train.py:2745`, not merely filtering the optimiser. So with only
`blocks.3.` trainable, block 0's inputs come from the frozen trunk and frozen input projections,
nothing below block 3 requires grad, and **autograd builds no graph through blocks 0-2 at all**.

### Estimate

| | min | why |
|---|---|---|
| `gpu_forward` | 69.2 → **69.2** | UNCHANGED — all 4 blocks still run; you need block 3's output |
| `gpu_backward` | 110.0 → **~28** | graph through 1 block instead of 4 |
| overhead | 53 → **53** | unchanged |
| epoch | 232 → **~150** | **1.55x** |

The backward division by 4 is fair because the 4 blocks are byte-identical in size and everything
outside the stack is 3.8% of the head.

**Second-order win.** Measured backward is 3.2x forward per pose (0.574 vs 0.181 s) where theory
says ~2x. The gap is the checkpoint re-forward on the 696 of 4,600 units above the 855-token
threshold. Freezing cuts retained activations ~4x, which should let checkpointing go off entirely
and recover that gap, pushing toward **1.67x**.

Quote the range as **1.5-1.7x**: epoch 3.87 h → ~2.4 h (plan on 2.5 h), a 10-epoch arm 38.7 h →
~25 h, saving **~14 h per arm**.

### Why ~20 prior arms did not already get this

~20 prior arms already use `--train-only`, but **every one targets INPUT-SIDE parameters**
(`linear_no_bias_d`, `pose_perblock`, `frame_dir`) which sit at the **BOTTOM** of the stack. The
gradient still has to traverse all 4 blocks to reach them, so those arms pay the full backward
while updating almost nothing. This proposal is the first configuration that puts the trainable set
at the **TOP**, which is exactly where the saving comes from.

### Costs and risks

- **This is a CAPACITY CUT, not a free optimisation.** 4.75M trainable params instead of 19.75M. It
  needs its own held-out control arm; it is **not** comparable against `mqird6`.
- It may win on generalisation anyway — 19.8M params against 840 structures.
- It makes per-block injection work untrainable in blocks 0-2 (`--pose-perblock`, `frame_dir` at
  all 4 depths).
- **The estimate is analytical and unvalidated.** An earlier same-method speed projection made this
  same day predicted 1.35x where the measured value was 1.10x — 25% optimistic. Discount
  accordingly.
- **Falsification:** run ~20 steps with the flag and compare s/step against `mqird6`'s measured
  **62.1 s/step** mean. If backward collapses you see ~40 s immediately. Needs a free card.

### How to turn it on

`FREEZE_BLOCKS=1` in `shell_scripts/launch_default_arm.sh`, in the style of the existing `MEMLOG`
and `CKPT_THRESH` switches. It appends `--train-only 'pairformer_stack\.blocks\.3\.'` and appends a
paragraph to the run's `NOTE` saying it is a different arm — the same rule `MEMLOG` follows, because
a frozen-blocks run recorded under the default note would be a false record of what was trained.
`FREEZE_PAT` overrides the pattern, so "freeze 0-1" needs no edit.

**It also freezes the PAE readout, which is easy to miss.** "Everything outside `blocks.3`" includes
`linear_no_bias_pae` (64×256) and `pae_ln` (256 weight + 256 bias) — the two tensors the objective's
own output passes through. They sit **above** the stack (`confidence.py` applies them to `z_pair`
after `pairformer_stack` returns), so unfreezing them adds **no backward depth at all**: their
gradient needs only their input and their output gradient. If the arm should keep an adaptable
readout:

```
FREEZE_PAT='pairformer_stack\.blocks\.3\.,^linear_no_bias_pae\.,^pae_ln\.'
```

4.75M + ~16.9k parameters, same expected speedup. Not the default only because the 1.5-1.7x above
was computed for one block exactly.

### The corollary nobody should have to rediscover: checkpointing can probably come off

With blocks 0-2 frozen they build no graph, so there are **no activations to retain** for them and
only block 3 is left to checkpoint. A rough extrapolation puts the largest remaining unit (1,650
tokens) at **~61 GB** unchecked against the ~90 GB per-pose budget, where today's measured peak is
**81.30 GB with checkpointing on**.

**Treat that 61 GB as an extrapolation and nothing more.** It cannot be derived from today's
numbers: the only measured scaling law for this workload, `peak_GB = 2.02 + 2.896e-05 * tokens^2`
from `--mem-log` run `tvqobh`, was fitted **with checkpointing on** (it predicts 80.9 GB at 1,650
tokens, which is why the measured 81.30 GB matches it so well), and there is no fit for the
unchecked-and-frozen regime. Measure it with `MEMLOG=1 EPOCHS=1` before raising
`--ckpt-token-threshold`, and remember what happened the last time a threshold was carried to a
population it was not measured on: `tztukd` died of CUDA OOM 2.6 h into epoch 1. Worth roughly
**9 min/epoch** if it holds — the gap between the measured 3.2x backward/forward ratio and the ~2x
theory is the checkpoint re-forward on the 696 of 4,600 units above the 855-token line.

## The 31 minutes nobody could name, and the timer that will name them

`unit_setup` was 37.0 min of epoch 3, with **0.2 min** of that in named leaves, and 31.0 of the
epoch's 35.9 min residual was placed INSIDE units and no further. The reason is simple and
embarrassing: on an uncropped arm the only per-unit I/O is `RC.load_unit(trunk_p, dev)`, and **that
call carried no timer of any kind**. The cropped path has four (`cpucrop_label`, `cpucrop_trunk`,
`cpucrop_tokens`, `cpucrop_slice`); the uncropped path, which is the default arm, had none.

`load_unit` (`phases/phase_j_rescore_cache.py`) is `np.load(npz)` and then 11 `torch.from_numpy(a)
.to(device)` calls, each a **synchronous pageable host-to-device copy**. At a mean 243 MB per unit
and 4,600 `load_unit` calls per rank per epoch that is **~1.1 TB each way, per rank, per epoch**.

**Now split into three buckets**, using the existing `_tick`/`_T`/`_TN` machinery so they print in
the same epoch line as everything else:

| bucket | what it is | count/epoch |
|---|---|---|
| `trunk_npz_open` | `np.load`, which only parses the zip central directory | 4,600 |
| `trunk_npz_read` | `z[key]`, the disk read and the numpy materialisation | 50,600 |
| `trunk_h2d` | `.to(device)`, the pageable copy | 46,000 |

**Why three and not one.** `np.load` on an npz is lazy — the open reads no member — so the read and
the transfer are two separate passes over the same bytes, limited by different things: page cache
and readahead for the first, PCIe and the absence of pinning for the second. **That split is what
decides whether items 4 and 5 are worth building**, which is why it matters more than its zero
direct saving. If the 31 min is mostly `trunk_npz_read`, prefetching it (item 4) hides it. If it is
mostly `trunk_h2d`, only pinned staging would, and item 4's payoff shrinks to the halving.

The three buckets are added to `_EXCL` (the exclusive residual sum) and to the in-unit `_leaves`
sum, so the printed residual shrinks by exactly what they now name. Leaving them out of `_leaves`
would have reported the very time they measure as still unaccounted — the measurement would have
looked like it changed nothing.

**One caveat on reading them.** With `--trunk-memmap` on, a mapping is lazy, so the disk read
**moves out of `trunk_npz_read` and into `trunk_h2d`**. Read their sum when comparing runs; read
them separately only at the default.

- **Saving: none.** It is the measurement.
- **Numerics: unchanged.** A `perf_counter` pair is ~100 ns against a 0.40 s call.
- **Confidence: high.** Always on, like every other bucket.

## The fingerprint disk cache threw itself away on every code edit

`fingerprint_hash` was 5.5 min of epoch 3 and ~43 min off the front of **every launch**, and the
reason was a one-line policy in `_fp_cache_load()`: a persisted rank file was discarded **entirely**
when its stored `code_fp` or `loss_fp` differed from this process's. Every edit to
`phase_j_build_cache.py` or to the loss config therefore threw away all ~11,040 entries at once and
the run re-SHA'd ~243 MB per unit from scratch — the trunk, the job cif and the bioassembly pickle,
for every unit, again.

But `code` and `loss` are **three short strings the process already holds**, and they cannot change
what `sha256(trunk)` is. So `unit_fingerprint` is split in two:

- `unit_content_fingerprint(a, pdb, trunk_p)` → `cif=…|bio=…|idx=…|trunk=…` — the entire cost, a
  pure function of four files.
- `unit_fingerprint(a, pdb, trunk_p, code_fp, loss_fp, content=None)` → `schema=…|code=…|loss=…`
  joined onto the content half with the same separator and the same field order.

Only the content half is persisted (`_FP_CACHE_VER = 2`), keyed by the existing size+`mtime_ns`
stat signature over the trunk, the cif, the bioassembly pickle and the indices file. `code_fp` and
`loss_fp` are still written into the file as a record of what wrote it, and are no longer read for
invalidation.

### The provenance guard, and how its preservation was verified

This is **the highest-risk change in the document** and the one most needing review. A label stores
a per-pose **permutation** of its own trunk's poses, so pairing a label with a different root's
same-named trunk trains pose *i* against pose *j*'s label. Review round 8 reproduced exactly that
with a `fold_regen` label against a `fold` trunk. The guard works by comparing the label's **stored
fingerprint string** against one recomputed now, so the only acceptable refactor is one that leaves
that string byte-identical.

What was done to establish it:

1. **`phases/test_unit_fingerprint_split.py`** asserts the composed string against a **literal copy
   of the pre-split function body**, not against a description of it — field order, separator and
   all seven fields. It also asserts that the content half does not move when `code_fp`/`loss_fp`
   move, and that it *does* move when the trunk, the cif, the bioassembly or the structure's own
   indices rows move. It is CPU-only, on temporary files it creates, and does not read the real
   cache. It passes.
2. **Every comparison site is unchanged in form.** `resolve_trunk`'s candidate loop, its
   two-candidate ambiguity `SystemExit`, its no-candidate `SystemExit` with the per-field `diff`,
   the `--allow-code-drift` branch, and `run_unit`'s per-unit re-assertion all still compare a
   composed string against `str(np.asarray(lab["fingerprint"]))`. What changed is that they now go
   through one helper, `_unit_fp`, instead of each carrying its own `if key not in cache` dance —
   two call sites meant two chances for the memo and the timing to drift apart.
3. **The prefix is recomposed from THIS process's values on every call.** A cache file written by a
   different checkout cannot smuggle its `code_fp` into a comparison: only `cif|bio|idx|trunk` is
   adopted, and `schema|code|loss` is always this process's own claim. So a stale `code_fp` still
   mismatches the label and still raises, exactly as before — the saving is that it no longer
   destroys 11,040 trunk hashes on the way there.
4. **Nothing is adopted without the stat signature matching.** That check is unchanged and still
   covers every file the content half hashes.

What is genuinely given up is what version 1 already gave up and documented: a file rewritten with
its size and `mtime_ns` preserved would go unnoticed. Everything that rewrites these files — a
refold, a rebuild, the 2026-10-14 geometry fix — changes mtime.

### And the ranks now pool it

`_fp_cache_load()` ran **once**, at startup, while `_fp_cache_save()` runs at **every epoch
boundary** and the loader globs every rank's file. So from epoch 2 onward each rank re-hashed every
unit the shuffle had moved onto it, while the hash it needed sat on disk beside its own. The three
ranks jointly cover all 6,903 units in epoch 1, so reloading after the save at each boundary makes
epoch 2 free for all three instead of one. Entries this process computed are never replaced — the
loader skips a key already in memory before it even looks at the signature — and each rank's file is
written atomically through `os.replace`, so a sibling reads the previous version or the new one and
never a torn one. A miss costs one re-hash and the next boundary picks it up.

- **Saving: ~43 min off every launch** (the whole cold-start hash), plus **~5.5 min/epoch** from
  epoch 2 once the ranks pool.
- **Numerics: unchanged.** The compared string is byte-identical.
- **Confidence: high** on the saving, and the gate above is the reason to trust the safety. It is
  always on; the only way to get the old behaviour is to delete the `.json` files.

## The step-scoped trunk cache: pass A and pass B each read every trunk

The cross-seed objective runs **two passes over every step's units** — a `no_grad` scoring pass A
whose scores the relative-iTM coefficients need (a pose's coefficient depends on every other pose of
its interface, so all scores must exist before any backward), then pass B, which rebuilds each pose
and backwards it. Both passes call `run_unit`, so **both call `load_unit`**: 4,600 calls for 2,300
units per rank per epoch. Each unit's ~243 MB is read and copied to the device **twice per step**.

`--trunk-cache-gb N` holds a step's trunks as **host numpy arrays** for the span of that step. Pass A
fills the cache (a miss reads inline and offers the result); pass B hits it and does only the device
copy. On top of that, `--prefetch-workers` threads warm the step's remaining units at the step
boundary, while the GPU is busy with pass A's first unit — so most of the surviving half goes behind
compute too. `phase_j_rescore_cache.load_unit` is split into `read_unit_host` (numpy only, no torch,
no device — which is what makes it safe on a worker thread) and `unit_from_host` (the only CUDA
half), with `load_unit(path, device, host=…)` composing them.

### The two hazards, and what was done about each

**Hazard 1 — `_pf_take` POPS, and that bug cost ~3 hours an epoch.** The existing crop prefetch is a
pop queue keyed by label with a 120 s blocking wait. Pass A walking the same units as pass B
consumed each unit's result and left pass B waiting out the full budget per unit: flat CPU, 0% GPU,
~3 h an epoch, observed on probe `dsswrj` and recorded in the comment the uncropped path now sits
beside. **So this is a dict keyed by `trunk_p`, never a queue.** Both passes *look up* the same key;
a lookup cannot block, cannot pop, and a miss reads inline. The cache is **purely advisory** — every
consumer falls back to the ordinary read — so a warmer that is slow, failed, capped out or never
started costs time and nothing else. There is no wait, no budget and no eviction cascade to get
wrong.

**Hazard 2 — the pending cap.** `_pf_pending`'s `max(16, 8*workers)` would be 32 entries, and at
243 MB per uncropped trunk that is **~7.8 GB of RSS per rank** against `--max-rss-gb 80`. This cache
caps on **bytes**, defaults to **off**, and **refuses rather than evicts** when full: every entry in
a step is wanted twice, so dropping the oldest would drop one pass B is about to ask for, while
refusing the newest only forgoes a saving. A typical step is 10 units at the median 582 tokens
(~2.4 GB); a deep step of the largest remaining structures is 10 × 1,650 tokens ≈ 8.5 GB. **4.0 GiB
is the suggested setting**: it covers a typical step and partially covers the worst, which is the
intended behaviour rather than a shortfall.

Three further guards, all static:

- **Cleared at every step boundary**, after joining the previous step's warmers, which is what bounds
  it to one step. A warmer still blocked in a read past the 30 s join keeps its entry — and that is
  *safe*, not merely tolerated: the key is a trunk **path** and a trunk's bytes do not change during
  a run, so a late put is correct content under a correct key. It costs cap, not correctness.
- **CUDA is part of the precondition.** `unit_from_host` **aliases** the host arrays when the device
  is CPU (`.to(cpu)` is a no-op), so two passes sharing one dict on CPU would share any in-place
  write. `.to(cuda)` always copies. `dev` is cuda at `phase_k_train.py:1528`; `_tr_on` says so out
  loud rather than relying on it.
- **Only where there is a pass A.** Without `--rank-cross-seed`, `_xs_at` is empty, the warmer is
  never started, every unit is read once and there is nothing to hit — the cache would fill to its
  cap and hold the RSS for no benefit. The same condition that gates `_xs_prepare` gates this.

A per-epoch line reports it. **How to read the rate:** 50% is what the cache alone gives — pass A
misses, fills, pass B hits. A unit the *warmer* reached before pass A is a hit in **both** passes,
so the achievable range is **50% (the warmers contributed nothing) to 100% (every unit warmed in
time)**. At `TRUNK_CACHE_GB=4` a 30-unit step is ~7.3 GB, so only about half its units fit and the
practical target is **~75%**, with `refused for cap` saying how many were turned away. `crop_slice`
not moving is how the first version of the crop prefetch's total failure went unnoticed, and this is
the same guard against the same mistake.

### What review found in the first version of this, and what it now does instead

Worth recording, because all three were lifecycle bugs rather than design errors:

- **A shared `threading.Event` could un-cancel a stale warmer.** `set()` → `join(timeout=30)` →
  `clear()` meant a warmer still inside a 243 MB read when the join expired resumed its loop, found
  the flag **cleared**, and went on warming the *previous* step's units — spending the new step's
  cap on trunks nobody would ask for. It was never joined again either, so up to
  `--prefetch-workers` orphans could leak **per step boundary** and live for the rest of the run,
  writing `_fp_cache`, `_fp_content` and `_T` unlocked while the main thread iterated them at the
  epoch boundary. It is a **generation counter** now: a worker compares its own `gen` against one
  cell and a stale worker can only exit.
- **The per-step join was the one place the main thread blocked.** W warmers joined with sequential
  30 s timeouts, 230 times an epoch, inside the critical path this exists to shorten. Nothing is
  joined per step now; the epoch boundary joins once, so no warmer races `_fp_cache_save`.
- **The cap bounded the cache, not the process.** Reserving `_tr_inflight` before the read and
  refusing in `_tr_put` after it meant the full 243 MB was already on the heap when the refusal
  happened: ~1 GB over the cap at four warmers. The size is now checked *before* the read, from the
  npz's own `os.path.getsize` (a faithful proxy, since every member is stored uncompressed at the
  dtype it is read back as).

  **Moving the check before the read was not enough, and the second review round is why.**
  `_tr_bytes[0]` counts only **admitted** entries, so four warmers arriving inside the same window
  all evaluated `_tr_bytes[0] + _sz > _tr_cap` against the **same low total**, all passed, and all
  went on to allocate their whole ~243 MB each — `TRUNK_CACHE_GB` bounded the cache while the
  process sat **~4x over it**, which is the original bug with one extra step in front of it. So
  `_tr_inflight` is now a **dict of path → reserved bytes** rather than a set of paths
  (`phase_k_train.py:4510-4520`): a set records *that* a read is in flight, and a byte cap needs to
  know *how much*. There is a `_tr_resv[0]` running total beside `_tr_bytes[0]`, the reservation is
  taken **under the same lock, in the same critical section in which the check passes**
  (`phase_k_train.py:4611-4616`), and it is released in a `finally` (`:4619-4621`) **and again on
  the `except BaseException` path** (`:4622-4632`) — both, because an exception between the
  reservation and the `finally` (`resolve_trunk` raising, a read error) would otherwise leave those
  bytes reserved for the rest of the run, and after a handful of failures the cache would refuse
  every put while reporting a near-empty occupancy. `_tr_put` **discounts the offering path's own
  reservation** before testing the cap (`:4566-4574`): the bytes it reserved before the read are the
  very bytes it is now offering, and charging both would refuse a put the reservation had already
  made room for. The cap therefore bounds **admitted + in-flight**, which is what host RSS actually
  is — the number `--max-rss-gb 80` is defending.

  **`_tr_reset` deliberately does NOT clear `_tr_inflight`** (`phase_k_train.py:4634-4645`). It
  bumps the generation and drops the admitted arrays, but a warmer from the **retired** generation
  may still be inside its 243 MB read — the same warmer the generation counter above exists to
  tolerate. Clearing the dict underneath it would make its `finally`'s `pop` a no-op and leave
  `_tr_resv[0]` permanently above zero, and that fails **closed**: a cache that refuses everything
  for the rest of the run while `_tr_bytes[0]` says it is empty. Stopping **new** reservations is
  all that retiring a step has to do, and the generation check already does that.

Two accounting fixes came out of the same review. `fingerprint_hash` is in `_EXCL`, the sum
subtracted from wall clock to get the residual — and warmers hashing concurrently with the main
thread can put **more than the epoch's wall clock** into it and drive the residual negative, which
is exactly the double-count the `rank_pass_a` comment on `_EXCL` was written about. Worker hashes go
to `fingerprint_hash_warm`, which is in neither `_EXCL` nor `_leaves`. And `_fp_cache_save`
re-`stat`ed at the boundary rather than at hash time, so a file rewritten in between was persisted
with a signature describing the **new** file and content describing the **old** — the one direction
a provenance cache must never be wrong in. The signature is now taken beside the hash.

- **Saving: 10-20 min/epoch**, as a range because the split between `trunk_npz_read` and
  `trunk_h2d` is not yet measured. The halving of the reads is structural; how much of the rest
  hides behind the GPU is not.
- **Numerics: unchanged.** The same bytes reach the device; `phases/test_trunk_memmap.py` gates the
  host-dict path bit-identical to a fresh read, and gates a **second** build from a once-read dict
  identical to the first, which is exactly what pass A and pass B do.
  **What that gate cannot prove:** on CPU `.to(cpu)` is a no-op, so both builds *alias* the host
  dict and the equality is trivially true. The test now asserts the aliasing explicitly and says so,
  because the property the cache actually depends on — `.to(cuda)` always copying — needs a card.
- **Host peak per unit rises from ~187 MB to ~243 MB**, the one behavioural side effect of splitting
  `load_unit`: it used to read one member, move it and drop it, so its host peak was `z_trunk`
  alone; the dict materialises all 11 first. Deliberate, since the dict is what gets handed around,
  and additive with the cap above.
- **Confidence: medium.** The design is static and the fallbacks are unconditional, but it is
  threaded code that has never run on a GPU.

## `np.memmap` the `z_trunk` member

`z[key]` `read()`s a member into a fresh heap buffer and the device copy then reads that buffer: two
passes over 243 MB per unit. A mapping lets the host-to-device copy read page cache directly and the
heap pass disappears.

**Both enabling facts were verified on real files, not assumed.** On three trunks (66.4, 104.4 and
191.8 MB) every npz member reports `compress_type 0` — **STORED, not deflated** — and `z_trunk.npy`
is the **first** member at zip `header_offset 0` with its payload at a fixed **61-byte** offset
(30-byte local header + 11-byte name + 20-byte extra field, byte-identical in all three). So the
member is one contiguous run of the file and `np.memmap` can address it in place **without rewriting
the 1.7 TB cache**.

The implementation assumes none of that at runtime: the payload offset is read from the member's own
local header, and the shape, dtype and order from its own npy header. A deflated member, a
fortran-ordered member, an object dtype, a short file or **any exception at all** returns `None` and
the caller falls back to `np.load`'s own path. `mode="c"` rather than `"r"`, because a read-only
mapping hands back a non-writeable array and `torch.from_numpy` then warns on every unit;
copy-on-write is writeable, is never written, and never writes back.

Two hardenings came out of review. The span is required to **exhaust the member exactly** —
`(off − payload_start) + n·itemsize == info.file_size`, not merely to fit inside the file — because
fitting inside the file would let an offset error map *across a member boundary* and hand the head a
reinterpretation of the wrong bytes, which does not raise and is the worst failure mode available
here. And the `except` is `Exception`, not a narrow list: a corrupt npy header raises `SyntaxError`
out of `read_array_header_*` and a bad `descr` raises `TypeError`, neither of which was caught, so
"map this if you can" was really "kill the run" — with a working `np.load` path one line away.

**Only `z_trunk` is mapped.** It is ~99% of the bytes (187.4 MB of 191.8 on `9mnq`); every other
member is at or under ~1.5 MB, where the extra `open`/`mmap` syscalls cost more than the copy they
would avoid.

**It does not compose with item 4 for `z_trunk`.** A mapping is lazy, so a *warmed* memmap does no
read on the worker and hands the bytes straight back to the main thread's device copy. The warmer
therefore always reads into a heap buffer, and `--trunk-memmap` affects only the inline misses.
Measure one at a time.

**THE MAPPING IS UNALIGNED, AND THAT MAY COST THE WHOLE SAVING.** The payload starts at 61 bytes
(zip local header) + a 128-byte npy header = **byte 189, which is odd**. So a uint16/bf16 array
mapped here has `flags.aligned == False` and an odd `data_ptr()`, on **every** trunk, structurally —
and a pageable `cudaMemcpy` from a 1-byte-aligned host pointer gives up the wide-load staging path.
It is bit-exact (verified), but the ~6 min could shrink or invert. **When it is measured on a GPU,
compare `trunk_npz_read + trunk_h2d` summed across the two arms, not `trunk_h2d` alone**, and be
prepared for no win. If it loses, the alternative is `np.frombuffer` on an `mmap.mmap` sliced to a
64-byte-aligned base plus a view offset. A `trunk_memmap_fallback` bucket counts members that could
not be mapped, so a run cannot honestly report `--trunk-memmap` while never mapping anything.

- **Saving: ~6 min/epoch, and this is the least trustworthy number in the document** — see the
  alignment paragraph above.
- **Numerics: unchanged, and gated.** `phases/test_trunk_memmap.py` compares both paths element by
  element on real trunks — bf16 through a `uint16` view, because comparing bf16 directly would test
  float equality on the very reinterpretation the gate exists to catch — and asserts that the
  mapping actually happened, so a silently-falling-back `npy_memmap` cannot pass by doing nothing.
  It passes on CPU.
- **Confidence: medium.** Correct on CPU on real data; the device copy from a mapping is the part no
  one has timed.

## The prefetch that was running for nothing

`_pf_take` is reached **only** inside `if a.crop_margin > 0 or a.crop_tokens`. The default arm is
fully uncropped and passes `--prefetch-units 2 --prefetch-workers 4`, so four worker threads ran a
full `np.load` plus `interface_crop_tokens` plus `slice_cached_unit` per unit and **nobody ever
consumed one** — then blocked forever on a full `put()` once 8 results had piled up, holding those 8
slices in RSS for the rest of the epoch. `_pf_start` now returns early when cropping is off.

- **Saving: small — 8 wasted crops and 8 retained slices per epoch per rank.** Listed because it is
  free and because the next person to read `--prefetch-units 2` on an uncropped arm will otherwise
  assume it is doing something.
- **Numerics: unchanged.** No uncropped code path ever read the queue.
- **Confidence: high.** The gate is the same expression `run_unit` already uses to decide whether to
  take from the queue at all.
- The epoch report is keyed on whether `_pf_start` actually started, not on `--prefetch-units` —
  otherwise every uncropped epoch would print `prefetch (4 worker(s)): 0/0 hit (0.0%)`, which reads
  as a regression to 0% rather than as "there is no prefetch on this path".

## One incidental fix: `--help` was broken

`--rank-pool`'s help string contains a bare `67.7% of`. argparse runs every help string through
`% params`, `"% o"` is a conversion, and the whole `--help` died with
`TypeError: %o format: an integer is required, not dict`. Present at `72148ac3` and found only
because the two new flags above have help text nobody could otherwise have read. Two characters.

## Idea 2 — cache blocks 0-2's output between pass A and pass B (DEPENDS ON IDEA 1)

The trainer runs **two passes per step**. Pass A is a `torch.no_grad()` forward per pose that yields
the scores the relative-iTM coefficients need — a pose's coefficient depends on every other pose in
its interface, so all scores must exist before any backward. Pass B then rebuilds each pose and
backwards it. So **every pose is forwarded TWICE within the same step**; that is why `gpu_forward`
shows 23,000 calls against 11,500 backwards.

With blocks 0-2 frozen their output is **identical in both passes**, so stash it in pass A and start
pass B from it. Validity rests on `pairformer_dropout = 0.0`: no dropout means pass A and pass B
compute a bit-identical representation for the same pose. **This is EXACT, not an approximation.**
Pass A already computes the tensor, so the stash is free — you just don't free it.

Saves blocks 0-2's forward in pass B = 3/4 of pass B's 34.6 min ≈ **26 min/epoch**.

### Memory, which is the whole difficulty

The pair rep is n² × 256 in bf16, and a rank holds 5 seeds × 5 poses = 25 poses of each of the 2
structures in a deep cross-seed step.

| tokens | per pose | per structure | per step |
|---|---|---|---|
| median 582 | 165 MiB | 4.0 GiB | 8 GiB |
| p90 1004 | — | ~12 GiB | ~24 GiB |
| max 1650 | 1329 MiB | 32.5 GiB | 65 GiB |

Freezing drops peak memory a long way (only block 3 retained), so expect ~55-60 GB of headroom.
**Median and p90 fit; the largest structures do not.** So it wants a **TOKEN THRESHOLD**, the same
pattern as the existing `--ckpt-token-threshold`: cache below the line, recompute above it.
Coverage by budget: ≤409 tok = 12.3% of units, ≤579 = 49.3%, ≤819 = 82.9%, ≤1158 = 93.9%.

### Stacked with Idea 1 (these two only)

Freeze alone 232 → ~150 min; plus cache at ~85% coverage → **~128 min**, about **1.8x** overall,
10 epochs in ~21 h instead of 38.7. For the whole programme rather than these two, see *The stacked
estimate, and why the items are NOT additive* below — which quotes **~2 h/epoch (1.7-2.2 h)**, not
the arithmetic sum, and says why.

### Caveats

- **This is the first item needing real code rather than a flag** — it changes the two-pass
  structure.
- It is an estimate stacked on an estimate. Trust it less than Idea 1.
- **It is strictly dependent on Idea 1.** Without frozen blocks 0-2, pass B must run them anyway to
  build the graph, so there is nothing to cache.
- **It COMPETES WITH TURNING CHECKPOINTING OFF for the same freed memory.** Freezing frees headroom
  once, and both ideas want to spend it: raising `--ckpt-token-threshold` spends it on retained
  activations, and this cache spends it on stashed pair representations. On a large structure they
  cannot both have it. **The cache wins on value — ~26 min/epoch against ~9** — so the sensible
  shape is *both below a token threshold and cache-only above it*, with the same
  `--ckpt-token-threshold` pattern deciding where the line falls. Do not reason about either in
  isolation; the headroom is one budget.
- **It is still only documented.** This is the one item here that needs real code rather than a
  flag, it changes the two-pass structure, and it needs its own design review before anyone writes
  it. A documented design beats speculative code in the two-pass loop, which is where the ~3-hour
  `dsswrj` bug lived.

## The rank-2 straggler — documented, and `--numa-map` is NOT to be changed

~13 min/epoch sits in `loss_stats_collective_plus_wait` (13.5 min of epoch 3) and that bucket is
deliberately a wait: every rank blocks at the step's first collective until the slowest arrives.
Rank 2 is the slowest, and **get the direction right, because a previous pass at this got it exactly
backwards**:

- `nvidia-smi topo -m`: **GPU0-3 are on NUMA node 0, GPU4-7 on node 1.**
- The run uses cards **4, 5 and 7 — all three on node 1.**
- The map is `--numa-map 0:0,1:0,2:1`. So **rank 2 is the ONLY correctly co-located rank**, and
  ranks 0 and 1 are the cross-socket ones.

Rank 2 nonetheless carries **61.5 min more system time** and **15 min more `unit_setup`** than its
siblings, with equal user time and equal GPU work. Equal user time with that much more system time
is not a locality problem — a cross-socket memory path shows up as slower *user* work, not as time
in the kernel. The surviving explanation is **external contention from another tenant on socket 1**:
rank 2 competes for that socket's memory bandwidth and page cache with somebody else's job, while
ranks 0 and 1 pay a predictable cross-socket penalty on a quiet socket.

**The tension, stated plainly so nobody "fixes" half of it.** NUMA locality wants rank 2 where it
is. Contention wants it off node 1. These point in opposite directions and the measurement cannot
yet say which dominates, because the contention is another tenant's and is not under our control or
observation. **Do not change `--numa-map`**: moving rank 2 to node 0 would make all three ranks
cross-socket on the basis of a hypothesis about someone else's job.

What would settle it: `/proc/pressure/memory` or a `numastat` sample on both sockets during a step,
taken while the contention is present. Until then this is ~13 min/epoch that is understood well
enough not to be mistaken for a collective cost, and not well enough to act on.

- **Saving if solved: ~13 min/epoch.** **Confidence: low.** **Numerics: unchanged.**

## The stacked estimate, and why the items are NOT additive

Best case, everything above working as estimated: **232 min → ~95 min**.

**Quote it as ~2 h/epoch, range 1.7-2.2 h.** Not ~1.6 h. The reason is specific and recent: an
earlier projection by this same method, made the same day, predicted **1.35x where the measured
figure was 1.10x** — 25% optimistic. Discounting the stack by that much is the honest reading, and
if the real figure lands at 2.2 h that is still a 10-epoch arm going from 38.7 h to ~22 h.

**The items overlap, so do not add them up:**

- **#4 and #7 overlap.** The slow rank is slow *because of the loading* — rank 2's extra time is
  system time and `unit_setup`, which is the trunk read. Hide the read behind the GPU and part of
  the straggler disappears with it. Counting both in full double-counts the same minutes.
- **Freezing (#3) SHRINKS #4's payoff.** #4 works by hiding host I/O behind GPU time. Cut the
  backward from 110 min to ~28 and there is **less GPU time left to hide it behind** — the same
  I/O against a shorter compute shadow. The more #3 succeeds, the more of #4's 10-20 min becomes
  visible again rather than absorbed.
- **#5 and #4 do not compose for `z_trunk`** (a warmed mapping does no read; see that section).
- **#6 and checkpointing-off compete for one memory budget** (see Idea 2's caveats).

So the stack is not `232 - 82 - 20 - 6 - 26 - 13`. The defensible claim is: #2 removes a launch
cost, #1 costs nothing and tells us whether #4 and #5 were worth building, #3 is the one large
lever and the one with an experimental cost, and #4/#5 are worth having mainly *before* #3 lands,
because #3 eats their shadow.

## Ruled out — do not re-propose

1. **Caching blocks 0-2 across EPOCHS to disk. DEAD TWICE OVER.** Capacity: 34,515 poses × n² × 256
   × 2 B ≈ **~7.8 TB** in bf16, against 721 GiB free and a 3,072 GiB hard quota — it would not fit
   on an empty disk. Bandwidth: even granting infinite disk, reading ~7.8 TB back each epoch exceeds
   the ~52 min of forward compute it would save.
2. **Skipping unused confidence outputs. ALREADY DONE** and live in `mqird6`: `lean_readouts = True`
   and `pae_only = True`. The log prints "LEAN READOUTS: pde, pLDDT and resolved are NOT COMPUTED"
   and "pae-only: all-atom lddt SKIPPED; measured 3.4 ms and up to 1.66 GB per pose". Worth
   recording why it mattered: at the shipped weights PAE is only ~30% of the loss (pLDDT 42%,
   PDE 27%), so most of the gradient used to go to outputs ipTM never reads.
3. **Restricting the PAE readout to the scored interface submatrix.** Worth only 1-3% — a single
   linear against an O(n³) stack. Not worth the code.
4. **Engineering overhead generally.** 77% of the epoch is the model's own forward and backward on
   real poses, so the ceiling on pure overhead elimination is about **1.3x**. There is no large win
   hiding in the data path.
5. **NCCL instead of gloo.** `gloo` is **deliberate**, not an oversight. A reproducible CUDA illegal
   memory access under NCCL is documented at `phase_k_train.py:1017-1026`. And there is nothing to
   win: the payload is **48 bytes**. Whatever `loss_stats_collective_plus_wait` is, it is not
   transport.
6. **`torch.compile`.** Token counts run **128 to 1,650** and every distinct shape is a recompile.
   That is a recompile storm, not a speedup.
7. **TF32 and the cuequivariance kernels. ALREADY ON.**
8. **Fused AdamW.** `apply_step_total` is **1.8 min** of 232. There is nothing there.
9. **`set_to_none`. ALREADY USED.**
10. **Removing `--mem-log` overhead.** Under **1 min/epoch**, and it is off on the arm anyway.
11. **Fusing or deferring the collectives.** `loss_stats_collective_plus_wait` and
    `grad_collective_plus_wait` are 13.5 and 1.1 min, and both are mostly **straggler wait**:
    fusing them relocates the wait to whichever collective is left. The fix for a wait is the
    straggler (#7), not the collective.
12. **Restricting the PAE readout to the scored interface submatrix.** Worth **1-3%** — one linear
    against an O(n³) stack. Not worth the code. (Listed twice now, at item 3 above and here,
    because it keeps being re-proposed.)
13. **Cross-epoch disk caching of activations. ~7.8 TB**, against 721 GiB free and a 3,072 GiB hard
    quota. Dead on capacity and dead again on bandwidth; see item 1.
14. **Lean readouts. ALREADY LIVE**; see item 2.
