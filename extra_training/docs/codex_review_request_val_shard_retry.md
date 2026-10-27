# Review request: held-out PAE shard retry (commits `ddeafe89`..`07d6b656`)

Please review two commits that change how a failed held-out-PAE shard is handled, plus one
one-off recovery tool. Prior rounds of this review found four real P1/P2s that I had missed, so
please be adversarial; I am more interested in what I got wrong than in confirmation.

## What happened

`tv6hue` epoch 4 spawned 3 held-out PAE shards over cards 4,5,7. Shard 0 finished normally
(2.1 min, 3.80 GB peak). Shards 1 and 2 both died inside `torch._C._cuda_init()`:

- shard 1: `RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.`
- shard 2: `RuntimeError: No CUDA GPUs are available`

`heldout_pae()` voided the whole epoch's number, by design, because pooling the survivors would
report a mean over a biased subset of the held-out set.

Ruled out: not OOM (that raises `CUDA out of memory`); not compute-mode exclusivity (cards 4,5,7
all report `Default`); no Xid in dmesg; the three GPU reservation holders stayed alive; ECC
uncorrected = 0; MemAvailable 806 GB. **First occurrence in the project** — no other run's log
contains either string. Root cause NOT established.

## The three changes

**1. `phases/phase_k_train.py` — retry + stagger (parent side).**
Failed shards are re-run up to `--val-loss-shard-retries` (default 2) before the epoch is voided;
children are spawned 2 s apart instead of simultaneously; a previous attempt's `shard.log` is
rotated to `shard.log.failed<N>` rather than overwritten, because the pooling scans `shard.log`
for the `SHARD COVERAGE` line and a stale log would be read as the live one. Only the failed
indices are re-run. `procs` is rebuilt from the final handles.

**2. `run_logging/recover_val_pae.py`** — re-runs only the failed shards of a PAST epoch against
the same `resume_ep<N>.pt` and pools with the parent's guards (disjointness, one coverage line per
shard, population agreement, all units covered, exact pose count). It rebuilds the parent's
command from `args.json` and then compares the re-run shard's written `args.json` field-by-field
against a shard the parent launched, allowing only `out`/`manifest_shard`/`resume_note` to differ.
Used once: tv6hue epoch 4 recovered to **2.752259 over 7,950 poses**.

**3. `run_logging/val_shard_retry_repro.py`** — 5-case test of the control flow with `Popen`
faked, plus `assert_source_matches()` over 7 load-bearing lines.

## What I checked

- 5/5 retry cases pass, including "fails forever -> still voids, 5 spawns, no infinite loop".
- `phases/test_guard_fields.py` passes. `val_loss_shard_retries` is deliberately NOT in the resume
  guard's `_mine` allowlist (it is a retry count, not an objective), so old bundles stay resumable.
- `run_logging/test_records.py` 10/10.
- The module **executes** (not merely parses) — load-bearing because an arm is running right now
  and its eval children re-read this file at spawn.

## What I want you to attack

1. **`_spawn` rotating the log.** Is there a path where the rotation loses the only record of a
   failure, or where a `.failed` file is later read as live? The pooling reads `shard.log` by path
   from `procs`; is that still true after a retry?
2. **Is `procs = [_handles[_i] for _i in range(_nsh)]` correct** when some handles are first-attempt
   and some are retries? Could it raise `KeyError`, or hold a closed file handle the pooling then
   uses?
3. **`_await` closes `_fh`, and a retried shard reopens the same path.** Any fd or truncation
   hazard, especially if the first attempt's child is somehow still writing?
4. **The stagger is a mitigation, not a diagnosis.** Do you see a plausible root cause I missed
   for a CUDA context-creation failure under these conditions (3 children created simultaneously
   against cards whose training ranks hold ~67 GB each, persistence mode Disabled)? If the real
   cause would ALSO defeat the retry, I want to know now.
5. **`recover_val_pae.py` rebuilding the parent's command.** It duplicates `heldout_pae()`'s
   construction by hand. The args-comparison is meant to catch divergence — is it sufficient, or
   is there a field that could differ and NOT show up in the child's written `args.json`?
6. **Did appending the recovered row to `val_loss.csv` by hand do anything wrong?** The row is
   `4,976,7950,2.752259,resume_ep4.pt`; the writer is append-only; step 976 was inferred from the
   244/epoch cadence and cross-checked against `head_ep4_step976.pt`.

## Scope

`extra_training/phases/phase_k_train.py` (the `if _nsh > 1:` block in `heldout_pae()`, and the new
`--val-loss-shard-retries` argument), `extra_training/run_logging/recover_val_pae.py`,
`extra_training/run_logging/val_shard_retry_repro.py`. Read-only; an arm is mid-flight on cards
4,5,7 — please do not run anything on a GPU.
