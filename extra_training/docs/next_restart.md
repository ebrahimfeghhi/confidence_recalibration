# What is waiting for the next restart

`ocqtyt` is running epochs at 35 min with the code it loaded at launch. Everything below is
committed and on disk but takes effect only when a process starts fresh. **Smoke test before
relaunching** — none of it has run on a GPU, and a NameError at an epoch boundary is how x3k2m4
lost 77.9 minutes.

## 1. Multi-worker prefetch — the reason to restart

`--prefetch-workers N`, default 1. Epochs 7 and 8 hit a 100% prefetch hit rate and *still* spent
11-16 min per epoch waiting on the worker: one thread needs ~0.7 s/unit against ~0.84 s of GPU,
and the GIL eats the margin. The crop did not disappear, it moved from the main thread into the
wait. Expect roughly 35 → 25 min/epoch; unverified.

**This is a threading change and the failure mode to fear is a deadlock at the epoch boundary**,
which would hang silently against `--dist-timeout-min 240`. One case is already guarded: the
queue is drained *between* joins rather than once before them, because a worker blocked on a
full `put()` never sees the stop flag and the second join would hang as soon as the first refills
the queue. **Smoke test at least TWO epochs** so `_pf_start`/`_pf_end` run across a boundary more
than once. `--epochs 8.06` from `resume_ep8.pt` gives two short epochs.

Suggested: `--prefetch-workers 4`. The box allows 64 cores; 3 ranks x 4 workers is 12.

## 2. The vestigial deletion

117 lines gone: `item_cache` and its RAM-floor eviction, `train_ds`, `byp`, the `nodata` check,
`item_build_s`, `ram_floor`, three live-feature branches. `--alignment-preflight` is now required
unconditionally rather than only under `--minimal-items`, because with the per-structure path
deleted, omitting the flag would drop `check_alignment` entirely. `--minimal-items` is accepted
and ignored.

## 3. Two capacity-control arms, neither started

`shell_scripts/launch_pose_focused_arm.sh` — **run this one first**. `--train-only` on the
complete pose pathway plus the PAE readout: 26,624 parameters, 0.135% of the head. Consider
appending `,pae_ln` (+512 params), and note lr 1e-2 is inherited from a 19.8M-parameter problem
and is untuned for this one. A flat held-out curve will not distinguish success from
underfitting — the training-PAE curve is what separates them.

`shell_scripts/launch_lora_arm.sh` — rank 8, 567,112 parameters, 4.42% of the head.

Both need the cards, so neither can run beside `ocqtyt`. **weight_decay is 0.0 in both and always
has been**; until a weight-decay arm exists, "LoRA helped" and "any regularisation helped" are
not distinguishable.

## 4. Backfill epoch 7's held-out PAE

`shell_scripts/backfill_val_pae.sh 7` — refuses while a trainer is live, because mid-epoch the
cards sit at ~70 of 97.9 GB and an OOM there costs the run, not just the measurement. Run it
when the cards are free. Epoch 7 is marked incomplete until then.

## Already live on `ocqtyt`, no restart needed

- `--loss-only` exempt from the mandatory `--resume-note` (the val subprocess reads this file
  fresh each boundary, so it picked the fix up at epoch 8 — confirmed, `pae 2.6451 poses 1500`).
- The remainder fix in `build_epoch`, same route: without it a held-out pass measured 1,445 poses
  instead of 1,500.
- `metrics_io` completeness: per-segment step expectation, and an epoch only publishes once a
  later epoch exists.
