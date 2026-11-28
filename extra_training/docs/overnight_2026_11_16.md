# Overnight: the zsoftmax arm — what ran, and what to check first

**Live arm: `y565bm`**, launched 12:15 AM EST on 2026-11-16, chain pid 3800969,
`shell_scripts/fullset_chain_zsoft.sh`, watcher `training/runs/arm_watch.log` (10-minute ticks).

## What to read first

1. `training/runs/fullset_chain_zsoft4.log` — one `=== fs_epN ===` table per stop, this head vs
   shipped with gained/lost.
2. `figures/fullset_arm.png` — regenerate with `python -I run_logging/plot_fullset.py`.
3. The per-epoch `Z-SOFTMAX:` line in `training/runs/y565bm/run.log` — acceptable mass, score
   spread and gradient norm. **Read the spread and the norm, not the loss**: m and d are detached,
   so a flat loss is consistent with a live gradient, and the coefficients go as 1/spread.

## The arm

PAE bin loss + z-scored softmax, both at weight 1.0, 60 epochs, shipped init, constant lr 1e-3,
pair dropout 0.25, per-item class balance at DockQ 0.23. Val (141 items, 30 seeds) at epochs
1/2/4/7/10/20/30/40/50/60; the 280-item in-sample slice at the ten-epoch points.

    z_i = (s_i - m)/d   over an interface's candidates, m and d CONSTANTS
    p   = softmax(z)    no temperature
    L   = -log(sum of p over the candidates with DockQ >= 0.23)

Items whose candidates are all acceptable or all unacceptable are skipped and keep their PAE loss.

**alpha 1 is measured parity on the fixed code**: median `alpha_for_parity` **1.001** over the 43
full-coverage steps of probe `f4fkmx` (lr 0, frozen shipped weights, every step measured). Median
cosine between the two gradients **−0.198**, negative on 28 of 43 — the terms lean against each
other, so parity is the largest defensible weight rather than a neutral one.

I predicted the divisor fix would move parity ~2.5x and it did not move at all. The measurement
stands; the prediction was wrong.

## Two defects fixed before this arm started (`70ee2945`)

- **d0 was taken on the cropped pair in the score being differentiated.** Pass A pinned
  `n_pair_tokens` to the uncropped chain pair; pass B and the grad probe did not, so the objective
  descended `Σ c_i(s_A)·s_B,i` rather than the gradient of the loss pass A evaluated. All four
  call sites now agree. **This affected every `rel` arm identically** — their parity numbers are
  stale for this reason too.
- **The ranking term divided by contributing items, not by the step's items**, so a surviving
  item's weight swung ~4x with whatever else landed in its step. It is now a mean over the items
  the step held, matching the PAE term.

## One defect fixed during the previous attempt (`ad68dd3c`)

The chain reused the reference arm's scratchpad TAG, so both arms' row files landed in one
directory. Every report would have been **empty at all ten stops** — silently, because the exit
status came from `tee` — and the PAE-only arm's val/slice numbers would have been overwritten.
The baseline is now copied to `training/runs/pae_ref_scores/` and its tables are tracked in git.

## Known, not fixed

- `phases/trainer/` is a parallel refactor still carrying both of the above defects
  (`unit.py:506`, `:560`, `helpers.py:1228`). It is not what runs, but both return if it is cut over.
- `score_checkpoint.sh` has no `set -o pipefail`, so a report-generation failure still exits 0.
  Deliberately left: adding it overnight would convert a silent empty report into a chain abort
  and idle the cards.
- The `--grad-probe` measurement itself has two biases: `g_pae` only accumulates on poses that
  carried a ranking coefficient, and it measures the unweighted PAE loss while the step applies the
  class-balance weight. Parity is good to ~1.2-1.5x, not two digits.
- `test_guard_fields.py` fails on `epoch_balance`/`pose_batch` — pre-existing half-wiring, not mine.

## The number that reframes the loss

The ranking term trains on **15-seed structures only**, ~665 items an epoch, because of how the
skip rule lands:

| declared seeds | items | all-good | mixed | all-bad |
|---|---|---|---|---|
| 15 | 934 | 22.3% | **71.2%** | 6.5% |
| 1 | 728 | 59.5% | **0.0%** | 40.5% |

None of the 728 single-seed items straddles 0.23 — closest approach 0.2230 below and 0.2550 above,
median within-item DockQ range 0.021. Five poses from one seed are genuinely always one class, so
this is the data and not a label artefact. PAE still trains on all 846 structures.

## Reference arm, for comparison (PAE-only, stopped at epoch 50)

| epoch | slice top-1 / mean q | val top-1 / mean q |
|---|---|---|
| shipped | 30.7% / 0.2505 | 72.3% / 0.5816 |
| 10 | — | 72.3% / 0.5873 |
| 20 | 35.4% / — | 70.9% / 0.5757 |
| 30 | 34.6% / — | — / 0.5680 |
| 40 | 40.7% / 0.2931 | 70.9% / 0.5670 |
| 50 | 49.6% / 0.3208 | 68.8% / 0.5591 |

In-sample rose steadily, held-out fell monotonically after epoch 10 and never cleared shipped.
That is the arm this one has to beat.

## zsoftmax epoch 1 from the previous attempt (same code except the two fixes)

Val 71.6% / 0.5763 against shipped 72.3% / 0.5816, 0 gained / 1 lost; easy bin 95% vs 96%.
Flat after one epoch, nothing collapsed.
