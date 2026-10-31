# Uncropped rel-iTM arm (`erb7nd`) — state for review

Launched 2026-10-30 3:22 PM EDT. Live. Everything below is reproducible from this repo.

## What this run is

The same mean-centred iTM regression arm as `o2qczq`, with **exactly one training variable
changed**: `--crop-margin 10 --crop-contact 5` → `--crop-margin 0`.

    L = PAE + 75 * (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2

per scored chain pair over all 50 poses of a structure, `s` = MEAN-pooled predicted ipTM,
`g` = the same from the true error bins.

Launch script: `shell_scripts/launch_rel_itm_beta75_uncropped.sh`
Config as run: `training/runs/erb7nd/args.json`

## Why

`paper_results.md` §8–§9c, in order:

- §8 — scored UNCROPPED, the cropped arm is **−0.0087** in-sample against the shipped head.
- §8b — its ranking gain sits entirely outside the top of each pose list: Spearman over all 50
  poses +0.2819, over the **top 10** +0.0326, picks the best pose 3.79% against shipped 3.98%.
- §8c — but `phase_k_eval` had no crop path at all, while the trainer cropped **every** pose
  (199,800 of 199,800, median 53% of tokens kept). So §8 read a cropped-trained head off its
  training distribution.
- §9 — with the crop MATCHED on both heads, in-sample goes **−0.0087 → +0.0733**, and the top-10
  Spearman +0.0326 → +0.2040.
- §9b — held out, matched crop: **+0.0376 mean DockQ (t=+2.86)**, **+0.0845 Spearman (t=+4.10)**,
  frac acceptable 0.7826 vs 0.7283.
- §9c — **the crop is built from `label_coordinate`, i.e. GROUND TRUTH.** No number in §9/§9b is a
  deployment number. This run removes the crop from training instead, which needs no label.

## What changed in the code to make it runnable

- §10 — uncropped training OOMed in the FORWARD on all three ranks at ~91 GiB of 94.97, inside
  `triangle_multiplicative_update`. Not fragmentation (reserved-but-unallocated < 0.1% of the card).
- `1c77b619` — per-block orientation is now correct under activation checkpointing instead of
  refused. `_prep_blocks` captures `pose_direction._CALL["dir"]`; `_add_dir` takes it as an
  argument, so a block recomputed during backward sees what its original forward saw.
  Gate: `run_logging/test_perblock_ckpt_equiv.py` — forward identical to 9 decimal places, **all 18
  parameter gradients at worst 0.000e+00**, all four orientation projections with NON-ZERO weights
  and non-zero gradients, 3 poses with different geometry, cache cleared between poses, and a
  NEGATIVE CONTROL that restores the pre-fix path and fails as it must.
- §11 — with `--blocks-per-ckpt 1` the nine-largest probe completed forward, backward and 3
  optimiser steps. Peak reserved 91.94 GB; device-free bottomed at 1,137 MiB on the tightest card.

## The gate that refused this launch

`shell_scripts/greenlight_uncropped.sh` ran the three pre-registered checks and said NO LAUNCH:

    PASS  no OOM, no guard STOP, 3 optimiser steps, no non-finite gradients
    FAIL  peak reserved below 91.94 GB            worst 91.94
    FAIL  >=5120 MiB free at the tightest moment  card4 1882  card5 3137  card7 1137

**It was launched anyway, on the user's explicit override.** The case: peak is driven by the
largest SINGLE unit, `9tuz` at 1543 tokens is the largest structure in the whole manifest and it
survived the probe, and a crash costs one epoch rather than the run.

## How this differs from the cropped arm, in full

| | cropped (`o2qczq`) | uncropped (`erb7nd`) |
|---|---|---|
| crop | margin 10, contact 5 | **none** |
| activation checkpointing | `--blocks-per-ckpt 0` | **`--blocks-per-ckpt 1`** |
| eval cadence | every 5 epochs | **every epoch** |
| everything else | — | identical |

Checkpointing is an implementation change, gated to produce identical gradients. The eval cadence
is a measurement change. Both are stated rather than folded into "one variable".

## Known open issues

- **47 of the 738 training interfaces have NO acceptable pose** (best DockQ < 0.23 over seeds
  500000–500009), e.g. `7uds_FL` 0.046, `7ugq_CG` 0.054. The structure-level filter kept them
  because another interface of the same structure is winnable. They are 6.4% of the training
  signal and no selector can score above 0 on them. NOT changed for this run, because the cropped
  arm trained on this exact pool and removing them would add a second variable.
- `plot_val_series.py` picks the argmax with `max()`, which resolves ties by row order;
  `stored_iptm` is recorded at 4dp so poses tie. This moved the shipped baseline by 0.0001 between
  epochs and trips the figure's own drift warning. Immaterial at that size, and the mechanism
  should be removed rather than tolerated — the same tie-handling already exists in
  `run_logging/rank_gain_location.py`.
- Health and per-epoch push both inherited a `>= 5500 poses` epoch-complete threshold from an arm
  with a different epoch size; on a 6100-pose epoch it fired at 5600 and published epoch 1 with
  `rel`, `rho` and `sd_ratio` all `-`. Fixed in `ae9dcd19` and `9521adf1`; both now trigger on the
  trainer's own `[rank 0] epoch N rank-loss` line.

## Where the numbers are

    figures/train_series_erb7nd.png      training metrics vs epoch
    figures/val_series_erb7nd.png        held-out metrics vs epoch
    training/runs/erb7nd_table.txt       the per-epoch table
    training/runs/erb7nd/args.json       the config as run
    training/runs/erb7nd/run.log         epoch summaries, memory, timings
    training/runs/erb7nd/metric_valfull_ep<N>.json   held-out metric per epoch

## Results so far

    TRAINING   ep       PAE        rel      rho  sd_ratio
                1    3.1671    0.01230  +0.1652     0.240
                2    3.0691    0.01189  +0.1970     0.344

    HELD OUT   ep  ours MEAN  ours MAX  shipped    dMEAN     dMAX
                1     0.5832    0.5816   0.5830  +0.0002  -0.0014
                2     0.5848    0.5847   0.5831  +0.0017  +0.0015

    held-out Spearman (92 interfaces, 150 candidates)
                1   MEAN +0.2074  MAX +0.1948   shipped +0.1709
                2   MEAN +0.2001  MAX +0.2125   shipped +0.1709

Two points. The held-out Spearman series swung ±0.08 between adjacent epochs on the cropped arm,
so nothing here is yet distinguishable from zero. The decision point is epoch 10.
