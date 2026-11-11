# Cropped training arm: the plan

Status: **proposed, not built.** No GPU work, no change to `87ja6a`.

## The recipe

- **Fixed crop size 256** — Protenix's own `train_crop_size`.
- **Pair-restricted anchor at 15 A.** A token may anchor only if it lies within 15 A of a token on
  the OTHER member of a scored pair. Protenix's rule accepts a contact with ANY distinct chain, so
  an antibody heavy-chain token qualifies by packing against the light chain and the crop centres
  away from the epitope. That is the measured cause of its 27.6% unusable rate (section 21).
- **Only the anchor is restricted.** k-nearest still runs over the WHOLE assembly, same crop size,
  same `arange * 1e-3` tie-break. Third chains can still supply context. The vendored Protenix
  implementation is untouched; a local sampler is selected explicitly.
- **One crop per STRUCTURE**, shared by every pose and every scored interface. Per-pose crops make
  the iTM gap between two poses measure the crop; per-interface crops hide the fact that 82% of
  structures carry more than one scored interface and would force a second forward pass.
- **60 crops per structure, pre-computed, one drawn per epoch.** NOT a single deterministic crop:
  the model should see multiple views, which is most of why Protenix crops randomly at all.
- **Keep only VALID crops.** A crop is valid when every scored interface of the structure still has
  at least one surviving cross-chain pair. Sample until 60 valid ones exist.
- **Rebuild `--rel-labels` per crop**, with `d0` PINNED at the uncropped scored chain-pair token
  count.
- **Validation stays uncropped.** Every historical number is uncropped and the comparison must not
  move.

## Why these choices, with the measurements behind them

**Why not screen on regret.** Screening out crops that produce bad DockQ regret would select crops
using the answer. It is also unnecessary: only **2.0%** of crops cost more than 0.10 DockQ and
**4.0%** more than 0.05 (section 23 thresholds), and a fresh crop is drawn every epoch, so a
structure sees roughly ONE bad draw in 60. That is noise that averages out, which is the whole
point of keeping it random. The validity screen is pure geometry and carries no such risk.

**Why 256 and not the two-tier 256/512.** The only thing the 512 tier bought was joint coverage,
0.967 -> 0.993. The validity screen gives **1.000 by construction**, because an invalid crop is
simply not kept. So 256 + screen is better than two-tier on coverage AND keeps the full **3.56x**
against two-tier's 3.00x. The two-tier recommendation in section 23 is superseded by this.

**Why 60 valid crops is cheap.**

    draws needed to reach 60 valid:  mean 62.1,  p99 104.7,  max 161.3
    all 434 structures reach 60 valid comfortably
    label storage for 60 crop sets:  0.09 GB against a 3,072 GB quota

**Why the label rebuild is ~15 minutes, not hours.** The expensive step is
`PAELoss.calculate_label` on every pose, and it runs ONCE regardless of crop count. Evaluating a
crop is then re-averaging the cached bin indices over a different pair subset -- one sgemm. That
is why the 886-interface x 1000-crop analysis took ten minutes rather than ten times something.
60 label sets is one bin pass plus 60 cheap re-averages.

**Expected speed.** 3.56x on the training pose work, from the measured per-pose cost
`C0 + K2*N^2 + K3*N^3` fitted on 87ja6a epoch 12. `C0 = 0.081 s/pose` is the non-scaling part
(trunk H2D, optimiser step, collective) and is already inside that figure -- 15.5 min of the
94.3 min epoch. The held-out eval (~12.5 min every 5 epochs) does NOT shrink, so a 5-epoch block
goes 484 -> 145 min rather than 484/3.56.

## Build order

1. `run_logging/build_crop_pool.py` -- per structure, sample pair-anchored 256-token crops until 60
   are valid; write `<pdb>.npz` with `[60, n_token]` bool masks plus a `__meta` record carrying the
   anchor rule, crop size, contact threshold, seed, the unit SHA digest and the builder commit.
   `--verify-rule` re-derives one mask from scratch and asserts bit-identity.
2. `run_logging/build_itm_labels_cropped.py` -- one `calculate_label` pass per pose, then 60
   re-averages per interface; writes `itm_labels_crop/<pdb>.npz` keyed by
   `(crop_index, iface, seed, sample)`. d0 from the UNCROPPED scored chain-pair count.
3. Trainer: `--crop-pool <dir>` selects crop `epoch % 60` for each structure, and `--rel-labels`
   points at the cropped label directory. Refuse at startup if the pool and the labels disagree on
   crop count, or if any structure's selected crop leaves a scored interface empty.
4. Gates before any GPU time: full-mask equivalence (crop 0 == uncropped when the mask is all
   ones), label-vs-analysis agreement on real pose keys, and a dry-run that prints the per-epoch
   crop index and token count for a sample of structures.

## What this does NOT establish

Everything measured so far is about the **label**: the target a cropped head would be trained
against still orders poses correctly, slightly better than uncropped. Sections 13 and 14 found
cropped training improved rho WITHOUT improving selection, three times over. Nothing here
contradicts that, and the arm has to be run to find out.
