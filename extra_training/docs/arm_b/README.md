# ARM B — six pose-injection sites + `pae_ln`, PAE-only, lr 1e-3

**This is the best arm so far.** Run `w57hjo`, 20 epochs, 2026-11-18 00:00–01:44 EST.
Its learning-rate fork is `32khid` (Arm B decay), resumed from Arm B's own epoch 10.

`armB_w57hjo_val_slice.csv` and `armB_decay_32khid_val_slice.csv` hold one row per eval pass,
32 columns each, recomputed from the per-candidate rows.

## What the arm is

Six pose-injection sites: one before each of the four pairformer blocks (block 0 its own
pretrained injection, blocks 1–3 copy-initialised from it), a fifth before the PAE readout's
layer norm, and a sixth **after** it, zero-initialised. The sixth is the one that can *scale*
the pair feature — pre-norm can only rotate it, since the layer norm divides magnitude out.
**`pae_ln` is unfrozen**, which is the only difference from Arm A.

**20 trainable tensors, 66,560 of 19,797,070 parameters (0.34% of the head).** Everything else
frozen. Loss is PAE only — no ranking term. Pair-dropout 0.25. `--pae-split-accept 0.23` tallies
the good/bad split; **`--pae-balance-accept` is 0, so the objective is NOT balanced** (deliberate,
for comparison against the four baselines that had it at 0.23).

Training population: 846 structures, 34,610 poses — 412 structures at 1 seed and 434 at 15, which
is the trim, not an accident.

## Results

Shipped baseline, measured on the same rows: val q 0.5819, top-1 72.3%, rho 0.1882;
slice q 0.2505, top-1 30.7%, rho 0.0819.

| arm | ep | val q (mean-pool) | val top-1 | val rho | slice q | slice top-1 |
|---|---|---|---|---|---|---|
| Arm B | 10 | **0.5940** | **73.0%** | 0.2166 | 0.2471 | 30.4% |
| Arm B | 20 | 0.5908 | **73.0%** | 0.2158 | 0.2473 | 29.6% |
| Arm B decay | 20 | 0.5930 | **73.0%** | 0.2176 | — | — |

Max-pooled for the same rows is in the CSV; it is flat at 72.3% top-1 everywhere, for every arm,
including shipped. **The entire top-1 gain exists only under mean pooling.**

## THE OPEN PROBLEM

**Good-pose PAE loss stops improving at epoch 12 and then rises**, while bad-pose loss keeps
falling for the whole 20 epochs:

| | min | at epoch | epoch 20 | drift |
|---|---|---|---|---|
| good poses | 3.1010 | 12 | 3.1148 | **+0.0138** |
| bad poses | — | — | 2.6472 | falls throughout, 2.8336 → 2.6472 |

Held-out mean DockQ peaks at epoch 10 (0.5940) and gives some back by 20 (0.5908), so the
good-pose minimum is close to a leading indicator of the metric we actually care about.

### Two explanations tested and RULED OUT

1. **Learning rate.** Every arm ran at a constant 1e-3 — `lr_final` defaults to 0, which sends
   `lr_at_step` down its "hold lr flat" branch. Arm B decay forked at epoch 10 and cosine-decayed
   to 1e-4 by epoch 20. **The training loss is identical to four decimals**: both bottom at 3.1010
   at epoch 12, both end at ~3.115. It is not the optimiser bouncing at too large a step.
2. **Capacity.** Arms C and D added rank-4 and rank-8 LoRA on `blocks.3.pair_transition`
   (+15,360 and +30,720 parameters, 97,280 total for rank 8 — a 46% increase over Arm B).
   The rank-8 arm's good-pose curve bottoms at **the same epoch 12** and rises by **+0.0145**,
   almost exactly Arm B's +0.0138. More capacity lowers the floor ~0.009 and changes nothing
   about the shape.

### The measurement that ties them together

The weights move a lot while the metrics do not:

- flat vs decay, epochs 11–20: `‖Δw‖` = **50.45** vs **29.62** from the shared epoch-10
  checkpoint, and the two final heads are **21.36** apart — different models, training loss
  identical to 4 dp.
- rank-4 vs rank-8 heads: **12.87** apart on the 20 shared dense tensors, val mean q **0.5853 vs
  0.5853** and rho **0.2143 vs 0.2146**.

That is a flat basin: the objective does not care where in it the head lands.

## WHAT TO CHALLENGE

1. The good/bad split excludes ~6% of poses that straddle (a pose can be acceptable for one
   interface of a structure and not another, and one crop is shared by all of them, so it has two
   contradictory labels and is in neither group). Does that bias the turn-up?
2. `--pae-balance-accept` is the obvious next lever and **has never been A/B'd on any arm**. Is
   reweighting good-vs-bad the right fix, or does it just trade the same flat basin for another?
3. Is a +0.0022 mean-q difference (decay vs flat at epoch 20) distinguishable from noise on 141
   interfaces? It is roughly one interface changing hands.
4. Mean pooling carries the whole top-1 gain and max pooling carries none. Which should be the
   headline, given the shipped convention is max?
5. Slice (in-sample) sits BELOW shipped for Arms A and B while val sits above. The pools differ in
   difficulty — slice is 34% single-good-pose, val is 75% all-five-good — but the sign flip holds
   *within* bin 5 alone (val +0.0136, slice −0.0129), so composition does not explain it.
