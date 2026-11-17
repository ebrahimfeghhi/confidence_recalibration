# Choosing alpha_rank for the 5-site pose-injection rel arm

`hz9ujj_grad_split.csv` is the raw per-step gradient-split measurement from run `hz9ujj`.
**The question: what single `--alpha-rank` should the 60-epoch arm use?**

## What the run is

One epoch, 153 optimiser steps, measured at EVERY step (`--grad-norm-every 1`). The architecture
is the arm's exactly: five pose-injection sites (one before each of the four pairformer blocks,
block 0 using its own pretrained injection and blocks 1-3 copy-initialised from it, plus a fifth
before the PAE readout's layer norm), orientation at all of them, and **only** those sites
trainable -- 15 of 261 head tensors, 55,040 of 19,797,070 parameters, 0.278% of the head.
lr 1e-4, pair-dropout 0.25, loss = PAE + mean-centred regression on the true iTM
(`--rank-form rel`). Warmup is 20 steps.

## Columns

| column | meaning |
|---|---|
| `rank` | which of the 3 data-parallel ranks wrote the row; all three see the same all-reduced gradient, so they are near-duplicates and rank 0 alone is sufficient |
| `step` | optimiser step within the epoch. **Step 1 is absent** -- the recorder arms only after a step completes |
| `poses` | poses in the step (75 = 15 units x nsample 5); constant, so no pose is skipped |
| `g_pae` | L2 norm of the PAE term's gradient over the 15 trainable tensors, pre-clipping |
| `g_rank` | same for the rel term, **with `alpha_rank` already applied** |
| `ratio` | `g_rank / g_pae` |
| `cos` | cosine between the two gradient vectors |
| `alpha_rank` | the alpha this probe ran at (75 throughout) |
| `alpha_for_parity` | `alpha_rank / ratio` -- the alpha that would have equalised the two norms AT THAT STEP. `g_rank` is exactly linear in alpha, so this column is alpha-independent and is the one to read |

## What we see, and why we are asking

- Pooled over all steps the median `alpha_for_parity` is ~100, but the **per-step spread is ~30x**
  (p10 ~22, p90 ~700). Every previous alpha in this project came from ONE accumulation window,
  which could have landed anywhere in that range; 75 was chosen against the mid-range of five such
  draws and sits near this distribution's p25.
- The early steps differ from the rest: steps 2-21 (warmup) median ~170, steps 22+ median ~77.
  One reading is that the head moves under alpha 75 so the arm's *starting* alpha should be the
  early figure. Against that, split into fifths the series runs 189 -> 92 -> 79 -> 72 -> 168, which
  comes back up rather than trending, which looks more like a noisy quantity than a trajectory.
- `g_rank` is a near-cancelling residual: the rel coefficients sum to exactly zero within an
  interface group, so poses cancel against each other. Dropout noise inflates a residual's norm
  proportionally more than it inflates `g_pae`, which biases `ratio` **up** and therefore
  `alpha_for_parity` **down** -- so these numbers are plausibly a lower bound.
- `cos` is small and changes sign (median ~+0.15, negative on roughly a third of steps), i.e. the
  two terms are close to orthogonal and sometimes oppose each other.

## What we would value an opinion on

1. Which statistic of this column should set a single alpha -- median, geometric mean, an early-step
   figure, or something robust to the 30x spread?
2. Is **gradient-norm parity** even the right target? Nothing here establishes that equal norms is
   optimal rather than merely scale-free-sounding. Is there a better-posed criterion given the
   near-cancellation and the near-orthogonality?
3. Does a 0.278%-trainable head make the whole exercise moot -- i.e. is the arm underpowered to move
   held-out DockQ at any alpha? Three prior arms had 113-119 trainable tensors and only matched the
   shipped baseline.
