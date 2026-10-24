# hronsr: WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS, --rank-within-weight 8, --rank-topk-decay 0, 25 epochs

Regenerate with `python -I run_logging/render_findings.py --rid hronsr`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-6 (6 of 25.0).  **Interfaces:** 18.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.027 |
| alpha_rank / alpha_pae | 1.0 / 0.0 |
| trainable (from the run's own log) | **12 tensor(s), 0.0M of 464.5M parameters (0.0%)** |
| `--train-only` patterns | 3 |
| L2-SP lambda | 0.00025 |
| L2-SP excluded | `frame_dir` -- 4 tensor(s) out of the anchor, 8 anchored |
| pose injection | per-block=True copies=True orientation=True |
| lr / warmup / epochs | 0.01 / 20 / 25.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.2900 +- 0.0574 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.5556 +- 0.1205 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.2400 +- 0.0667 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.04661 | 0.04887 +- 0.01106 | 0.2374 | 0.9 | 0.0002 | 3.1449 |
| 2 | 0.03915 | 0.04028 +- 0.00499 | 0.2225 | 3.7 | 0.0009 | 3.5360 |
| 3 | 0.03817 | 0.03818 +- 0.00352 | 0.2188 | 4.9 | 0.0012 | 3.7282 |
| 4 | 0.03774 | 0.03900 +- 0.00426 | 0.2209 | 3.6 | 0.0009 | 3.6362 |
| 5 | 0.03769 | 0.03883 +- 0.00423 | 0.2211 | 3.2 | 0.0008 | 3.6155 |
| 6 | 0.03743 | 0.03827 +- 0.00382 | 0.2197 | 3.8 | 0.0009 | 3.6569 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2737 +- 0.0601 | 0.5000 +- 0.1213 | 0.2004 +- 0.0703 | 0.0070 |
| 2 | 0.2808 +- 0.0608 | 0.5556 +- 0.1205 | 0.2076 +- 0.0729 | 0.0052 |
| 3 | 0.2607 +- 0.0570 | 0.5556 +- 0.1205 | 0.2335 +- 0.0715 | 0.0047 |
| 4 | 0.2686 +- 0.0574 | 0.5556 +- 0.1205 | 0.2208 +- 0.0698 | 0.0048 |
| 5 | 0.2849 +- 0.0577 | 0.5556 +- 0.1205 | 0.2294 +- 0.0681 | 0.0046 |
| 6 | 0.2900 +- 0.0574 | 0.5556 +- 0.1205 | 0.2400 +- 0.0667 | 0.0043 |

## Paired per-interface test, epoch 1 -> 6

Per-interface selected-pose DockQ (argmax ipTM), 18 interfaces paired.

- paired mean **+0.0163 +- 0.0286** (0.57 sigma)
- 6 up, 2 worsened, 10 unchanged

| interface | ep1 | ep6 | delta |
|---|---|---|---|
| `9bx7_AC` | 0.184 | 0.587 | +0.403 |
| `8og0_AC` | 0.655 | 0.370 | -0.285 |
| `7r58_AB` | 0.011 | 0.087 | +0.076 |
| `7qu2_BC` | 0.015 | 0.070 | +0.055 |
| `9bx7_BC` | 0.492 | 0.528 | +0.036 |

## Rank of the first acceptable pose, ordering by ipTM

`frac acceptable` only asks whether rank 1 happens to be acceptable, so it cannot see an
arm that lifts the best pose from rank 90 to rank 2. This can. Rank is 1-indexed and
**lower is better**; rank 1 everywhere is frac 1.0.

Scored over 150 poses per interface, threshold DockQ >= 0.23, 0 interface(s) with no acceptable pose anywhere (those have no rank and are excluded; a non-zero count here would bias the mean optimistically, since the hardest interfaces would be the ones dropped).

The unpaired s.e. is large because a few interfaces first become acceptable in the 90s
while most are acceptable at rank 1, so between-interface variance swamps the effect.
**The paired column is the one that can resolve a change** -- every interface is scored
by both orderings over the same poses, so the pairing is exact.

| epoch | mean rank +- s.e. | median | at rank 1 | paired vs shipped | better/worse/same |
|---|---|---|---|---|---|
| 1 | 44.89 +- 13.91 | 1.5 | 9/18 | +7.44 +- 4.77 | 3/6/9 |
| 2 | 41.94 +- 13.85 | 1.0 | 10/18 | +4.44 +- 3.02 | 2/5/11 |
| 3 | 40.89 +- 13.56 | 1.0 | 10/18 | +3.44 +- 3.19 | 3/4/11 |
| 4 | 42.28 +- 14.11 | 1.0 | 10/18 | +4.78 +- 3.40 | 3/4/11 |
| 5 | 42.44 +- 14.18 | 1.0 | 10/18 | +5.00 +- 3.25 | 3/4/11 |
| 6 | 43.39 +- 14.43 | 1.0 | 10/18 | +5.89 +- 3.57 | 3/4/11 |

Shipped on the same interfaces: mean rank 37.50 +- 12.56, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep6.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 3.9871e+00 |
| `pose_perblock_frame_dir.0.weight` | 3.9802e+00 |
| `pose_perblock_frame_dir.1.weight` | 3.2943e+00 |
| `pose_perblock_frame_dir.2.weight` | 3.1876e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS, --rank-within-weight 8, --rank-topk-decay 0, 25 epochs. Identical to 24u34d/24u342 in every other setting, so this is a one-variable-pair comparison against the only arm that has ever beaten shipped on frac acceptable. WHAT CHANGES: (1) --rank-pairs all, so the loss forms BOTH cross-seed and within-seed pairs where 24u342 formed cross-seed only -- deployment selects argmax ipTM over ALL poses regardless of seed, so ordering inside a seed is part of what decides the pick and has never entered this objective; (2) --rank-within-weight 8 scales within-seed pairs in the numerator AND the normaliser W, making L a SHARE knob that cannot act as a second --alpha-rank (test_rank_loss.py asserts L normalises away when every pair is within-seed, has no effect under --rank-pairs cross, leaves the pair count alone, and keeps coefficients summing to zero); (3) --rank-tau 0.0270, the median |ipTM gap| over ALL 1,146,184 training pairs of both families across 497 structures, replacing the cross-seed-only DEFAULT_TAU of 0.0305. At L=8 within-seed pairs carry 37.1% of the pairwise score-space gradient mass against 7.0% at L=1 (reproduce: run_logging/within_weight.py --tau 0.0270), while mass reaching INVERTED pairs falls only 75.0% to 71.6% between L=1 and L=8. WHY THE WEIGHT AND NOT A SECOND TAU: within-seed pairs were never short of gradient -- at tau 0.0270 a median within-seed pair on overfit30 (gap 0.0137) weights 0.376 correctly ordered against 0.624 inverted, real discrimination, and both directions push toward the better-DockQ pose so they ADD rather than cancel. What they lack is SHARE: 7.9% of pairs by count. A multiplier is monotone in the quantity being targeted and can be set from a measurement; a second tau changes the weighting's SHAPE and its effect on mass share is not monotone. WHY TAU IS NOT RETUNED PER ARM: tau is fixed to a property of the data rather than to the knob, so the two arms differ in one thing. It does NOT equal the w_ij-weighted median (0.1329 at L=1, 0.0794 at L=8, both above tau) and no rule requires it to; DEFAULT_TAU has always been an unweighted median while the loss ran weight_by_gap=True. PARAMETERS, 44,032 over 12 tensors, unchanged from 24u34d: ANCHORED at lambda 2.5e-4, linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 plus 3 per-block copies = 30,720, so 40,960 over 8 tensors; NOT ANCHORED, frame_dir_proj LinearNoBias(3 -> 256) = 768 plus 3 per-block orientation copies = 2,304 over 4 tensors, excluded via --l2-sp-exclude 'frame_dir' because they are ZERO-INIT and anchoring them to their init would be plain decay toward zero, penalising the term being added. pae_ln and linear_no_bias_pae stay FROZEN. --blocks-per-ckpt 0 IS LOAD-BEARING: the per-block orientation copies read pose_direction's cache, which the outer forward clears after the stack, so a checkpointed block would pair one pose's directions with another's gradient. NOT A RESUME, so --l2-sp snapshots the anchor at init from the SHIPPED head; the log should say 'anchor SNAPSHOT at init' here, which is the opposite of what a resume requires. CONTROLS: 24u342 (this arm minus within-seed pairs, tau 0.0305, 50 epochs, final DockQ 0.2790 +- 0.0568, frac 0.5556, rho 0.2163, paired first-acceptable-rank negative in all 33 evaluated epochs 18-50 with mean -1.25) and the companion arm at the other L. CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, derived from this run's own shards. Spearman is reported but is NOT the criterion. TOP-k RANK DECAY K=0: 0 means off; above 0 each pair is weighted by max(v(r_i), v(r_j)) with v(r)=exp(-(r-1)/K) over the current ipTM order, concentrating the loss on the pairs that can change the argmax -- at K=2, 39.3% of weight on pairs touching rank 1 against 1.6% undecayed, and 92.8% inside the top 5. Ranks come from DETACHED scores, so the weight is constant in the backward and this is not the exact gradient of a fixed objective. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: L=8 landing on L=1 within error on DockQ and frac, which would say the within-seed family carries no usable signal at any weight this objective can give it.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.0071, 0 non-finite gradient(s).

