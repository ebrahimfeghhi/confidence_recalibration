# 6v6bbf: WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS, --rank-within-weight 8, --rank-topk-decay 2, 15 epochs

Regenerate with `python -I run_logging/render_findings.py --rid 6v6bbf`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-8 (8 of 15.0).  **Interfaces:** 18.  **Budget:** 30.

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
| lr / warmup / epochs | 0.01 / 20 / 15.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.2612 +- 0.0560 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.5000 +- 0.1213 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.2206 +- 0.0698 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.07000 | 0.04671 +- 0.00981 | 0.2331 | 2.8 | 0.0007 | 3.1423 |
| 2 | 0.04683 | 0.03872 +- 0.00362 | 0.2180 | 5.7 | 0.0014 | 3.6279 |
| 3 | 0.04139 | 0.03735 +- 0.00233 | 0.2124 | 9.3 | 0.0023 | 3.9591 |
| 4 | 0.03994 | 0.03732 +- 0.00221 | 0.2109 | 9.1 | 0.0023 | 4.0270 |
| 5 | 0.03955 | 0.03729 +- 0.00219 | 0.2103 | 8.0 | 0.0020 | 4.0169 |
| 6 | 0.03927 | 0.03738 +- 0.00207 | 0.2087 | 8.1 | 0.0020 | 4.0484 |
| 7 | 0.03915 | 0.03721 +- 0.00213 | 0.2100 | 7.0 | 0.0017 | 4.0117 |
| 8 | 0.03904 | 0.03721 +- 0.00217 | 0.2104 | 6.3 | 0.0016 | 3.9551 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2821 +- 0.0596 | 0.5000 +- 0.1213 | 0.2087 +- 0.0693 | 0.0069 |
| 2 | 0.2676 +- 0.0572 | 0.5556 +- 0.1205 | 0.1800 +- 0.0805 | 0.0049 |
| 3 | 0.2794 +- 0.0593 | 0.5000 +- 0.1213 | 0.1806 +- 0.0773 | 0.0041 |
| 4 | 0.2806 +- 0.0590 | 0.5000 +- 0.1213 | 0.1895 +- 0.0774 | 0.0038 |
| 5 | 0.2806 +- 0.0590 | 0.5000 +- 0.1213 | 0.2038 +- 0.0761 | 0.0036 |
| 6 | 0.2807 +- 0.0591 | 0.5000 +- 0.1213 | 0.2073 +- 0.0740 | 0.0034 |
| 7 | 0.2807 +- 0.0591 | 0.5000 +- 0.1213 | 0.2252 +- 0.0700 | 0.0035 |
| 8 | 0.2612 +- 0.0560 | 0.5000 +- 0.1213 | 0.2206 +- 0.0698 | 0.0035 |

## Paired per-interface test, epoch 1 -> 8

Per-interface selected-pose DockQ (argmax ipTM), 18 interfaces paired.

- paired mean **-0.0208 +- 0.0322** (0.65 sigma)
- 3 up, 8 worsened, 7 unchanged

| interface | ep1 | ep8 | delta |
|---|---|---|---|
| `9bx7_AC` | 0.184 | 0.587 | +0.403 |
| `8og0_AC` | 0.655 | 0.370 | -0.285 |
| `9bx7_BC` | 0.492 | 0.231 | -0.261 |
| `8og0_BC` | 0.716 | 0.646 | -0.070 |
| `9ima_BD` | 0.275 | 0.209 | -0.066 |

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
| 1 | 41.89 +- 13.73 | 1.5 | 9/18 | +4.44 +- 4.93 | 4/5/9 |
| 2 | 40.56 +- 13.22 | 1.0 | 10/18 | +3.06 +- 3.69 | 3/5/10 |
| 3 | 41.61 +- 13.03 | 1.5 | 9/18 | +4.17 +- 3.02 | 2/7/9 |
| 4 | 41.78 +- 12.92 | 1.5 | 9/18 | +4.28 +- 3.15 | 2/7/9 |
| 5 | 42.33 +- 12.87 | 1.5 | 9/18 | +4.89 +- 3.41 | 2/6/10 |
| 6 | 43.39 +- 12.85 | 1.5 | 9/18 | +5.89 +- 3.52 | 2/6/10 |
| 7 | 43.44 +- 13.33 | 1.5 | 9/18 | +5.94 +- 3.37 | 2/6/10 |
| 8 | 42.89 +- 13.32 | 1.5 | 9/18 | +5.44 +- 3.02 | 2/7/9 |

Shipped on the same interfaces: mean rank 37.44 +- 12.54, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep8.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 4.8339e+00 |
| `pose_perblock_frame_dir.0.weight` | 4.8280e+00 |
| `pose_perblock_frame_dir.1.weight` | 4.6169e+00 |
| `pose_perblock_frame_dir.2.weight` | 4.6999e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS, --rank-within-weight 8, --rank-topk-decay 2, 15 epochs. Identical to 24u34d/24u342 in every other setting, so this is a one-variable-pair comparison against the only arm that has ever beaten shipped on frac acceptable. WHAT CHANGES: (1) --rank-pairs all, so the loss forms BOTH cross-seed and within-seed pairs where 24u342 formed cross-seed only -- deployment selects argmax ipTM over ALL poses regardless of seed, so ordering inside a seed is part of what decides the pick and has never entered this objective; (2) --rank-within-weight 8 scales within-seed pairs in the numerator AND the normaliser W, making L a SHARE knob that cannot act as a second --alpha-rank (test_rank_loss.py asserts L normalises away when every pair is within-seed, has no effect under --rank-pairs cross, leaves the pair count alone, and keeps coefficients summing to zero); (3) --rank-tau 0.0270, the median |ipTM gap| over ALL 1,146,184 training pairs of both families across 497 structures, replacing the cross-seed-only DEFAULT_TAU of 0.0305. At L=8 within-seed pairs carry 37.1% of the pairwise score-space gradient mass against 7.0% at L=1 (reproduce: run_logging/within_weight.py --tau 0.0270), while mass reaching INVERTED pairs falls only 75.0% to 71.6% between L=1 and L=8. WHY THE WEIGHT AND NOT A SECOND TAU: within-seed pairs were never short of gradient -- at tau 0.0270 a median within-seed pair on overfit30 (gap 0.0137) weights 0.376 correctly ordered against 0.624 inverted, real discrimination, and both directions push toward the better-DockQ pose so they ADD rather than cancel. What they lack is SHARE: 7.9% of pairs by count. A multiplier is monotone in the quantity being targeted and can be set from a measurement; a second tau changes the weighting's SHAPE and its effect on mass share is not monotone. WHY TAU IS NOT RETUNED PER ARM: tau is fixed to a property of the data rather than to the knob, so the two arms differ in one thing. It does NOT equal the w_ij-weighted median (0.1329 at L=1, 0.0794 at L=8, both above tau) and no rule requires it to; DEFAULT_TAU has always been an unweighted median while the loss ran weight_by_gap=True. PARAMETERS, 44,032 over 12 tensors, unchanged from 24u34d: ANCHORED at lambda 2.5e-4, linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 plus 3 per-block copies = 30,720, so 40,960 over 8 tensors; NOT ANCHORED, frame_dir_proj LinearNoBias(3 -> 256) = 768 plus 3 per-block orientation copies = 2,304 over 4 tensors, excluded via --l2-sp-exclude 'frame_dir' because they are ZERO-INIT and anchoring them to their init would be plain decay toward zero, penalising the term being added. pae_ln and linear_no_bias_pae stay FROZEN. --blocks-per-ckpt 0 IS LOAD-BEARING: the per-block orientation copies read pose_direction's cache, which the outer forward clears after the stack, so a checkpointed block would pair one pose's directions with another's gradient. NOT A RESUME, so --l2-sp snapshots the anchor at init from the SHIPPED head; the log should say 'anchor SNAPSHOT at init' here, which is the opposite of what a resume requires. CONTROLS: 24u342 (this arm minus within-seed pairs, tau 0.0305, 50 epochs, final DockQ 0.2790 +- 0.0568, frac 0.5556, rho 0.2163, paired first-acceptable-rank negative in all 33 evaluated epochs 18-50 with mean -1.25) and the companion arm at the other L. L8 (hronsr) RAN FIRST AT 15 EPOCHS: DockQ 0.3024 against shipped 0.2909, paired +0.0115 +- 0.0088 over the same 18 interfaces (1.3 sigma, 8 better / 4 worse / 6 same), POSITIVE IN ALL EIGHT EPOCHS 8-15 with mean +0.0114 and a minimum of +0.0084, frac unmoved at 0.5556 throughout. Not significant at any single epoch and the epochs are serially correlated, so it is a consistent sign rather than a result. It is also the first arm in this project to put DockQ ABOVE shipped at all -- 24u342's only win was frac, from one interface, with DockQ below. L1 is the control that says whether that came from the within-seed WEIGHTING or merely from tau 0.0270 and the presence of within-seed pairs. CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, derived from this run's own shards. Spearman is reported but is NOT the criterion. TOP-k RANK DECAY K=2: 0 means off; above 0 each pair is weighted by max(v(r_i), v(r_j)) with v(r)=exp(-(r-1)/K) over the current ipTM order, concentrating the loss on the pairs that can change the argmax -- at K=2, 39.3% of weight on pairs touching rank 1 against 1.6% undecayed, and 92.8% inside the top 5. Ranks come from DETACHED scores, so the weight is constant in the backward and this is not the exact gradient of a fixed objective. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: L=8 landing on L=1 within error on DockQ and frac, which would say the within-seed family carries no usable signal at any weight this objective can give it.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.0090, 0 non-finite gradient(s).

