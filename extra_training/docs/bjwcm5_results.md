# bjwcm5: WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS, --rank-within-weight 1, --rank-topk-decay 0, 15 epochs

Regenerate with `python -I run_logging/render_findings.py --rid bjwcm5`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-13 (13 of 15.0).  **Interfaces:** 18.  **Budget:** 30.

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
| selected DockQ | 0.2841 +- 0.0541 | 0.2909 +- 0.0620 | 0.5968 |
| frac > 0.23 | 0.6111 +- 0.1182 | 0.5556 +- 0.1205 | 1.0000 |
| Spearman rho | 0.2468 +- 0.0674 | 0.1958 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.04928 | 0.04859 +- 0.01084 | 0.2368 | 1.1 | 0.0003 | 3.1496 |
| 2 | 0.03965 | 0.04036 +- 0.00500 | 0.2220 | 4.0 | 0.0010 | 3.5172 |
| 3 | 0.03843 | 0.03822 +- 0.00344 | 0.2178 | 5.6 | 0.0014 | 3.7249 |
| 4 | 0.03778 | 0.03919 +- 0.00413 | 0.2192 | 4.3 | 0.0011 | 3.6078 |
| 5 | 0.03766 | 0.03945 +- 0.00434 | 0.2191 | 3.8 | 0.0010 | 3.5271 |
| 6 | 0.03729 | 0.03894 +- 0.00378 | 0.2169 | 4.4 | 0.0011 | 3.6039 |
| 7 | 0.03724 | 0.03966 +- 0.00424 | 0.2170 | 4.2 | 0.0010 | 3.4650 |
| 8 | 0.03680 | 0.03939 +- 0.00388 | 0.2160 | 4.0 | 0.0010 | 3.5242 |
| 9 | 0.03695 | 0.03971 +- 0.00407 | 0.2165 | 3.7 | 0.0009 | 3.4873 |
| 10 | 0.03690 | 0.03895 +- 0.00344 | 0.2150 | 4.3 | 0.0011 | 3.6148 |
| 11 | 0.03663 | 0.03971 +- 0.00404 | 0.2156 | 4.0 | 0.0010 | 3.3991 |
| 12 | 0.03671 | 0.03864 +- 0.00358 | 0.2162 | 4.4 | 0.0011 | 3.4732 |
| 13 | 0.03680 | 0.03825 +- 0.00344 | 0.2175 | 4.1 | 0.0010 | 3.5737 |

Shipped, on these same interfaces: held-out ranking loss 0.08032, E[DockQ] 0.2572 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.2737 +- 0.0601 | 0.5000 +- 0.1213 | 0.1999 +- 0.0704 | 0.0069 |
| 2 | 0.2853 +- 0.0615 | 0.5556 +- 0.1205 | 0.2096 +- 0.0720 | 0.0051 |
| 3 | 0.2591 +- 0.0573 | 0.5556 +- 0.1205 | 0.2328 +- 0.0711 | 0.0046 |
| 4 | 0.2774 +- 0.0578 | 0.5556 +- 0.1205 | 0.2182 +- 0.0690 | 0.0047 |
| 5 | 0.2876 +- 0.0570 | 0.5556 +- 0.1205 | 0.2429 +- 0.0656 | 0.0047 |
| 6 | 0.2684 +- 0.0567 | 0.5556 +- 0.1205 | 0.2415 +- 0.0694 | 0.0047 |
| 7 | 0.2859 +- 0.0543 | 0.6111 +- 0.1182 | 0.2234 +- 0.0685 | 0.0049 |
| 8 | 0.2792 +- 0.0557 | 0.6111 +- 0.1182 | 0.2235 +- 0.0692 | 0.0049 |
| 9 | 0.2848 +- 0.0539 | 0.6111 +- 0.1182 | 0.2180 +- 0.0677 | 0.0050 |
| 10 | 0.2845 +- 0.0541 | 0.6111 +- 0.1182 | 0.2281 +- 0.0674 | 0.0047 |
| 11 | 0.2848 +- 0.0539 | 0.6111 +- 0.1182 | 0.2163 +- 0.0672 | 0.0049 |
| 12 | 0.2845 +- 0.0541 | 0.6111 +- 0.1182 | 0.2342 +- 0.0682 | 0.0046 |
| 13 | 0.2841 +- 0.0541 | 0.6111 +- 0.1182 | 0.2468 +- 0.0674 | 0.0046 |

## Paired per-interface test, epoch 1 -> 13

Per-interface selected-pose DockQ (argmax ipTM), 18 interfaces paired.

- paired mean **+0.0104 +- 0.0338** (0.31 sigma)
- 6 up, 4 worsened, 8 unchanged

| interface | ep1 | ep13 | delta |
|---|---|---|---|
| `9bx7_AC` | 0.184 | 0.587 | +0.403 |
| `8og0_AC` | 0.655 | 0.370 | -0.285 |
| `9bx7_BC` | 0.492 | 0.231 | -0.261 |
| `9c7x_BC` | 0.071 | 0.241 | +0.170 |
| `7r58_AB` | 0.011 | 0.087 | +0.076 |

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
| 1 | 45.61 +- 14.02 | 1.5 | 9/18 | +8.17 +- 5.03 | 3/6/9 |
| 2 | 42.44 +- 13.92 | 1.0 | 10/18 | +4.94 +- 2.89 | 3/5/10 |
| 3 | 41.39 +- 13.79 | 1.0 | 10/18 | +3.94 +- 3.31 | 3/4/11 |
| 4 | 42.28 +- 14.15 | 1.0 | 10/18 | +4.78 +- 2.97 | 2/4/12 |
| 5 | 40.78 +- 13.40 | 1.0 | 10/18 | +3.33 +- 2.42 | 1/6/11 |
| 6 | 39.39 +- 13.48 | 1.0 | 10/18 | +1.89 +- 2.79 | 3/4/11 |
| 7 | 38.17 +- 12.93 | 1.0 | 11/18 | +0.67 +- 2.99 | 4/4/10 |
| 8 | 37.22 +- 12.97 | 1.0 | 11/18 | -0.22 +- 2.63 | 3/4/11 |
| 9 | 37.56 +- 12.83 | 1.0 | 11/18 | +0.11 +- 2.71 | 3/5/10 |
| 10 | 37.11 +- 12.97 | 1.0 | 11/18 | -0.39 +- 2.97 | 4/4/10 |
| 11 | 37.61 +- 12.70 | 1.0 | 11/18 | +0.11 +- 2.83 | 4/4/10 |
| 12 | 38.22 +- 12.87 | 1.0 | 11/18 | +0.78 +- 3.00 | 3/5/10 |
| 13 | 36.83 +- 12.97 | 1.0 | 11/18 | -0.67 +- 2.83 | 4/4/10 |

Shipped on the same interfaces: mean rank 37.50 +- 12.56, median 1.0, 10/18 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep13.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 8.4020e+00 |
| `pose_perblock_frame_dir.0.weight` | 8.3895e+00 |
| `pose_perblock_frame_dir.1.weight` | 7.6254e+00 |
| `pose_perblock_frame_dir.2.weight` | 7.2594e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS, --rank-within-weight 1, --rank-topk-decay 0, 15 epochs. Identical to 24u34d/24u342 in every other setting, so this is a one-variable-pair comparison against the only arm that has ever beaten shipped on frac acceptable. WHAT CHANGES: (1) --rank-pairs all, so the loss forms BOTH cross-seed and within-seed pairs where 24u342 formed cross-seed only -- deployment selects argmax ipTM over ALL poses regardless of seed, so ordering inside a seed is part of what decides the pick and has never entered this objective; (2) --rank-within-weight 1 scales within-seed pairs in the numerator AND the normaliser W, making L a SHARE knob that cannot act as a second --alpha-rank (test_rank_loss.py asserts L normalises away when every pair is within-seed, has no effect under --rank-pairs cross, leaves the pair count alone, and keeps coefficients summing to zero); (3) --rank-tau 0.0270, the median |ipTM gap| over ALL 1,146,184 training pairs of both families across 497 structures, replacing the cross-seed-only DEFAULT_TAU of 0.0305. At L=1 within-seed pairs carry 7.0% of the pairwise score-space gradient mass against 7.0% at L=1 (reproduce: run_logging/within_weight.py --tau 0.0270), while mass reaching INVERTED pairs falls only 75.0% to 71.6% between L=1 and L=8. WHY THE WEIGHT AND NOT A SECOND TAU: within-seed pairs were never short of gradient -- at tau 0.0270 a median within-seed pair on overfit30 (gap 0.0137) weights 0.376 correctly ordered against 0.624 inverted, real discrimination, and both directions push toward the better-DockQ pose so they ADD rather than cancel. What they lack is SHARE: 7.9% of pairs by count. A multiplier is monotone in the quantity being targeted and can be set from a measurement; a second tau changes the weighting's SHAPE and its effect on mass share is not monotone. WHY TAU IS NOT RETUNED PER ARM: tau is fixed to a property of the data rather than to the knob, so the two arms differ in one thing. It does NOT equal the w_ij-weighted median (0.1329 at L=1, 0.0794 at L=8, both above tau) and no rule requires it to; DEFAULT_TAU has always been an unweighted median while the loss ran weight_by_gap=True. PARAMETERS, 44,032 over 12 tensors, unchanged from 24u34d: ANCHORED at lambda 2.5e-4, linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 plus 3 per-block copies = 30,720, so 40,960 over 8 tensors; NOT ANCHORED, frame_dir_proj LinearNoBias(3 -> 256) = 768 plus 3 per-block orientation copies = 2,304 over 4 tensors, excluded via --l2-sp-exclude 'frame_dir' because they are ZERO-INIT and anchoring them to their init would be plain decay toward zero, penalising the term being added. pae_ln and linear_no_bias_pae stay FROZEN. --blocks-per-ckpt 0 IS LOAD-BEARING: the per-block orientation copies read pose_direction's cache, which the outer forward clears after the stack, so a checkpointed block would pair one pose's directions with another's gradient. NOT A RESUME, so --l2-sp snapshots the anchor at init from the SHIPPED head; the log should say 'anchor SNAPSHOT at init' here, which is the opposite of what a resume requires. CONTROLS: 24u342 (this arm minus within-seed pairs, tau 0.0305, 50 epochs, final DockQ 0.2790 +- 0.0568, frac 0.5556, rho 0.2163, paired first-acceptable-rank negative in all 33 evaluated epochs 18-50 with mean -1.25) and the companion arm at the other L. L8 (hronsr) RAN FIRST AT 15 EPOCHS: DockQ 0.3024 against shipped 0.2909, paired +0.0115 +- 0.0088 over the same 18 interfaces (1.3 sigma, 8 better / 4 worse / 6 same), POSITIVE IN ALL EIGHT EPOCHS 8-15 with mean +0.0114 and a minimum of +0.0084, frac unmoved at 0.5556 throughout. Not significant at any single epoch and the epochs are serially correlated, so it is a consistent sign rather than a result. It is also the first arm in this project to put DockQ ABOVE shipped at all -- 24u342's only win was frac, from one interface, with DockQ below. L1 is the control that says whether that came from the within-seed WEIGHTING or merely from tau 0.0270 and the presence of within-seed pairs. CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, derived from this run's own shards. Spearman is reported but is NOT the criterion. TOP-k RANK DECAY K=0: 0 means off; above 0 each pair is weighted by max(v(r_i), v(r_j)) with v(r)=exp(-(r-1)/K) over the current ipTM order, concentrating the loss on the pairs that can change the argmax -- at K=2, 39.3% of weight on pairs touching rank 1 against 1.6% undecayed, and 92.8% inside the top 5. Ranks come from DETACHED scores, so the weight is constant in the backward and this is not the exact gradient of a fixed objective. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: L=8 landing on L=1 within error on DockQ and frac, which would say the within-seed family carries no usable signal at any weight this objective can give it.

## Trajectory, latest epoch

15 optimiser step(s), max `grad_norm` 0.0073, 0 non-finite gradient(s).

