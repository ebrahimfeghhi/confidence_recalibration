# 8fkrsu: STRADDLE-ONLY RANKING LOSS ON THE FULL TRAINING FOLD, lambda 5e-05, 15 epochs

Regenerate with `python -I run_logging/render_findings.py --rid 8fkrsu`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-3 (3 of 15.0).  **Interfaces:** 92.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.1429 |
| alpha_rank / alpha_pae | 1.0 / 0.0 |
| trainable (from the run's own log) | **8 tensor(s), 0.0M of 464.5M parameters (0.0%)** |
| `--train-only` patterns | 2 |
| L2-SP lambda | 5e-05 |
| L2-SP excluded | `(none)` |
| pose injection | per-block=True copies=True orientation=False |
| lr / warmup / epochs | 0.01 / 20 / 15.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.5758 +- 0.0379 | 0.5830 +- 0.0387 | 0.6926 |
| frac > 0.23 | 0.7283 +- 0.0466 | 0.7391 +- 0.0460 | 0.8587 |
| Spearman rho | 0.2475 +- 0.0367 | 0.2218 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.14452 | 0.11987 +- 0.01859 | 0.5732 | 14.0 | 0.0007 | 3.0041 |
| 2 | 0.14570 | 0.11468 +- 0.01376 | 0.5688 | 20.3 | 0.0010 | 3.2424 |
| 3 | 0.14146 | 0.11592 +- 0.01275 | 0.5677 | 34.7 | 0.0017 | 3.1977 |

Shipped, on these same interfaces: held-out ranking loss 0.11632, E[DockQ] 0.5719 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.5735 +- 0.0387 | 0.7174 +- 0.0472 | 0.2270 +- 0.0358 | 0.0055 |
| 2 | 0.5787 +- 0.0381 | 0.7391 +- 0.0460 | 0.2464 +- 0.0355 | 0.0066 |
| 3 | 0.5758 +- 0.0379 | 0.7283 +- 0.0466 | 0.2475 +- 0.0367 | 0.0064 |

## Paired per-interface test, epoch 1 -> 3

Per-interface selected-pose DockQ (argmax ipTM), 92 interfaces paired.

- paired mean **+0.0023 +- 0.0037** (0.63 sigma)
- 25 up, 23 worsened, 44 unchanged

| interface | ep1 | ep3 | delta |
|---|---|---|---|
| `9qrv_AB` | 0.434 | 0.598 | +0.164 |
| `9mz6_BC` | 0.964 | 0.843 | -0.121 |
| `9cfd_BC` | 0.285 | 0.393 | +0.108 |
| `9mmj_BC` | 0.953 | 0.846 | -0.107 |
| `7q4q_BE` | 0.909 | 0.811 | -0.098 |

## Rank of the first acceptable pose, ordering by ipTM

`frac acceptable` only asks whether rank 1 happens to be acceptable, so it cannot see an
arm that lifts the best pose from rank 90 to rank 2. This can. Rank is 1-indexed and
**lower is better**; rank 1 everywhere is frac 1.0.

Scored over 150 poses per interface, threshold DockQ >= 0.23, 13 interface(s) with no acceptable pose anywhere (those have no rank and are excluded; a non-zero count here would bias the mean optimistically, since the hardest interfaces would be the ones dropped).

The unpaired s.e. is large because a few interfaces first become acceptable in the 90s
while most are acceptable at rank 1, so between-interface variance swamps the effect.
**The paired column is the one that can resolve a change** -- every interface is scored
by both orderings over the same poses, so the pairing is exact.

| epoch | mean rank +- s.e. | median | at rank 1 | paired vs shipped | better/worse/same |
|---|---|---|---|---|---|
| 1 | 13.03 +- 3.92 | 1.0 | 66/79 | +1.77 +- 0.84 | 3/11/65 |
| 2 | 12.09 +- 3.88 | 1.0 | 68/79 | +0.82 +- 0.79 | 6/6/67 |
| 3 | 11.77 +- 3.71 | 1.0 | 67/79 | +0.51 +- 0.88 | 4/8/67 |

Shipped on the same interfaces: mean rank 11.27 +- 3.58, median 1.0, 68/79 at rank 1.

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> STRADDLE-ONLY RANKING LOSS ON THE FULL TRAINING FOLD, lambda 5e-05, 15 epochs. THE OBJECTIVE: --rank-straddle 0.23 keeps a pair only when exactly one of its two poses has DockQ above 0.23. The reported criterion is the FRACTION of interfaces whose selected pose clears 0.23, and that number moves ONLY when selection crosses the threshold -- two acceptable poses, or two failures, can be reordered freely without changing it, so 87.5% of the pairs the loss used to weight cannot affect the metric. This aims the gradient at the subspace that can. 143,680 of 1,146,184 pairs survive. WHAT IT COSTS, stated rather than hidden: mean DockQ is NOT invariant to the dropped pairs -- promoting 0.61 over 0.38 raises the mean and leaves the fraction untouched -- so this optimises the FRACTION and treats mean DockQ as a free variable; both are reported because they can now move in opposite directions. AND 590 OF 1028 INTERFACES GO SILENT: only 438 are usable, the rest having every pose on one side of 0.23, so they form no straddling pair and contribute nothing. That is the biggest risk here and the first thing to check if it fails. BOTH FAMILIES, cross-seed AND within-seed (--rank-pairs all at within_weight 1): the straddle restriction is orthogonal to the family split and a within-seed pair crossing 0.23 is exactly as selection-relevant as a cross-seed one. TAU IS 0.1429, the median |ipTM gap| over the pairs THIS ARM ACTUALLY TRAINS (measure_tau.py --pairs all --straddle 0.23, full 497-structure fold, after the DockQ-tie filter). It is 4.7x DEFAULT_TAU and that is not an error: poses on opposite sides of 0.23 are genuinely different and the shipped head already separates them more; training them at 0.0305 would put nearly all of them deep in saturation. NO ORIENTATION, NO PAE: --frame-direction is dropped so the trainable set is 8 tensors / 40,960 parameters -- linear_no_bias_d + pose_perblock, distance at all four pairformer blocks and nothing else -- and --alpha-pae 0 removes the PAE term, which the oe6sru verdict supports (paired -0.0209, positive 0 of 8, against a +0.0116 threshold fixed in advance). L2-SP COVERS EVERY TRAINABLE TENSOR, no --l2-sp-exclude, because the only tensors zg5fnw excluded were frame_dir and frame_dir is not trained here; the anchored set is 40,960 parameters, the SAME count zg5fnw anchored, so lambda transfers on the measured gradient ratio alone with a size factor of exactly 1. CONTROL: zg5fnw, this same fold and this same 44,032-parameter family (it additionally trained frame_dir), cross-seed at tau 0.0305 with alpha_pae 1 and lambda 1e-05, which reached paired +0.00052 positive 4 of 7 before being stopped at epoch 7 -- its training ranking loss never descended, 0.05525 at epoch 1 to 0.05865 at epoch 7. CRITERION: FRACTION ABOVE 0.23 over valfull's 92 interfaces is the primary number for this arm, against shipped 0.7391; selected mean DockQ against shipped 0.5831 is reported but is explicitly NOT what this objective optimises. NOTE 13 of valfull's 92 interfaces (14.1%) have NO acceptable pose at any rank and are censored in the rank metric. STEPS WITH NO TASK GRADIENT ARE NOT TAKEN. 52.3% of the fold's structures have no straddling pair at all, so with --rank-structs-per-step 2 about 27% of steps accumulate nothing from the ranking term, and with --alpha-pae 0 there is then no task gradient whatsoever. The first attempt at this arm (t7dvob) handed those steps to AdamW with only the L2-SP term, and because AdamW is scale-invariant that is a near-full lr-sized step straight back toward the anchor rather than a small one: 16 of its first 69 taken steps had grad_norm < 1e-3 against a typical 0.01-0.20, the late ones clustered at ~2.5e-4 and rising as the weights drifted. Those steps are now skipped and counted in the epoch line. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: fraction at or below shipped 0.7391 over epochs 10-15, which would say that concentrating the gradient on threshold-crossing pairs buys nothing even when it is the metric being optimised.

## Trajectory, latest epoch

168 optimiser step(s), max `grad_norm` 0.1892, 75 non-finite gradient(s).

