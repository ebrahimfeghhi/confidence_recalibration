# zg5fnw: THE ORIENTATION ARM ON THE FULL TRAINING FOLD, SCORED ON THE FULL VALIDATION SET

Regenerate with `python -I run_logging/render_findings.py --rid zg5fnw`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-7 (7 of 12.0).  **Interfaces:** 92.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `softplus`, tau 0.0305 (the `softplus` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 1.0 / 1.0 |
| trainable (from the run's own log) | **12 tensor(s), 0.0M of 464.5M parameters (0.0%)** |
| `--train-only` patterns | 3 |
| L2-SP lambda | 1e-05 |
| L2-SP excluded | `frame_dir` -- 4 tensor(s) out of the anchor, 8 anchored |
| pose injection | per-block=True copies=True orientation=True |
| lr / warmup / epochs | 0.01 / 20 / 12.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.5848 +- 0.0382 | 0.5831 +- 0.0387 | 0.6926 |
| frac > 0.23 | 0.7391 +- 0.0460 | 0.7391 +- 0.0460 | 0.8587 |
| Spearman rho | 0.2439 +- 0.0348 | 0.2218 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | 0.05525 | 0.05661 +- 0.00887 | 0.5742 | 443.2 | 0.0044 | 2.8566 |
| 2 | 0.05956 | 0.05395 +- 0.00862 | 0.5726 | 374.8 | 0.0037 | 2.9321 |
| 3 | 0.05869 | 0.05929 +- 0.00925 | 0.5737 | 429.4 | 0.0043 | 2.8221 |
| 4 | 0.06054 | 0.05616 +- 0.00845 | 0.5735 | 364.0 | 0.0036 | 2.8293 |
| 5 | 0.05729 | 0.05239 +- 0.00753 | 0.5732 | 365.2 | 0.0037 | 2.8253 |
| 6 | 0.05764 | 0.05470 +- 0.00830 | 0.5737 | 402.4 | 0.0040 | 2.8007 |
| 7 | 0.05865 | 0.05441 +- 0.00837 | 0.5735 | 400.4 | 0.0040 | 2.8312 |

Shipped, on these same interfaces: held-out ranking loss 0.06063, E[DockQ] 0.5719 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.5790 +- 0.0387 | 0.7283 +- 0.0466 | 0.2393 +- 0.0360 | 0.0066 |
| 2 | 0.5787 +- 0.0388 | 0.7283 +- 0.0466 | 0.2409 +- 0.0358 | 0.0062 |
| 3 | 0.5813 +- 0.0384 | 0.7391 +- 0.0460 | 0.2397 +- 0.0353 | 0.0094 |
| 4 | 0.5871 +- 0.0382 | 0.7500 +- 0.0454 | 0.2393 +- 0.0353 | 0.0086 |
| 5 | 0.5878 +- 0.0380 | 0.7609 +- 0.0447 | 0.2409 +- 0.0355 | 0.0078 |
| 6 | 0.5868 +- 0.0383 | 0.7391 +- 0.0460 | 0.2433 +- 0.0351 | 0.0075 |
| 7 | 0.5848 +- 0.0382 | 0.7391 +- 0.0460 | 0.2439 +- 0.0348 | 0.0073 |

## Paired per-interface test, epoch 1 -> 7

Per-interface selected-pose DockQ (argmax ipTM), 92 interfaces paired.

- paired mean **+0.0058 +- 0.0040** (1.45 sigma)
- 22 up, 17 worsened, 53 unchanged

| interface | ep1 | ep7 | delta |
|---|---|---|---|
| `9bx7_BC` | 0.172 | 0.492 | +0.320 |
| `9c7x_BC` | 0.074 | 0.217 | +0.143 |
| `9mz6_AC` | 0.773 | 0.723 | -0.050 |
| `7zra_AC` | 0.869 | 0.917 | +0.048 |
| `7y8j_BC` | 0.963 | 0.917 | -0.046 |

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
| 1 | 12.00 +- 3.77 | 1.0 | 67/79 | +0.75 +- 0.70 | 5/8/66 |
| 2 | 11.75 +- 3.74 | 1.0 | 67/79 | +0.48 +- 0.84 | 5/8/66 |
| 3 | 11.99 +- 3.75 | 1.0 | 68/79 | +0.73 +- 0.68 | 5/7/67 |
| 4 | 11.84 +- 3.82 | 1.0 | 69/79 | +0.58 +- 0.54 | 4/7/68 |
| 5 | 11.62 +- 3.76 | 1.0 | 70/79 | +0.35 +- 0.57 | 5/5/69 |
| 6 | 11.72 +- 3.75 | 1.0 | 68/79 | +0.46 +- 0.72 | 5/7/67 |
| 7 | 11.48 +- 3.74 | 1.0 | 68/79 | +0.22 +- 0.59 | 5/7/67 |

Shipped on the same interfaces: mean rank 11.27 +- 3.58, median 1.0, 68/79 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude frame_dir` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep7.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `frame_dir_proj.weight` | 2.0527e+01 |
| `pose_perblock_frame_dir.0.weight` | 2.0702e+01 |
| `pose_perblock_frame_dir.1.weight` | 2.1301e+01 |
| `pose_perblock_frame_dir.2.weight` | 1.7346e+01 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> THE ORIENTATION ARM ON THE FULL TRAINING FOLD, SCORED ON THE FULL VALIDATION SET. Identical to 24u342 in every setting except the data: cross-seed softplus ranking loss at tau 0.0305 with --alpha-pae 0, distance AND orientation injected at all four pairformer blocks with untied per-block copies, --train-only 'linear_no_bias_d,pose_perblock,frame_dir' = 12 tensors / 44,032 parameters, --l2-sp-exclude 'frame_dir' leaving the 4 zero-init orientation tensors trainable but out of the anchor, lr 1e-2, warmup 20, --blocks-per-ckpt 0. TRAINING DATA: training/full_train (488 structures) instead of overfit30 (30), roughly 16x the units and 16x the optimiser steps per epoch, which is why this is 12 epochs rather than 50 -- on the order of 4x the total optimisation 24u342 received, with twelve held-out evaluations. HELD OUT: training/valfull (92 interfaces) instead of val10 (18). THESE TWO POOLS ARE NOT COMPARABLE and no number from any previous arm transfers to this one; shipped 0.2909 / 0.5556 are VAL10 figures. The reporting derives shipped from this run's own eval shards via their stored_iptm column, so the reference follows the pool automatically. WHY: six arms have ended at parity on val10 and every one of those nulls is ambiguous, because 30 structures was never enough to distinguish 'the reordering does not generalise' from 'the objective is wrong'. 24u342 made the case concrete -- training loss flat from epoch 9, frac pinned at 11/18 for fifteen epochs, and a stable 4-better/4-worse split across the eight hard interfaces that looks exactly like an un-generalised signal. If more data breaks that symmetry, the objective is fine and the pool was the limit; if the symmetry holds at 1,028 interfaces, the objective is the problem and no amount of data fixes it. Either answer is worth more than another overfit30 arm. LAMBDA was re-measured by a fresh --grad-probe on this data even though the parameter count is unchanged and the sqrt factor is exactly 1, because |g_rank| can differ on a new distribution. CRITERION: selected DockQ and fraction above 0.23 on the 92 valfull interfaces against shipped DERIVED FROM THIS RUN'S SHARDS. Spearman is reported but is NOT the criterion. CONTROL: 24u342, this arm on 30 structures scored on 18 interfaces.

## Trajectory, latest epoch

243 optimiser step(s), max `grad_norm` 2.1628, 0 non-finite gradient(s).

