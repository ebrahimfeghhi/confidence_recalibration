# dntxmn: ONE VARIABLE AGAINST j583i8/uhyypy: THE FOUR INJECTION SITES GET FOUR INDEPENDENT WEIGHT SETS INSTEAD OF ONE

Regenerate with `python -I run_logging/render_findings.py --rid dntxmn`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-17 (17 of 20.0).  **Interfaces:** 92.  **Budget:** 30.

## Configuration

| | |
|---|---|
| objective | `linear`, tau 0.0305 (the `linear` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 1.0 / 1.0 |
| trainable (from the run's own log) | **8 tensor(s), 0.0M of 464.5M parameters (0.0%)** |
| `--train-only` patterns | 2 |
| L2-SP lambda | 0.0 |
| L2-SP excluded | `(none)` |
| pose injection | per-block=True copies=True orientation=False |
| lr / warmup / epochs | 0.01 / 20 / 20.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.5861 +- 0.0383 | 0.5830 +- 0.0387 | 0.6926 |
| frac > 0.23 | 0.7391 +- 0.0460 | 0.7391 +- 0.0460 | 0.8587 |
| Spearman rho | 0.2562 +- 0.0344 | 0.2218 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | nan | -0.10867 +- 0.02072 | 0.5741 | nan | nan | 2.8283 |
| 2 | nan | -0.10289 +- 0.01954 | 0.5734 | nan | nan | 2.7991 |
| 3 | nan | -0.10978 +- 0.02086 | 0.5738 | nan | nan | 2.7785 |
| 4 | nan | -0.11632 +- 0.02231 | 0.5739 | nan | nan | 2.7883 |
| 5 | nan | -0.10897 +- 0.02074 | 0.5736 | nan | nan | 2.7736 |
| 6 | nan | -0.11550 +- 0.02229 | 0.5737 | nan | nan | 2.7688 |
| 7 | nan | -0.10833 +- 0.02106 | 0.5716 | nan | nan | 2.7933 |
| 8 | nan | -0.11219 +- 0.02175 | 0.5740 | nan | nan | 2.7655 |
| 9 | nan | -0.10564 +- 0.02077 | 0.5627 | nan | nan | 2.7892 |
| 10 | nan | -0.10711 +- 0.02071 | 0.5739 | nan | nan | 2.7638 |
| 11 | nan | -0.10250 +- 0.01972 | 0.5738 | nan | nan | 2.7768 |
| 12 | nan | -0.10690 +- 0.02035 | 0.5730 | nan | nan | 2.7617 |
| 13 | nan | -0.11275 +- 0.02169 | 0.5719 | nan | nan | 2.7580 |
| 14 | nan | -0.11740 +- 0.02261 | 0.5746 | nan | nan | 2.7609 |
| 15 | nan | -0.11123 +- 0.02129 | 0.5737 | nan | nan | 2.7538 |
| 16 | nan | -0.11382 +- 0.02202 | 0.5749 | nan | nan | 2.7645 |
| 17 | nan | -0.11241 +- 0.02160 | 0.5741 | nan | nan | 2.7680 |

Shipped, on these same interfaces: held-out ranking loss -0.10567, E[DockQ] 0.5719 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.5870 +- 0.0382 | 0.7500 +- 0.0454 | 0.2386 +- 0.0366 | 0.0098 |
| 2 | 0.5836 +- 0.0386 | 0.7391 +- 0.0460 | 0.2466 +- 0.0355 | 0.0088 |
| 3 | 0.5837 +- 0.0385 | 0.7391 +- 0.0460 | 0.2495 +- 0.0351 | 0.0081 |
| 4 | 0.5868 +- 0.0387 | 0.7391 +- 0.0460 | 0.2500 +- 0.0349 | 0.0078 |
| 5 | 0.5820 +- 0.0389 | 0.7283 +- 0.0466 | 0.2481 +- 0.0352 | 0.0082 |
| 6 | 0.5865 +- 0.0387 | 0.7391 +- 0.0460 | 0.2433 +- 0.0352 | 0.0077 |
| 7 | 0.5833 +- 0.0387 | 0.7283 +- 0.0466 | 0.2432 +- 0.0353 | 0.0073 |
| 8 | 0.5851 +- 0.0386 | 0.7391 +- 0.0460 | 0.2501 +- 0.0348 | 0.0083 |
| 9 | 0.5733 +- 0.0392 | 0.7174 +- 0.0472 | 0.2263 +- 0.0342 | 0.0123 |
| 10 | 0.5863 +- 0.0386 | 0.7283 +- 0.0466 | 0.2415 +- 0.0351 | 0.0079 |
| 11 | 0.5856 +- 0.0385 | 0.7391 +- 0.0460 | 0.2539 +- 0.0343 | 0.0078 |
| 12 | 0.5825 +- 0.0387 | 0.7283 +- 0.0466 | 0.2452 +- 0.0351 | 0.0061 |
| 13 | 0.5821 +- 0.0387 | 0.7283 +- 0.0466 | 0.2478 +- 0.0349 | 0.0073 |
| 14 | 0.5863 +- 0.0385 | 0.7391 +- 0.0460 | 0.2560 +- 0.0348 | 0.0075 |
| 15 | 0.5846 +- 0.0386 | 0.7391 +- 0.0460 | 0.2537 +- 0.0348 | 0.0073 |
| 16 | 0.5848 +- 0.0386 | 0.7283 +- 0.0466 | 0.2521 +- 0.0351 | 0.0077 |
| 17 | 0.5861 +- 0.0383 | 0.7391 +- 0.0460 | 0.2562 +- 0.0344 | 0.0079 |

## Paired per-interface test, epoch 1 -> 17

Per-interface selected-pose DockQ (argmax ipTM), 92 interfaces paired.

- paired mean **-0.0009 +- 0.0042** (0.21 sigma)
- 29 up, 26 worsened, 37 unchanged

| interface | ep1 | ep17 | delta |
|---|---|---|---|
| `29tj_BG` | 0.400 | 0.166 | -0.234 |
| `9bx7_BC` | 0.492 | 0.661 | +0.169 |
| `9qrv_AB` | 0.571 | 0.434 | -0.137 |
| `7tzh_AC` | 0.678 | 0.779 | +0.101 |
| `8tbq_DF` | 0.597 | 0.666 | +0.069 |

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
| 1 | 12.10 +- 3.87 | 1.0 | 69/79 | +0.85 +- 0.70 | 4/7/68 |
| 2 | 12.00 +- 3.85 | 1.0 | 68/79 | +0.73 +- 0.58 | 3/8/68 |
| 3 | 11.16 +- 3.71 | 1.0 | 68/79 | -0.09 +- 0.75 | 4/7/68 |
| 4 | 11.11 +- 3.63 | 1.0 | 68/79 | -0.15 +- 0.87 | 6/4/69 |
| 5 | 11.68 +- 3.74 | 1.0 | 67/79 | +0.42 +- 0.58 | 6/6/67 |
| 6 | 11.80 +- 3.78 | 1.0 | 68/79 | +0.54 +- 0.66 | 5/5/69 |
| 7 | 11.08 +- 3.70 | 1.0 | 67/79 | -0.19 +- 0.88 | 7/5/67 |
| 8 | 11.46 +- 3.67 | 1.0 | 68/79 | +0.20 +- 0.79 | 4/6/69 |
| 9 | 10.42 +- 3.51 | 1.0 | 66/79 | -0.84 +- 1.44 | 4/9/66 |
| 10 | 11.68 +- 3.71 | 1.0 | 67/79 | +0.43 +- 0.74 | 4/8/67 |
| 11 | 11.54 +- 3.70 | 1.0 | 68/79 | +0.28 +- 0.68 | 4/7/68 |
| 12 | 11.67 +- 3.74 | 1.0 | 67/79 | +0.41 +- 0.72 | 5/7/67 |
| 13 | 10.73 +- 3.40 | 1.0 | 67/79 | -0.53 +- 0.69 | 6/6/67 |
| 14 | 11.73 +- 3.69 | 1.0 | 68/79 | +0.47 +- 0.79 | 5/6/68 |
| 15 | 11.81 +- 3.71 | 1.0 | 68/79 | +0.56 +- 0.80 | 5/6/68 |
| 16 | 11.61 +- 3.66 | 1.0 | 67/79 | +0.34 +- 0.69 | 6/6/67 |
| 17 | 11.78 +- 3.69 | 1.0 | 68/79 | +0.52 +- 0.82 | 5/6/68 |

Shipped on the same interfaces: mean rank 11.27 +- 3.58, median 1.0, 68/79 at rank 1.

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> ONE VARIABLE AGAINST j583i8/uhyypy: THE FOUR INJECTION SITES GET FOUR INDEPENDENT WEIGHT SETS INSTEAD OF ONE. --pose-perblock-copy initialises every copy from block 0 own pretrained linear_no_bias_d and linear_no_bias_d_wo_onehot, and install() ASSERTS the copies equal block 0 to the bit -- so STEP 0 IS IDENTICAL to the share arm and the only thing that differs is capacity, 40,960 parameters across 8 tensors against 10,240 across 2. Everything else is held: PAE only, full training fold, same selection and skip list, same crop, same lr 0.01 flat after warmup (lr_final 0.0 is NO DECAY), same seeds, same valfull held-out set. --train-only linear_no_bias_d,linear_no_bias_d_wo_onehot needs no change: the match is an unanchored regex search, so it reaches pose_perblock_linear_no_bias_d.* as well; the launch log must print 8 tensor(s) / 40,960 parameters and NOT 2 / 10,240. WHY THIS CELL IS EMPTY AND WORTH FILLING: all 30 runs that have ever used --pose-perblock-copy carried the RANKING loss (24u342, 8fkrsu, zg5fnw, ykhvzs, hronsr, oe6sru and 24 others), and every PAE-only arm has been share. The ranking arms were later shown to be lowering their loss by MARGIN COLLAPSE rather than by ordering -- x5frz6 drove within-seed std 0.00593 -> 0.00147 and the top1-top2 gap 0.00205 -> 0.00015 while its loss fell -- so copy has never been tried under the one objective measured to improve sign-only ordering. copy x PAE is the empty cell. WHAT THE CONTROL SAYS: uhyypy (share x PAE, this arm at 1x capacity) sits LEVEL with shipped on valfull through epoch 9 -- paired dDockQ +0.0000 +- 0.0021 at epoch 6 and +0.0007 +- 0.0025 at epoch 8 -- while reshuffling 37 of 92 interface selections. Its held-out PAE also flattened early, -0.01425/epoch over epochs 1-5 then -0.00088 over 5-8, 16x slower. PREDICTION: if the flattening is a CAPACITY limit, 4x the weight sets should extend the descent past epoch 5; if it is a DATA or OBJECTIVE limit, copy flattens at the same epoch and merely overfits sooner, and the tied injection was never the bottleneck. That is the falsifier and it is readable by epoch 10. CRITERION: selected DockQ and fraction above 0.23 over valfull 92 interfaces against shipped DERIVED FROM THIS RUN OWN SHARDS. VAL10 AND VALFULL ARE NOT COMPARABLE. TEN-EPOCH MINIMUM before any trend claim.

## Trajectory, latest epoch

244 optimiser step(s), max `grad_norm` 1.3061, 0 non-finite gradient(s).

