# tv6hue: RESUME OF ij24wr AT EPOCH 2 TO FINISH THE 20-EPOCH HORIZON

Regenerate with `python -I run_logging/render_findings.py --rid tv6hue`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-17 (17 of 20.0).  **Interfaces:** 92.  **Budget:** 30.

**Resume chain:** `ij24wr` -> `tv6hue`. A resume writes to a fresh directory
while continuing the parent's epoch numbering, so every table here spans the chain; the
parent is found from `args.json`'s recorded `--resume` checkpoint path, not from the
run id. An epoch trained but stopped before its held-out pass finished has no row here
rather than a carried-forward one.

## Configuration

| | |
|---|---|
| objective | `linear`, tau 0.0305 (the `linear` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 1.0 / 1.0 |
| trainable (from the run's own log) | **254 tensor(s), 19.8M of 464.5M parameters (4.3%)** |
| `--train-only` patterns | none (whole head) |
| L2-SP lambda | 0.001 |
| L2-SP excluded | `linear_no_bias_d` -- 8 tensor(s) out of the anchor, 246 anchored |
| pose injection | per-block=True copies=True orientation=False |
| lr / warmup / epochs | 0.01 / 20 / 20.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.5701 +- 0.0383 | 0.5830 +- 0.0387 | 0.6926 |
| frac > 0.23 | 0.7391 +- 0.0460 | 0.7391 +- 0.0460 | 0.8587 |
| Spearman rho | 0.2275 +- 0.0355 | 0.2218 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | nan | -0.09644 +- 0.01932 | 0.5658 | 59.2 | 0.0592 | 2.8207 |
| 2 | nan | -0.06290 +- 0.01187 | 0.5638 | 48.2 | 0.0482 | 2.8167 |
| 3 | nan | -0.11753 +- 0.02268 | 0.5677 | 49.2 | 0.0492 | 2.7660 |
| 4 | nan | -0.09304 +- 0.01794 | 0.5704 | 44.7 | 0.0447 | nan |
| 5 | nan | -0.09941 +- 0.01920 | 0.5717 | 48.6 | 0.0486 | 2.7174 |
| 6 | nan | -0.10723 +- 0.02113 | 0.5727 | 45.5 | 0.0455 | 2.7715 |
| 7 | nan | -0.05057 +- 0.01006 | 0.5409 | 49.3 | 0.0493 | 2.7913 |
| 8 | nan | -0.10417 +- 0.01983 | 0.5743 | 47.4 | 0.0474 | 2.7583 |
| 9 | nan | -0.08719 +- 0.01784 | 0.5630 | 50.7 | 0.0507 | 2.7265 |
| 10 | nan | -0.07618 +- 0.01425 | 0.5619 | 51.6 | 0.0516 | 2.7318 |
| 11 | nan | -0.11266 +- 0.02204 | 0.5709 | 44.5 | 0.0445 | 2.7551 |
| 12 | nan | -0.11085 +- 0.02159 | 0.5637 | 47.7 | 0.0477 | 2.7309 |
| 13 | nan | -0.10758 +- 0.02147 | 0.5721 | 46.9 | 0.0469 | 2.7555 |
| 14 | nan | -0.11622 +- 0.02280 | 0.5723 | 55.3 | 0.0553 | 2.7245 |
| 15 | nan | -0.11577 +- 0.02160 | 0.5712 | 54.5 | 0.0545 | 2.7447 |
| 16 | nan | -0.10304 +- 0.02175 | 0.5721 | 55.3 | 0.0553 | 2.7854 |
| 17 | nan | -0.09243 +- 0.01804 | 0.5696 | 51.0 | 0.0510 | 2.7657 |

Shipped, on these same interfaces: held-out ranking loss -0.10567, E[DockQ] 0.5719 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.5736 +- 0.0385 | 0.7283 +- 0.0466 | 0.2120 +- 0.0343 | 0.0071 |
| 2 | 0.5798 +- 0.0382 | 0.7391 +- 0.0460 | 0.2305 +- 0.0353 | 0.0096 |
| 3 | 0.5774 +- 0.0384 | 0.7391 +- 0.0460 | 0.2253 +- 0.0334 | 0.0060 |
| 4 | 0.5803 +- 0.0381 | 0.7500 +- 0.0454 | 0.2216 +- 0.0359 | 0.0095 |
| 5 | 0.5772 +- 0.0382 | 0.7391 +- 0.0460 | 0.2348 +- 0.0364 | 0.0062 |
| 6 | 0.5755 +- 0.0386 | 0.7391 +- 0.0460 | 0.2195 +- 0.0355 | 0.0091 |
| 7 | 0.5614 +- 0.0392 | 0.7065 +- 0.0477 | 0.2271 +- 0.0350 | 0.0085 |
| 8 | 0.5681 +- 0.0388 | 0.7174 +- 0.0472 | 0.2160 +- 0.0361 | 0.0085 |
| 9 | 0.5790 +- 0.0386 | 0.7283 +- 0.0466 | 0.2351 +- 0.0356 | 0.0070 |
| 10 | 0.5626 +- 0.0385 | 0.7174 +- 0.0472 | 0.2437 +- 0.0345 | 0.0095 |
| 11 | 0.5704 +- 0.0379 | 0.7500 +- 0.0454 | 0.2321 +- 0.0349 | 0.0087 |
| 12 | 0.5658 +- 0.0386 | 0.7391 +- 0.0460 | 0.2243 +- 0.0357 | 0.0071 |
| 13 | 0.5777 +- 0.0384 | 0.7283 +- 0.0466 | 0.2223 +- 0.0367 | 0.0107 |
| 14 | 0.5733 +- 0.0380 | 0.7391 +- 0.0460 | 0.2241 +- 0.0362 | 0.0073 |
| 15 | 0.5638 +- 0.0389 | 0.7283 +- 0.0466 | 0.2225 +- 0.0357 | 0.0098 |
| 16 | 0.5679 +- 0.0383 | 0.7283 +- 0.0466 | 0.2230 +- 0.0371 | 0.0129 |
| 17 | 0.5701 +- 0.0383 | 0.7391 +- 0.0460 | 0.2275 +- 0.0355 | 0.0101 |

## Paired per-interface test, epoch 1 -> 17

Per-interface selected-pose DockQ (argmax ipTM), 92 interfaces paired.

- paired mean **-0.0035 +- 0.0037** (0.94 sigma)
- 28 up, 33 worsened, 31 unchanged

| interface | ep1 | ep17 | delta |
|---|---|---|---|
| `9bx7_AC` | 0.587 | 0.363 | -0.224 |
| `9mz6_BC` | 0.965 | 0.843 | -0.122 |
| `9bx7_BC` | 0.172 | 0.274 | +0.102 |
| `7zmk_VX` | 0.692 | 0.778 | +0.086 |
| `9mmj_AC` | 0.784 | 0.710 | -0.074 |

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
| 1 | 10.32 +- 3.44 | 1.0 | 67/79 | -0.94 +- 0.88 | 7/5/67 |
| 2 | 10.41 +- 3.43 | 1.0 | 68/79 | -0.86 +- 1.16 | 7/3/69 |
| 3 | 9.65 +- 3.30 | 1.0 | 68/79 | -1.61 +- 1.22 | 7/4/68 |
| 4 | 8.58 +- 3.04 | 1.0 | 69/79 | -2.68 +- 1.20 | 9/1/69 |
| 5 | 9.75 +- 3.54 | 1.0 | 68/79 | -1.52 +- 1.28 | 6/4/69 |
| 6 | 9.44 +- 3.26 | 1.0 | 68/79 | -1.81 +- 1.13 | 9/2/68 |
| 7 | 8.51 +- 3.10 | 1.0 | 65/79 | -2.76 +- 1.18 | 10/5/64 |
| 8 | 9.15 +- 3.16 | 1.0 | 66/79 | -2.10 +- 0.95 | 9/4/66 |
| 9 | 9.61 +- 3.43 | 1.0 | 67/79 | -1.65 +- 1.29 | 7/5/67 |
| 10 | 9.38 +- 3.42 | 1.0 | 66/79 | -1.87 +- 1.06 | 8/5/66 |
| 11 | 9.11 +- 3.38 | 1.0 | 69/79 | -2.15 +- 1.25 | 8/3/68 |
| 12 | 9.33 +- 3.37 | 1.0 | 68/79 | -1.94 +- 1.39 | 7/5/67 |
| 13 | 7.86 +- 2.74 | 1.0 | 67/79 | -3.41 +- 1.22 | 11/2/66 |
| 14 | 9.68 +- 3.32 | 1.0 | 68/79 | -1.58 +- 1.07 | 7/3/69 |
| 15 | 8.39 +- 2.71 | 1.0 | 67/79 | -2.86 +- 1.29 | 10/3/66 |
| 16 | 8.66 +- 3.05 | 1.0 | 67/79 | -2.61 +- 1.30 | 9/4/66 |
| 17 | 8.67 +- 3.25 | 1.0 | 68/79 | -2.59 +- 1.36 | 9/2/68 |

Shipped on the same interfaces: mean rank 11.27 +- 3.58, median 1.0, 68/79 at rank 1.

## Tensors excluded from the anchor

`--l2-sp-exclude linear_no_bias_d` leaves these trainable but outside
`||W - W0||^2`, so the drift column above does **not** cover them. They are zero-init, so
their norm is the direct test of whether the loss finds any use for them. From `resume_ep17.pt`:

| tensor | \|\|W\|\| |
|---|---|
| `linear_no_bias_d.weight` | 1.1970e+02 |
| `linear_no_bias_d_wo_onehot.weight` | 2.1976e+00 |
| `pose_perblock_linear_no_bias_d.0.weight` | 1.1863e+02 |
| `pose_perblock_linear_no_bias_d.1.weight` | 1.2789e+02 |
| `pose_perblock_linear_no_bias_d.2.weight` | 1.2845e+02 |
| `pose_perblock_linear_no_bias_d_wo_onehot.0.weight` | 2.1593e+00 |
| `pose_perblock_linear_no_bias_d_wo_onehot.1.weight` | 1.8440e+00 |
| `pose_perblock_linear_no_bias_d_wo_onehot.2.weight` | 1.8555e+00 |

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> RESUME OF ij24wr AT EPOCH 2 TO FINISH THE 20-EPOCH HORIZON. Epoch 1 was stopped at the user's direction so the ipTM join verification could run on genuinely free cards -- its --compare checks PER-POSE ipTM against the eval's own rows, which only means anything if no other job is sharing a GPU. NOTHING ABOUT THIS ARM CHANGES: every argument is copied from launch_fullhead_l2sp_posefree.sh except --resume, so the guard should report no drift. Full confidence head trains, the 8 pose tensors stay OUT of the anchor and the other 246 are anchored at 1e-3 * ||W - W_shipped||^2; the launch log must print 8 excluded / 246 anchored and must say the anchor was INHERITED from ij24wr. Speed flags stay on: --numa-map 0:0,1:0,2:1 and --prefetch-workers 4, both measured on epoch 1 with the cards to themselves.

## Trajectory, latest epoch

244 optimiser step(s), max `grad_norm` 3.4992, 0 non-finite gradient(s).

