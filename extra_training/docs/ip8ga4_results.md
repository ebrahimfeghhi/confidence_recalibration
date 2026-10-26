# ip8ga4: RESUME OF j583i8 AT EPOCH 8, SPEED ONLY

Regenerate with `python -I run_logging/render_findings.py --rid ip8ga4`. Every number is read
back from this run's own records; the shipped baselines are derived from the same eval shards
via their `stored_iptm` column, not carried over by hand.

**Epochs evaluated:** 1-16 (16 of 20.0).  **Interfaces:** 92.  **Budget:** 30.

**Resume chain:** `j583i8` -> `uhyypy` -> `ip8ga4`. A resume writes to a fresh directory
while continuing the parent's epoch numbering, so every table here spans the chain; the
parent is found from `args.json`'s recorded `--resume` checkpoint path, not from the
run id. An epoch trained but stopped before its held-out pass finished has no row here
rather than a carried-forward one.

## Configuration

| | |
|---|---|
| objective | `linear`, tau 0.0305 (the `linear` default, --rank-tau not passed) |
| alpha_rank / alpha_pae | 1.0 / 1.0 |
| trainable (from the run's own log) | **2 tensor(s), 0.0M of 464.4M parameters (0.0%)** |
| `--train-only` patterns | 2 |
| L2-SP lambda | 0.0 |
| L2-SP excluded | `(none)` |
| pose injection | per-block=True copies=False orientation=False |
| lr / warmup / epochs | 0.01 / 20 / 20.0 |

## Criterion: selected DockQ and fraction above 0.23

Spearman is reported but is **not** the criterion.

| | trained | shipped | oracle |
|---|---|---|---|
| selected DockQ | 0.5853 +- 0.0385 | 0.5830 +- 0.0387 | 0.6926 |
| frac > 0.23 | 0.7391 +- 0.0460 | 0.7391 +- 0.0460 | 0.8587 |
| Spearman rho | 0.2467 +- 0.0348 | 0.2218 | |

## Loss, per epoch

`held-out ranking loss` is the held-out value of this arm's objective; E[DockQ] is reported alongside it for
comparability with the expected-DockQ arms and is **not** what this arm optimises.

| epoch | training | held-out ranking loss | E[DockQ] | drift \|\|W-W0\|\|^2 | penalty | val PAE |
|---|---|---|---|---|---|---|
| 1 | nan | -0.10969 +- 0.02113 | 0.5747 | nan | nan | 2.8305 |
| 2 | nan | -0.10140 +- 0.01919 | 0.5735 | nan | nan | 2.8038 |
| 3 | nan | -0.10803 +- 0.02071 | 0.5736 | nan | nan | 2.7797 |
| 4 | nan | -0.11624 +- 0.02242 | 0.5738 | nan | nan | 2.7887 |
| 5 | nan | -0.10620 +- 0.02045 | 0.5734 | nan | nan | 2.7735 |
| 6 | nan | -0.11200 +- 0.02171 | 0.5737 | nan | nan | 2.7719 |
| 7 | nan | -0.10754 +- 0.02099 | 0.5720 | nan | nan | 2.8016 |
| 8 | nan | -0.10831 +- 0.02105 | 0.5738 | nan | nan | 2.7708 |
| 9 | nan | -0.11221 +- 0.02203 | 0.5711 | nan | nan | 2.7857 |
| 10 | nan | -0.10251 +- 0.01981 | 0.5738 | nan | nan | 2.7862 |
| 11 | nan | -0.10198 +- 0.01975 | 0.5735 | nan | nan | 2.7826 |
| 12 | nan | -0.10527 +- 0.02048 | 0.5727 | nan | nan | 2.7680 |
| 13 | nan | -0.11536 +- 0.02251 | 0.5732 | nan | nan | 2.7621 |
| 14 | nan | -0.11368 +- 0.02213 | 0.5743 | nan | nan | 2.7775 |
| 15 | nan | -0.10754 +- 0.02081 | 0.5733 | nan | nan | 2.7654 |
| 16 | nan | -0.11232 +- 0.02191 | 0.5745 | nan | nan | 2.7674 |

Shipped, on these same interfaces: held-out ranking loss -0.10567, E[DockQ] 0.5719 (tau 0.05).

## Selection metrics, per epoch

| epoch | selected DockQ | frac > 0.23 | Spearman rho | spread |
|---|---|---|---|---|
| 1 | 0.5868 +- 0.0382 | 0.7500 +- 0.0454 | 0.2430 +- 0.0361 | 0.0087 |
| 2 | 0.5876 +- 0.0382 | 0.7500 +- 0.0454 | 0.2397 +- 0.0364 | 0.0094 |
| 3 | 0.5886 +- 0.0382 | 0.7500 +- 0.0454 | 0.2437 +- 0.0355 | 0.0083 |
| 4 | 0.5834 +- 0.0384 | 0.7391 +- 0.0460 | 0.2404 +- 0.0353 | 0.0077 |
| 5 | 0.5824 +- 0.0385 | 0.7283 +- 0.0466 | 0.2409 +- 0.0358 | 0.0081 |
| 6 | 0.5832 +- 0.0384 | 0.7391 +- 0.0460 | 0.2388 +- 0.0358 | 0.0076 |
| 7 | 0.5868 +- 0.0387 | 0.7391 +- 0.0460 | 0.2288 +- 0.0361 | 0.0072 |
| 8 | 0.5839 +- 0.0385 | 0.7283 +- 0.0466 | 0.2412 +- 0.0355 | 0.0082 |
| 9 | 0.5851 +- 0.0388 | 0.7391 +- 0.0460 | 0.2352 +- 0.0353 | 0.0081 |
| 10 | 0.5850 +- 0.0385 | 0.7283 +- 0.0466 | 0.2380 +- 0.0351 | 0.0082 |
| 11 | 0.5849 +- 0.0385 | 0.7391 +- 0.0460 | 0.2506 +- 0.0348 | 0.0080 |
| 12 | 0.5856 +- 0.0385 | 0.7391 +- 0.0460 | 0.2313 +- 0.0356 | 0.0077 |
| 13 | 0.5864 +- 0.0386 | 0.7391 +- 0.0460 | 0.2372 +- 0.0354 | 0.0079 |
| 14 | 0.5866 +- 0.0384 | 0.7391 +- 0.0460 | 0.2551 +- 0.0347 | 0.0081 |
| 15 | 0.5868 +- 0.0384 | 0.7391 +- 0.0460 | 0.2429 +- 0.0352 | 0.0074 |
| 16 | 0.5853 +- 0.0385 | 0.7391 +- 0.0460 | 0.2467 +- 0.0348 | 0.0084 |

## Paired per-interface test, epoch 1 -> 16

Per-interface selected-pose DockQ (argmax ipTM), 92 interfaces paired.

- paired mean **-0.0015 +- 0.0036** (0.41 sigma)
- 35 up, 26 worsened, 31 unchanged

| interface | ep1 | ep16 | delta |
|---|---|---|---|
| `29tj_BG` | 0.400 | 0.197 | -0.203 |
| `9qrv_AB` | 0.571 | 0.434 | -0.137 |
| `7tzh_AC` | 0.678 | 0.779 | +0.101 |
| `8tbq_DF` | 0.597 | 0.666 | +0.069 |
| `9q1l_EF` | 0.795 | 0.728 | -0.067 |

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
| 1 | 12.16 +- 3.82 | 1.0 | 69/79 | +0.91 +- 0.70 | 4/7/68 |
| 2 | 12.27 +- 3.88 | 1.0 | 69/79 | +1.00 +- 0.79 | 3/7/69 |
| 3 | 11.87 +- 3.83 | 1.0 | 69/79 | +0.62 +- 0.67 | 4/7/68 |
| 4 | 11.57 +- 3.69 | 1.0 | 68/79 | +0.30 +- 0.64 | 6/5/68 |
| 5 | 11.76 +- 3.77 | 1.0 | 67/79 | +0.49 +- 0.64 | 4/8/67 |
| 6 | 11.57 +- 3.69 | 1.0 | 68/79 | +0.32 +- 0.70 | 5/6/68 |
| 7 | 11.82 +- 3.81 | 1.0 | 68/79 | +0.56 +- 0.59 | 6/5/68 |
| 8 | 11.62 +- 3.67 | 1.0 | 67/79 | +0.37 +- 0.61 | 3/9/67 |
| 9 | 11.71 +- 3.75 | 1.0 | 68/79 | +0.46 +- 0.70 | 6/4/69 |
| 10 | 12.13 +- 3.76 | 1.0 | 67/79 | +0.87 +- 0.64 | 3/9/67 |
| 11 | 12.03 +- 3.78 | 1.0 | 68/79 | +0.76 +- 0.80 | 4/6/69 |
| 12 | 11.84 +- 3.71 | 1.0 | 68/79 | +0.57 +- 0.61 | 6/5/68 |
| 13 | 11.90 +- 3.68 | 1.0 | 68/79 | +0.63 +- 0.77 | 5/5/69 |
| 14 | 11.67 +- 3.66 | 1.0 | 68/79 | +0.41 +- 0.87 | 3/7/69 |
| 15 | 12.08 +- 3.79 | 1.0 | 68/79 | +0.82 +- 0.79 | 5/6/68 |
| 16 | 11.72 +- 3.70 | 1.0 | 68/79 | +0.46 +- 0.90 | 6/5/68 |

Shipped on the same interfaces: mean rank 11.27 +- 3.58, median 1.0, 68/79 at rank 1.

## The arm as launched

Verbatim `--note` from `config.json`, written before any result was seen:

> RESUME OF j583i8 AT EPOCH 8, SPEED ONLY. Same arm, same data, same objective: PAE on the full training fold, --pose-perblock-share, 2 tensors / 10,240 parameters, 15 epochs total with cosine lr 1e-2 -> 0. WHAT IS DIFFERENT IS WALL CLOCK, NOT THE MODEL: (1) the held-out PAE pass shards over cards 4,5,7 instead of running on one while two sit at a barrier -- 12.8 min -> ~4.3 expected, same 7,950 poses pooled by count under a SHARD COVERAGE assert; (2) readahead is issued from the thread that reads, so the first pose of a unit does not wait on disk the GPU could have been given; (3) the training crop is built from pose 0 rather than the union over all five, which is the SAME MASK -- crop_pose_equivalence.py finds 0 differences and identical kept-token counts, because the per-pose label permutation is within-residue (symmetric atoms swapped inside one token, so the token's atom-position set and therefore its contact distances are unchanged) -- for 88.4 -> 19.1 ms/unit of CPU. PREDICTION: cycle 46 -> ~37 min, and the held-out PAE and DockQ curves CONTINUE across the 7/8 boundary with no step. A visible step falsifies one of the three claims and is the thing to look for first. WHERE IT STOOD AT EPOCH 7, all on valfull's 92 interfaces against shipped DERIVED FROM THIS RUN'S OWN SHARDS: held-out PAE 2.8305 -> 2.7719 over epochs 1-6 then 2.8016 at epoch 7; selected DockQ 0.5868 / 0.5876 / 0.5886 / 0.5834 / 0.5824 / 0.5832 / 0.5868 against shipped 0.5831; frac 0.7500 x3 then 0.7391 / 0.7283 / 0.7391 / 0.7391 against shipped 0.7391. Paired against shipped on the same interfaces the arm is level: +0.0053 +- 0.0037 at epoch 3, +0.0000 +- 0.0021 at epoch 6, while 37 of 92 interfaces select a DIFFERENT pose -- it reshuffles 40% of the picks and the wins and losses cancel. Spearman is the one steady gain, 0.243 vs shipped 0.2218. TEN-EPOCH MINIMUM still applies and epochs 1-7 carry over, so epoch 10 is the first verdict. GUARD: crop_all_poses drifts by design; resume_ep7.pt predates the flag and migrates to the pre-change True.

## Trajectory, latest epoch

244 optimiser step(s), max `grad_norm` 2.4912, 0 non-finite gradient(s).

