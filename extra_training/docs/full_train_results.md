# Full training set: the four metrics, per epoch

Generated 2026-10-21 03:50 PDT by `run_logging/results_table.py` from `x3k2m4+jhgfxy+2w5to5+ocqtyt`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

## The lineage is one experiment in three parts

Epoch numbers are continuous, but the segments do NOT share an implementation, so **throughput and memory are not comparable across them**. The metrics are.

| segment | epochs | configuration |
|---|---|---|
| `x3k2m4` | 1 | checkpointing ON (blocks_per_ckpt=1); per-epoch fingerprint hashing; prefetch_issue double-counted the hash |
| `jhgfxy` | 2 | checkpointing ON; fresh process so hashing re-paid; prefetch choice-cache added |
| `2w5to5` | 3–6 | checkpointing OFF (blocks_per_ckpt=None); persistent fingerprint cache; empty_cache at the boundary; exclusive timers + residual + memory peaks |

## Epoch 0 is the shipped head

| metric | epoch 0 | oracle |
|---|---|---|
| held-out PAE | 2.9756 | — |
| val10 mean DockQ | 0.2909 | 0.5968 |
| val10 frac-acceptable | 0.5556 | 1.0000 |
| train PAE | **unmeasured** | — |

Budget 30, 18 interfaces, 10 validation structures. **Not** the budget-100 figures (0.2626 / 0.5000, oracle 0.6662 / 1.0000) — never quote one against the other. The shipped head's train PAE over these 488 structures has never been measured; it needs a `--loss-only` pass over 4,880 units.

## Per epoch

| epoch | seg | complete | train PAE | held-out PAE | vs ep0 | val10 DockQ | vs ep0 | frac-acc | vs ep0 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `x3k2m4` | yes | 2.7384 | 2.6953 | -0.2803 | 0.2999 | +0.0090 | 0.5556 | +0.0000 |
| 2 | `jhgfxy` | yes | 2.6623 | 2.6779 | -0.2977 | 0.2543 | -0.0366 | 0.3333 | -0.2223 |
| 3 | `2w5to5` | yes | 2.6061 | 2.7359 | -0.2397 | 0.2864 | -0.0045 | 0.4444 | -0.1112 |
| 4 | `2w5to5` | yes | 2.5784 | 2.6931 | -0.2825 | 0.2908 | -0.0001 | 0.4444 | -0.1112 |
| 5 | `2w5to5` | yes | 2.5509 | 2.6576 | -0.3180 | 0.3219 | +0.0310 | 0.5556 | +0.0000 |
| 6 | `2w5to5` | yes | 2.5270 | 2.6717 | -0.3039 | 0.3134 | +0.0225 | 0.5556 | +0.0000 |
| 7 | `ocqtyt` | **NO** | 2.4691 | — | — | 0.3211 | +0.0302 | 0.5000 | -0.0556 |
| 8 | `ocqtyt` | yes | 2.4318 | 2.6451 | -0.3305 | 0.3235 | +0.0326 | 0.5556 | +0.0000 |
| 9 | `ocqtyt` | yes | 2.4048 | 2.6728 | -0.3028 | 0.3303 | +0.0394 | 0.5000 | -0.0556 |
| 10 | `ocqtyt` | yes | 2.3749 | 2.7112 | -0.2644 | 0.3238 | +0.0329 | 0.6111 | +0.0555 |
| 11 | `ocqtyt` | yes | 2.3421 | 2.7181 | -0.2575 | 0.2978 | +0.0069 | 0.6111 | +0.0555 |
| 12 | `ocqtyt` | yes | 2.3185 | 2.7085 | -0.2671 | 0.2670 | -0.0239 | 0.5000 | -0.0556 |
| 13 | `ocqtyt` | yes | 2.2925 | 2.7162 | -0.2594 | 0.2987 | +0.0078 | 0.6111 | +0.0555 |
| 14 | `ocqtyt` | yes | 2.2744 | 2.6973 | -0.2783 | 0.3122 | +0.0213 | 0.6111 | +0.0555 |
| 15 | `ocqtyt` | yes | 2.2532 | 2.7176 | -0.2580 | 0.3122 | +0.0213 | 0.6111 | +0.0555 |
| 16 | `ocqtyt` | yes | 2.2383 | 2.7864 | -0.1892 | 0.3431 | +0.0522 | 0.5556 | +0.0000 |
| 17 | `ocqtyt` | yes | 2.2171 | 2.7505 | -0.2251 | 0.2911 | +0.0002 | 0.5000 | -0.0556 |
| 18 | `ocqtyt` | yes | 2.2046 | 2.7995 | -0.1761 | 0.2981 | +0.0072 | 0.5000 | -0.0556 |
| 19 | `ocqtyt` | yes | 2.1854 | 2.8185 | -0.1571 | 0.3382 | +0.0473 | 0.6111 | +0.0555 |
| 20 | `ocqtyt` | yes | 2.1756 | 2.7585 | -0.2171 | 0.3372 | +0.0463 | 0.6111 | +0.0555 |
| 21 | `ocqtyt` | yes | 2.1639 | 2.8251 | -0.1505 | 0.2809 | -0.0100 | 0.5556 | +0.0000 |
| 22 | `ocqtyt` | yes | 2.1474 | 2.8566 | -0.1190 | 0.3057 | +0.0148 | 0.5556 | +0.0000 |

Held-out PAE is a loss, so **lower is better** and a negative delta is an improvement. DockQ and frac-acceptable are the opposite.

`complete` means all four metrics are present AND the held-out pass covered its expected 1500 poses. An epoch marked **NO** is mid-boundary or lost a metric to a failure — its em-dashes are missing measurements, not zeros.

## Why held-out PAE is the metric to read

On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics showed nothing the entire time because ipTM barely orders poses within an interface — Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE curve is not evidence a run is working.

**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is the thing that would settle whether selection improved, and it needs several epochs of checkpoints to be worth running.

## Per-epoch wall clock and memory

| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |
|---|---|---|---|---|---|---|---|---|
| 1 | `x3k2m4` | 2 | 77.9 | — | — | — | — | — |
| 2 | `jhgfxy` | 0 | 79.4 | — | — | — | — | — |
| 3 | `2w5to5` | 2 | 78.3 | 78.2m | 0.1m | 65.70 | 66.80 | None |
| 4 | `2w5to5` | 2 | 78.5 | 78.5m | 0.0m | 65.70 | 66.47 | None |
| 5 | `2w5to5` | 2 | 73.1 | 73.0m | 0.0m | 65.70 | 66.47 | None |
| 6 | `2w5to5` | 2 | 65.4 | 65.3m | 0.1m | 65.70 | 66.47 | None |
| 7 | `ocqtyt` | 1 | 37.9 | 19.6m | 18.3m | 65.70 | 66.80 | None |
| 8 | `ocqtyt` | 1 | 35.1 | 19.1m | 15.9m | 65.70 | 66.47 | None |
| 9 | `ocqtyt` | 2 | 34.1 | 22.7m | 11.4m | 65.70 | 66.47 | None |
| 10 | `ocqtyt` | 1 | 37.5 | 18.8m | 18.7m | 65.70 | 66.47 | None |
| 11 | `ocqtyt` | 1 | 28.8 | 19.0m | 9.8m | 65.70 | 66.47 | None |
| 12 | `ocqtyt` | 1 | 28.6 | 19.1m | 9.4m | 65.70 | 66.47 | None |
| 13 | `ocqtyt` | 1 | 30.4 | 18.8m | 11.6m | 65.70 | 66.47 | None |
| 14 | `ocqtyt` | 1 | 33.9 | 19.0m | 14.9m | 65.70 | 66.47 | None |
| 15 | `ocqtyt` | 1 | 30.5 | 18.8m | 11.7m | 65.70 | 66.47 | None |
| 16 | `ocqtyt` | 1 | 33.3 | 19.0m | 14.4m | 65.70 | 66.47 | None |
| 17 | `ocqtyt` | 2 | 31.0 | 23.0m | 8.1m | 65.70 | 66.47 | None |
| 18 | `ocqtyt` | 2 | 33.9 | 23.6m | 10.3m | 65.70 | 66.47 | None |
| 19 | `ocqtyt` | 2 | 31.2 | 23.1m | 8.1m | 65.70 | 66.47 | None |
| 20 | `ocqtyt` | 1 | 33.8 | 19.5m | 14.3m | 65.70 | 66.47 | None |
| 21 | `ocqtyt` | 2 | 37.2 | 26.5m | 10.8m | 65.70 | 66.47 | None |
| 22 | `ocqtyt` | 2 | 35.7 | 25.2m | 10.5m | 65.70 | 66.47 | None |

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 20** (`ocqtyt`)

- rank 0: apply_step_total 0.7m/244  crop_slice 1.5m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.7m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 0.4m/244  pose_total 16.4m/7320  prefetch_issue 0.0m/1463
- rank 1: apply_step_total 0.7m/244  crop_slice 1.5m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.7m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 0.8m/244  pose_total 16.4m/7320  prefetch_issue 0.0m/1463
- rank 2: apply_step_total 0.7m/244  crop_slice 1.6m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.8m/7320  grad_collective_plus_wait 0.5m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 3.8m/244  pose_total 16.5m/7320  prefetch_issue 0.0m/1463

**epoch 21** (`ocqtyt`)

- rank 0: apply_step_total 0.7m/244  crop_slice 1.5m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.7m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 0.4m/244  pose_total 16.4m/7320  prefetch_issue 0.0m/1463
- rank 1: apply_step_total 0.7m/244  crop_slice 1.5m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.7m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 0.8m/244  pose_total 16.3m/7320  prefetch_issue 0.0m/1463
- rank 2: apply_step_total 0.7m/244  crop_slice 1.6m/1464  cropped_tokens mean 326  gpu_backward 12.5m/7320  gpu_forward 5.8m/7320  grad_collective_plus_wait 0.5m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 7.6m/244  pose_total 16.5m/7320  prefetch_issue 0.0m/1463

**epoch 22** (`ocqtyt`)

- rank 0: apply_step_total 0.7m/244  crop_slice 1.6m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.7m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 0.3m/244  pose_total 16.3m/7320  prefetch_issue 0.0m/1463
- rank 1: apply_step_total 0.7m/244  crop_slice 1.6m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.7m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 0.5m/244  pose_total 16.3m/7320  prefetch_issue 0.0m/1463
- rank 2: apply_step_total 0.7m/244  crop_slice 1.7m/1464  cropped_tokens mean 326  gpu_backward 12.5m/7320  gpu_forward 5.8m/7320  grad_collective_plus_wait 0.5m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 6.4m/244  pose_total 16.5m/7320  prefetch_issue 0.0m/1463

