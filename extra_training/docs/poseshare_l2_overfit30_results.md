# Full training set: the four metrics, per epoch

Generated 2026-10-22 04:55 PDT by `run_logging/results_table.py` from `4x8t44`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

## The lineage is one experiment in three parts

Epoch numbers are continuous, but the segments do NOT share an implementation, so **throughput and memory are not comparable across them**. The metrics are.

| segment | epochs | configuration |
|---|---|---|
| `x3k2m4` | — | checkpointing ON (blocks_per_ckpt=1); per-epoch fingerprint hashing; prefetch_issue double-counted the hash |
| `jhgfxy` | — | checkpointing ON; fresh process so hashing re-paid; prefetch choice-cache added |
| `2w5to5` | — | checkpointing OFF (blocks_per_ckpt=None); persistent fingerprint cache; empty_cache at the boundary; exclusive timers + residual + memory peaks |

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
| 1 | `4x8t44` | yes | 2.5406 | 2.9088 | -0.0668 | 0.2774 | -0.0135 | 0.5000 | -0.0556 |
| 2 | `4x8t44` | yes | 2.4278 | 2.8827 | -0.0929 | 0.2602 | -0.0307 | 0.4444 | -0.1112 |
| 3 | `4x8t44` | yes | 2.4211 | 2.8689 | -0.1067 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 4 | `4x8t44` | yes | 2.3797 | 2.8699 | -0.1057 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 5 | `4x8t44` | yes | 2.3970 | 2.8819 | -0.0937 | 0.2615 | -0.0294 | 0.4444 | -0.1112 |
| 6 | `4x8t44` | yes | 2.3785 | 2.8560 | -0.1196 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 7 | `4x8t44` | yes | 2.3731 | 2.8506 | -0.1250 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 8 | `4x8t44` | yes | 2.3774 | 2.8560 | -0.1196 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 9 | `4x8t44` | yes | 2.3641 | 2.8516 | -0.1240 | 0.2607 | -0.0302 | 0.4444 | -0.1112 |
| 10 | `4x8t44` | yes | 2.3694 | 2.8365 | -0.1391 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 11 | `4x8t44` | yes | 2.3883 | 2.8432 | -0.1324 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 12 | `4x8t44` | yes | 2.3719 | 2.8373 | -0.1383 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 13 | `4x8t44` | yes | 2.3719 | 2.8442 | -0.1314 | 0.2602 | -0.0307 | 0.4444 | -0.1112 |
| 14 | `4x8t44` | yes | 2.3561 | 2.8399 | -0.1357 | 0.2602 | -0.0307 | 0.4444 | -0.1112 |
| 15 | `4x8t44` | yes | 2.3676 | 2.8247 | -0.1509 | 0.2866 | -0.0043 | 0.5000 | -0.0556 |
| 16 | `4x8t44` | yes | 2.3492 | 2.8286 | -0.1470 | 0.2858 | -0.0051 | 0.5000 | -0.0556 |
| 17 | `4x8t44` | yes | 2.3655 | 2.8243 | -0.1513 | 0.2858 | -0.0051 | 0.5000 | -0.0556 |
| 18 | `4x8t44` | yes | 2.3566 | 2.8396 | -0.1360 | 0.2864 | -0.0045 | 0.5000 | -0.0556 |
| 19 | `4x8t44` | yes | 2.3709 | 2.8478 | -0.1278 | 0.2859 | -0.0050 | 0.5000 | -0.0556 |
| 20 | `4x8t44` | yes | 2.3528 | 2.8469 | -0.1287 | 0.2852 | -0.0057 | 0.5000 | -0.0556 |
| 21 | `4x8t44` | yes | 2.3458 | 2.8339 | -0.1417 | 0.2859 | -0.0050 | 0.5000 | -0.0556 |
| 22 | `4x8t44` | yes | 2.3838 | 2.8171 | -0.1585 | 0.2524 | -0.0385 | 0.4444 | -0.1112 |
| 23 | `4x8t44` | yes | 2.3468 | 2.8406 | -0.1350 | 0.2595 | -0.0314 | 0.4444 | -0.1112 |
| 24 | `4x8t44` | yes | 2.3484 | 2.8396 | -0.1360 | 0.2859 | -0.0050 | 0.5000 | -0.0556 |
| 25 | `4x8t44` | yes | 2.3421 | 2.8679 | -0.1077 | 0.2588 | -0.0321 | 0.4444 | -0.1112 |
| 26 | `4x8t44` | yes | 2.3645 | 2.8547 | -0.1209 | 0.2596 | -0.0313 | 0.4444 | -0.1112 |
| 27 | `4x8t44` | yes | 2.3275 | 2.8305 | -0.1451 | 0.2595 | -0.0314 | 0.4444 | -0.1112 |
| 28 | `4x8t44` | yes | 2.3634 | 2.8332 | -0.1424 | 0.2512 | -0.0397 | 0.4444 | -0.1112 |
| 29 | `4x8t44` | yes | 2.3375 | 2.8360 | -0.1396 | 0.2549 | -0.0360 | 0.4444 | -0.1112 |
| 30 | `4x8t44` | yes | 2.3550 | 2.8384 | -0.1372 | 0.2597 | -0.0312 | 0.4444 | -0.1112 |
| 31 | `4x8t44` | yes | 2.3653 | 2.8308 | -0.1448 | 0.2595 | -0.0314 | 0.4444 | -0.1112 |
| 32 | `4x8t44` | yes | 2.3534 | 2.8190 | -0.1566 | 0.2469 | -0.0440 | 0.4444 | -0.1112 |
| 33 | `4x8t44` | yes | 2.3598 | 2.8327 | -0.1429 | 0.2473 | -0.0436 | 0.4444 | -0.1112 |
| 34 | `4x8t44` | yes | 2.3277 | 2.8145 | -0.1611 | 0.2473 | -0.0436 | 0.4444 | -0.1112 |
| 35 | `4x8t44` | yes | 2.3544 | 2.8265 | -0.1491 | 0.2443 | -0.0466 | 0.4444 | -0.1112 |
| 36 | `4x8t44` | yes | 2.3585 | 2.8649 | -0.1107 | 0.2593 | -0.0316 | 0.4444 | -0.1112 |
| 37 | `4x8t44` | yes | 2.3258 | 2.8570 | -0.1186 | 0.2549 | -0.0360 | 0.4444 | -0.1112 |
| 38 | `4x8t44` | yes | 2.3543 | 2.8358 | -0.1398 | 0.2472 | -0.0437 | 0.4444 | -0.1112 |
| 39 | `4x8t44` | yes | 2.3496 | 2.8641 | -0.1115 | 0.2588 | -0.0321 | 0.4444 | -0.1112 |
| 40 | `4x8t44` | yes | 2.3263 | 2.8430 | -0.1326 | 0.2638 | -0.0271 | 0.4444 | -0.1112 |
| 41 | `4x8t44` | yes | 2.3248 | 2.8284 | -0.1472 | 0.2472 | -0.0437 | 0.4444 | -0.1112 |
| 42 | `4x8t44` | yes | 2.3220 | 2.8268 | -0.1488 | 0.2437 | -0.0472 | 0.4444 | -0.1112 |
| 43 | `4x8t44` | yes | 2.3426 | 2.8326 | -0.1430 | 0.2473 | -0.0436 | 0.4444 | -0.1112 |
| 44 | `4x8t44` | yes | 2.3406 | 2.8269 | -0.1487 | 0.2474 | -0.0435 | 0.4444 | -0.1112 |
| 45 | `4x8t44` | yes | 2.3291 | 2.8299 | -0.1457 | 0.2436 | -0.0473 | 0.4444 | -0.1112 |
| 46 | `4x8t44` | yes | 2.3429 | 2.8418 | -0.1338 | 0.2472 | -0.0437 | 0.4444 | -0.1112 |
| 47 | `4x8t44` | yes | 2.3357 | 2.8541 | -0.1215 | 0.2588 | -0.0321 | 0.4444 | -0.1112 |
| 48 | `4x8t44` | yes | 2.3278 | 2.8323 | -0.1433 | 0.2551 | -0.0358 | 0.4444 | -0.1112 |
| 49 | `4x8t44` | yes | 2.3462 | 2.8393 | -0.1363 | 0.2512 | -0.0397 | 0.4444 | -0.1112 |

Held-out PAE is a loss, so **lower is better** and a negative delta is an improvement. DockQ and frac-acceptable are the opposite.

`complete` means all four metrics are present AND the held-out pass covered its expected 1500 poses. An epoch marked **NO** is mid-boundary or lost a metric to a failure — its em-dashes are missing measurements, not zeros.

## Why held-out PAE is the metric to read

On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics showed nothing the entire time because ipTM barely orders poses within an interface — Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE curve is not evidence a run is working.

**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is the thing that would settle whether selection improved, and it needs several epochs of checkpoints to be worth running.

## Per-epoch wall clock and memory

| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |
|---|---|---|---|---|---|---|---|---|
| 1 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 39.01 | None |
| 2 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 3 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 4 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 5 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 6 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 7 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 8 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 9 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 10 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 11 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 12 | `4x8t44` | 2 | 1.7 | 1.6m | 0.1m | 37.44 | 38.49 | None |
| 13 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 14 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 15 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 16 | `4x8t44` | 2 | 1.8 | 1.6m | 0.2m | 37.44 | 38.49 | None |
| 17 | `4x8t44` | 1 | 1.7 | 1.6m | 0.1m | 37.44 | 38.49 | None |
| 18 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 19 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 20 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 21 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 22 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 23 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 24 | `4x8t44` | 1 | 1.8 | 1.6m | 0.2m | 37.44 | 38.49 | None |
| 25 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 26 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 27 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 28 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 29 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 30 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 31 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 32 | `4x8t44` | 2 | 5.3 | 1.8m | 3.4m | 37.44 | 38.49 | None |
| 33 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 34 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 35 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 36 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 37 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 38 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 39 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 40 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 41 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 42 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 43 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 44 | `4x8t44` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 45 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 46 | `4x8t44` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 47 | `4x8t44` | 1 | 2.1 | 1.5m | 0.5m | 37.44 | 38.49 | None |
| 48 | `4x8t44` | 1 | 1.9 | 1.6m | 0.3m | 37.44 | 38.49 | None |
| 49 | `4x8t44` | 1 | 2.0 | 1.6m | 0.4m | 37.44 | 38.49 | None |
| 50 | `?` | 1 | 1.8 | 1.6m | 0.3m | 37.44 | 38.49 | None |

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 48** (`4x8t44`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.3m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.3m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

**epoch 49** (`4x8t44`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.4m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.4m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

**epoch 50** (`?`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.2m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.2m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

