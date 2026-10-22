# Full training set: the four metrics, per epoch

Generated 2026-10-21 13:56 PDT by `run_logging/results_table.py` from `mdv9x7`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

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
| 1 | `mdv9x7` | yes | 2.3765 | 2.7457 | -0.2299 | 0.2712 | -0.0197 | 0.4444 | -0.1112 |
| 2 | `mdv9x7` | yes | 2.2646 | 2.7466 | -0.2290 | 0.2774 | -0.0135 | 0.5000 | -0.0556 |
| 3 | `mdv9x7` | yes | 2.2407 | 2.7127 | -0.2629 | 0.2444 | -0.0465 | 0.5000 | -0.0556 |
| 4 | `mdv9x7` | yes | 2.1777 | 2.7555 | -0.2201 | 0.2800 | -0.0109 | 0.5000 | -0.0556 |
| 5 | `mdv9x7` | yes | 2.1997 | 2.7448 | -0.2308 | 0.2658 | -0.0251 | 0.4444 | -0.1112 |
| 6 | `mdv9x7` | yes | 2.1678 | 2.7649 | -0.2107 | 0.2571 | -0.0338 | 0.5000 | -0.0556 |
| 7 | `mdv9x7` | yes | 2.1318 | 2.8060 | -0.1696 | 0.2842 | -0.0067 | 0.5000 | -0.0556 |
| 8 | `mdv9x7` | yes | 2.1452 | 2.7428 | -0.2328 | 0.2755 | -0.0154 | 0.5000 | -0.0556 |
| 9 | `mdv9x7` | yes | 2.1094 | 2.7149 | -0.2607 | 0.2576 | -0.0333 | 0.4444 | -0.1112 |
| 10 | `mdv9x7` | yes | 2.1259 | 2.7290 | -0.2466 | 0.2792 | -0.0117 | 0.4444 | -0.1112 |
| 11 | `mdv9x7` | yes | 2.1283 | 2.7317 | -0.2439 | 0.2717 | -0.0192 | 0.5556 | +0.0000 |
| 12 | `mdv9x7` | yes | 2.0804 | 2.6881 | -0.2875 | 0.2689 | -0.0220 | 0.4444 | -0.1112 |
| 13 | `mdv9x7` | yes | 2.0832 | 2.7199 | -0.2557 | 0.2624 | -0.0285 | 0.6111 | +0.0555 |
| 14 | `mdv9x7` | yes | 2.0423 | 2.7359 | -0.2397 | 0.2870 | -0.0039 | 0.5556 | +0.0000 |
| 15 | `mdv9x7` | yes | 2.0417 | 2.7402 | -0.2354 | 0.2508 | -0.0401 | 0.5556 | +0.0000 |
| 16 | `mdv9x7` | yes | 2.0035 | 2.6969 | -0.2787 | 0.2693 | -0.0216 | 0.6111 | +0.0555 |
| 17 | `mdv9x7` | yes | 2.0309 | 2.7817 | -0.1939 | 0.2505 | -0.0404 | 0.5556 | +0.0000 |
| 18 | `mdv9x7` | yes | 2.0474 | 2.7314 | -0.2442 | 0.2481 | -0.0428 | 0.5000 | -0.0556 |
| 19 | `mdv9x7` | yes | 2.0178 | 2.7664 | -0.2092 | 0.2381 | -0.0528 | 0.5000 | -0.0556 |
| 20 | `mdv9x7` | yes | 1.9694 | 2.7517 | -0.2239 | 0.2441 | -0.0468 | 0.5556 | +0.0000 |
| 21 | `mdv9x7` | yes | 1.9385 | 2.8285 | -0.1471 | 0.2072 | -0.0837 | 0.5000 | -0.0556 |
| 22 | `mdv9x7` | yes | 1.9866 | 2.7211 | -0.2545 | 0.2018 | -0.0891 | 0.5000 | -0.0556 |
| 23 | `mdv9x7` | yes | 1.9388 | 2.7865 | -0.1891 | 0.2904 | -0.0005 | 0.5556 | +0.0000 |
| 24 | `mdv9x7` | yes | 1.9685 | 2.8045 | -0.1711 | 0.2170 | -0.0739 | 0.5000 | -0.0556 |
| 25 | `mdv9x7` | yes | 1.9217 | 2.8063 | -0.1693 | 0.1790 | -0.1119 | 0.3889 | -0.1667 |
| 26 | `mdv9x7` | yes | 1.9166 | 2.7532 | -0.2224 | 0.1883 | -0.1026 | 0.4444 | -0.1112 |
| 27 | `mdv9x7` | yes | 1.8788 | 2.8199 | -0.1557 | 0.2712 | -0.0197 | 0.6111 | +0.0555 |
| 28 | `mdv9x7` | yes | 1.8934 | 3.1125 | +0.1369 | 0.1699 | -0.1210 | 0.2222 | -0.3334 |
| 29 | `mdv9x7` | yes | 1.8679 | 2.8310 | -0.1446 | 0.2019 | -0.0890 | 0.3889 | -0.1667 |
| 30 | `mdv9x7` | **NO** | 1.8666 | 2.8182 | -0.1574 | — | — | — | — |

Held-out PAE is a loss, so **lower is better** and a negative delta is an improvement. DockQ and frac-acceptable are the opposite.

`complete` means all four metrics are present AND the held-out pass covered its expected 1500 poses. An epoch marked **NO** is mid-boundary or lost a metric to a failure — its em-dashes are missing measurements, not zeros.

## Why held-out PAE is the metric to read

On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics showed nothing the entire time because ipTM barely orders poses within an interface — Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE curve is not evidence a run is working.

**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is the thing that would settle whether selection improved, and it needs several epochs of checkpoints to be worth running.

## Per-epoch wall clock and memory

| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |
|---|---|---|---|---|---|---|---|---|
| 1 | `mdv9x7` | 2 | 2.0 | 1.8m | 0.2m | 44.82 | 45.67 | None |
| 2 | `mdv9x7` | 2 | 2.0 | 1.8m | 0.2m | 44.82 | 45.34 | None |
| 3 | `mdv9x7` | 1 | 1.9 | 1.8m | 0.1m | 44.82 | 45.34 | None |
| 4 | `mdv9x7` | 2 | 1.8 | 1.7m | 0.1m | 44.82 | 45.34 | None |
| 5 | `mdv9x7` | 2 | 2.0 | 1.7m | 0.2m | 44.82 | 45.34 | None |
| 6 | `mdv9x7` | 2 | 2.1 | 1.8m | 0.4m | 44.82 | 45.34 | None |
| 7 | `mdv9x7` | 2 | 2.0 | 1.7m | 0.3m | 44.82 | 45.34 | None |
| 8 | `mdv9x7` | 2 | 1.9 | 1.7m | 0.2m | 44.82 | 45.34 | None |
| 9 | `mdv9x7` | 2 | 2.1 | 1.8m | 0.3m | 44.82 | 45.34 | None |
| 10 | `mdv9x7` | 2 | 1.9 | 1.7m | 0.1m | 44.82 | 45.34 | None |
| 11 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 12 | `mdv9x7` | 2 | 1.8 | 1.7m | 0.0m | 44.82 | 45.34 | None |
| 13 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 14 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 15 | `mdv9x7` | 2 | 1.8 | 1.7m | 0.1m | 44.82 | 45.34 | None |
| 16 | `mdv9x7` | 2 | 1.9 | 1.7m | 0.2m | 44.82 | 45.34 | None |
| 17 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 18 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 19 | `mdv9x7` | 2 | 1.8 | 1.7m | 0.1m | 44.82 | 45.34 | None |
| 20 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 21 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 22 | `mdv9x7` | 1 | 1.8 | 1.7m | 0.0m | 44.82 | 45.34 | None |
| 23 | `mdv9x7` | 2 | 1.8 | 1.7m | 0.0m | 44.82 | 45.34 | None |
| 24 | `mdv9x7` | 2 | 1.8 | 1.7m | 0.0m | 44.82 | 45.34 | None |
| 25 | `mdv9x7` | 2 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 26 | `mdv9x7` | 0 | 1.8 | 1.7m | 0.0m | 44.82 | 45.34 | None |
| 27 | `mdv9x7` | 2 | 1.8 | 1.7m | 0.0m | 44.82 | 45.34 | None |
| 28 | `mdv9x7` | 1 | 1.8 | 1.8m | 0.0m | 44.82 | 45.34 | None |
| 29 | `mdv9x7` | 2 | 2.1 | 1.8m | 0.3m | 44.82 | 45.34 | None |
| 30 | `mdv9x7` | 1 | 1.8 | 1.7m | 0.0m | 44.82 | 45.34 | None |

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 28** (`mdv9x7`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89  [rank 1] epoch 28 memory: peak allocated 44.82 GB, peak reserved 45.34 GB, blocks_per_ckpt=None

**epoch 29** (`mdv9x7`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.1m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89

**epoch 30** (`mdv9x7`)

- rank 0: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89

