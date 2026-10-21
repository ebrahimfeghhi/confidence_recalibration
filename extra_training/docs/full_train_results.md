# Full training set: the four metrics, per epoch

Generated 2026-10-20 17:37 PDT by `run_logging/results_table.py` from `x3k2m4+jhgfxy+2w5to5+ocqtyt`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

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

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 5** (`2w5to5`)

- rank 0: apply_step_total 1.0m/289  crop_slice 23.0m/1620  cropped_tokens mean 329  fingerprint_hash 3.8m/640  gpu_backward 14.1m/8100  gpu_forward 6.2m/8100  grad_collective_plus_wait 0.7m/290  label_open 0.0m/1620  loss_feature_prep 0.0m/1620  loss_stats_collective_plus_wait 24.8m/289  pose_total 17.8m/8100  prefetch_issue 0.0m/1619  trunk_read_plus_crop_tokens 2.6m/1620
- rank 1: apply_step_total 1.0m/289  crop_slice 20.6m/1620  cropped_tokens mean 339  fingerprint_hash 3.8m/710  gpu_backward 14.8m/8100  gpu_forward 6.5m/8100  grad_collective_plus_wait 0.7m/290  label_open 0.0m/1620  loss_feature_prep 0.0m/1620  loss_stats_collective_plus_wait 26.3m/289  pose_total 18.7m/8100  prefetch_issue 0.0m/1619  trunk_read_plus_crop_tokens 2.6m/1620
- rank 2: apply_step_total 1.0m/289  crop_slice 18.9m/1620  cropped_tokens mean 311  fingerprint_hash 4.9m/810  gpu_backward 12.9m/8100  gpu_forward 5.8m/8100  grad_collective_plus_wait 0.7m/290  label_open 0.0m/1620  loss_feature_prep 0.0m/1620  loss_stats_collective_plus_wait 29.1m/289  pose_total 16.5m/8100  prefetch_issue 0.0m/1619  trunk_read_plus_crop_tokens 2.5m/1620

**epoch 6** (`2w5to5`)

- rank 0: apply_step_total 1.1m/289  crop_slice 19.3m/1620  cropped_tokens mean 313  fingerprint_hash 3.4m/560  gpu_backward 12.8m/8100  gpu_forward 5.7m/8100  grad_collective_plus_wait 0.8m/290  label_open 0.0m/1620  loss_feature_prep 0.0m/1620  loss_stats_collective_plus_wait 22.6m/289  pose_total 16.3m/8100  prefetch_issue 0.0m/1619  trunk_read_plus_crop_tokens 2.7m/1620
- rank 1: apply_step_total 1.1m/289  crop_slice 17.4m/1620  cropped_tokens mean 346  fingerprint_hash 3.0m/440  gpu_backward 16.5m/8100  gpu_forward 7.2m/8100  grad_collective_plus_wait 0.7m/290  label_open 0.0m/1620  loss_feature_prep 0.0m/1620  loss_stats_collective_plus_wait 19.8m/289  pose_total 21.4m/8100  prefetch_issue 0.0m/1619  trunk_read_plus_crop_tokens 2.7m/1620
- rank 2: apply_step_total 1.1m/289  crop_slice 16.3m/1620  cropped_tokens mean 321  fingerprint_hash 2.9m/490  gpu_backward 13.3m/8100  gpu_forward 6.0m/8100  grad_collective_plus_wait 0.8m/290  label_open 0.0m/1620  loss_feature_prep 0.0m/1620  loss_stats_collective_plus_wait 25.3m/289  pose_total 17.1m/8100  prefetch_issue 0.0m/1619  trunk_read_plus_crop_tokens 2.6m/1620

**epoch 7** (`ocqtyt`)

- rank 0: apply_step_total 0.6m/244  crop_slice 1.6m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.8m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 4.7m/244  pose_total 16.6m/7320  prefetch_issue 0.0m/1463
- rank 1: apply_step_total 0.6m/244  crop_slice 1.6m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.9m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 0.7m/244  pose_total 16.6m/7320  prefetch_issue 0.0m/1463
- rank 2: apply_step_total 0.6m/244  crop_slice 1.6m/1464  cropped_tokens mean 326  gpu_backward 12.4m/7320  gpu_forward 5.8m/7320  grad_collective_plus_wait 0.4m/244  label_open 0.0m/1464  loss_feature_prep 0.0m/1464  loss_stats_collective_plus_wait 4.7m/244  pose_total 16.5m/7320  prefetch_issue 0.0m/1463

