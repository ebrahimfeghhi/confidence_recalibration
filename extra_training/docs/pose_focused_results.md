# Full training set: the four metrics, per epoch

Generated 2026-10-21 10:38 PDT by `run_logging/results_table.py` from `ggfhj5+zcuiab`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

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
| 1 | `ggfhj5` | **NO** | 2.4282 | — | — | 0.3008 | +0.0099 | 0.5556 | +0.0000 |
| 2 | `ggfhj5` | yes | 2.3127 | 2.8063 | -0.1693 | 0.3099 | +0.0190 | 0.5556 | +0.0000 |
| 3 | `ggfhj5` | yes | 2.3064 | 2.7935 | -0.1821 | 0.3063 | +0.0154 | 0.5556 | +0.0000 |
| 4 | `ggfhj5` | yes | 2.2660 | 2.8011 | -0.1745 | 0.3035 | +0.0126 | 0.5556 | +0.0000 |
| 5 | `ggfhj5` | yes | 2.2819 | 2.7996 | -0.1760 | 0.3099 | +0.0190 | 0.5556 | +0.0000 |
| 6 | `ggfhj5` | yes | 2.2650 | 2.7790 | -0.1966 | 0.2899 | -0.0010 | 0.5556 | +0.0000 |
| 7 | `ggfhj5` | yes | 2.2500 | 2.7750 | -0.2006 | 0.2837 | -0.0072 | 0.5556 | +0.0000 |
| 8 | `ggfhj5` | yes | 2.2541 | 2.7684 | -0.2072 | 0.2858 | -0.0051 | 0.5556 | +0.0000 |
| 9 | `ggfhj5` | yes | 2.2420 | 2.7714 | -0.2042 | 0.2837 | -0.0072 | 0.5556 | +0.0000 |
| 10 | `ggfhj5` | yes | 2.2542 | 2.7571 | -0.2185 | 0.2751 | -0.0158 | 0.5556 | +0.0000 |
| 11 | `ggfhj5` | yes | 2.2744 | 2.7528 | -0.2228 | 0.2820 | -0.0089 | 0.5556 | +0.0000 |
| 12 | `zcuiab` | yes | 2.2485 | 2.7525 | -0.2231 | 0.2820 | -0.0089 | 0.5556 | +0.0000 |
| 13 | `zcuiab` | yes | 2.2518 | 2.7558 | -0.2198 | 0.2820 | -0.0089 | 0.5556 | +0.0000 |
| 14 | `zcuiab` | yes | 2.2322 | 2.7536 | -0.2220 | 0.2820 | -0.0089 | 0.5556 | +0.0000 |
| 15 | `zcuiab` | yes | 2.2481 | 2.7510 | -0.2246 | 0.2506 | -0.0403 | 0.5000 | -0.0556 |
| 16 | `zcuiab` | yes | 2.2302 | 2.7633 | -0.2123 | 0.2798 | -0.0111 | 0.5556 | +0.0000 |
| 17 | `zcuiab` | yes | 2.2518 | 2.7547 | -0.2209 | 0.2569 | -0.0340 | 0.5000 | -0.0556 |
| 18 | `zcuiab` | yes | 2.2389 | 2.7589 | -0.2167 | 0.2518 | -0.0391 | 0.5000 | -0.0556 |
| 19 | `zcuiab` | yes | 2.2511 | 2.7548 | -0.2208 | 0.2643 | -0.0266 | 0.5000 | -0.0556 |
| 20 | `zcuiab` | yes | 2.2323 | 2.7642 | -0.2114 | 0.2838 | -0.0071 | 0.5556 | +0.0000 |
| 21 | `zcuiab` | yes | 2.2329 | 2.7521 | -0.2235 | 0.2734 | -0.0175 | 0.5000 | -0.0556 |

Held-out PAE is a loss, so **lower is better** and a negative delta is an improvement. DockQ and frac-acceptable are the opposite.

`complete` means all four metrics are present AND the held-out pass covered its expected 1500 poses. An epoch marked **NO** is mid-boundary or lost a metric to a failure — its em-dashes are missing measurements, not zeros.

## Why held-out PAE is the metric to read

On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics showed nothing the entire time because ipTM barely orders poses within an interface — Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE curve is not evidence a run is working.

**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is the thing that would settle whether selection improved, and it needs several epochs of checkpoints to be worth running.

## Per-epoch wall clock and memory

| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |
|---|---|---|---|---|---|---|---|---|
| 1 | `ggfhj5` | 1 | 3.4 | 1.8m | 1.6m | 36.72 | 37.82 | None |
| 2 | `ggfhj5` | 2 | 1.7 | 1.7m | 0.0m | 36.72 | 37.30 | None |
| 3 | `ggfhj5` | 2 | 1.7 | 1.7m | 0.0m | 36.72 | 37.30 | None |
| 4 | `ggfhj5` | 2 | 1.8 | 1.8m | 0.0m | 36.72 | 37.30 | None |
| 5 | `ggfhj5` | 1 | 1.7 | 1.7m | 0.0m | 36.72 | 37.30 | None |
| 6 | `ggfhj5` | 1 | 1.9 | 1.6m | 0.2m | 36.72 | 37.30 | None |
| 7 | `ggfhj5` | 2 | 1.9 | 1.8m | 0.1m | 36.72 | 37.32 | None |
| 8 | `ggfhj5` | 2 | 1.7 | 1.7m | 0.0m | 36.72 | 37.30 | None |
| 9 | `ggfhj5` | 2 | 1.7 | 1.7m | 0.0m | 36.72 | 37.30 | None |
| 10 | `ggfhj5` | 1 | 1.9 | 1.7m | 0.2m | 36.72 | 37.30 | None |
| 11 | `ggfhj5` | 2 | 2.1 | 1.9m | 0.2m | 36.72 | 37.30 | None |
| 12 | `zcuiab` | 1 | 2.1 | 2.1m | 0.0m | 36.72 | 37.84 | None |
| 13 | `zcuiab` | 1 | 1.7 | 1.7m | 0.0m | 36.72 | 37.32 | None |
| 14 | `zcuiab` | 1 | 1.8 | 1.8m | 0.0m | 36.72 | 37.32 | None |
| 15 | `zcuiab` | 1 | 1.8 | 1.8m | 0.0m | 36.72 | 37.32 | None |
| 16 | `zcuiab` | 1 | 1.6 | 1.6m | 0.0m | 36.72 | 37.32 | None |
| 17 | `zcuiab` | 1 | 1.7 | 1.6m | 0.1m | 36.72 | 37.32 | None |
| 18 | `zcuiab` | 1 | 1.6 | 1.6m | 0.0m | 36.72 | 37.32 | None |
| 19 | `zcuiab` | 2 | 2.0 | 2.0m | 0.0m | 36.72 | 37.32 | None |
| 20 | `zcuiab` | 2 | 1.9 | 1.8m | 0.0m | 36.72 | 37.32 | None |
| 21 | `zcuiab` | 1 | 1.9 | 1.9m | 0.0m | 36.72 | 37.32 | None |

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 19** (`zcuiab`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.4m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.4m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

**epoch 20** (`zcuiab`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.3m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89  [rank 2] epoch 20 accounted 1.8m of 1.9m, residual 0.0m (exclusive buckets only; gpu_* overlap pose_total)
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.3m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

**epoch 21** (`zcuiab`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.3m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.3m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

