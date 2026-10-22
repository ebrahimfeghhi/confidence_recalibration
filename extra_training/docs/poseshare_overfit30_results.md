# Full training set: the four metrics, per epoch

Generated 2026-10-21 23:46 PDT by `run_logging/results_table.py` from `8eiek8`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

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
| 1 | `8eiek8` | yes | 2.5389 | 2.9056 | -0.0700 | 0.2782 | -0.0127 | 0.5000 | -0.0556 |
| 2 | `8eiek8` | yes | 2.4173 | 2.8776 | -0.0980 | 0.2607 | -0.0302 | 0.4444 | -0.1112 |
| 3 | `8eiek8` | yes | 2.4027 | 2.8692 | -0.1064 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 4 | `8eiek8` | yes | 2.3565 | 2.8915 | -0.0841 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 5 | `8eiek8` | yes | 2.3702 | 2.8880 | -0.0876 | 0.2614 | -0.0295 | 0.4444 | -0.1112 |
| 6 | `8eiek8` | yes | 2.3550 | 2.8552 | -0.1204 | 0.2624 | -0.0285 | 0.4444 | -0.1112 |
| 7 | `8eiek8` | yes | 2.3383 | 2.8478 | -0.1278 | 0.2902 | -0.0007 | 0.5000 | -0.0556 |
| 8 | `8eiek8` | yes | 2.3401 | 2.8479 | -0.1277 | 0.3018 | +0.0109 | 0.5556 | +0.0000 |
| 9 | `8eiek8` | yes | 2.3254 | 2.8406 | -0.1350 | 0.3017 | +0.0108 | 0.5556 | +0.0000 |
| 10 | `8eiek8` | yes | 2.3311 | 2.8338 | -0.1418 | 0.3246 | +0.0337 | 0.6111 | +0.0555 |
| 11 | `8eiek8` | yes | 2.3444 | 2.8327 | -0.1429 | 0.3052 | +0.0143 | 0.5556 | +0.0000 |
| 12 | `8eiek8` | yes | 2.3273 | 2.8225 | -0.1531 | 0.2845 | -0.0064 | 0.5556 | +0.0000 |
| 13 | `8eiek8` | yes | 2.3270 | 2.8345 | -0.1411 | 0.3062 | +0.0153 | 0.5556 | +0.0000 |
| 14 | `8eiek8` | yes | 2.3086 | 2.8147 | -0.1609 | 0.2828 | -0.0081 | 0.5556 | +0.0000 |
| 15 | `8eiek8` | yes | 2.3187 | 2.8033 | -0.1723 | 0.2701 | -0.0208 | 0.4444 | -0.1112 |
| 16 | `8eiek8` | yes | 2.3008 | 2.8117 | -0.1639 | 0.3103 | +0.0194 | 0.5556 | +0.0000 |
| 17 | `8eiek8` | yes | 2.3194 | 2.8057 | -0.1699 | 0.3049 | +0.0140 | 0.5000 | -0.0556 |
| 18 | `8eiek8` | yes | 2.3073 | 2.8242 | -0.1514 | 0.3110 | +0.0201 | 0.6111 | +0.0555 |
| 19 | `8eiek8` | yes | 2.3214 | 2.8257 | -0.1499 | 0.3098 | +0.0189 | 0.5556 | +0.0000 |
| 20 | `8eiek8` | yes | 2.3066 | 2.8283 | -0.1473 | 0.3081 | +0.0172 | 0.5556 | +0.0000 |
| 21 | `8eiek8` | yes | 2.2993 | 2.7979 | -0.1777 | 0.3067 | +0.0158 | 0.5556 | +0.0000 |
| 22 | `8eiek8` | yes | 2.3354 | 2.7900 | -0.1856 | 0.3081 | +0.0172 | 0.5556 | +0.0000 |
| 23 | `8eiek8` | yes | 2.2976 | 2.7972 | -0.1784 | 0.3064 | +0.0155 | 0.5556 | +0.0000 |
| 24 | `8eiek8` | yes | 2.3001 | 2.8205 | -0.1551 | 0.3122 | +0.0213 | 0.5556 | +0.0000 |
| 25 | `8eiek8` | yes | 2.2926 | 2.8399 | -0.1357 | 0.2996 | +0.0087 | 0.5000 | -0.0556 |
| 26 | `8eiek8` | yes | 2.3149 | 2.8125 | -0.1631 | 0.3004 | +0.0095 | 0.5000 | -0.0556 |
| 27 | `8eiek8` | yes | 2.2806 | 2.7876 | -0.1880 | 0.2978 | +0.0069 | 0.5000 | -0.0556 |
| 28 | `8eiek8` | yes | 2.3138 | 2.8056 | -0.1700 | 0.3108 | +0.0199 | 0.5556 | +0.0000 |
| 29 | `8eiek8` | yes | 2.2887 | 2.7919 | -0.1837 | 0.2929 | +0.0020 | 0.4444 | -0.1112 |
| 30 | `8eiek8` | yes | 2.3027 | 2.7938 | -0.1818 | 0.2974 | +0.0065 | 0.5000 | -0.0556 |
| 31 | `8eiek8` | yes | 2.3107 | 2.7844 | -0.1912 | 0.2974 | +0.0065 | 0.5000 | -0.0556 |
| 32 | `8eiek8` | yes | 2.3019 | 2.7715 | -0.2041 | 0.2957 | +0.0048 | 0.5000 | -0.0556 |
| 33 | `8eiek8` | yes | 2.3074 | 2.7900 | -0.1856 | 0.2974 | +0.0065 | 0.5000 | -0.0556 |
| 34 | `8eiek8` | yes | 2.2727 | 2.7750 | -0.2006 | 0.2967 | +0.0058 | 0.5000 | -0.0556 |
| 35 | `8eiek8` | yes | 2.2988 | 2.7740 | -0.2016 | 0.2967 | +0.0058 | 0.5000 | -0.0556 |
| 36 | `8eiek8` | yes | 2.3037 | 2.8098 | -0.1658 | 0.2954 | +0.0045 | 0.5000 | -0.0556 |
| 37 | `8eiek8` | yes | 2.2699 | 2.7997 | -0.1759 | 0.2969 | +0.0060 | 0.5000 | -0.0556 |
| 38 | `8eiek8` | yes | 2.2983 | 2.8055 | -0.1701 | 0.2977 | +0.0068 | 0.5000 | -0.0556 |
| 39 | `8eiek8` | yes | 2.2938 | 2.8458 | -0.1298 | 0.2977 | +0.0068 | 0.5000 | -0.0556 |
| 40 | `8eiek8` | yes | 2.2787 | 2.7754 | -0.2002 | 0.2969 | +0.0060 | 0.5000 | -0.0556 |
| 41 | `8eiek8` | yes | 2.2708 | 2.7884 | -0.1872 | 0.2946 | +0.0037 | 0.4444 | -0.1112 |
| 42 | `8eiek8` | yes | 2.2670 | 2.7690 | -0.2066 | 0.2846 | -0.0063 | 0.5000 | -0.0556 |
| 43 | `8eiek8` | yes | 2.2844 | 2.7786 | -0.1970 | 0.2974 | +0.0065 | 0.5000 | -0.0556 |
| 44 | `8eiek8` | yes | 2.2785 | 2.7702 | -0.2054 | 0.2979 | +0.0070 | 0.5000 | -0.0556 |
| 45 | `8eiek8` | yes | 2.2753 | 2.7709 | -0.2047 | 0.2893 | -0.0016 | 0.5000 | -0.0556 |
| 46 | `8eiek8` | yes | 2.2832 | 2.7991 | -0.1765 | 0.2984 | +0.0075 | 0.5000 | -0.0556 |
| 47 | `8eiek8` | yes | 2.2788 | 2.7953 | -0.1803 | 0.2979 | +0.0070 | 0.5000 | -0.0556 |
| 48 | `8eiek8` | yes | 2.2650 | 2.7804 | -0.1952 | 0.3001 | +0.0092 | 0.5000 | -0.0556 |
| 49 | `8eiek8` | yes | 2.2850 | 2.7820 | -0.1936 | 0.2911 | +0.0002 | 0.5000 | -0.0556 |
| 50 | `8eiek8` | **NO** | 2.2556 | 2.8047 | -0.1709 | — | — | — | — |

Held-out PAE is a loss, so **lower is better** and a negative delta is an improvement. DockQ and frac-acceptable are the opposite.

`complete` means all four metrics are present AND the held-out pass covered its expected 1500 poses. An epoch marked **NO** is mid-boundary or lost a metric to a failure — its em-dashes are missing measurements, not zeros.

## Why held-out PAE is the metric to read

On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics showed nothing the entire time because ipTM barely orders poses within an interface — Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE curve is not evidence a run is working.

**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is the thing that would settle whether selection improved, and it needs several epochs of checkpoints to be worth running.

## Per-epoch wall clock and memory

| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |
|---|---|---|---|---|---|---|---|---|
| 1 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 39.01 | None |
| 2 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 3 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 4 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 5 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 6 | `8eiek8` | 1 | 1.7 | 1.6m | 0.1m | 37.44 | 38.49 | None |
| 7 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 8 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 9 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 10 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 11 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 12 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 13 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 14 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 15 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 16 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 17 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 18 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 19 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 20 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 21 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 22 | `8eiek8` | 1 | 1.6 | 1.6m | 0.1m | 37.44 | 38.49 | None |
| 23 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 24 | `8eiek8` | 1 | 2.8 | 1.6m | 1.2m | 37.44 | 38.49 | None |
| 25 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 26 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 27 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 28 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 29 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 30 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 31 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 32 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 33 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 34 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 35 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 36 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 37 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 38 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 39 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 40 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 41 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 42 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 43 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 44 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 45 | `8eiek8` | 2 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 46 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 47 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 48 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 49 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |
| 50 | `8eiek8` | 1 | 1.6 | 1.6m | 0.0m | 37.44 | 38.49 | None |

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 48** (`8eiek8`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

**epoch 49** (`8eiek8`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

**epoch 50** (`8eiek8`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89  [rank 2] epoch 50 took 1.6 min -- apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.0m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.4m/450  prefetch_issue 0.0m/89

