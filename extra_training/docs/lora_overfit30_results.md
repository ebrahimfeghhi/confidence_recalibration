# Full training set: the four metrics, per epoch

Generated 2026-10-21 18:30 PDT by `run_logging/results_table.py` from `eewbr5`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

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
| 1 | `eewbr5` | yes | 2.4163 | 2.8052 | -0.1704 | 0.2822 | -0.0087 | 0.5000 | -0.0556 |
| 2 | `eewbr5` | yes | 2.2906 | 2.7774 | -0.1982 | 0.2659 | -0.0250 | 0.4444 | -0.1112 |
| 3 | `eewbr5` | yes | 2.2651 | 2.7576 | -0.2180 | 0.2854 | -0.0055 | 0.5000 | -0.0556 |
| 4 | `eewbr5` | yes | 2.2023 | 2.7980 | -0.1776 | 0.3087 | +0.0178 | 0.5556 | +0.0000 |
| 5 | `eewbr5` | yes | 2.1959 | 2.7634 | -0.2122 | 0.1989 | -0.0920 | 0.3889 | -0.1667 |
| 6 | `eewbr5` | yes | 2.2045 | 2.7818 | -0.1938 | 0.2997 | +0.0088 | 0.5000 | -0.0556 |
| 7 | `eewbr5` | yes | 2.1716 | 2.7287 | -0.2469 | 0.2625 | -0.0284 | 0.4444 | -0.1112 |
| 8 | `eewbr5` | yes | 2.1686 | 2.7203 | -0.2553 | 0.2047 | -0.0862 | 0.4444 | -0.1112 |
| 9 | `eewbr5` | yes | 2.1386 | 2.7574 | -0.2182 | 0.2414 | -0.0495 | 0.5000 | -0.0556 |
| 10 | `eewbr5` | yes | 2.1339 | 2.7331 | -0.2425 | 0.2809 | -0.0100 | 0.4444 | -0.1112 |
| 11 | `eewbr5` | yes | 2.1356 | 2.7424 | -0.2332 | 0.2331 | -0.0578 | 0.4444 | -0.1112 |
| 12 | `eewbr5` | yes | 2.0962 | 2.7228 | -0.2528 | 0.2333 | -0.0576 | 0.4444 | -0.1112 |
| 13 | `eewbr5` | yes | 2.1185 | 2.7057 | -0.2699 | 0.2256 | -0.0653 | 0.3889 | -0.1667 |
| 14 | `eewbr5` | yes | 2.0873 | 2.7217 | -0.2539 | 0.2319 | -0.0590 | 0.4444 | -0.1112 |
| 15 | `eewbr5` | yes | 2.0781 | 2.7306 | -0.2450 | 0.2486 | -0.0423 | 0.4444 | -0.1112 |
| 16 | `eewbr5` | yes | 2.0429 | 2.7394 | -0.2362 | 0.2373 | -0.0536 | 0.3889 | -0.1667 |
| 17 | `eewbr5` | yes | 2.0673 | 2.7628 | -0.2128 | 0.2499 | -0.0410 | 0.3889 | -0.1667 |
| 18 | `eewbr5` | yes | 2.0418 | 2.8336 | -0.1420 | 0.2004 | -0.0905 | 0.3333 | -0.2223 |
| 19 | `eewbr5` | yes | 2.0680 | 2.7632 | -0.2124 | 0.2862 | -0.0047 | 0.5000 | -0.0556 |
| 20 | `eewbr5` | yes | 2.0173 | 2.7475 | -0.2281 | 0.2307 | -0.0602 | 0.4444 | -0.1112 |
| 21 | `eewbr5` | yes | 1.9946 | 2.7609 | -0.2147 | 0.2879 | -0.0030 | 0.4444 | -0.1112 |
| 22 | `eewbr5` | yes | 2.0241 | 2.7589 | -0.2167 | 0.2475 | -0.0434 | 0.4444 | -0.1112 |
| 23 | `eewbr5` | yes | 1.9721 | 2.7285 | -0.2471 | 0.2416 | -0.0493 | 0.4444 | -0.1112 |
| 24 | `eewbr5` | yes | 1.9870 | 2.8066 | -0.1690 | 0.2793 | -0.0116 | 0.4444 | -0.1112 |
| 25 | `eewbr5` | yes | 1.9863 | 2.7468 | -0.2288 | 0.2578 | -0.0331 | 0.4444 | -0.1112 |
| 26 | `eewbr5` | yes | 1.9878 | 2.7592 | -0.2164 | 0.2905 | -0.0004 | 0.4444 | -0.1112 |
| 27 | `eewbr5` | yes | 1.9740 | 2.7431 | -0.2325 | 0.3338 | +0.0429 | 0.5556 | +0.0000 |
| 28 | `eewbr5` | yes | 2.0151 | 2.7681 | -0.2075 | 0.2863 | -0.0046 | 0.5556 | +0.0000 |
| 29 | `eewbr5` | yes | 1.9502 | 2.8515 | -0.1241 | 0.2394 | -0.0515 | 0.5000 | -0.0556 |
| 30 | `eewbr5` | yes | 1.9568 | 2.7929 | -0.1827 | 0.3014 | +0.0105 | 0.5000 | -0.0556 |

Held-out PAE is a loss, so **lower is better** and a negative delta is an improvement. DockQ and frac-acceptable are the opposite.

`complete` means all four metrics are present AND the held-out pass covered its expected 1500 poses. An epoch marked **NO** is mid-boundary or lost a metric to a failure — its em-dashes are missing measurements, not zeros.

## Why held-out PAE is the metric to read

On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics showed nothing the entire time because ipTM barely orders poses within an interface — Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE curve is not evidence a run is working.

**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is the thing that would settle whether selection improved, and it needs several epochs of checkpoints to be worth running.

## Per-epoch wall clock and memory

| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |
|---|---|---|---|---|---|---|---|---|
| 1 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.64 | None |
| 2 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 3 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 4 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 54.76 | None |
| 5 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 6 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 7 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 54.76 | None |
| 8 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 9 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 10 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 54.76 | None |
| 11 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 54.76 | None |
| 12 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 13 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 14 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 15 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 16 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 17 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 18 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 19 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 20 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 54.76 | None |
| 21 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.16 | None |
| 22 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 23 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 24 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 54.76 | None |
| 25 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 54.76 | None |
| 26 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 27 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 28 | `eewbr5` | 2 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 29 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |
| 30 | `eewbr5` | 1 | 2.6 | 2.6m | 0.0m | 52.37 | 53.15 | None |

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 28** (`eewbr5`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89

**epoch 29** (`eewbr5`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89

**epoch 30** (`eewbr5`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.1m/90  cropped_tokens mean 407  gpu_backward 1.5m/450  gpu_forward 1.0m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 2.4m/450  prefetch_issue 0.0m/89

