# Full training set: the four metrics, per epoch

Generated 2026-10-22 09:24 PDT by `run_logging/results_table.py` from `wpg274`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

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
| 1 | `wpg274` | yes | 2.3764 | 2.7463 | -0.2293 | 0.2687 | -0.0222 | 0.4444 | -0.1112 |
| 2 | `wpg274` | yes | 2.2663 | 2.7456 | -0.2300 | 0.2774 | -0.0135 | 0.5000 | -0.0556 |
| 3 | `wpg274` | yes | 2.2435 | 2.7115 | -0.2641 | 0.2444 | -0.0465 | 0.5000 | -0.0556 |
| 4 | `wpg274` | yes | 2.1821 | 2.7519 | -0.2237 | 0.2857 | -0.0052 | 0.5556 | +0.0000 |
| 5 | `wpg274` | yes | 2.2065 | 2.7442 | -0.2314 | 0.2734 | -0.0175 | 0.5000 | -0.0556 |
| 6 | `wpg274` | yes | 2.1736 | 2.7618 | -0.2138 | 0.2880 | -0.0029 | 0.5556 | +0.0000 |
| 7 | `wpg274` | yes | 2.1424 | 2.8005 | -0.1751 | 0.2725 | -0.0184 | 0.5000 | -0.0556 |
| 8 | `wpg274` | yes | 2.1485 | 2.7692 | -0.2064 | 0.2546 | -0.0363 | 0.5556 | +0.0000 |
| 9 | `wpg274` | yes | 2.1295 | 2.7371 | -0.2385 | 0.2711 | -0.0198 | 0.5000 | -0.0556 |
| 10 | `wpg274` | yes | 2.1450 | 2.7307 | -0.2449 | 0.2621 | -0.0288 | 0.4444 | -0.1112 |
| 11 | `wpg274` | yes | 2.1422 | 2.7235 | -0.2521 | 0.2536 | -0.0373 | 0.5000 | -0.0556 |
| 12 | `wpg274` | yes | 2.1018 | 2.7066 | -0.2690 | 0.2844 | -0.0065 | 0.6111 | +0.0555 |
| 13 | `wpg274` | yes | 2.1238 | 2.7161 | -0.2595 | 0.2645 | -0.0264 | 0.6111 | +0.0555 |
| 14 | `wpg274` | yes | 2.1070 | 2.6952 | -0.2804 | 0.2404 | -0.0505 | 0.5556 | +0.0000 |
| 15 | `wpg274` | yes | 2.0813 | 2.7178 | -0.2578 | 0.2364 | -0.0545 | 0.5556 | +0.0000 |
| 16 | `wpg274` | yes | 2.0381 | 2.7222 | -0.2534 | 0.2889 | -0.0020 | 0.5556 | +0.0000 |
| 17 | `wpg274` | yes | 2.0522 | 2.7110 | -0.2646 | 0.2484 | -0.0425 | 0.5000 | -0.0556 |
| 18 | `wpg274` | yes | 2.0489 | 2.7189 | -0.2567 | 0.2367 | -0.0542 | 0.5556 | +0.0000 |
| 19 | `wpg274` | yes | 2.0455 | 2.7717 | -0.2039 | 0.2167 | -0.0742 | 0.5556 | +0.0000 |
| 20 | `wpg274` | yes | 2.0050 | 2.7625 | -0.2131 | 0.2167 | -0.0742 | 0.5000 | -0.0556 |
| 21 | `wpg274` | yes | 1.9867 | 2.7318 | -0.2438 | 0.1999 | -0.0910 | 0.3333 | -0.2223 |
| 22 | `wpg274` | yes | 2.0441 | 2.6912 | -0.2844 | 0.2473 | -0.0436 | 0.5556 | +0.0000 |
| 23 | `wpg274` | yes | 1.9874 | 2.7077 | -0.2679 | 0.2266 | -0.0643 | 0.5000 | -0.0556 |
| 24 | `wpg274` | yes | 1.9697 | 2.7135 | -0.2621 | 0.2216 | -0.0693 | 0.5000 | -0.0556 |
| 25 | `wpg274` | yes | 1.9661 | 2.8004 | -0.1752 | 0.2171 | -0.0738 | 0.3889 | -0.1667 |
| 26 | `wpg274` | yes | 1.9589 | 2.7763 | -0.1993 | 0.2503 | -0.0406 | 0.5000 | -0.0556 |
| 27 | `wpg274` | yes | 1.9449 | 2.7935 | -0.1821 | 0.2270 | -0.0639 | 0.5000 | -0.0556 |
| 28 | `wpg274` | yes | 1.9596 | 2.8112 | -0.1644 | 0.2120 | -0.0789 | 0.5000 | -0.0556 |
| 29 | `wpg274` | yes | 1.9294 | 2.9149 | -0.0607 | 0.2651 | -0.0258 | 0.5556 | +0.0000 |
| 30 | `wpg274` | yes | 1.9340 | 2.9060 | -0.0696 | 0.1970 | -0.0939 | 0.3333 | -0.2223 |
| 31 | `wpg274` | yes | 1.9296 | 2.8267 | -0.1489 | 0.2315 | -0.0594 | 0.3889 | -0.1667 |
| 32 | `wpg274` | yes | 1.9174 | 2.9005 | -0.0751 | 0.2323 | -0.0586 | 0.5000 | -0.0556 |
| 33 | `wpg274` | **NO** | 1.8980 | 2.8174 | -0.1582 | — | — | — | — |

Held-out PAE is a loss, so **lower is better** and a negative delta is an improvement. DockQ and frac-acceptable are the opposite.

`complete` means all four metrics are present AND the held-out pass covered its expected 1500 poses. An epoch marked **NO** is mid-boundary or lost a metric to a failure — its em-dashes are missing measurements, not zeros.

## Why held-out PAE is the metric to read

On the 30-structure arm, training PAE fell 38% from epoch 1 to 142 while held-out PAE ROSE 51% (2.8290 → 4.2639 against 2.9756 untrained), and the DockQ metrics showed nothing the entire time because ipTM barely orders poses within an interface — Spearman 0.20 for the shipped head, 0.10 for the trained one. A falling train-PAE curve is not evidence a run is working.

**But held-out PAE is necessary, not sufficient.** It is a loss on the statistic ipTM derives from, not a measure of ranking. DockQ carries about ±0.05 on 18 interfaces and frac-acceptable only moves in steps of 1/18 = 0.0556, so single-epoch moves in those columns are noise — which cuts both ways: epoch 2's fall from 10/18 to 6/18 acceptable is as uninterpretable alone as a rise would be. A paired per-interface ranking analysis (`analysis/oneoff/05_val10_rank_correlation.py`) is the thing that would settle whether selection improved, and it needs several epochs of checkpoints to be worth running.

## Per-epoch wall clock and memory

| epoch | seg | slowest rank | wall (min) | accounted | residual | peak alloc (GB) | peak reserved (GB) | ckpt |
|---|---|---|---|---|---|---|---|---|
| 1 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.71 | None |
| 2 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |
| 3 | `wpg274` | 2 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |
| 4 | `wpg274` | 2 | 1.7 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 5 | `wpg274` | 2 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 6 | `wpg274` | 2 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 7 | `wpg274` | 2 | 1.9 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 8 | `wpg274` | 2 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 9 | `wpg274` | 2 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 10 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |
| 11 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |
| 12 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 13 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |
| 14 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 15 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 16 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 17 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |
| 18 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |
| 19 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 20 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 21 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 22 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 23 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 24 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 25 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 26 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 27 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 28 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 29 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 30 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 31 | `wpg274` | 1 | 1.8 | 1.7m | 0.0m | 44.89 | 45.42 | None |
| 32 | `wpg274` | 1 | 1.8 | 1.7m | 0.1m | 44.89 | 45.42 | None |
| 33 | `wpg274` | 1 | 1.8 | 1.8m | 0.0m | 44.89 | 45.42 | None |

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 31** (`wpg274`)

- rank 0: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89

**epoch 32** (`wpg274`)

- rank 0: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.0m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.1m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89

**epoch 33** (`wpg274`)

- rank 0: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 1: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89
- rank 2: apply_step_total 0.1m/15  crop_slice 0.2m/90  cropped_tokens mean 407  gpu_backward 1.2m/450  gpu_forward 0.5m/450  grad_collective_plus_wait 0.0m/15  label_open 0.0m/90  loss_feature_prep 0.0m/90  loss_stats_collective_plus_wait 0.0m/15  pose_total 1.5m/450  prefetch_issue 0.0m/89

