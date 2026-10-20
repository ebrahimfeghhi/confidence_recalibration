# Full training set: the four metrics, per epoch

Generated 2026-10-20 11:41 PDT by `run_logging/results_table.py` from `x3k2m4+jhgfxy+2w5to5`, at **budget 30**. Regenerated automatically when an epoch's fourth metric lands. Do not edit.

## The lineage is one experiment in three parts

Epoch numbers are continuous, but the segments do NOT share an implementation, so **throughput and memory are not comparable across them**. The metrics are.

| segment | epochs | configuration |
|---|---|---|
| `x3k2m4` | 1 | checkpointing ON (blocks_per_ckpt=1); per-epoch fingerprint hashing; prefetch_issue double-counted the hash |
| `jhgfxy` | 2 | checkpointing ON; fresh process so hashing re-paid; prefetch choice-cache added |
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
| 1 | `x3k2m4` | yes | 2.7384 | 2.6953 | -0.2803 | 0.2999 | +0.0090 | 0.5556 | +0.0000 |
| 2 | `jhgfxy` | yes | 2.6623 | 2.6779 | -0.2977 | 0.2543 | -0.0366 | 0.3333 | -0.2223 |

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

**Wall, accounted and residual all come from the SAME rank** — the one with the longest wall clock — so accounted + residual equals wall. Taking each column's maximum independently produced a row that did not add up under a heading that said 'worst rank'. Memory stays an across-rank maximum on purpose: there the question is whether ANY rank came near the card.

`ckpt` is the live `blocks_per_ckpt`: `1` recomputes every pairformer block's forward during backward, `None` stores activations instead. The residual is wall clock minus the mutually exclusive buckets; `gpu_forward` and `gpu_backward` are CUDA-event spans that overlap `pose_total` and are excluded from the sum.

### Bucket detail, per rank

**epoch 1** (`x3k2m4`)

- rank 0: crop_slice 12.9m/1620  cropped_tokens mean 334  fingerprint_hash 8.1m/1620  gpu_backward 20.7m/8100  gpu_forward 6.7m/8100  grad_collective_plus_wait 0.6m/290  pose_total 25.0m/8100  prefetch_issue 8.1m/1619  trunk_read_plus_crop_tokens 2.5m/1620
- rank 1: crop_slice 7.8m/1620  cropped_tokens mean 324  fingerprint_hash 7.6m/1620  gpu_backward 19.0m/8100  gpu_forward 6.1m/8100  grad_collective_plus_wait 0.7m/290  pose_total 23.0m/8100  prefetch_issue 7.6m/1619  trunk_read_plus_crop_tokens 2.5m/1620
- rank 2: crop_slice 15.0m/1620  cropped_tokens mean 319  fingerprint_hash 8.5m/1620  gpu_backward 19.1m/8100  gpu_forward 6.1m/8100  grad_collective_plus_wait 0.7m/290  pose_total 23.1m/8100  prefetch_issue 8.6m/1619  trunk_read_plus_crop_tokens 2.6m/1620

**epoch 2** (`jhgfxy`)

- rank 0: crop_slice 10.1m/1620  cropped_tokens mean 325  fingerprint_hash 9.5m/1620  gpu_backward 20.2m/8100  gpu_forward 6.5m/8100  grad_collective_plus_wait 0.7m/290  pose_total 24.5m/8100  prefetch_issue 9.5m/1619  trunk_read_plus_crop_tokens 2.8m/1620
- rank 1: crop_slice 14.3m/1620  cropped_tokens mean 320  fingerprint_hash 9.3m/1620  gpu_backward 19.1m/8100  gpu_forward 6.2m/8100  grad_collective_plus_wait 0.8m/290  pose_total 23.2m/8100  prefetch_issue 9.3m/1619  trunk_read_plus_crop_tokens 2.6m/1620
- rank 2: crop_slice 11.8m/1620  cropped_tokens mean 333  fingerprint_hash 9.2m/1620  gpu_backward 21.9m/8100  gpu_forward 7.0m/8100  grad_collective_plus_wait 0.7m/290  pose_total 26.8m/8100  prefetch_issue 9.2m/1619  trunk_read_plus_crop_tokens 2.7m/1620

