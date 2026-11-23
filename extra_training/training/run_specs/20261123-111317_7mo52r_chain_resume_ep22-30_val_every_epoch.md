# RUN SPEC -- 7mo52r chain resume ep22-30 val every epoch

- generated 2026-11-23 11:13 AM PST
- derived from the argv block of `/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad/ep30b_dry4.log` (argv block), not from any banner or summary
- compared against `training/runs/ep30_launch.log` (argv block), shown as **ggd8uv (ep19-21)**

**29 flag(s) differ from ggd8uv (ep19-21).** Everything else is identical.

| flag | ggd8uv (ep19-21) | 7mo52r chain resume ep22-30 val every epoch | |
|---|---|---|---|
| `--alignment-preflight` | _not passed_ | **`training/alignment_preflight_sharded.json`** | CHANGED |
| `--allow-code-drift` | _not passed_ | **on** | CHANGED |
| `--blocks-per-ckpt` | _not passed_ | **`1`** | CHANGED |
| `--ckpt-token-threshold` | _not passed_ | **`1250`** | CHANGED |
| `--clip-norm` | _not passed_ | **`25.0`** | CHANGED |
| `--dist-backend` | _not passed_ | **`gloo`** | CHANGED |
| `--dist-timeout-min` | _not passed_ | **`240`** | CHANGED |
| `--expect-trainable` | _not passed_ | **`113`** | CHANGED |
| `--fold` | _not passed_ | **`training/fold`** | CHANGED |
| `--indices` | _not passed_ | **`training/prep/all857_indices.csv`** | CHANGED |
| `--lean-readouts` | _not passed_ | **on** | CHANGED |
| `--manifest-selection` | _not passed_ | **`train_val_splits/cache_train_structures_extended.csv`** | CHANGED |
| `--manifest-skip` | _not passed_ | **`4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m`** | CHANGED |
| `--max-rss-gb` | _not passed_ | **`80`** | CHANGED |
| `--mem-log` | _not passed_ | **on** | CHANGED |
| `--pae-scored-pairs` | _not passed_ | **`training/combined/pool.csv`** | CHANGED |
| `--pae-scored-roles` | _not passed_ | **`training/fold/chain_roles_all.csv`** | CHANGED |
| `--pair-dropout` | _not passed_ | **`0.25`** | CHANGED |
| `--prefetch-units` | _not passed_ | **`2`** | CHANGED |
| `--prefetch-workers` | _not passed_ | **`16`** | CHANGED |
| `--preload-min-free-gb` | _not passed_ | **`250`** | CHANGED |
| `--require-replay` | _not passed_ | **`training/extended_mixed_replay.log`** | CHANGED |
| `--resume` | _not passed_ | **`/home/ebrahim/confidence_recalibration/extra_training/training/runs/ggd8uv/resume_ep21.pt`** | CHANGED |
| `--save-every` | _not passed_ | **`250`** | CHANGED |
| `--seed` | _not passed_ | **`0`** | CHANGED |
| `--train-eval-every` | `2` | **`1`** | CHANGED |
| `--train-only` | _not passed_ | **`^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.`** | CHANGED |
| `--trunk-memmap` | _not passed_ | **on** | CHANGED |
| `--warmup-steps` | _not passed_ | **`20`** | CHANGED |

## Full spec -- 58 flags

### Resume & horizon

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--epochs` | `30` | same |
| `--lr` | `1e-3` | same |
| `--resume` | `/home/ebrahim/confidence_recalibration/extra_training/training/runs/ggd8uv/resume_ep21.pt` | **was _not passed_** |
| `--seed` | `0` | **was _not passed_** |
| `--warmup-steps` | `20` | **was _not passed_** |

### Data & crop

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--alignment-preflight` | `training/alignment_preflight_sharded.json` | **was _not passed_** |
| `--crop-allow` | `training/crop_final/ost_usable_crops.json` | same |
| `--crop-pool` | `training/crop_final` | same |
| `--epoch-balance` | on | same |
| `--epoch-cover-all` | on | same |
| `--fold` | `training/fold` | **was _not passed_** |
| `--indices` | `training/prep/all857_indices.csv` | **was _not passed_** |
| `--manifest-selection` | `train_val_splits/cache_train_structures_extended.csv` | **was _not passed_** |
| `--manifest-skip` | `4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m` | **was _not passed_** |
| `--minimal-items` | on | same |
| `--nsample` | `5` | same |
| `--pae-scored-pairs` | `training/combined/pool.csv` | **was _not passed_** |
| `--pae-scored-roles` | `training/fold/chain_roles_all.csv` | **was _not passed_** |
| `--require-replay` | `training/extended_mixed_replay.log` | **was _not passed_** |

### Objective

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--alpha-confidence` | `1.0` | same |
| `--alpha-pae` | `1.0` | same |
| `--alpha-rank` | `59` | same |
| `--pae-only` | on | same |
| `--rank-cross-seed` | on | same |
| `--rank-form` | `rel` | same |
| `--rank-labels` | `training/fold/results.csv` | same |
| `--rank-loss` | on | same |
| `--rank-pool` | `mean` | same |
| `--rank-structs-per-step` | `3` | same |
| `--rel-labels` | `training/fold/itm_labels_crop` | same |

### Architecture

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--expect-trainable` | `113` | **was _not passed_** |
| `--frame-direction` | on | same |
| `--lean-readouts` | on | **was _not passed_** |
| `--lean-single` | on | same |
| `--pose-perblock` | on | same |
| `--pose-perblock-copy` | on | same |
| `--skip-dead-ops` | on | same |
| `--train-only` | `^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.` | **was _not passed_** |

### Regularisation

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--clip-norm` | `25.0` | **was _not passed_** |
| `--pair-dropout` | `0.25` | **was _not passed_** |

### Evaluation

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--train-eval-budget` | `30` | same |
| `--train-eval-every` | `1` | **was `2`** |
| `--train-eval-fold` | `validation/fold` | same |
| `--train-eval-pool` | `training/valfull83/pool.csv` | same |
| `--train-eval-seed-base` | `400000` | same |

### Runtime

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--accum` | `240` | same |
| `--allow-code-drift` | on | **was _not passed_** |
| `--blocks-per-ckpt` | `1` | **was _not passed_** |
| `--ckpt-token-threshold` | `1250` | **was _not passed_** |
| `--dist-backend` | `gloo` | **was _not passed_** |
| `--dist-timeout-min` | `240` | **was _not passed_** |
| `--max-rss-gb` | `80` | **was _not passed_** |
| `--mem-log` | on | **was _not passed_** |
| `--prefetch-units` | `2` | **was _not passed_** |
| `--prefetch-workers` | `16` | **was _not passed_** |
| `--preload-min-free-gb` | `250` | **was _not passed_** |
| `--save-every` | `250` | **was _not passed_** |
| `--trunk-memmap` | on | **was _not passed_** |
