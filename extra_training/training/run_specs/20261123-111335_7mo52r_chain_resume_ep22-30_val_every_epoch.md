# RUN SPEC -- 7mo52r chain resume ep22-30 val every epoch

- generated 2026-11-23 11:13 AM PST
- derived from the argv block of `/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad/ep30b_dry4.log` (argv block), not from any banner or summary
- compared against `training/runs/ep30_launch.log` (argv block), shown as **ggd8uv (ep19-21)**

**3 flag(s) differ from ggd8uv (ep19-21).** Everything else is identical.

| flag | ggd8uv (ep19-21) | 7mo52r chain resume ep22-30 val every epoch | |
|---|---|---|---|
| `--out` | `/home/ebrahim/confidence_recalibration/extra_training/training/runs/ggd8uv` | **_not passed_** | CHANGED |
| `--resume` | `training/runs/7mo52r/resume_ep18.pt` | **`/home/ebrahim/confidence_recalibration/extra_training/training/runs/ggd8uv/resume_ep21.pt`** | CHANGED |
| `--train-eval-every` | `2` | **`1`** | CHANGED |

## Full spec -- 58 flags

### Resume & horizon

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--epochs` | `30` | same |
| `--lr` | `1e-3` | same |
| `--resume` | `/home/ebrahim/confidence_recalibration/extra_training/training/runs/ggd8uv/resume_ep21.pt` | **was `training/runs/7mo52r/resume_ep18.pt`** |
| `--seed` | `0` | same |
| `--warmup-steps` | `20` | same |

### Data & crop

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--alignment-preflight` | `training/alignment_preflight_sharded.json` | same |
| `--crop-allow` | `training/crop_final/ost_usable_crops.json` | same |
| `--crop-pool` | `training/crop_final` | same |
| `--epoch-balance` | on | same |
| `--epoch-cover-all` | on | same |
| `--fold` | `training/fold` | same |
| `--indices` | `training/prep/all857_indices.csv` | same |
| `--manifest-selection` | `train_val_splits/cache_train_structures_extended.csv` | same |
| `--manifest-skip` | `4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m` | same |
| `--minimal-items` | on | same |
| `--nsample` | `5` | same |
| `--pae-scored-pairs` | `training/combined/pool.csv` | same |
| `--pae-scored-roles` | `training/fold/chain_roles_all.csv` | same |
| `--require-replay` | `training/extended_mixed_replay.log` | same |

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
| `--expect-trainable` | `113` | same |
| `--frame-direction` | on | same |
| `--lean-readouts` | on | same |
| `--lean-single` | on | same |
| `--pose-perblock` | on | same |
| `--pose-perblock-copy` | on | same |
| `--skip-dead-ops` | on | same |
| `--train-only` | `^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.` | same |

### Regularisation

| flag | value | vs ggd8uv (ep19-21) |
|---|---|---|
| `--clip-norm` | `25.0` | same |
| `--pair-dropout` | `0.25` | same |

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
| `--allow-code-drift` | on | same |
| `--blocks-per-ckpt` | `1` | same |
| `--ckpt-token-threshold` | `1250` | same |
| `--dist-backend` | `gloo` | same |
| `--dist-timeout-min` | `240` | same |
| `--max-rss-gb` | `80` | same |
| `--mem-log` | on | same |
| `--prefetch-units` | `2` | same |
| `--prefetch-workers` | `16` | same |
| `--preload-min-free-gb` | `250` | same |
| `--save-every` | `250` | same |
| `--trunk-memmap` | on | same |

### Passed by the reference and NOT here

| flag | ggd8uv (ep19-21) |
|---|---|
| `--out` | `/home/ebrahim/confidence_recalibration/extra_training/training/runs/ggd8uv` |
