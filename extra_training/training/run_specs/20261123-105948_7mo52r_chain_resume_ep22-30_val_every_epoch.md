# RUN SPEC -- 7mo52r chain resume ep22-30, val every epoch

- generated 2026-11-23 10:59 AM PST
- derived from the argv block of `/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad/ep30b_dry3.log` (argv block), not from any banner or summary
- compared against `training/runs/ep30_launch.log` (argv block), shown as **ggd8uv (ep19-21)**

**1 flag(s) differ from ggd8uv (ep19-21).** Everything else is identical.

| flag | ggd8uv (ep19-21) | 7mo52r chain resume ep22-30, val every epoch | |
|---|---|---|---|
| `--train-eval-every` | `2` | **`1`** | CHANGED |

## Full spec (30 flags)

| flag | value | ggd8uv (ep19-21) |
|---|---|---|
| `--accum` | `240` | same |
| `--alpha-confidence` | `1.0` | same |
| `--alpha-pae` | `1.0` | same |
| `--alpha-rank` | `59` | same |
| `--crop-allow` | `training/crop_final/ost_usable_crops.json` | same |
| `--crop-pool` | `training/crop_final` | same |
| `--epoch-balance` | on | same |
| `--epoch-cover-all` | on | same |
| `--epochs` | `30` | same |
| `--frame-direction` | on | same |
| `--lean-single` | on | same |
| `--lr` | `1e-3` | same |
| `--minimal-items` | on | same |
| `--nsample` | `5` | same |
| `--pae-only` | on | same |
| `--pose-perblock` | on | same |
| `--pose-perblock-copy` | on | same |
| `--rank-cross-seed` | on | same |
| `--rank-form` | `rel` | same |
| `--rank-labels` | `training/fold/results.csv` | same |
| `--rank-loss` | on | same |
| `--rank-pool` | `mean` | same |
| `--rank-structs-per-step` | `3` | same |
| `--rel-labels` | `training/fold/itm_labels_crop` | same |
| `--skip-dead-ops` | on | same |
| `--train-eval-budget` | `30` | same |
| `--train-eval-every` | `1` | **`2`** |
| `--train-eval-fold` | `validation/fold` | same |
| `--train-eval-pool` | `training/valfull83/pool.csv` | same |
| `--train-eval-seed-base` | `400000` | same |
