# One-off analysis, 2026-10-20

Scripts behind the numbers reported while the lr 1e-2 anchor (`335psh` + `2u8tf4`) was training.
Each is here so a figure can be traced back to the code that produced it. They are NOT part of any
pipeline; the durable changes they depend on live in `phases/`.

Run them from `extra_training/`.

| script | what it produced |
|---|---|
| `01_build_val10_labels.sh` | `validation/fold/labels/` -- 10 seeds x 10 val10 structures = 100 label sets, 18 MB. Validation had no PAE labels at all, which is why held-out loss had never been measured. |
| `02_val10_alignment_preflight.sh` | `training/val10/alignment_preflight.json` -- 300 units, 10 structures, 0 mismatches, 1.1 min. Required because `--crop-margin` needs `--minimal-items`, and without the same crop the val loss is not comparable to the training loss. |
| `03_val10_loss_only.sh` | val10 pose-weighted PAE for a given bundle, via the trainer's `--loss-only`. ep1 = 2.8290 over 500 poses. |
| `04_train_selection_ep142.sh` | `training/overfit30/metric_train_ep142.json` -- mean DockQ and frac-acceptable of the ep142 head on the 30 TRAINING structures, budget 10. |
| `05_val10_rank_correlation.py` | the per-interface Spearman(ipTM, DockQ) table: trained 0.103 vs shipped 0.199 across 18 interfaces, 9 better / 9 worse, with 9qrv +0.495 and 9bx7 -0.415 on selected DockQ nearly cancelling. |

## Durable changes these rely on

- `phases/phase_k_train.py` -- `--loss-only`: forward and loss, no backward, no optimiser, one pass.
  Lives in the trainer rather than a separate script so the crop, the `--pae-scored-pairs` mask, the
  `--pae-only` stub and the loss weights are the same code; a reimplementation would drift and the
  training and held-out numbers would stop being comparable.
- `phases/phase_k_eval.py` -- the shipped-head baseline is derived from the pool actually scored
  instead of a hardcoded 0.5775 / 0.7283 (the full pool's numbers, printed on every val10 log).
- `phases/phase_k_alignment_preflight.py` -- refuses to write a marker over an empty manifest. It
  had printed `PREFLIGHT PASSED: 0 structure(s)` with the sha256 of the empty string, because a
  selection csv without a `tokens` column drops every row silently.
