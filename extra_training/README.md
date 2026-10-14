# extra_training — recalibrating Protenix v2's confidence head

Protenix's confidence head never sees, during training, the kind of pose it is asked to score at
inference. Per training example the trunk runs, a 20-step mini-rollout produces one structure under
`no_grad`, and the head predicts its error against that example's own ground truth
(`protenix/model/protenix.py:1675-1740`). AlphaFold 3 is the same. At inference the head scores
200-step samples on complexes it has never seen — including confidently wrong poses the mini-rollout
rarely produces.

This project re-runs Protenix's FT3 stage on **our** pose distribution: 501 SAbDab2 structures × 10
seeds, folded at inference settings, with the trunk tensors cached so the finetune does not refold.
The pool is held out from FoldBench and PXM-2025 at the cluster level — no CDR-H3 or antigen homology
to any benchmark complex.

| folder | what it is |
|---|---|
| `SAbDab2_data/` | the SAbDab2 source download, which everything is selected from |
| `train_val_splits/` | which clusters train, which validate, and what was excluded for homology |
| `building_msa/` | the sequences searched for MSAs, and the MSAs produced — read by every fold |
| `training/` | the training side: jobs, references, labels, and the 5,010-fold cache |
| `validation/` | the held-out 40-cluster set it will be measured on |
| `protenix_assets/` | template database and training indices downloaded from Protenix |
| `phases/` | the pipeline, A→J — one script per stage |
| `shell_scripts/` | the watchers running a fold, plus MSA-database setup |
| `analysis/` | the seed-depth curves |
| `docs/` | the write-ups: methods, validation set, training stages, cached training |

Each folder has its own README. `METHODS.md` is the procedure as implemented; `MSA_RUNBOOK.md`
rebuilds the MSAs on a fresh machine.
