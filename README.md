# confidence_recalibration

Protenix v2's confidence head is trained on a pose distribution it never sees at inference: per
training example a 20-step mini-rollout produces one structure under `no_grad`, and the head predicts
its error against that example's own ground truth. At inference it scores 200-step samples on
complexes it has never seen — including confidently wrong poses the mini-rollout rarely produces.

This repo re-runs Protenix's FT3 stage on that real distribution, and measures whether it helps on two
antibody–antigen benchmarks.

| | |
|---|---|
| `extra_training/` | the finetune: data selection, MSAs, the 5,010-fold training cache, the held-out validation set, and the A→J pipeline that builds them |
| `experimental_data/` | the benchmarks — FoldBench and PXM-2025 — one results table each, plus the inputs needed to score or refold them |
| `run_model/` | the fold worker: `folder.py` claims units from a queue, folds them, and hands each to `scorer.py` |
| `eval/` | DockQ scoring — `ost_dockq.py` (OpenStructure) and `pxm_dockq.py` (pxmeter) |
| `tools/`, `envs/` | the downloaded binaries and virtualenvs the MSA pipeline shells out to |
| `paths.py` | every location in one place; code reaches data through `paths.DATASETS`, never a literal |
| `archive/` | superseded code and data, kept for provenance |

Each folder has its own README. Start with `extra_training/README.md`.
