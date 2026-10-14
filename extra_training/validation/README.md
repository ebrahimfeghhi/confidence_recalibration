# validation

The held-out set the finetune is measured on: 40 post-cutoff SAbDab2 clusters, disjoint from training.

| | |
|---|---|
| `jobs/` | one JSON per structure — what a fold is launched from |
| `refs/` | the deposited structures DockQ scores against |
| `fold/` | the fold's output: poses, trunk tensors, `results.csv` |
