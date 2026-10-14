# training

The training side of the finetune: what to fold, what to score against, and what came out.

| | |
|---|---|
| `jobs/` | one JSON per structure — the sequences, MSAs and templates a fold is launched from |
| `refs/` | the deposited structures DockQ scores each fold against |
| `prep/` | per-job mmCIFs and bioassemblies — the **labels** the confidence head trains on |
| `fold/` | the fold's output: poses, trunk tensors, `results.csv`, and the queue/claims that drive it |
