# FoldBench

Free-fold results for the FoldBench antibody–antigen benchmark: 104 complexes, 160 interfaces.

| file | what it is |
|---|---|
| `foldbench_baseline_results.csv` | **the results** — every free pose's DockQ and pair ipTM, 83,368 rows over 160 interfaces, with a `metadata` column naming the run each row came from. |
| `chain_roles.csv` | the interface registry: one row per interface giving its two chains and which is antibody and which antigen; `folder.py` reads this to know what a fold must be scored on. |
| `chain_maps.csv` | per `(pdb, seed, sample)`, how the model's chain letters map onto the reference's — the two are not the same and scoring has to translate. |
| `ost_json` | the raw OpenStructure output behind each DockQ number, one JSON per pose per interface (48,120 files), kept so a disputed score can be re-examined rather than re-folded. |
| `foldbench_baseline_remaining_folds/` | the 612 folds still needed to bring all 104 complexes to 100 seeds, as a `folder.py` queue; the training-cache watcher launches it when its own queue drains. |
| `setup_audit.md` | the one-off audit of how this benchmark's inputs were built. |
| `setup.log` | the log of that setup run. |
| `rebalance.log` | the GPU-allocation watcher's log from the October arm scheduling. |

## What happened to the rest

The per-block directories (`free`, `free_extra10`, `free_top100`, `free_top100b`,
`free_other_seeds`, `free_pool`) and fifteen finished experiment directories were deleted on
2026-10-14 once their scores were folded into `foldbench_baseline_results.csv`. Their poses are gone
and folds do not reproduce. Two exceptions worth knowing:

- `free_other_seeds`' 31,395 rows were **not** carried over — that block sampled only 48 of the 160
  interfaces, and only complexes that had been experimented on before, so it was excluded from the
  baseline rather than pooled into it.
- the result tables of the fifteen experiment directories are preserved at
  `archive/foldbench_retired_experiment_results/`, which is what the write-ups citing them resolve
  against.

`pool.csv`, the published 50-seed baseline, was deleted too: all 39,998 of its rows are in
`foldbench_baseline_results.csv` under `metadata` "pool, seeds 101-150". It had been doing double
duty as the interface registry, which is now `chain_roles.csv`.
