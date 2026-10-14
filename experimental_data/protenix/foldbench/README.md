# FoldBench

Free-fold results for the FoldBench antibody–antigen benchmark: 104 complexes, 160 interfaces.

| file | what it is |
|---|---|
| `foldbench_baseline_results.csv` | **the results** — every free pose's DockQ and pair ipTM, 83,368 rows over 160 interfaces, with a `metadata` column naming the run each row came from. |
| `chain_roles.csv` | the interface registry: one row per interface giving its two chains and which is antibody and which antigen; `folder.py` reads this to know what a fold must be scored on. |
| `chain_maps.csv` | per `(pdb, seed, sample)`, how the model's chain letters map onto the reference's — the two are not the same and scoring has to translate. |
| `ost_json` | the raw OpenStructure output behind each DockQ number, one JSON per pose per interface (48,120 files), kept so a disputed score can be re-examined rather than re-folded. |
| `foldbench_baseline_remaining_folds/` | the 612 folds still needed to bring all 104 complexes to 100 seeds, as a `folder.py` queue; the training-cache watcher launches it when its own queue drains. |
