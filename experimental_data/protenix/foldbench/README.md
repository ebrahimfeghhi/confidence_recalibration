# FoldBench

Free-fold results for the FoldBench antibody–antigen benchmark: 104 complexes, 160 interfaces.

**Every interface now carries at least 100 seeds** (min 100, max 150). 28 of the 160 sat at 60–93
until the top-off queue was merged into the baseline table on 2026-10-20; anything that reads only
the pre-merge table under-counts those 28.

| file | what it is |
|---|---|
| `foldbench_baseline_results.csv` | **the results** — every free pose's DockQ and pair ipTM, 87,918 rows over 160 interfaces, with a `metadata` column naming the run each row came from. |
| `chain_roles.csv` | the interface registry: one row per interface giving its two chains and which is antibody and which antigen; `folder.py` reads this to know what a fold must be scored on. |
| `chain_maps.csv` | per `(pdb, seed, sample)`, how the model's chain letters map onto the reference's — the two are not the same and scoring has to translate. |
| `ost_json` | the raw OpenStructure output behind each DockQ number, one JSON per pose per interface (48,120 files), kept so a disputed score can be re-examined rather than re-folded. |
| `foldbench_baseline_remaining_folds/` | **done, and merged in.** It was the queue that brought the 28 short interfaces to 100 seeds; its 4,550 scored poses are now rows of the baseline table above, tagged `metadata=foldbench_baseline_remaining_folds`, and its own `results.csv` is kept as the record of that fold. |
