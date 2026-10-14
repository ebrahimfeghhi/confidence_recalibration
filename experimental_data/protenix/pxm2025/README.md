# PXM-2025

Free-fold results for the PXM-2025 antibody–antigen benchmark — 123 complexes, 292 interfaces — and
the inputs needed to produce and score them. Analysis built on top of these results lives in
`archive/pxm2025_analysis/`.

| file | what it is |
|---|---|
| `pxm2025_baseline_results.csv` | **the results** — every free pose's DockQ and pair ipTM, 177,375 rows over 292 interfaces, with a `metadata` column naming the run each row came from. |
| `dockq_components_pxm2025.csv` | the parts each DockQ score is built from — `fnat`, `fnonnat`, `irmsd`, `lrmsd` per pose. |
| `pxm2025_interfaces.csv` | the interface registry: one row per folded interface with its two chains and which is antibody and which antigen; `folder.py` reads this to know what a fold must be scored on. |
| `chain_maps.csv` | per `(pdb, seed, sample)`, how the model's chain letters map onto the reference's — the two are not the same and scoring has to translate. |
| `chain_roles.csv` | every antibody–antigen chain pair the benchmark's construction found, 314 of them, with the Ig-domain evidence for each; a superset of the 292 actually folded. |
| `cif/` | the reference structures DockQ is scored against. Contents gitignored. |
| `jsons/` | the job files a fold is launched from. Contents gitignored. |
| `prep/` | the MSA search output those jobs were built from. Contents gitignored. |
| `templates/` | the structure templates available to a fold, with their release dates. Contents gitignored. |
| `pxm_json/` | per-pose pxmeter output, the raw scoring result behind each DockQ number. Contents gitignored. |
