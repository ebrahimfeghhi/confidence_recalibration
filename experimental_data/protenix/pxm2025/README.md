# PXM-2025

Free-fold results for the PXM-2025 antibody–antigen benchmark: 123 complexes, 292 interfaces.

| file | what it is |
|---|---|
| `pxm2025_baseline_results.csv` | **the results** — every free pose's DockQ and pair ipTM, 177,375 rows over 292 interfaces, with a `metadata` column naming the run each row came from. |
| `pxm2025_interfaces.csv` | the interface registry: one row per folded interface with its two chains and which is antibody and which antigen; `folder.py` reads this to know what a fold must be scored on. |
| `dockq_components_pxm2025.csv` | the parts each DockQ score is built from — `fnat`, `fnonnat`, `irmsd`, `lrmsd` per pose. |
| `chain_maps.csv` | per `(pdb, seed, sample)`, how the model's chain letters map onto the reference's — the two are not the same and scoring has to translate. |
| `chain_roles.csv` | every antibody–antigen chain pair the benchmark's construction found, 314 of them, with the Ig-domain evidence for each; a superset of the 292 actually folded. |
| `interfaces.csv` | the raw interface list from the source entries, before the antibody–antigen filter. |
| `epitope_overlap_pxm.csv` | per pose, how much of the native epitope the prediction recovers (`recall`, `jaccard`). |
| `paratope_overlap_pxm.csv` | the same measurement for the paratope. |
| `pxm2025_selection_protocol.csv` | per interface, what each selection rule would have picked and the epitope overlap it would have got. |
| `DEAD_COMPLEXES.txt` | complexes deliberately excluded from the benchmark, one per line with the reason. |
| `ids.txt` | the benchmark's complex ids. |
| `order.txt` | the fold order those ids were queued in. |
| `chain_maps.log` | the log of building `chain_maps.csv`. |
| `epiwatch.log` | the log of the epitope-overlap run. |
| `fold_vs_epitope.png` | DockQ against epitope recall, per pose. |
| `iptm_vs_epitope_worst.png` | pair ipTM against epitope recall for the worst-scoring interfaces. |
| `spread_conf_vs_best.png` | per-interface confidence spread against the best pose available. |
| `cif/` | the reference structures DockQ is scored against. Contents gitignored. |
| `jsons/` | the job files a fold is launched from. Contents gitignored. |
| `jsons_test/` | a two-file sample of the above, for smoke tests. Contents gitignored. |
| `prep/` | the MSA search output the jobs were built from (uniref). Contents gitignored. |
| `pxm_json/` | per-pose pxmeter scoring output, the raw result behind each DockQ number. Contents gitignored. |
| `templates/` | the structure templates available to a fold, with their release dates. Contents gitignored. |
