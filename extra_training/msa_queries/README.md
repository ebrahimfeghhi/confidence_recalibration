# msa_queries

Written by `phases/phase_c_extract_msa_queries.py`. The deduplicated protein sequences that MSAs were
searched for, before any folding.

| file | what it is |
|---|---|
| `train.fasta` | the training pool's query sequences |
| `bench.fasta` | the benchmarks' query sequences |
| `complexes_all.fasta`, `complexes.csv` | the per-complex chain sequences they were deduplicated from |
| `manifest.csv`, `training_instances.txt` | which SAbDab2 instance each sequence came from |

**Why it matters:** the search these drove produced `msa_work/` (52 GB), and every fold reads those
MSAs. This is the record of exactly which sequences the MSAs cover — without it there is no way to
say what was searched, only what came back.
