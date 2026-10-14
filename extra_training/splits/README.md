# splits

Which clusters train, which validate, and what was excluded. Written by phases A, B and J0; read by
the later phases rather than recomputed.

| file | what it is |
|---|---|
| `train_clusters.csv` | the 483 training clusters, after leakage removal and the validation split |
| `train_clusters.meta.json` | provenance for the above, including a sha256 of the validation file it was built against |
| `val_split_proposal.csv` | the 40-cluster validation split: one named representative per cluster |
| `val_split_proposal_clusters.csv` | just those 40 cluster ids |
| `val_split_second_reps.csv` | the second representative per cluster, where one exists |
| `excluded_homolog_clusters.csv` | the 34 clusters dropped for homology with the benchmarks, with the matching query and identity |
| `cache_train_structures.csv` | the 501 structures selected for the training cache fold |
| `archive/` | the superseded 50-cluster validation list |
