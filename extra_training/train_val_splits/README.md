# train_val_splits

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
| `refold_internal_omissions.csv` | **the refold list, training**: 126 structures whose folded query omits an internal native stretch, one row each with the runs and the exact sequence to restore |
| `refold_internal_omissions_validation.csv` | the same for validation: 16 of 53 structures |
| `excluded_interdomain_linkers.csv` | runs deliberately NOT restored, training: 4 single-chain-construct tethers (`GGGGSGGGGSGGGGS` and the like) that belong to neither variable domain |
| `excluded_interdomain_linkers_validation.csv` | the same for validation: 1 |
| `archive/` | the superseded 50-cluster validation list |

The four refold/linker files are written by `phases/phase_j_refold_list.py` from
`phases/phase_j_job_cifs.py --native-gaps`, so they can be regenerated rather than trusted:

    python phases/phase_j_job_cifs.py --native-gaps <gaps.csv>            # per-run report
    python phases/phase_j_refold_list.py --gaps <gaps.csv> --out ... \
        --linkers ... --seeds 10 --mb-per-unit 280                        # the list + its cost

`--seeds` is 10 for training and 100 for validation; `--mb-per-unit` is the measured trunk size,
280 MB for training folds and 136 MB for validation ones.
