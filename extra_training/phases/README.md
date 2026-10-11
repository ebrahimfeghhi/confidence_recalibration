# phases/ — one script per phase of the confidence-head finetune

Each file is a phase of the pipeline, named for what it does. Run them in order; each states its
inputs, outputs and gates in its own docstring. The plain-language account of what every phase is
*for* is in `extra_training/README.md`, under **Phase-by-phase walkthrough**.

| phase | script | writes |
|---|---|---|
| A | `phase_a_define_train_set.py` | `results/excluded_homolog_clusters.csv`; exports `training_clusters()` |
| B | `phase_b_define_val_clusters.py` | `results/val_clusters.csv` |
| C | `phase_c_extract_msa_queries.py` | `msa_work/query.fasta`, `msa_work/query_short.fasta`, `msa_queries/*` |

Phases D through I (UniRef search, env pass, taxids and pairing, job JSONs, templates, folding) still
live in `extra_training/scripts/` and `diffusion_commit_clean_v2/`. Moving them is pending.

## Why A exports the training set

`training_clusters()` is defined in phase A and imported by B, C and phase G (`build_val_jobs.py`).
Phase A *is* the definition of the training set, so that is where the derivation belongs. The
alternative — each phase re-deriving it, or reading a cached cluster list — gives either four copies
that can drift apart or a file that goes stale when the exclusion changes. Neither is acceptable in a
pipeline whose cluster count appears in a paper.

## Why `BENCHMARKS` is written out by hand

`paths.DATASET_NAMES` is `('pxm2025', 'foldbench', 'sabdab_val')`. `sabdab_val` is *derived* from this
pool, and it has a readable `pool.csv`, so iterating that tuple fails silently rather than loudly: in
phase A it would exclude the validation clusters from the pool they are drawn from, and in phase C it
would file the 79 validation structures as "benchmark" sequences. Both phases name the two evaluation
sets explicitly instead.

## Verification

Every phase was re-run after the move and its output diffed against the artifact the pre-move code
produced. All byte-identical:

| artifact | result |
|---|---|
| `results/excluded_homolog_clusters.csv` | identical |
| `results/val_clusters.csv` | identical |
| `msa_queries/{train,bench}.fasta`, `manifest.csv`, `complexes{_all.fasta,.csv}` | identical |
| `msa_work/query.fasta` | identical to the `cat`-produced original |
| `msa_work/query_short.fasta` | identical |

`msa_work/query.fasta` used to be produced by a bare `cat train.fasta bench.fasta` at a shell, which
left a step of the method with no code behind it. Phase C now writes it, and the check above is that
it writes the same bytes.
