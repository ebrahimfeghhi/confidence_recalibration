# Training and validation data, after leakage removal

The SAbDab2 antibody–antigen pool restricted to the 533 clusters that survived the homology audit
against FoldBench and PXM-2025, split into the 483 training and 50 validation clusters.

| cutoff | clusters | structures | instances |
|---|---|---|---|
| pre-cutoff | 392 | 1,860 | 3,166 |
| post-cutoff | 248 | 1,757 | 2,817 |
| **total (distinct)** | **533** | **3,617** | **5,983** |

- **clusters** — distinct `ab_ag_cluster`. The era rows sum past 533 because 107 clusters hold
  structures from both eras, so a cluster is counted in both.
- **structures** — distinct `PDB_ID`.
- **instances** — rows of `abag_split.csv`: one antibody–antigen pairing. A single PDB entry can
  contribute several.
- **cutoff** — `PDBdepo` against 2021-09-30, Protenix v2's wwPDB training cutoff. Post-cutoff
  structures cannot have been in its training pool; pre-cutoff ones are all present in the index
  Protenix ships (`indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz`), though that index is
  the pool it weight-samples from, not a log of what was drawn.

## By split

| set | cutoff | clusters | structures | instances |
|---|---|---|---|---|
| training | pre | 358 | 1,538 | 2,582 |
| training | post | 224 | 1,594 | 2,453 |
| training | **total** | 483 | 3,132 | 5,035 |
| validation | pre | 34 | 324 | 584 |
| validation | post | 24 | 163 | 364 |
| validation | **total** | 50 | 487 | 948 |

Train and validation clusters are disjoint, and neither intersects the exclusion list.

## What is actually folded, versus what is available

The tables above are the pool. The fold queues take one structure per cluster, because folding every
instance at 20 seeds is about 17 days on two GPUs against 2.3 days for one per cluster.

| | clusters | structures folded | seeds | interfaces |
|---|---|---|---|---|
| validation (`val_jobs/`, `--reps 2`) | 50 | 77 | 30 | 157 |
| training (`train_jobs/`, `--reps 1`) | 483 | 483 | not folded yet | 922 |

Of the 77 folded validation structures, 49 are pre-cutoff and 28 post. Of the 483 training
structures, 314 are pre-cutoff and 169 post.

So the post-cutoff validation depth is a sampling choice, not a data limit: 163 post-cutoff
structures exist in the validation clusters and 28 are folded.

## Provenance

| | |
|---|---|
| pool table | `data/zenodo_splits/splits_final/abag_split.csv` (15,810 instances) |
| cluster lists | `results/train_clusters.csv`, `results/val_clusters.csv`, `results/excluded_homolog_clusters.csv` |
| fold queues | `train_jobs/manifest.csv`, `val_jobs/manifest.csv` (both carry per-structure `deposited` and `post_cutoff`) |
| how the clusters were chosen | `phases/README.md`, `phases/phase_a_define_train_set.py`, `phases/phase_b_define_val_clusters.py` |
