# Training and validation data, after leakage removal

The SAbDab2 antibody–antigen pool restricted to the 533 clusters that survived the homology audit
against FoldBench and PXM-2025.

**Current state, 2026-10-13.** Validation is **40 clusters / 53 structures / 91 scored interfaces at
100 seeds**, folded and verified as dataset `sabdab_val2`. How those clusters were chosen, and the
result, is in [`validation_set.md`](validation_set.md) -- the canonical file. Per-cluster era
composition for the whole pool is in [`cluster_composition.md`](cluster_composition.md).

This file describes the POOL the split was drawn from, not the split. Two earlier validation sets
appear below for context and both are superseded: phase B's draw of 50 clusters (chosen without
regard to deposition date, still what `train_val_splits/val_clusters.csv` contains) and the 21-cluster
post-cutoff subset of it that was folded on 2026-10-12. The 21-cluster fold remains reproducible
under `val2_fold` / `sabdab_val2`.

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

The validation rows are the phase B draw of 50 clusters, NOT the live validation set. 21 of those
clusters are live; see the current-state note above.

Train and validation clusters are disjoint, and neither intersects the exclusion list.

## What is actually folded, versus what is available

The tables above are the pool. The fold queues take one structure per cluster, because folding every
instance at 20 seeds is about 17 days on two GPUs against 2.3 days for one per cluster.

| | clusters | structures folded | seeds | interfaces |
|---|---|---|---|---|
| **validation, LIVE (`val2_jobs/`)** | **40** | **53** | **100** | **91** |
| validation, superseded: the 21-cluster subset (`val_jobs/`) | 21 | 28 | 30 | 54 |
| validation, superseded: phase B's draw (`val_jobs/`, `--reps 2`) | 50 | 77 | 20-30 | 157 |
| training (`train_jobs/`, `--reps 1`) | 483 | 483 | not folded yet | 922 |

Of the 77 folded validation structures, 49 are pre-cutoff and 28 post. Of the 483 training
structures, 314 are pre-cutoff and 169 post.

So validation depth is a sampling choice, not a data limit. The live set takes 53 structures from
40 clusters; 13 of those clusters contribute a second representative and the other 27 have only one
eligible post-cutoff structure in the PDB.

## Provenance

| | |
|---|---|
| pool table | `data/zenodo_splits/splits_final/abag_split.csv` (15,810 instances) |
| cluster lists | `train_val_splits/train_clusters.csv`, `train_val_splits/val_clusters.csv`, `train_val_splits/excluded_homolog_clusters.csv` |
| fold queues | `train_jobs/manifest.csv`, `val_jobs/manifest.csv` (both carry per-structure `deposited` and `post_cutoff`) |
| how the clusters were chosen | `phases/README.md`, `phases/phase_a_define_train_set.py`, `phases/phase_b_define_val_clusters.py` |

## The validation set

See **[`validation_set.md`](validation_set.md)**, the single file for how it was chosen and what it
contains: 40 clusters (24 all-new, 16 mixed), 53 structures, 100 seeds, dataset `sabdab_val2`, folded
and verified 2026-10-13. Result, cluster-averaged then weighted equally across clusters:

| arm | clusters | decoy | acceptable | no acceptable | coverage | regret |
|---|---|---|---|---|---|---|
| all new | 24 | 21% | 59% | 20% | 80% | 0.165 |
| mixed | 16 | 11% | 83% | 6% | 94% | 0.116 |
| all 40 | 40 | 17% | 69% | 15% | 85% | 0.145 |

Everything the split implies for the training side is there too: the 493 non-selected clusters,
whole-cluster disjointness, and why a selected cluster's pre-cutoff structures must also stay out of
training.
