# Training and validation data, after leakage removal

The SAbDab2 antibody–antigen pool restricted to the 533 clusters that survived the homology audit
against FoldBench and PXM-2025.

**Current state, as of 2026-10-12.** Validation is **21 clusters / 28 structures / 54 scored
interfaces** -- the post-cutoff structures only. Every table below that says "50 validation
clusters" describes the ORIGINAL phase B draw, which is still what `results/val_clusters.csv`
contains: 50 clusters chosen without regard to deposition date. 26 of those 50 hold no post-cutoff
structure at all and now contribute nothing, and 3 more hold one that was never folded, so 21 are
live. See "Pending" at the bottom. The exact clusters and structures are in
[`cluster_composition.md`](cluster_composition.md).

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
| validation, phase B draw (`val_jobs/`, `--reps 2`) | 50 | 77 | — | 157 |
| validation, LIVE (post-cutoff only) | **21** | **28** | 100 | **54** |
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

## The validation set

Superseded by **[`validation_set.md`](validation_set.md)**, which is the single file for how the
validation set was chosen and what it contains. In short: validation is now **40 post-cutoff
clusters** (24 all-new, 16 mixed), one named representative each, 100 seeds, folding as dataset
`sabdab_val2` since 2026-10-12. The earlier 21-cluster set remains reproducible under `val_fold` /
`sabdab_val`.

Everything the split implies for the training side is there too: the 493 non-selected clusters,
whole-cluster disjointness, and why a selected cluster's pre-cutoff structures must also stay out of
training.
