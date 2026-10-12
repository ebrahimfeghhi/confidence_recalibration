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

## Pending: the post-cutoff-only validation split (decided 2026-10-12, partly unexecuted)

Validation was restricted to post-cutoff structures, because a pre-cutoff structure is in the pool
Protenix sampled from and cannot test generalisation. The 49 pre-cutoff structures are recorded in
`val_fold/skipped.csv` with their deposition dates, and their trunk caches were deleted (240 GB);
their poses and scores remain.

**Currently folding:** 28 structures over 21 clusters, 55 interfaces, to 100 seeds.

Two changes follow from the decision and are NOT yet done:

1. **Add three clusters to validation.** 24 of the 50 validation clusters hold a post-cutoff
   structure; we folded one for only 21. Clusters **47, 371 and 766** hold 46, 6 and 54 post-cutoff
   structures respectively, and phase G happened to pick pre-cutoff representatives. Adding one
   structure each gives 24 clusters / 31 structures — every eligible cluster covered.
   Their MSAs already exist (candidates with MSAs: `7ugq 9bio 9mic 8say` for 47, `9djw 8z8m 9bn7`
   for 371, `7t0w 7t0z 9fg3 9ffy` for 766), so only phase G and phase H are needed, then ~3 GPU-h.
   **Deferred for disk**: the running job lands near 1,348 GB of 1,536 (88%), leaving 34 GB under the
   90% stop threshold. Do it after freeing space. Prefer the SMALLEST post-cutoff structure in each
   cluster — the trunk cache scales as N², so the representative's size dominates the cost.

2. **Move 26 clusters from validation to training.** 26 of the 50 validation clusters contain no
   post-cutoff structure anywhere in the pool, so they can never contribute to a post-cutoff
   validation set; held out, they are only lost training data. Moving them gives **509 training
   clusters / 24 validation clusters**, still 533, with zero train/val overlap and zero intersection
   with the 34 exclusions.

   **Whole clusters only.** The pre-cutoff structures of the 24 clusters that STAY in validation must
   not go into training: training on a 2015 structure of a system validated on its 2023 structure is
   cluster-level leakage, which is what the 533-cluster audit exists to prevent. Of the 49 skipped
   structures, the 20 belonging to the 26 moving clusters could become training poses; the other 29
   can serve neither purpose.
