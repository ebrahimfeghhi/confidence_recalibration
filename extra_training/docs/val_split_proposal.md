# Proposed 40-cluster post-cutoff validation split

Status: **PROPOSAL, not applied.** `results/val_clusters.csv` is unchanged and the running fold is
untouched. Generated 2026-10-12 by `scripts/propose_val_split.py`, which is the authority on the
procedure; this file records what it does and what it produced.

## Why redesign at all

Validation was restricted to post-cutoff structures, which left 21 of the original 50 clusters live
(26 hold no post-cutoff structure at all, 3 hold one that was never folded). 21 clusters is a thin
holdout when a single interface is ~2% of the signal, and phase B's draw never controlled the things
that matter for a cached validation set: complex size, how much training data a cluster withholds,
and redundancy between clusters.

## The rules, applied identically to retained and newly sampled clusters

| rule | threshold | why |
|---|---|---|
| **R1** size | representative <= 1000 tokens | the cache grows as N^2: `z_trunk` is `[N, N, 256]` in bf16, 96% of the npz and incompressible, so one 1,200-token structure costs as much as six at 500 |
| **R2** withholding | cluster holds <= 5 post-cutoff structures | holding a cluster out removes every structure in it from training; the post-cutoff ones are the costly half, since the pre-cutoff ones are what Protenix already fits |
| **R3** composition | 24 all-new + 16 mixed | *all new* = no structure in the cluster predates 2021-09-30, so no relative of it can have been in Protenix's training pool. *mixed* = a post-cutoff structure of a system Protenix has older structures of. The two test different things and are reported separately |
| **R4** redundancy | no two clusters whose representatives share a deposition date and token count | same study, different cluster id -- redundancy the cluster id does not catch |

**Selection is blind to prediction scores.** The representative is the smallest post-cutoff
structure in the cluster by token count, and token count is a property of the input. Sampling is
stratified by token count into one bin per needed cluster, one cluster drawn per bin, under seed
20261012 -- so the set spans the size range instead of collecting at the cheap end.

**Token counts come from the job json where we have one** (`val_jobs/manifest.csv`) and from summed
chain sequence lengths otherwise. The proxy underestimates assemblies with repeated chains by up to
2x -- `9dzf` is 1,081 by proxy and 2,289 in fact -- and that is a 4x error in cache bytes, so any
newly sampled cluster must be re-priced from its real job json before folding.

## The existing 21 were audited, not grandfathered

They came from phase B's frozen draw: the 533 clusters stratified into 5 bins by median deposition
date, 20,000 random draws under seed 20261010, rejecting any draw where one cluster exceeded 15% of
the target, all before anything was folded and with no reference to predictions. That makes them
admissible candidates. It does not exempt them, because phase B controlled for date and nothing
else.

Four were rejected:

| cluster | representative | rejected by | cache it would have cost |
|---|---|---|---|
| 1867 | `9dzh` | R1 (1,261 tokens) **and** R2 (withholds 6) | 79 GB |
| 2611 | `9p4f` | R1 (1,172 tokens) | 68 GB |
| 764 | `9hvw` | R2 (withholds 11) | 27 GB |
| 1882 | `8t05` | R4 (duplicate of 1833: both 2023-05-31, both 448 tokens) | 10 GB |

Cluster 1867 matters beyond its own 79 GB: its second structure `9dzf` is 2,289 tokens and 244 GB
at 100 seeds, so the cluster the rules reject most firmly is also the most expensive thing in the
fold currently running.

## Retained-plus-sampled versus a fully redrawn set

Checked before settling on reuse, because retaining clusters could preserve a selection bias:

| set | clusters | all-new | mixed | tokens min/med/max | cache @100 seeds | post withheld | pre withheld |
|---|---|---|---|---|---|---|---|
| retain 17 + sample 23 | 40 | 24 | 16 | 190/452/829 | 505 GB | 83 | 96 |
| fully redrawn 40 | 40 | 24 | 16 | 157/439/858 | 478 GB | 72 | 118 |

Median size differs by 13 tokens, cache by 27 GB, post-cutoff withholding by 11 structures. The
eligible pool is 124 all-new and 80 mixed clusters, so nothing forces either choice. Reuse was taken
because it saves refolding 17 clusters, and is recorded here as a decision rather than an
assumption.

## The selected set

40 clusters: 24 all-new, 16 mixed. 17 retained, 23 sampled. Tokens 222/452/799 (min/median/max). Cache at 100 seeds **493 GB**. Withholds 84 post-cutoff and 113 pre-cutoff structures from training. 17 of 40 are already folded.

At 80 seeds the same set is 395 GB, which is the lever to pull before dropping clusters if the budget tightens.

### All new (24 clusters)

| cluster | origin | representative | deposited | tokens | GB @100 seeds | post withheld | pre withheld |
|---|---|---|---|---|---|---|---|
| 1855 | retained | `8y31` | 2024-01-28 | 782 | 30.3 | 2 | 0 |
| 1389 | retained | `7t2p` | 2021-12-06 | 765 | 29.0 | 1 | 0 |
| 1461 | sampled | `pdb_00007t82` | 2021-12-15 | 725 | 26.1 | 1 | 0 |
| 1738 | sampled | `pdb_00007yix` | 2022-07-18 | 707 | 24.8 | 1 | 0 |
| 2243 | sampled | `pdb_00009mla` | 2024-12-18 | 586 | 17.0 | 1 | 0 |
| 1877 | retained | `8omz` | 2023-03-31 | 564 | 15.8 | 1 | 0 |
| 2393 | retained | `9qrv` | 2025-04-04 | 550 | 15.0 | 1 | 0 |
| 2124 | retained | `9e6k` | 2024-10-30 | 546 | 14.8 | 1 | 0 |
| 2347 | sampled | `pdb_00009vnp` | 2025-06-30 | 528 | 13.8 | 1 | 0 |
| 1360 | sampled | `pdb_00007qia` | 2021-12-14 | 518 | 13.3 | 1 | 0 |
| 2381 | retained | `9q1l` | 2025-08-14 | 482 | 11.5 | 1 | 0 |
| 1369 | retained | `7q4q` | 2021-11-01 | 452 | 10.1 | 1 | 0 |
| 1944 | sampled | `pdb_00009eot` | 2024-03-15 | 452 | 10.1 | 3 | 0 |
| 1833 | retained | `8t04` | 2023-05-31 | 448 | 10.0 | 4 | 0 |
| 2312 | sampled | `pdb_00009e7l` | 2024-11-02 | 432 | 9.3 | 2 | 0 |
| 2111 | retained | `9mfg` | 2024-12-09 | 424 | 8.9 | 2 | 0 |
| 2157 | retained | `9fvb` | 2024-06-26 | 424 | 8.9 | 3 | 0 |
| 2377 | sampled | `pdb_00009ulp` | 2025-04-20 | 390 | 7.5 | 1 | 0 |
| 2593 | sampled | `pdb_00009tuy` | 2026-01-10 | 337 | 5.6 | 1 | 0 |
| 2596 | retained | `29tj` | 2026-04-07 | 250 | 3.1 | 1 | 0 |
| 2340 | sampled | `pdb_00009yc3` | 2025-09-18 | 245 | 3.0 | 1 | 0 |
| 1720 | retained | `7y8j` | 2022-06-24 | 243 | 2.9 | 4 | 0 |
| 2103 | retained | `9egn` | 2024-11-21 | 231 | 2.6 | 2 | 0 |
| 1968 | sampled | `pdb_00008r9w` | 2023-11-30 | 226 | 2.5 | 2 | 0 |

### Mixed (16 clusters)

| cluster | origin | representative | deposited | tokens | GB @100 seeds | post withheld | pre withheld |
|---|---|---|---|---|---|---|---|
| 1332 | retained | `8df1` | 2022-06-21 | 799 | 31.7 | 1 | 1 |
| 1055 | sampled | `pdb_00009ryf` | 2025-07-15 | 692 | 23.7 | 2 | 5 |
| 1139 | sampled | `pdb_00009eth` | 2024-03-26 | 624 | 19.3 | 2 | 1 |
| 682 | retained | `7qu2` | 2022-01-17 | 600 | 17.9 | 5 | 2 |
| 926 | sampled | `pdb_00007xy8` | 2022-06-01 | 532 | 14.0 | 3 | 2 |
| 34 | sampled | `pdb_00009fkd` | 2024-06-03 | 500 | 12.4 | 1 | 6 |
| 234 | retained | `7shy` | 2021-10-11 | 482 | 11.5 | 3 | 14 |
| 13 | retained | `9mmj` | 2024-12-20 | 463 | 10.6 | 5 | 41 |
| 978 | sampled | `pdb_00007y3j` | 2022-06-11 | 447 | 9.9 | 2 | 4 |
| 239 | sampled | `pdb_00008vgn` | 2023-12-27 | 437 | 9.5 | 2 | 6 |
| 584 | sampled | `pdb_00009avo` | 2024-03-04 | 428 | 9.1 | 2 | 6 |
| 605 | sampled | `pdb_00009dq4` | 2024-09-23 | 425 | 9.0 | 4 | 12 |
| 993 | sampled | `pdb_00009pav` | 2025-06-25 | 414 | 8.5 | 4 | 6 |
| 938 | sampled | `pdb_00007u9p` | 2022-03-11 | 321 | 5.1 | 2 | 1 |
| 778 | sampled | `pdb_00009ezh` | 2024-04-12 | 239 | 2.8 | 5 | 4 |
| 593 | sampled | `pdb_00008h3c` | 2022-10-08 | 222 | 2.4 | 2 | 2 |

## Before this is applied

1. Re-price every sampled cluster from its real job json; the proxy can be 2x low on assemblies.
2. Build jobs and references for the 23 sampled clusters (phase G) and date-filter their templates
   (phase H). MSA coverage must be checked per chain -- it is not guaranteed outside the folded set.
3. Decide the seed depth (100 or a uniform 80) and freeze it.
4. Write the split through phase B's guard so the exclusion check runs, and move every cluster not
   selected back into training -- whole clusters only. The pre-cutoff structures of a selected
   cluster must stay out of training: training on a 2015 structure of a system validated on its 2023
   structure is cluster-level leakage.
5. Freeze the file and record its hash before any model comparison.

## Reproduce

    cd extra_training && python3 scripts/propose_val_split.py

Flags: `--tok-cap`, `--post-cap`, `--n-allnew`, `--n-mixed`, `--seeds`, `--seed`, `--out`. Output
`results/val_split_proposal.csv`.

