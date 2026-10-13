# What seed scaling buys, by benchmark

Measured 2026-10-12. Each interface is one of three things at a given seed depth:

- **no acceptable** — no pose in the draw reaches DockQ 0.23. A *sampling* failure; reranking these
  poses cannot fix it.
- **decoy** — an acceptable pose is in the draw, but top pair-ipTM picks one below 0.23. A
  *selection* failure, winnable with no extra folding.
- **acceptable** — top pair-ipTM picks an acceptable pose.

Cells are interface counts, mean ± sd over 30 bootstrap draws of that many seeds, drawn without
replacement from each interface's own seed universe. The last row of each table is the single
possible draw, so it has no spread. Every interface contributes exactly the same number of seeds, so
each column is the same experiment across interfaces.

## PXM-2025 — 274 interfaces, universe of 100 seeds

| seeds | decoy | acceptable | no acceptable |
|---|---|---|---|
| 10 | 52.3 ± 4.9 | 121.0 ± 3.3 | 100.7 ± 4.8 |
| 20 | 65.0 ± 4.4 | 126.0 ± 2.4 | 83.0 ± 3.8 |
| 30 | 71.1 ± 4.6 | 129.5 ± 3.3 | 73.4 ± 2.5 |
| 40 | 75.5 ± 3.2 | 130.9 ± 1.7 | 67.6 ± 3.0 |
| 50 | 78.1 ± 2.7 | 132.6 ± 2.2 | 63.3 ± 1.9 |
| 60 | 79.8 ± 3.1 | 133.6 ± 2.5 | 60.6 ± 1.6 |
| 70 | 81.6 ± 3.1 | 134.1 ± 2.1 | 58.2 ± 2.0 |
| 80 | 83.1 ± 2.2 | 134.3 ± 1.4 | 56.6 ± 1.5 |
| 90 | 84.8 ± 1.4 | 133.8 ± 1.0 | 55.4 ± 1.1 |
| 100 | 86.0 | 134.0 | 54.0 |

18 of 292 interfaces are excluded for holding fewer than 100 usable seeds.

## FoldBench — 159 interfaces, universe of 60 seeds

| seeds | decoy | acceptable | no acceptable |
|---|---|---|---|
| 10 | 25.1 ± 3.3 | 105.3 ± 2.4 | 28.6 ± 2.5 |
| 20 | 25.2 ± 2.7 | 111.3 ± 2.4 | 22.4 ± 1.8 |
| 30 | 26.1 ± 2.8 | 114.1 ± 2.4 | 18.8 ± 1.7 |
| 40 | 25.9 ± 2.5 | 116.6 ± 2.1 | 16.5 ± 1.8 |
| 50 | 26.0 ± 1.6 | 118.2 ± 1.3 | 14.8 ± 1.2 |
| 60 | 27.0 | 119.0 | 13.0 |

1 of 160 interfaces is excluded for holding 59 usable seeds. FoldBench stops at 60 because that is
all the depth it has at settings comparable to its published baseline.

## What the two tables say together

Seed scaling reliably reduces the no-acceptable count on both benchmarks, and the decoy count rises
or stays flat — it never falls. Selection then decides whether a rescued interface is actually won:

| | no-acceptable rescued | of those, became acceptable | conversion |
|---|---|---|---|
| PXM-2025, 10 -> 100 seeds | 46.7 | 13.0 | **28%** |
| FoldBench, 10 -> 60 seeds | 15.6 | 13.7 | **88%** |

Same intervention, opposite bottleneck. FoldBench's ceiling is sampling: it keeps nearly everything
scaling finds. PXM's is selection: 34 of the 47 interfaces that gained a good pose land in the decoy
column, where no further folding helps and only reranking can.

This is also why the decoy count, not the acceptable count, is the metric to watch. Past 50 seeds
PXM's acceptable count is flat inside its own noise (132.6 ± 2.2 -> 134.0) while the recoverable
pool grows from 78 to 86 interfaces.

## Provenance

| | |
|---|---|
| table generator | `diffusion_commit_clean_v2/analysis/seed_depth_split.py` |
| data | `paths.DATASETS[ds].free_pool`, rows with `tier != "suspect"` |
| commands | `seed_depth_split.py --dataset pxm2025 --boots 30`<br>`seed_depth_split.py --dataset foldbench --depths 10,20,30,40,50,60 --universe 60 --boots 30` |

`tier != "suspect"` excludes FoldBench's 151-200 seed block: 16 interfaces over 10 complexes
disagree with the pool on how often a seed finds an acceptable pose, at p<0.001 and in both
directions. These are max-over-seeds statistics, which is exactly what such a disagreement moves.
See each benchmark's `free_pool/INDEX.md`.

## Queued: both benchmarks to 100 usable seeds

Not launched. Runs after the 40-cluster validation fold and its 13 second representatives.

| | folds | on 3 GPUs | poses |
|---|---|---|---|
| PXM-2025 | 290 (29 already done) | 1.5 h | ~1 GiB |
| FoldBench | 1,554 | 8.9 h | ~4 GiB |

### Why FoldBench costs 6x what PXM does, and what was decided

FoldBench's usable depth is 60 seeds (the 50-seed pool plus `extra10`), because its 151-200 block is
`suspect` in the pooled table. Three policies were priced:

| policy | folds | on 3 GPUs |
|---|---|---|
| (a) ignore the suspect block, fold 40 fresh seeds for all 104 structures | 4,161 | 35.7 h |
| (b) credit the suspect block entirely | 1,192 | 6.3 h |
| **(c) credit it except the 10 disagreeing complexes, which refold** | **1,554** | **8.9 h** |

**(c) was chosen.** The 10 are `8up2 7x8p 8vyl 8oni 7zoz 8be3 8urf 8che 8byu 9fww` -- the complexes
whose suspect block disagrees with the pool at p<0.001 on how often a seed finds an acceptable pose,
in both directions (`8up2_AC` 48/50 -> 23/50, `8vyl_AF` 15/50 -> 0/50). For the other 94 the paired
mean DockQ matches to 0.007, so crediting them is sound; (a) would spend 27 extra GPU-hours
re-deriving seeds we already have, and (b) would carry the disagreement into coverage counts, which
are max-over-seeds statistics and therefore exactly what it moves.

### How to launch

    cd diffusion_commit_clean_v2
    # foldbench queue already written by:
    python3 build_topup_queue.py --dataset foldbench --credit-suspect-except-flagged --seed0 600000 --apply
    # -> experimental_data/protenix/foldbench/free_top100/queue.tsv, seeds 600000-600039, 1554 folds

    # then one worker per card, e.g.
    tmux new-session -d -s fb_top100_0 "cd ~/Protenix && CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold \
      ~/Protenix/.venv/bin/python ~/tts_abag/diffusion_commit_clean_v2/folder.py \
      --queue <arm>/queue.tsv --out <arm>/poses --csv <arm>/results.csv --claim-dir <arm>/claims \
      --dataset foldbench --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> <arm>/w0.log 2>&1"

Workers share one queue through the claim directory, so three on one queue is the normal pattern.

**PXM resumes its existing arm, it is not regenerated.** `pxm2025/free_top100/queue.tsv` already
holds the right 290 folds at seeds 300601+ with 29 done; running the generator for pxm would allocate
310000+ and discard that work.

**FoldBench folds template-free.** `paths.DATASETS['foldbench'].free_templates = False`, matching the
pool its seeds join; `--dataset foldbench` enforces it. PXM folds with templates. Mixing the two
within a benchmark's pool would be a settings inconsistency of exactly the kind the `tier` column
exists to catch.

After both finish: rerun `build_free_pool.py --apply`, confirm every interface reaches 100 usable
seeds, and regenerate the tables above with `analysis/seed_depth_split.py --boots 30` -- for FoldBench
at `--universe 100` rather than 60.

## Validation set: interface class vs seed depth

Generated by `diffusion_commit_clean_v2/analysis/val_seed_depth_curve.py`; numbers in the matching
`.csv` beside each figure.

x = seeds drawn per interface. y = % of interfaces in each class, **every interface weighted
equally**. Band = ±1 sd over draws. k=1 enumerates all 100 single-seed draws (the i-th seed of each
interface); k=100 has one possible draw; otherwise 50 draws without replacement, seed 0. Acceptable
at DockQ ≥ 0.23, selection by top pair-ipTM.

**All 40 clusters, 91 interfaces**

![All 40 clusters, 91 interfaces](figures/val_seed_depth.png)

| seeds | draws | acceptable | decoy | no acceptable |
|---|---|---|---|---|
| 1 | 100 | 61.6 ± 3.3 | 4.1 ± 2.3 | 34.2 ± 3.5 |
| 5 | 50 | 69.6 ± 1.7 | 8.0 ± 2.1 | 23.0 ± 2.1 |
| 10 | 50 | 70.8 ± 1.4 | 10.0 ± 1.9 | 19.0 ± 1.5 |
| 20 | 50 | 71.8 ± 1.2 | 12.2 ± 1.5 | 16.4 ± 1.5 |
| 30 | 50 | 71.9 ± 1.0 | 13.1 ± 1.7 | 14.8 ± 1.1 |
| 40 | 50 | 71.8 ± 1.3 | 13.5 ± 1.3 | 14.5 ± 1.1 |
| 50 | 50 | 72.2 ± 0.9 | 14.2 ± 1.1 | 13.8 ± 0.7 |
| 60 | 50 | 72.2 ± 1.0 | 14.3 ± 1.2 | 13.8 ± 0.7 |
| 70 | 50 | 72.2 ± 0.9 | 13.9 ± 1.1 | 13.6 ± 0.6 |
| 80 | 50 | 72.3 ± 0.8 | 14.2 ± 0.9 | 13.5 ± 0.5 |
| 90 | 50 | 72.4 ± 0.7 | 14.4 ± 0.7 | 13.3 ± 0.3 |
| 100 | 1 | 72.5 | 14.3 | 13.2 |

**All-new clusters, 24 clusters, 52 interfaces**

![All-new clusters, 24 clusters, 52 interfaces](figures/val_seed_depth_allnew.png)

| seeds | draws | acceptable | decoy | no acceptable |
|---|---|---|---|---|
| 1 | 100 | 51.3 ± 4.6 | 5.6 ± 3.5 | 43.1 ± 4.6 |
| 5 | 50 | 60.9 ± 2.9 | 9.2 ± 3.3 | 30.9 ± 2.5 |
| 10 | 50 | 62.7 ± 2.5 | 10.1 ± 3.0 | 26.0 ± 2.6 |
| 20 | 50 | 63.6 ± 2.2 | 12.5 ± 2.6 | 23.3 ± 2.1 |
| 30 | 50 | 64.9 ± 2.0 | 13.1 ± 2.5 | 22.1 ± 1.5 |
| 40 | 50 | 64.5 ± 1.9 | 14.6 ± 2.5 | 21.1 ± 1.3 |
| 50 | 50 | 64.3 ± 1.9 | 14.5 ± 2.0 | 20.7 ± 1.2 |
| 60 | 50 | 64.8 ± 2.0 | 15.1 ± 2.1 | 20.5 ± 1.3 |
| 70 | 50 | 64.6 ± 1.7 | 15.5 ± 2.2 | 20.1 ± 1.0 |
| 80 | 50 | 64.8 ± 1.3 | 15.5 ± 1.5 | 19.7 ± 0.8 |
| 90 | 50 | 65.0 ± 1.0 | 15.3 ± 1.2 | 19.5 ± 0.7 |
| 100 | 1 | 65.4 | 15.4 | 19.2 |

**Mixed clusters, 16 clusters, 39 interfaces**

![Mixed clusters, 16 clusters, 39 interfaces](figures/val_seed_depth_mixed.png)

| seeds | draws | acceptable | decoy | no acceptable |
|---|---|---|---|---|
| 1 | 100 | 75.4 ± 4.6 | 2.1 ± 2.6 | 22.4 ± 5.0 |
| 5 | 50 | 81.7 ± 1.3 | 6.4 ± 3.1 | 12.6 ± 3.1 |
| 10 | 50 | 82.0 ± 0.8 | 8.9 ± 2.6 | 9.2 ± 2.8 |
| 20 | 50 | 81.8 ± 0.7 | 11.0 ± 1.9 | 6.6 ± 1.7 |
| 30 | 50 | 81.9 ± 0.6 | 12.2 ± 1.4 | 5.6 ± 1.1 |
| 40 | 50 | 82.0 | 12.7 ± 0.6 | 5.3 ± 0.6 |
| 50 | 50 | 82.0 | 12.8 ± 0.6 | 5.3 ± 0.7 |
| 60 | 50 | 82.0 | 12.8 | 5.1 |
| 70 | 50 | 82.0 | 12.8 | 5.1 |
| 80 | 50 | 82.0 | 12.8 | 5.1 |
| 90 | 50 | 82.0 | 12.8 | 5.1 |
| 100 | 1 | 82.0 | 12.8 | 5.1 |


## Validation set: each class across arms

Same data as above, transposed: one class per figure, one line per arm. `mixed` = the cluster also holds pre-2021-09-30 structures. Every interface weighted equally.

**acceptable**

![acceptable](figures/val_acceptable_by_arm.png)

| seeds | draws | all | all new | mixed |
|---|---|---|---|---|
| 1 | 100 | 61.6 ± 3.3 | 51.3 ± 4.6 | 75.4 ± 4.6 |
| 5 | 50 | 69.6 ± 1.6 | 59.7 ± 3.1 | 81.7 ± 1.4 |
| 10 | 50 | 70.8 ± 1.2 | 63.3 ± 2.4 | 81.9 ± 0.8 |
| 20 | 50 | 71.7 ± 1.2 | 63.8 ± 2.3 | 82.0 |
| 30 | 50 | 72.0 ± 1.0 | 64.8 ± 2.3 | 82.0 ± 0.5 |
| 40 | 50 | 72.0 ± 1.2 | 64.8 ± 1.9 | 82.0 |
| 50 | 50 | 72.0 ± 0.9 | 63.9 ± 2.1 | 82.0 |
| 60 | 50 | 72.4 ± 1.0 | 64.8 ± 1.7 | 82.0 |
| 70 | 50 | 72.4 ± 0.9 | 64.6 ± 1.5 | 82.0 |
| 80 | 50 | 72.4 ± 0.8 | 65.1 ± 1.5 | 82.0 |
| 90 | 50 | 72.4 ± 0.6 | 65.5 ± 1.1 | 82.0 |
| 100 | 1 | 72.5 | 65.4 | 82.0 |

**decoy**

![decoy](figures/val_decoy_by_arm.png)

| seeds | draws | all | all new | mixed |
|---|---|---|---|---|
| 1 | 100 | 4.1 ± 2.3 | 5.6 ± 3.5 | 2.1 ± 2.6 |
| 5 | 50 | 7.6 ± 2.2 | 9.3 ± 3.8 | 5.6 ± 2.6 |
| 10 | 50 | 9.6 ± 2.5 | 10.7 ± 2.8 | 8.6 ± 2.7 |
| 20 | 50 | 12.2 ± 1.4 | 12.6 ± 2.5 | 10.8 ± 1.8 |
| 30 | 50 | 13.0 ± 1.5 | 13.2 ± 2.4 | 12.4 ± 1.2 |
| 40 | 50 | 13.5 ± 1.6 | 13.9 ± 2.5 | 12.7 ± 0.6 |
| 50 | 50 | 14.0 ± 1.1 | 15.3 ± 2.2 | 12.7 ± 0.6 |
| 60 | 50 | 14.0 ± 1.1 | 15.0 ± 1.9 | 12.8 |
| 70 | 50 | 14.0 ± 0.9 | 15.4 ± 2.0 | 12.8 |
| 80 | 50 | 14.2 ± 0.9 | 15.3 ± 1.8 | 12.8 |
| 90 | 50 | 14.3 ± 0.7 | 15.1 ± 1.2 | 12.8 |
| 100 | 1 | 14.3 | 15.4 | 12.8 |

**no acceptable**

![no acceptable](figures/val_no_acceptable_by_arm.png)

| seeds | draws | all | all new | mixed |
|---|---|---|---|---|
| 1 | 100 | 34.2 ± 3.5 | 43.1 ± 4.6 | 22.4 ± 5.0 |
| 5 | 50 | 22.9 ± 1.8 | 30.9 ± 3.1 | 12.7 ± 2.8 |
| 10 | 50 | 19.6 ± 2.0 | 26.0 ± 2.3 | 9.5 ± 2.5 |
| 20 | 50 | 16.1 ± 1.0 | 23.6 ± 1.6 | 7.2 ± 1.8 |
| 30 | 50 | 15.0 ± 1.0 | 22.0 ± 1.7 | 5.6 ± 1.0 |
| 40 | 50 | 14.5 ± 0.9 | 21.3 ± 1.2 | 5.3 ± 0.6 |
| 50 | 50 | 14.0 ± 0.7 | 20.7 ± 1.2 | 5.3 ± 0.6 |
| 60 | 50 | 13.6 ± 0.6 | 20.2 ± 1.1 | 5.1 |
| 70 | 50 | 13.7 ± 0.5 | 20.0 ± 1.0 | 5.1 |
| 80 | 50 | 13.4 ± 0.4 | 19.6 ± 0.8 | 5.1 |
| 90 | 50 | 13.3 ± 0.4 | 19.4 ± 0.6 | 5.1 |
| 100 | 1 | 13.2 | 19.2 | 5.1 |

