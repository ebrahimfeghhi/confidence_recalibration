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
