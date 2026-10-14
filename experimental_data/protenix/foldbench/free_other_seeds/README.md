# FoldBench free folds outside the experiment's two seed blocks

Split out of `../free/` on 2026-10-06 by `diffusion_commit_clean_v2/split_free_pool.py`.
4,667 pose directories, 31,395 scored rows. **Kept, not archived** — some of this backs published
results.

## Why it was moved

The paper's experiment uses exactly two seed blocks, and `../free/` now holds only those: the pool
(101–150) and the compute-matched extras (151–200). Everything else lived in the same directory and
the same `results.csv`, which is a trap rather than a tidiness problem: `extra_free()` reads "every
seed outside the pool's first 50" and `matched_seeds()` then takes the **lowest-numbered available**.
While these rows shared that csv, the matched baseline drew seeds 0–100 for the handful of complexes
that had been experimented on before — and those extras measured 0.060 worse than the pool on mean
DockQ, 0.061 on ipTM, beating the pool on only 41% of interfaces against ~50% chance. That is why
adding free seeds appeared to make the baseline *worse*.

The leftovers were never evenly spread: 75 of 111 complexes held only pool seeds, while 8pn0 carried
449 extras, 7yv1 199, 7yxu 198, 8vui 197, 8ts0 197. So the contamination tracked which complexes
someone had poked at before, not anything about the complexes themselves.

## What is in here

| seed block | pose dirs | what it is |
|:---|---:|:---|
| 0–100 | 3,427 | leftovers below the pool |
| 100 | 33 | leftovers (seed 100 only; 101–150 stayed) |
| 201–339 | 798 | leftovers |
| 538–572 | 35 | leftovers |
| 739–861 | 158 | leftovers |
| 1327–1453 | 127 | **depth-robustness runs** |
| 1915–2036 | 122 | **depth-robustness runs** |

The two deep blocks are the ones behind "confident-decoy selection is depth-robust — 1024 seeds does
not rescue ipTM". Do not delete them.

## Reading it

`results.csv` is in the pool's own schema. To analyse these deliberately, point a reader at this
file instead of `../free/results.csv` — nothing does so by default, which is the point. The pool's
own scores are in `../pool.csv` and were never in either results.csv.
