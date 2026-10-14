# FoldBench seed depth

What exists, what counts, and what is still being folded. Measured from
`experimental_data/protenix/foldbench/free_pool/results.csv` on 2026-10-13.

## Blocks

| block | tier | counted | seeds | structures | seed range |
|---|---|---|---|---|---|
| `pool` | pool | yes | 50 | 104 | 101–150 |
| `extra10` | topup | yes | 10 | 104 | 501–510 |
| `top100` | topup | yes | 40 | 53 | 600000–600039 |
| `matched` | suspect | **no** | 50 | 91 | 151–200 |
| `other_seeds` | suspect | no | 647 | 30 | 0–2036 |

104 structures, 160 interfaces. "Counted" means `tier != "suspect"`, which is what every analysis
uses.

## Why `matched` is excluded

16 interfaces over 10 complexes disagree with the pool on how often a seed finds an acceptable pose,
at p<0.001, in **both** directions — 8up2_AC goes 48/50 to 23/50, 7zoz_AB goes 18/50 to 19/24. Timing
rules out reduced recycles and the paired mean DockQ matches to 0.007, so averages are unaffected; it
is the max-over-seeds statistics (decoy, no-acceptable) that move. That directory also holds
`resume_8vyl_s101`, `queue_refold_nan_8xnh.tsv` and two `claims_refold_*`: the block was re-run in
pieces. `free_extra10` and every pxm block show zero such interfaces.

The 10 complexes: 8up2, 7x8p, 8vyl, 8oni, 7zoz, 8be3, 8urf, 8che, 8byu, 9fww. All 10 are in `matched`.

## The decision, 2026-10-13

**`matched` stays suspect in full**, not just for the 10 flagged complexes.

Both options were costed first:

| rule | interfaces at ≥100 counted seeds | folds still needed | GPU hours |
|---|---|---|---|
| `matched` suspect in full (**chosen**) | 20 of 160 | 2,607 | 14.1 |
| `matched` counted except the 10 flagged | 132 of 160 | 612 | 3.3 |

The second is defensible — the problem is characterised and its complexes are named — but it puts a
tier judgement underneath a published number. 11 GPU-hours is a low price for not having to defend
that in review. Rates are measured, not estimated: FoldBench folds at **185 folds/h on 3 GPUs**
(1,554 folds over 8.40 h, stage 4).

## What is running

`experimental_data/protenix/foldbench/free_top100b`, 2,607 folds over 94 structures, seeds 601000+,
bringing every structure to 100 seeds **excluding** suspect. 135 done, 2,472 remaining, ~13.4 h.
Claim-based and resumable. It is queued behind the FT3 training cache; `extra_training/scripts/
watch_train_cache.sh` launches it when the cache queue is exhausted.

## A correction worth keeping

An earlier top-up (`free_top100`, the 1,554 folds of 2026-10-13) was built by
`build_topup_queue.py` crediting **suspect** seeds toward the target of 100, while every analysis
excludes them. It allocated 4–40 seeds per structure to 53 structures and left 51 untouched, so only
20 of 160 interfaces reached 100 countable seeds. The folds are fine and are counted; the target was
computed on a different rule from the one that reads the result.
