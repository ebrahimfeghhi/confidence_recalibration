# reffree_50 — how many extra folds the method costs

The reference-free self-templating run. Every number here is the **additional** compute on top of the
free pool, which already exists before the method starts.

## Total

| | |
|:---|---:|
| extra folds (donors) | **3,132** |
| complexes | 118 |
| **mean extra folds per complex** | **26.5** |
| median | 22 |

Each fold is one donor pose used as a template, at one seed (430000), producing 5 poses. So 3,132 folds
→ 15,660 additional poses.

## By class

Class is the complex's worst interface: `decoy` if any interface is a confident decoy, else `solved`,
else `nocov`.

| class | complexes | extra folds | mean per complex | median |
|:---|---:|---:|---:|---:|
| decoy | 46 | 1,615 | 35.1 | 36 |
| solved | 60 | 1,068 | 17.8 | 14 |
| nocov | 12 | 449 | 37.4 | 45 |

Solved complexes cost roughly half as much as decoys. That falls out of the algorithm rather than being
imposed: the diversity filter keeps a pose only if its pose-to-pose DockQ to everything already kept is
below 0.23, and on a solved complex the poses agree with each other, so the candidate pool exhausts long
before the cap of 50.

## What was actually folded this time

| | |
|:---|---:|
| folded in this run | 972 |
| reused from earlier runs | 2,160 |

Reuse is keyed on `(donor pose, seed, pinned chains)`. A fold's output is fixed by those three, so a
matching fold from an earlier experiment is carried over rather than repeated. All 2,160 were checked
against what this run's template builder would have produced: 1,625 of 1,625 whose recorded chains were
only a subset resolved to exactly the same template, so nothing needed refolding.

## Cost relative to the free pool

The free pool is 50 seeds × 5 samples = 250 poses per complex. The method adds a mean of 26.5 folds,
which is **+53% on the seed count** and +10.6% on the pose count.

## Caveat on the nocov row

The 12 nocov complexes consumed 449 folds and cannot succeed — no acceptable pose exists in their pool
to template from, and the measured result is 1 of 42 interfaces. They are kept because identifying them
in advance requires the crystal, so the method cannot skip them at test time. 368 nocov queue lines were
pruned by hand partway through this run, so the 449 figure is lower than the algorithm would have chosen.
