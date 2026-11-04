# Pruning degenerate structures, and what it buys on disk

Measured 2026-11-04. Nothing in this document has been deleted; it is a sizing exercise.

## The problem

The relative loss grades the **gaps between a structure's candidate poses**, and selection is
scored by which pose a head picks. A structure whose poses are all about equally good, or all
hopeless, offers no gap to grade and no choice to get right. It costs disk and training time and
contributes nothing in either direction.

Two ways a structure can be degenerate:

- **No spread.** Within-interface DockQ standard deviation below 0.05. Either every pose is
  near-native (unloseable) or every pose is wrong in the same way (uninformative).
- **No acceptable pose.** The best pose of the interface is below DockQ 0.23, so no selector,
  however good, can produce a usable answer.

How common this is, on what is currently on disk:

| | interfaces with sd < 0.05 | structures where EVERY interface is degenerate |
|---|---|---|
| training, 1,028 interfaces / 497 structures | 520 | 218 |
| validation, 92 interfaces / 53 structures | 45 | 26 |

On the validation side the degeneracy is sharply bimodal. Of the 20 interfaces below sd 0.02,
13 are saturated — `7sqw_BC` 0.9679, `8v8f_AC` 0.9650, `8gq1_AB` 0.9503 — and shipped ipTM picks
an acceptable pose on **13 of 13** of them, leaving a mean of 0.0196 DockQ on the table. The other
7 have no acceptable pose at all, so nothing can succeed on them. Those 20 interfaces are
unloseable or unwinnable; they cannot move a result.

More seeds does not repair this. Comparing within-interface sd over 30 seeds against all 100 on
the degenerate interfaces gives 0.0016 vs 0.0016, 0.0036 vs 0.0038, 0.0047 vs 0.0070 — the
degeneracy is a property of the complex, not of sampling depth. The fold is confidently right or
confidently wrong and extra draws land in the same place.

## Proposal 1 — drop every structure whose every interface is degenerate

Remove a structure when **all** of its interfaces satisfy `sd < 0.05 OR best pose < 0.23`. The
condition is applied per structure and must hold for every interface it carries, so a structure
with one discriminative interface is kept whatever else it contains.

| | structures dropped | freed | remaining |
|---|---|---|---|
| training | 220 of 500 | 628.3 GiB (45%) | 280 structures / 772.8 GiB |
| validation | 26 of 53 | 333.7 GiB (47%) | 27 structures / 369.7 GiB |
| **combined** | **246** | **962.0 GiB** | |

This supersedes rather than adds to the narrower "127 structures with zero acceptable poses" cut:
those 127 are a subset of the 220.

## Proposal 2 — on top of that, cut validation from 100 seeds to 30

The validation trunk cache holds 100 seed units per structure (seeds 400000–400099) while the
eval has only ever scored the first 30. The other 70 are 3,710 units that nothing reads.

Dropping to 30 seeds for the 27 surviving validation structures frees a further **258.7 GiB**.

**Which 30 is an open question.** The intent is to keep the 30 seeds that **maximise DockQ
spread**, so that each surviving interface is as discriminative as possible rather than merely a
uniform subsample. The selection rule is TBD. Two things are already known about it:

- The disk saving does not depend on the choice. Every unit of a structure carries the same token
  count, so size per unit is constant within a structure and keeping any 30 of 100 frees exactly
  70% of it. The selection governs benchmark quality, not space.
- Selection is **per structure, not per interface** — a seed unit holds every interface of its
  structure, and 92 interfaces live across 53 structures, so a multi-interface structure needs a
  compromise across its interfaces.

## Combined

| | freed |
|---|---|
| drop 220 degenerate training structures | 628.3 GiB |
| drop 26 degenerate validation structures | 333.7 GiB |
| trim the surviving 27 validation structures to 30 seeds | 258.7 GiB |
| **total** | **1,220.7 GiB** |

| | GiB |
|---|---|
| free now | 536.2 |
| **available after** | **1,756.9** |

Quota use would fall from 2,536 GiB to about 1,315 GiB of a 3,072 GiB allocation — 43% instead of
83%. What remains is 280 training structures (772.8 GiB) and 27 validation structures at 30 seeds
(110.9 GiB), all of them carrying at least one interface with real spread and a reachable
acceptable pose.

---

# Refilling the quota with new structures

## What pruning leaves behind

The 220 dropped training structures do not come out of the pool evenly — they take whole clusters
with them. After the cut:

| | |
|---|---|
| structures retained | 280 |
| clusters they still cover | **125** of 493 |
| (antibody, antigen) pairings held | 294 |
| distinct antigen clusters held | 160 |
| distinct chain sequences held | 907 |

So pruning costs 68 clusters outright, and refilling has to buy breadth back before it buys depth.

## What is foldable today

Every count below excludes anything already on disk and requires an antibody plus a polypeptide
antigen, at most 2560 tokens (Protenix v2's inference ceiling), and `pairing.a3m` +
`non_pairing.a3m` already resolvable through `seq_to_pdb_index.json`. **No MSA search is needed
for any of it** — the ColabFold uniref/envdb searches were run over the whole SAbDab sequence set,
and the date cutoff was applied downstream at structure selection.

| | structures | clusters | disk |
|---|---|---|---|
| training clusters, post-cutoff | 911 | **25** | 1,561.8 GiB |
| training clusters, pre-cutoff | 1,093 | 362 | 2,094.7 GiB |
| validation clusters, post-cutoff | 13 | 9 | 19.4 GiB |
| validation clusters, pre-cutoff | 210 | 16 | 243.4 GiB |

The asymmetry is the whole story: post-cutoff offers 911 structures but only in **25** clusters,
because the original cap-2 selection already took one or two from every post-cutoff cluster it
could reach. Pre-cutoff is the only source of cluster breadth.

## The plan being costed

**Group A — one structure for every cluster with no retained structure.** 323 clusters,
**556.1 GiB** at a median-sized pick.

**Group B — one additional post-cutoff structure per retained cluster.** Only **19** clusters have
an unfolded post-cutoff candidate, so this is 19 structures and **48.2 GiB**, not the ~190 a
first pass suggested. Those 19 clusters are deep, holding 988 candidates between them.

Combined: **342 structures, ~604 GiB, 3,420 units, roughly 19 h of fold on 3 cards**, against
1,756.9 GiB available. That would take training to 622 structures across 448 clusters.

## Selecting group B for maximum difference from what we keep

Worth doing deliberately: with 988 candidates for 19 slots, the choice is nearly free, and taking
the first of each cluster would mostly add near-duplicates of structures already retained. Four
criteria are computable from `abag_split.csv` alone, no folding required, listed strongest first:

| criterion | candidates satisfying it |
|---|---|
| **antigen cluster** not held by any retained structure | 45 |
| **antibody cluster** not held by any retained structure | 78 |
| **(antibody, antigen) pairing** not held by any retained structure | 81 |
| shares **no chain sequence** with any retained structure | 257 |

A new antigen cluster is the strongest signal, because it means a genuinely different epitope
family rather than another antibody against a target already represented. The pairing criterion is
the one the original selector already uses as its cap key, so reusing it keeps the new structures
consistent with how the frozen list was built. "No shared chain sequence" is the weakest but
broadest filter and is the natural tiebreaker.

Suggested rule: prefer a candidate offering a new antigen cluster; failing that, a new antibody
cluster; failing that, a new pairing; break remaining ties on no shared chain sequence, then on
smallest token count so the disk goes furthest. Not yet implemented.

**Caveat on group B's concentration.** 988 candidates in 19 clusters means those clusters are
heavily deposited targets, so even a maximally diverse pick stays within a narrow region of
antigen space. Group A is where the breadth actually comes from.

---

# WAVE 1 — SUPERSEDED, see "WAVE 1 — REVISED" below

> Kept for the reasoning. Three decisions changed on 2026-11-04: validation keeps ALL 53
> structures at 30 seeds (not 26 trimmed to 1), the 30 retained seeds are CHOSEN rather than the
> first 30, and variable-seed trainer support is explicitly not a prerequisite for folding.

Two waves. Wave 1 is specified below; its outcome sizes wave 2, which is deliberately left
unplanned because the degeneracy rate on pre-cutoff structures is not yet known.

## Step 1 — trim degenerate training structures to one seed

The 220 training structures whose every interface is degenerate (`sd < 0.05` OR best pose
`< 0.23`) are cut from 10 seed units to **1**, rather than deleted.

Why not delete: 118 clusters hold a degenerate structure and 63 of those have no unfolded
alternative anywhere in the PDB — 64 clusters contain only ONE usable post-cutoff structure in
total, we folded it, and it came back degenerate. Deleting would strand 73 clusters permanently.
Keeping one seed costs **62.8 GiB** and holds cluster coverage at 194 instead of 121.

Why one seed is enough: the loss is `PAE + 75 x rel`, and only the rel term needs gaps between
poses. The PAE term is absolute per-pose supervision and works fine on a degenerate structure. One
seed keeps the cluster visible to the PAE loss without flooding the epoch with redundant poses.

Frees **565.5 GiB**.

## Step 2 — validation cuts

Same treatment as training: nothing is deleted, everything is trimmed.

- The **26 degenerate validation structures** go from 100 seeds to **1**. Frees **330.2 GiB**,
  retains 3.3 GiB, and holds validation at **40 of 40 clusters** rather than dropping to 22.
- The **27 surviving structures** go from 100 seeds to **30**. Frees **258.7 GiB**, retains
  110.9 GiB. Which 30 is still TBD — the intent is to maximise DockQ spread; see the earlier
  section.

Validation freed: **588.9 GiB**, leaving 114.2 GiB on disk across all 53 structures.

## Step 3 — generate new training structures

Degenerate structures do NOT count toward any target, and are never refolded (the candidate pool
requires no trunk on disk, and all 220 have one).

| cluster type | target | rule |
|---|---|---|
| pre-cutoff only (299) | 1 | cheapest usable candidate |
| mixed (63) | 2 | **prefer post-cutoff** candidates; fall back to 1 if only 1 exists |
| post-cutoff only (131) | 2 | never refold a degenerate structure; **maximise the difference** between the two structures in a cluster; fall back to 1 |

"Maximise the difference" is scored over three computable signals, highest first: a different
antigen cluster (the strongest, since it means a different epitope family rather than another
antibody against the same target), a different antibody cluster, and no shared chain sequence.
Ties break on smallest token count so the disk goes furthest.

### What that costs

| type | clusters | to fold | disk |
|---|---|---|---|
| pre-only | 299 | 299 | 468.7 GiB |
| mixed | 63 | 51 | 88.2 GiB |
| post-only | 131 | 14 | 62.8 GiB |
| **total** | **493** | **364** | **619.6 GiB** |

3,640 units, roughly **20 h of fold on 3 cards**. Mixed needs 51 rather than 63 and post-only 14
rather than 131 because many of those clusters already hold two retained structures.

### Where it lands, before accounting for new degeneracy

| | clusters |
|---|---|
| end with >= 2 structures | 95 |
| end with exactly 1 | 349 |
| end with 0 | 49 |

The 49 are post-cutoff-only clusters whose sole usable structure is already folded and degenerate.
They survive as cluster coverage only through that one-seed structure.

## Step 4 — grow validation, post-cutoff only

Validation structures must be **post-cutoff**; the clusters they come from may be mixed.

| source | structures | clusters |
|---|---|---|
| new clusters, previously unassigned to train or val | 17 | 14 |
| existing validation clusters | 13 | 9 (3 of which currently hold only degenerate structures) |
| **total** | **30** | **23** |

All 30 have `pairing.a3m` + `non_pairing.a3m` verified present and non-empty for every chain.
Tokens 174–899, median 548. Validation goes from 40 to **54 clusters** and 53 to 83 structures.

That is the entire post-cutoff supply. The 2,080 clusters unassigned to either split hold 846
post-cutoff candidates, but 829 have no MSA built — their sequences were never submitted to the
ColabFold search, which only ran over the structures the original split selected.

**Folded at 100 seeds, then trimmed to 30**, matching how the existing set will be treated.

| | disk | units | fold |
|---|---|---|---|
| at 100 seeds (transient peak) | 445.2 GiB | 3,000 | ~16.7 h on 3 cards |
| retained after the trim to 30 | 133.6 GiB | 900 | — |

## Labels at 100 seeds

Selecting the best 30 of 100 by DockQ spread requires DockQ for every pose, so the label build,
the coordinate-based sample-index match and the rescore all run at 100 seeds — about 3.3x the
post-fold work. This applies to the 30 new structures AND to the 27 existing retained validation
structures, which currently carry labels only for seeds 400000-400029. CPU-only, so it does not
compete for cards; the stage has no measured rate.

## Full ledger, wave 1

| | GiB |
|---|---|
| free now | 536.2 |
| + trim 220 degenerate training structures to 1 seed | +565.5 |
| + trim 26 degenerate validation structures to 1 seed | +330.2 |
| + trim 27 validation structures to 30 seeds | +258.7 |
| **available** | **1,690.6** |
| − training fold, 364 structures at 10 seeds | −619.6 |
| − validation fold, 30 structures at 100 seeds | −445.2 |
| **spare at peak** | **625.8** |
| + trim the new validation structures to 30 seeds | +311.6 |
| **spare after wave 1** | **937.4** |

Fold total: 3,640 + 3,000 = **6,640 units, roughly 37 h on 3 cards.**


## Known risks, accepted

1. **The 44% degeneracy rate will apply to the new folds too.** It was measured on this pipeline
   on this data: 220 of 500. If it holds, roughly 160 of the 364 new structures come back
   degenerate, so "end with >= 2" of 95 is optimistic and perhaps 130 of the 299 pre-only clusters
   will be stranded on a degenerate single structure. Wave 2 exists to absorb this.
2. **The pre-cutoff rate is unmeasured.** All 44% comes from post-cutoff structures. Pre-cutoff
   complexes are inside Protenix's training set, so their poses may be systematically better and
   degenerate in the SATURATED direction instead, at an unknown rate. 299 of the 364 wave-1 folds
   are pre-cutoff, so wave 1 is the first measurement of it.
3. **Wave-to-wave latency is unmeasured.** Knowing which new structures are degenerate requires
   the fold AND the label build, the coordinate-based sample-index match, and the rescore. Only
   the fold stage has a measured rate (1.0 min/unit/worker); the rest does not.


---

# WAVE 1 — REVISED

The goal is **independent training-complex diversity**, not more regularisation on the existing
set. In-sample selection has improved repeatedly without transfer; cluster coverage is the lever
now. Folding is the critical path and nothing secondary may delay it.

## Step 1 — the validation seed selection

`run_logging/select_val_seeds.py`, algorithm `anchored-coverage-v1`, manifest digest
`adbcc352fb069b63`, written to `training/valfull/seed_manifest.json`.

**This is deliberate outcome-dependent construction of a DEVELOPMENT set.** It reads DockQ to
decide which seeds survive. It is not a simulation of 30 random seeds and no aggregate from it is
comparable to the old first-30 numbers. The final test set stays independent at all 100 seeds.

The algorithm, in order:

1. **Anchor on the oracle.** For every interface, force in the seed carrying its best pose. Makes
   oracle retention exact rather than incidental.
2. **Anchor on the threshold.** For every interface holding poses on both sides of DockQ 0.23,
   force in the seed carrying its best pose BELOW 0.23. Without this a spanning interface can be
   reduced to all-acceptable, deleting the decision the dev set exists to test.
3. **Fill the rest by capped, weighted bin coverage.** Bin edges `[0, .05, .15, .23, .35, .50,
   .70, 1.01]` with 0.23 ON a boundary; weights `[1, 1.5, 3, 3, 1.5, 1, 1]`, highest either side
   of the threshold. Each bin has a quota proportional to its share of the full 100-seed
   distribution, and a bin at quota earns nothing more — which is what stops the objective
   collapsing into variance maximisation.
4. **Joint across interfaces**, mean of the per-interface coverage score, so a 3-interface
   structure gets one compromise subset rather than the best subset for its first interface.
5. **Deterministic.** No RNG. Ties break on the lowest seed number, so a degenerate structure
   whose candidates are identical falls through to seeds 400000-400029.

### Method comparison, measured on all 53 structures / 92 interfaces

| method | bin coverage | histogram L1 | oracle lost | sd kept | >=1 acceptable | spans 0.23 | sd < 0.05 |
|---|---|---|---|---|---|---|---|
| first 30 (status quo) | 0.896 | 0.0267 | 0.0245 | 0.1010 | 79/80 | 47/50 | 47 |
| rank-stratified | 0.951 | 0.0165 | 0.0124 | 0.1087 | 79/80 | 49/50 | 42 |
| max-variance | 0.949 | **0.1045** | 0.0005 | 0.1396 | 80/80 | 50/50 | 35 |
| coverage greedy | 0.966 | 0.0117 | 0.0218 | 0.1050 | 80/80 | 50/50 | 43 |
| **anchored coverage** | **0.973** | **0.0117** | **0.0000** | 0.1070 | **80/80** | **50/50** | 43 |

Max-variance is the cautionary row: near-perfect oracle retention and a histogram nine times
further from the true distribution than anchored coverage, because it keeps the extremes and
discards the middle. Anchored coverage is better than the status quo on every column.

### The selected pool, as a new reference point

| | |
|---|---|
| shipped ipTM selection, mean DockQ | **0.5884** |
| shipped frac_acc | **0.7391** (68 of 92) |
| oracle mean DockQ | **0.7171** |
| oracle frac_acc | **0.8696** (80 of 92) |
| headroom | **0.1287** |
| within-interface DockQ sd | median 0.0545, mean 0.1070 |
| interfaces with sd < 0.05 | 43 of 92 |

**Not comparable to shipped 0.5831 / 0.7391 on the first-30 pool.** Different pose set, chosen
with labels. 50 of the 53 structures get a non-contiguous subset.

## Step 2 — validation keeps all 53 structures at 30 selected seeds

No validation structure is reduced to one seed, including the 26 degenerate ones. This holds all
40 existing clusters and keeps the cases where the generator produces no acceptable pose, which
are themselves a thing the head should not be rewarded for mis-ranking. Costs **96.7 GiB** more
than the superseded plan; frees **492.2 GiB**, retaining 210.9 GiB over 1,590 units.

## Step 3 — training trims, unchanged

The 220 degenerate training structures go from 10 seeds to 1. Frees **565.5 GiB**. Their five
remaining poses still carry PAE supervision AND may carry within-seed relative supervision, so
their ranking signal is NOT assumed to be zero.

## Step 4 — the fold queue

`run_logging/build_wave1_queue.py` -> `training/full_train/wave1_queue.json`.

**356 training structures, 3,560 units.** Per-cluster targets as before (1 pre-only, 2 mixed
preferring post-cutoff, 2 post-only), with candidates ranked by novelty first: a new antigen
cluster scores 4, a new antibody cluster 2, no shared chain sequence 1, then post-cutoff
preference, then smallest tokens. Where two are taken, the pair maximising mutual difference wins.
Every row carries `category`, `cutoff`, `tokens`, `ab_cluster`, `ag_cluster`, `ab_ag_cluster`.

356 rather than 364 because the novelty ranking dedups candidates that differ only by instance.

**The first 50 are the pilot, round-robined across (category, cutoff, token quartile) and
antigen-unique-first** — explicitly NOT the 50 smallest:

| | pilot (50) | full queue (356) |
|---|---|---|
| category/cutoff | mixed-post 2, mixed-pre 20, post-only-post 9, pre-only-pre 19 | 2 / 43 / 12 / 299 |
| token quartiles | 10 / 10 / 14 / 16 | 93 / 86 / 89 / 88 |
| distinct antigen clusters | **50 of 50** | 347 |
| distinct antibody clusters | **50 of 50** | 353 |
| tokens | 134 / 484 / 950 | 134 / 475 / 2289 |

Every pilot structure is a distinct antigen family and a distinct antibody family. After the pilot
the remainder runs smallest-first, since representativeness has already been bought.

**Validation folds:** the 30 post-cutoff structures at 100 seeds (peak 445.2 GiB), then the same
`anchored-coverage-v1` selection down to 30.

## Revised ledger

| | GiB |
|---|---|
| free now | 536.2 |
| + trim 220 degenerate training structures to 1 seed | +565.5 |
| + trim all 53 validation structures to 30 selected seeds | +492.2 |
| **available** | **1,593.9** |
| − training fold, 356 structures x 10 seeds | −619.6 |
| − validation fold, 30 structures x 100 seeds | −445.2 |
| **spare at peak** | **529.1** |
| + trim the new validation structures to 30 seeds | +311.6 |
| **spare after wave 1** | **840.7** |

Fold total **6,560 units, roughly 36 h on 3 cards.** Peak use is 2,543 GiB of the 3,072 GiB quota.

## Blockers before the queue can start

1. **319 selected validation units have no label.** Labels exist only for seeds 400000-400029 and
   the selection reaches outside that range on 50 of 53 structures. These must be built before any
   validation seed is trimmed, or the selection cannot be scored. CPU-only.
2. **The evaluator cannot read a non-contiguous seed set.** `phase_k_eval.py` enumerates
   `range(seed_base, seed_base + budget)` at lines 376 and 1008, asserts at line 1016 that every
   structure covers exactly `budget` seeds, and `build_union_crop_masks.py` does the same. A
   `--seed-manifest` path is needed in both before the selected pool can be evaluated. Does not
   block folding.
3. **Variable-seed training support.** `phase_k_train.py` assumes a uniform seed range and fixed
   optimiser-step quotas under `--rank-cross-seed`. Needed before training on the expanded set,
   explicitly NOT before folding.

## Data-retention safety

Nothing is deleted until its replacement is verified present:

- snapshot the current manifests and a full per-file disk inventory first;
- validation seeds 30-99 stay until the 319 missing labels exist AND the manifest digest is
  frozen (`adbcc352fb069b63`);
- every removal is staged: move to a quarantine path, verify the retained manifest still resolves
  through the downstream tools, then delete;
- the original manifest and an exact list of every removed file are kept;
- no deletion of a sole copy of anything needed to regenerate the validation selection.
