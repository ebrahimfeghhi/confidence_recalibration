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

# WAVE 1 — the agreed plan

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

## Ledger

| | GiB |
|---|---|
| free now | 536.2 |
| + trim 220 degenerate training structures to 1 seed | +565.5 |
| + trim 26 degenerate validation structures to 1 seed | +330.2 |
| + trim 27 validation structures to 30 seeds | +258.7 |
| **available** | **1,690.6** |
| − wave 1 fold | −619.6 |
| **spare after wave 1** | **1,071.0** |

Nothing is deleted anywhere in wave 1. Every cut is a seed trim, so every cluster that has any
folded structure keeps coverage: 194 of 493 in training (plus the 364 new folds) and 40 of 40 in
validation.

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
