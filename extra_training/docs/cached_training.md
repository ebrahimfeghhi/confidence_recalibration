# Cached training data for the FT3 confidence-head finetune

Proposal and open design questions. Every number below is either **measured** on this machine or
**estimated**, and each is labelled. Nothing here has been run.

**Notation.** A *structure* is one PDB entry. An *instance* is one antibody–antigen pairing within a
structure. A *seed* is one fold: one trunk pass at `N_cycle=10` plus `N_sample` diffusion rollouts at
200 steps. A *unit* is one cached (structure, seed): one `z_trunk` plus that seed's poses.
`ab_ag_cluster` is SAbDab2's transitively merged antibody/antigen cluster; it is the generalisation
unit. Cutoff means the Protenix-v2 training cutoff, **2021-09-30**.

## 0. The proposal in one block

Re-run Protenix's FT3 stage — confidence head only, trunk and diffusion frozen — on structures the
generator was never trained on, and serve it from a precomputed cache.

| | proposed |
|---|---|
| training set | post-cutoff structures in clusters holding ≥1 of them, redundancy-filtered |
| structures | **535** (antibody + polypeptide antigen, cap ≤2 per pairing); all MSA-ready |
| clusters represented | **193** |
| seeds per structure | **10** |
| each seed | one fold at `N_cycle=10`, 200 diffusion steps, `N_sample=5` poses |
| cached units | 5,350 at 10 seeds; **2,675 in the first build at 5** |
| poses cached | 5 per unit, sharing that unit's `z_trunk` |
| cached per unit | `z_trunk`, `s_trunk`, `s_inputs`, the 5 poses, feature masks |
| storage | **848 GB** at 5 seeds, 1,695 GB at 10, of 2,039 free — MEASURED, not z-only (ratio 1.0255 over 55 complete cache files) |
| build cost | 2,675 folds ≈ **6 h** at 5 seeds; 12.1 h at 10 |
| training cost after build | ~1 h per run, versus ~19 h uncached |

### Why this split of the budget

Equal-cost alternatives, all within reach of 2,040 GB:

| design | storage |
|---|---|
| **536 structures × 10 seeds** | **1,695 GB** |
| 1,594 structures × 5 seeds | 1,509 GB |
| 1,594 structures × 6 seeds | 1,811 GB |

Spending on seeds rather than structures, because seed is the axis the variance lives on: 87% of
ipTM variance is between seeds of the same interface, and the hard training examples — confident
representation, wrong pose — are produced by depth, not breadth (§3). The redundancy filter costs
almost nothing in return: the cap keeps every distinct antibody–antigen pairing,
so what is discarded is duplicate crystal forms.

Ten seeds also reaches the thin tail that matters: 41 of 292 PXM interfaces find an acceptable pose
in ≤5 of 100 seeds (§3). At 5 seeds per structure roughly half of those cases would never appear in
the cache at all.

**Seeds are incrementally addable.** Each seed is an independent fold, so extending 10 → 15 later
appends folds without rebuilding anything. The same is true of structures. What cannot be revised
after the build is rollout length, `N_cycle` and `N_sample`.

**What this is for.** Two claims. (1) Calibrating the confidence head on data the generator has not
seen improves selection; the control arm tests it. (2) Training the head on several poses that share
one trunk representation forces it to read the pose rather than the representation; the cache makes
that a one-line change, since 5 poses already share each cached `z`.

---

## 0a. The build plan, and what is ready

Ordered so nothing expensive precedes the thing that could invalidate it. **Working assumption,
pending the memory test: everything fits**, so no token cap is applied yet.

| step | where | cost | state |
|---|---|---|---|
| 0. fix the selection: polypeptide-antigen eligibility, input-tuple dedup accounting, diversity-ordered representatives | CPU | minutes | **done** — 535 structures / 193 clusters, all MSA-ready |
| 1. build Protenix training format: whole RCSB mmCIFs → bioassembly pickles + indices csv | CPU | ~1 h | **running** (`phases/phase_j_prepare_training_data.py`) |
| 2. memory test: forward+backward through the confidence head at 500 / 1,000 / 1,500 / 2,000 / 2,941 tokens | 1 GPU | ~1 h | queued behind the FoldBench fold |
| 3. one cached unit WITH labels, then a numerical replay against the uncached loss | 1 GPU | minutes | not written |
| 4. build the cache, 535 × **5 seeds**, smallest first | 3 GPUs | ~6 h | not queued |
| 5. extend to 10 seeds | 3 GPUs | ~6 h | only if a 5-seed run moves the head |

Step 1 is the prerequisite review round 3 exposed: `label_dict` — ground-truth coordinates in the
model's atom order, the coordinate and representative-atom masks, and the frames the PAE loss aligns
on — is built from the bioassembly dict. Without it there is no loss, so a cache of confidence-head
inputs cannot be trained against.

**Why 5 seeds first.** The open questions are whether the cache is *correct*, not whether it is deep
enough. 5 seeds is 848 GB against 1,695, leaves 1.2 TB of room to be wrong in, and halves the time to
the first real answer. Seeds append without rebuilding.

**Two things the prep found that the plan had wrong.** SAbDab2's per-instance cifs are trimmed and
carry no `pdbx_struct_assembly` block, so Protenix's parser raises on all 535 — the entries have to
come from RCSB. And the instance id, not `<pdb>_<Hchain>_<Lchain>`, is SAbDab2's file stem: 110 of
535 are single-domain antibodies whose light chain is written `+`.

## 0b. The structures, and how they were chosen

The list is frozen and in the repo: **`extra_training/results/cache_train_structures.csv`**, one row
per structure with its cluster, instance, antibody cluster, antigen cluster, token count, chain count
and MSA readiness. It is regenerated deterministically by
**`extra_training/scripts/select_cache_structures.py`** — same inputs, same list.

### The funnel

| step | source | result |
|---|---|---|
| SAbDab2 ML Dataset v0.2.0, antibody–antigen rows | `data/zenodo_splits/splits_final/abag_split.csv` | 15,810 instances |
| minus clusters holding any of our 227 benchmark complexes, minus 4 more on antigen homology | `phase_a_define_train_set.training_clusters()` | **533 clusters** |
| minus the live validation split (40 clusters) | `results/val_split_proposal_clusters.csv` | **493 clusters** / 2,506 structures |
| require an antibody chain AND ≥1 polypeptide antigen | `phase_a:263`'s filter | drops 548 rows |
| keep clusters holding ≥1 post-2021-09-30 structure | `PDBdepo > 2021-09-30` | 193 clusters |
| keep only the post-cutoff structures in them | | 1,594 → see cap |
| cap 2 structures per (antibody cluster, antigen cluster) pairing | `ab_cluster` × `agclusters` | **536 structures / 193 clusters** |

One row per structure: a PDB appearing under two `ab_ag_cluster`s is emitted once.

### Two things a reviewer should check first

**No validation leakage.** `train_clusters.csv` was built against the ORIGINAL 50-cluster phase-B
split and never regenerated when validation moved to 40 clusters on 2026-10-13; 22 of the live 40
were still listed as training. The first version of this selection therefore held 44 structures from
validation clusters, 28 of them structures actually folded into the validation set. Fixed: the
selector subtracts the live split before anything else and asserts none survives, phase B now writes
`train_clusters.meta.json` naming and hashing the validation file it subtracted, and the selector
refuses to run if that hash does not match. Verified on the current list: 0 validation clusters, 0
validation structures.

**Why the cap is per pairing, not per cluster.** `ab_ag_cluster`s are merged transitively, so size
does not measure redundancy: one cluster holds 1,029 structures of about 5 antibodies, another holds
183 structures spanning 108 distinct antibodies. A flat per-cluster cap would be free in the first and
destroy the second. Capping per (antibody, antigen) pairing keeps every distinct pairing.

### What is cached per structure

10 seeds. Each seed is one fold at `N_cycle=10`, 200 diffusion steps, `N_sample=5`, saving
`z_trunk`, `s_trunk`, `s_inputs`, the 5 poses and the feature masks — everything
`run_confidence_head` consumes, so a training step needs no trunk pass and no rollout.

### Cost, by MSA readiness

140 of the 695 have at least one antibody chain with no MSA; recovering them needs uniref30 and
colabfold_envdb re-downloaded (~1 TB transient, 8–13 h), since those were deleted under the old quota.

| set | structures | clusters | folds at 10 seeds | storage | build at 442 folds/h |
|---|---|---|---|---|---|
| MSA-ready now | 555 | 194 | 5,550 | **1,609 GB** | 12.6 h |
| blocked on MSA | 140 | 58 | 1,400 | 94 GB | 3.2 h |
| **all selected** | **695** | **207** | **6,950** | **1,703 GB** | **15.7 h** |

Free quota is 2,039 GB, so the whole set fits. The blocked 140 are small — 94 GB for 20% of the
structures — because their median is 425 tokens against 625 for the ready ones, and `z_trunk` goes as
N². Seeds and structures can both be appended later without rebuilding; rollout length, `N_cycle` and
`N_sample` cannot.

---

## 0c. Still open



Review round 3 (`reviews/round3_2026-10-13_cache.md`). The design stands; these gate the build.

| | finding | blocks |
|---|---|---|
| **high** | the cache stores confidence-head INPUTS only. The loss also needs ground-truth coordinates in model atom order, `coordinate_mask`, `pae_rep_atom_mask`, `frame_atom_index`, `has_frame`, and the PER-POSE symmetric permutation of the label (`protenix.py:1712-1720`). None is in the npz. | the build |
| **high** | backward-pass memory is untested above ~1,000 tokens. 77 of 535 structures exceed it and carry 45% of the bytes; the largest is 2,941 tokens, where one N×N×256 activation is 4.4 GB. **Assumed to fit for now**; step 2 settles it. | caching those 77 |
| medium | deduplication is per PDB id because the fold's unit key is the pdb (`folder.py:784`), which collapses **225** input-distinct instances. Recovering them needs a per-instance job name, not a change to the selection. Representatives are now ordered by input difference rather than alphabetically. | the selection |

Before any of it is generated: a **numerical replay** — one cached unit through the loss, reproducing
the uncached path's loss on the same structure and seed to within floating point. That is what makes
this precomputation rather than a different experiment.

## 1. What we propose to train on

Post-cutoff structures only, in clusters that hold at least one.

| step | result |
|---|---|
| SAbDab2 pool, minus our benchmarks, minus the 50 validation clusters | 483 clusters / 3,140 structure-slots |
| keep clusters with ≥1 post-cutoff structure | 224 clusters |
| keep only the post-cutoff structures within them | **1,594 structures** (2,453 instances) |
| of those, in clusters with no pre-cutoff member at all | 247 |
| of those, in clusters that also hold pre-cutoff structures | 1,347 |

Excluded: 259 clusters / 614 structures that are entirely pre-cutoff, and the 924 pre-cutoff
structures inside the 224 kept clusters.

**Why post-cutoff only.** The confidence head is trained, in stock FT3, on poses the frozen generator
produces for structures that generator was trained on. On those, a confident trunk representation
nearly always accompanies a correct pose, so the head learns that correlation. At deployment on a
novel antigen the trunk still commits confidently but can commit to the wrong epitope, and the head
has no training signal for that case. Calibrating on structures outside the generator's training
window is the only way those examples enter the training set.

The 924 excluded pre-cutoff structures are reserved as a **control arm**: a matched run trained on
seen data is what separates "held-out data is what matters" from "any antibody finetune helps".

### Redundancy filter

`ab_ag_cluster`s are merged transitively, so cluster size is not a measure of redundancy. One cluster
holds 183 structures spanning 108 distinct antibody clusters; another holds 1,029 structures
spanning 5. A flat per-cluster cap would delete biology in the first and nothing in the second. The
filter below instead caps structures per (antibody cluster, antigen cluster) pair.

| cap per (ab, ag) pair | structures | median tokens |
|---|---|---|
| ≤1 | 567 | 534 |
| ≤2 | **665** | 524 |
| ≤3 | 704 | 514 |
| uncapped | 1,594 | 572 |

567 is the number of distinct antibody–antigen pairings with any post-cutoff structure, so ≤1 keeps
one example of each. ≤2 adds 98 structures; ≤3 adds 39 more. The remaining 890 up to 1,594 are
duplicate crystal forms of pairings already represented.

(Storage figures elsewhere are computed over the 663 of 665 that carry a token count in
`abag_split.csv`; two have none, so their size is unknown rather than zero.)

**Every cluster survives the filter**, since a cluster qualifies only by holding a post-cutoff
structure, which is itself a pairing. Structures kept per cluster at cap ≤2:

| structures kept | clusters |
|---|---|
| 1 | 110 |
| 2 | 52 |
| 3 | 24 |
| 4 | 11 |
| 5 | 6 |
| 6 | 6 |
| 7–93 | 15 |

Half the clusters contribute one structure. The 93 is the cluster holding 108 distinct antibody
clusters — the case a flat per-cluster cap would have destroyed.

---

## 2. Why cache

### Cost (estimated)

In FT3 the trunk and the diffusion module are frozen (`protenix.py:283-287`, `:1750`), so `z_trunk`,
`s_trunk`, `s_inputs` and the rollout pose are pure functions of (structure, seed). Caching them is
precomputation, not approximation.

| | per training example | 10 passes over 1,594 structures (15,940 examples, 62 optimizer steps at batch 256) |
|---|---|---|
| uncached | ~13 s/GPU | **~19 h** on 3 GPUs |
| cached | <1 s/GPU | **~1.1 h** on 3 GPUs |
| cache build, one pass | — | 1 fold per unit |

The ~13 s is Protenix's own FT3 figure for an A100-80G (`docs/training_inference_instructions.md:252`),
where one step is 256 examples on 256 GPUs. Two corrections roughly cancel: our RTX PRO 6000 cards are
faster than an A100, and pinning `N_cycle=10` instead of the random 1–10 (mean 5.5) roughly doubles the
trunk work. **Not measured here** — all three GPUs were folding when this was written. One timed
micro-step replaces it.

Cross-check from our own folds: 442 folds/h on 3 cards (measured, §5), i.e. ~24 s per fold per card,
where a fold runs the same 10-recycle trunk plus 1,000 diffusion step-samples (200 steps × 5 samples).
A training example needs 20 step-samples, 2% of that, so it is trunk-dominated at 11–13 s.

### Stability

Folds do not reproduce bit-for-bit on this hardware. Uncached, every training run regenerates its own
poses, so two runs differ in both the head *and* the data they saw. With a cache, the inputs are
identical across runs and arms, and any difference in validation is attributable to the head alone.
This also makes the control arm a controlled comparison rather than two samples from different
distributions.

### What caching freezes

Decided at build time and immutable afterwards: seeds per structure, rollout length (20-step
mini-rollout as in FT3, or 200-step as at inference), `N_cycle`, crop, MSA subsampling, templates.
Experiments downstream of the trunk stay cheap: poses per `z`, learning rate, step count, loss
weights, EMA, held-out versus seen data.

---

## 3. Seed diversity is the reason for multiple seeds per structure

ipTM is overwhelmingly a seed property — 87% of its variance sits between seeds of the same
interface. The variation the head must learn to explain is therefore *within* a target, across seeds,
not between targets.

How many of 100 seeds contain an acceptable pose (DockQ ≥ 0.23), per interface:

### Validation, 91 interfaces at 100 seeds

| seeds of 100 with an acceptable pose | interfaces | % | expected seeds to first hit |
|---|---|---|---|
| 0 | 12 | 13% | never |
| 1 | 1 | 1% | ~100 |
| 2 | 1 | 1% | ~50 |
| 3–5 | 1 | 1% | ~20–33 |
| 6–10 | 5 | 5% | ~10–17 |
| 11–25 | 7 | 8% | ~4–9 |
| 26–50 | 3 | 3% | ~2–4 |
| 51–100 | 61 | 67% | ~1–2 |

### PXM-2025, 292 interfaces at 100 seeds

| seeds of 100 with an acceptable pose | interfaces | % | expected seeds to first hit |
|---|---|---|---|
| 0 | 54 | 18% | never |
| 1 | 12 | 4% | ~100 |
| 2 | 5 | 2% | ~50 |
| 3–5 | 24 | 8% | ~20–33 |
| 6–10 | 25 | 9% | ~10–17 |
| 11–25 | 36 | 12% | ~4–9 |
| 26–50 | 16 | 5% | ~2–4 |
| 51–100 | 120 | 41% | ~1–2 |

**41 of 292 PXM interfaces (14%) find an acceptable pose in 5 or fewer of 100 seeds**, and 12 in
exactly one. A single cached seed per structure samples one draw from this distribution and will
miss, for those interfaces, the only pose that matters.

The corresponding class fractions, same data:

| | 1 seed | 100 seeds |
|---|---|---|
| validation, acceptable | 61.6% | 72.5% |
| validation, decoy | 4.1% | 14.3% |
| PXM, acceptable | 38.8% | 50.7% |
| PXM, decoy | 4.5% | 30.8% |

The decoy fraction — an acceptable pose exists but top-ipTM picks below it — rises 3.5× to 7× with
depth. Confident-but-wrong examples, which are exactly what the head lacks training signal for, are
produced by *adding seeds*, not by adding structures.

---

## 4. Storage budget

Quota raised to **3,072 GB**; 1,032 GB in use; **2,040 GB free** (measured, `quota -s`).
`z_trunk` is `[N, N, 256]` bf16, so a unit costs N² × 512 bytes plus ~0.2 MB of poses; 97% of a unit
is `z_trunk` (measured: 94.7 MB of 97.5 MB at N=430).

### All post-cutoff structures, one job each

| seeds | storage | fits in 2,040 GB |
|---|---|---|
| 1 | 302 GB | yes |
| 3 | 905 GB | yes |
| 5 | 1,509 GB | yes |
| 6 | 1,811 GB | tight |
| 10 | 3,018 GB | no |

### Redundancy-filtered

| filter | structures | 5 seeds | 10 seeds |
|---|---|---|---|
| ≤1 per (ab, ag) | 567 | 681 GB | 1,362 GB |
| ≤2 | **665** | 771 GB | **1,542 GB — proposed** |
| ≤3 | 704 | 801 GB | 1,601 GB |

### Size is not concentrated in a tail

| token cap | structures kept | 10 seeds, all post-cutoff | share of bytes |
|---|---|---|---|
| ≤600 | 1,167 (73%) | 1,437 GB | 48% |
| ≤800 | 1,443 (91%) | 2,064 GB | 68% |
| ≤1000 | 1,526 (96%) | 2,387 GB | 79% |
| none | 1,594 | 3,018 GB | 100% |

Largest structure is 2,941 tokens = 4.4 GB for one seed.

### Levers not yet used

- **int8 `z_trunk`** with a per-row scale would halve every figure. Untested; the test is to quantise
  a cached `z`, rescore with the existing confidence path, and compare ipTM to the stored value.
- **Validation cache**: 704 GB for 53 structures × 100 seeds. Dropping 20 of the 100 seeds frees
  ~141 GB. Those seeds become permanently unrescorable, since folds do not reproduce, and published
  baseline tables quote 100 seeds.

### Generation cost

At the measured 442 folds/h on 3 GPUs:

| design | folds | build time |
|---|---|---|
| **665 structures × 10 seeds (proposed)** | **6,650** | **~15 h** |
| 1,594 × 5 seeds | 7,970 | ~18 h |
| 1,594 × 10 seeds | 15,940 | ~36 h |

---

## 5. Measured values and provenance

| quantity | value | source |
|---|---|---|
| fold throughput, 3 GPUs | 442 folds/h sustained, 160/h including gaps | mtimes of 5,300 `run.json` in `val2_fold/poses` |
| cached unit size at N=430 | 97.5 MB, of which `z_trunk` 94.7 MB | `val2_fold/trunks/22gi/free_400053.npz` |
| npz read throughput | 1.0–3.1 GB/s | timed loads, 4 files |
| validation cache total | 704 GB, 5,300 units | `du -sh val2_fold/trunks` |
| quota | 3,072 GB limit, 1,032 GB used | `quota -s` |
| `/home` capacity | 14 TB, 4.8 TB free | `df -h` |
| confidence head size | 19.8 M params, 4.3% of 464 M | parameter count of `protenix-v2.pt` |
| token distribution, 1,594 post-cutoff | median 572, mean 563, p90 784, max 2,941 | `abag_split.csv`, Hseq+Lseq+polypeptide antigens |
| class fractions vs seed depth | `docs/seed_scaling_impact.md` | pooled free tables, `tier != suspect` |

Cache contents, from an existing validation npz: `z_trunk [N,N,256] uint16`, `s_trunk [N,384]`,
`ci_s_inputs [N,449] f32`, `ci_coordinate [N_sample,N_atom,3] f32`, plus `distogram_rep_atom_mask`,
`atom_to_token_idx`, `atom_to_tokatom_idx`, `asym_id`, `has_frame`, `is_ligand`, and the RNG state.
This is everything `run_confidence_head` consumes, so rescoring needs no trunk and no diffusion.

---

## 6. Questions for review

1. **Is caching worth it at all?** It costs 15–36 h of generation and 0.7–1.5 TB to turn a 19-hour
   training run into a 1-hour one. If the answer is "run 10 optimizer steps once and look", the cache
   never amortises. If it is "sweep learning rate, step count, 1-vs-k poses, and run a control arm",
   it pays for itself on the second run. Which regime should we plan for?

2. **Given 2,040 GB, what is the better split of the budget:** all 1,594 structures at 5 seeds
   (1,509 GB), or 665 redundancy-filtered structures at 10 seeds (1,542 GB)? Same cost. We propose
   the second (§0). The first buys target diversity, the second buys seed diversity per target. §3
   argues the failure mode is within-target and between-seed; §1 argues generalisation is a cluster
   property and we have only 224 clusters either way.

3. **How many seeds per structure is enough?** This is measurable before spending anything: the
   validation cache is 53 structures × 100 seeds, so the fraction of a structure's error distribution
   captured at k = 1, 3, 5, 10, 20 can be computed directly. Should that run first?

4. **20-step mini-rollout poses or 200-step inference poses?** FT3 trains on 20-step rollouts; the
   head is deployed on 200-step poses. Training on what it will score seems right, but it is a
   deviation from the stage we claim to be re-running. We propose 200-step.

5. **Crop.** FT3 crops to 768 tokens before the trunk. A cache is uncropped, so a cached `z` was
   computed with full context. Is cropping a cached `z` afterwards acceptable, or does training
   uncropped (N up to 2,941) create a memory problem in the head's 4 pairformer blocks?

6. **Does pinning `N_cycle=10` bias the result?** We infer at 10 and the cache is a depth-10
   representation, so it matches deployment, but the released head was trained across depths 1–10.
   An improvement could come from depth-matching rather than from held-out data. The control arm
   trains at `N_cycle=10` too, which we believe handles it.

7. **Should the control arm be cached as well?** Training on the 924 pre-cutoff structures is what
   makes the central claim a claim. Uncached it is another 19-hour run; cached, after filtering,
   perhaps 400 structures and ~250 GB at 5 seeds.

8. **Is int8 `z_trunk` safe?** It halves every storage figure. The test is cheap and described above.

9. **Is it acceptable to drop 20 of the validation set's 100 seeds** to free ~141 GB, given folds do
   not reproduce and the published baselines are at 100 seeds? We would rather not, but it is
   available.

10. **One `z` per seed is forced** — MSA subsampling differs per seed, so k seeds means k trunk
    representations and k × 167 MB. There is no way to get seed diversity at the cost of pose
    diversity. Is there a representation of `z` cheaper than N² × 512 bytes that the head would
    accept?
