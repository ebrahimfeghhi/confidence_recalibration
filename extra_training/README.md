# extra_training — a SAbDab2-derived training pool for the confidence head

Training data for fine-tuning Protenix v2's confidence head, held out from our two evaluation
benchmarks at the cluster level.

**Headline: 557 training clusters / ~2,900 structures**, with no CDR-H3 or antigen homology to any
FoldBench or PXM-2025 complex.

---

## Why we need this

Protenix's confidence head is trained in a way that never sees the kind of pose it is asked to score
at inference. Per training example the trunk runs, a **20-step mini-rollout** produces one structure
under `torch.no_grad()`, that structure is detached, and the head predicts its error against the same
example's ground truth (`Protenix/protenix/model/protenix.py:1675-1740`, `loss.py:1634`). AlphaFold 3
is identical in design. No held-out structure is ever used for confidence training.

At inference the head scores 200-step samples, 5 per seed, on complexes it has never seen — including
confidently wrong poses, which the mini-rollout rarely produces. The pose distribution it was
calibrated on is not the one it is deployed on.

That matters because selection, not sampling, is our measured bottleneck. From
`experimental_results/budget_decomposition.md`: a perfect selector at 10 seeds already matches real
pair_iptm at 50 seeds on 80–87% of solved interfaces, and the gap between what ipTM picks and the best
pose present is ~0.15 DockQ at *both* budgets. More seeds does not shrink it.

## The stage we would be re-running

Protenix documents its recipe in `Protenix/docs/training_inference_instructions.md:238-250`. The
confidence head is trained jointly at a token weight throughout, and then in a **dedicated
confidence-only final stage**:

| hyperparameter | Initial | FT 1 | FT 2 | **FT 3** |
|---|---|---|---|---|
| `train_crop_size` | 384 | 640 | 768 | 768 |
| `alpha_confidence` | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `alpha_diffusion` | 4.0 | 4.0 | 4.0 | **0** |
| `alpha_distogram` | 0.03 | 0.03 | 0.03 | **0** |
| `alpha_pae` | 0 | 0 | 0 | **1.0** |
| `train_confidence_only` | False | False | False | **True** |
| throughput (A100-80G) | ~12 s/step | ~30 | ~44 | **~13** |
| peak memory | ~34 GB | ~35 | ~48 | **~24 GB** |

Three consequences:

- **ipTM is learned almost entirely in FT3.** `alpha_pae` is 0 for the first three stages and only
  turns on at 1.0 in FT3; ipTM is derived from PAE. The joint confidence weight of 1e-4 against a
  diffusion weight of 4.0 means joint training barely shapes the head.
- **FT3 runs on a converged trunk.** Diffusion and distogram losses are off and
  `train_confidence_only=True`, so only the head learns — on a trunk that has already fit the training
  set. The head's whole PAE calibration is learned in the regime where the trunk is accurate.
- **It is cheap and separable.** ~13 s/step at ~24 GB, and `finetune_optim_configs` sets
  `max_steps: 20000` — roughly 72 A100-hours, confidence head only. What we want is **their own FT3 with
  a different pose distribution**, not a new procedure. Crop size 768 sits just under our ~800-token
  median, so cropping is minimal.

## What is in here

```
extra_training/
  data/
    zenodo_splits/
      splits_v0.2.0.tar.gz        SAbDab2 ML Dataset v0.2.0, Zenodo record 22019991, 849 MB
      splits_final/               the four CSVs (the 15,812 cifs are still inside the tarball)
        abag_split.csv            15,810 rows, 63 cols, split column `ab_ag_split`
        ab_split.csv              same rows, split column `ab_split`
        abag_split_sd.csv         3,382 single-domain rows, subset of abag_split.csv
        ab_split_sd.csv           same, with `ab_split`
    rcsb_unlisted/                39 mmCIFs from RCSB, for the complexes SAbDab2 omits
  results/
    excluded_homolog_clusters.csv  the clusters dropped by sequence, with query and identity
  README.md                        this file
```

Code lives with the rest of the pipeline, not here:

- `extra_training/phases/phase_a_define_train_set.py` — builds the pool and runs the exclusion
- `diffusion_commit_clean_v2/budget_decomposition.py` — the selection-vs-coverage measurement

Tools fetched for this work are in `tts_abag/tools/`: `mmseqs/` (17-b804f),
`hmmer-3.1b2-linux-intel-x86_64/`, and `anarci_hmms/` (ANARCI's `ALL.hmm`, re-pressed with 3.1b2).

## Which file to use, and why

`abag_split.csv`. It and `ab_split.csv` hold the **same** 15,810 instances; only the train/test line
differs, and `abag_split`'s is far stricter:

| split | held out on | train antigen clusters | test | shared |
|---|---|---|---|---|
| `ab_ag_split` | antibody **and** antigen | 639 | 248 | **0 (0% of test)** |
| `ab_split` | antibody only | 733 | 314 | 160 (**51% of test**) |

Under `ab_split`, half the test antigens appear in training bound to a different antibody. Since our
failure mode is picking the wrong *surface on the antigen*, a head trained that way could recall an
epitope rather than generalise. `abag_split` forecloses it.

The `_sd` files are a subset (single-domain, `type` in SD-H/SD-L), with identical split labels —
verified 688/688 test and 2,694/2,694 train agree. Use `abag_split.csv` and filter on `type`; do not
concatenate, or the 3,382 single-domain instances are counted twice.

## Phase-by-phase walkthrough

Written for a reader without a structural-biology background. Each phase says what it is for, what it
produced, and how we know the output is right. The sections after this one give the same ground in
full detail; this is the orientation.

### Vocabulary, once

- An **antibody** is a protein shaped like a Y, built from a **heavy** and a **light** chain. Only the
  tips of the Y touch anything.
- The **antigen** is what it grabs — typically a piece of a virus or a cell-surface protein.
- Every protein is a string of letters, one per amino acid. **Sequence identity** is the fraction of
  positions that match once two strings are aligned: 0.97 means 97% of letters agree.
- The tip of the Y is the **variable domain**; inside it one loop varies far more than the rest, the
  **CDR-H3**. It is the best single fingerprint of what an antibody binds.
- A **complex** is one antibody plus its antigen. Our benchmarks are 227 complexes.
- An **instance** is one antibody-antigen pairing. One PDB entry can hold several, which is why
  instance counts exceed structure counts throughout.

### Phase A — build the training pool

**For.** Retraining the confidence head needs antibody-antigen structures. If any of them also appear
in FoldBench or PXM, our reported DockQ is meaningless, because the model would be tuned on its own
test set. So the pool is SAbDab2 minus everything that touches the benchmarks.

**How SAbDab2 groups structures.** Three levels, no code and no documented thresholds shipped, so the
scheme was read back off the data (full evidence in *How SAbDab2 clusters* below):

| column | clusters, whole file | groups |
|---|---|---|
| `cdrh3_cluster` | 4,635 | antibodies by CDR-H3 at 95% identity |
| `agclusters` | 1,631 | antigens, typed by prefix: `P` protein, `E` peptide, `D` DNA, `R` RNA |
| `ab_ag_cluster` | 2,641 | the complex — the one we use |

The complex-level cluster behaves as a **connected component**: two structures share a cluster if
their antibodies cluster together *or* their antigens do, chained transitively. Verified on the whole
15,810-row file — 0 of 4,285 antibody clusters are split across complex clusters, and only 9 of 1,631
antigen clusters are. (Counts above are for the whole file; the sections below quote the
polypeptide-antigen subset, which is smaller.)

That makes clusters conservative, which is what a holdout wants, and lumpy: median 2 structures,
largest 2,812.

Clusters, not structures, are the unit throughout. Two structures in one cluster are near-duplicates,
so they are one data point.

**Filter 1, by name.** 188 of the 227 benchmark complexes have a row in SAbDab2. Look up each one's
`ab_ag_cluster` and drop it: **140 clusters removed**, fewer than 188 because several benchmark
complexes share a cluster.

**Filter 2, by sequence, for the 39 that are not listed.** The other 39 benchmark complexes have no
row at all — no cluster to delete. Absence is not safety: a training cluster can carry *the same
antigen* bound by a different antibody, and training on that means studying the exact surface we then
test on. Since there is no ID to match, we match spelling. For each of the 39: fetch the deposited
mmCIF from RCSB, take every polymer chain, use `hmmscan` against ANARCI's own HMMs to decide which
chains are antibody (the same recognizer SAbDab2 used), then search the rest against the pool's
antigens and the antibody chains against the pool's antibodies. Thresholds are **ours, not
SAbDab2's**: 40% identity against protein antigens, 80% against peptides (short strings match by
coincidence far more easily), 95% over the antibody variable domain, 50% coverage, identity measured
over the shorter sequence. 40% sounds permissive, but at that level two proteins are usually the same
protein from another species or variant — the same target in practice.

**4 clusters removed**, all through the antigen search:

| cluster | caught by | identity |
|---|---|---|
| 398 | 9isf | 0.973 |
| 1496 | 8xnh | 0.966 |
| 1497 | 9kur | 0.827 |
| 499 | 8zp2 | 0.578 |

Nothing was caught by the antibody or peptide searches, so no training cluster shares an antibody with
an unlisted benchmark complex.

**Filter 3, CDR-H3 near-identity.** The mirror question to filter 2: do we train on a benchmark
*antibody*? The same antibody against a different antigen still leaks. Compare every training
antibody's CDR-H3 against every benchmark antibody's; drop at 95% or above. **0 clusters removed** —
evidence, rather than an assumption, that filters 1 and 2 were already thorough.

**Out.** **557 clusters / 4,555 instances / 2,886 unique PDB entries.**

### Phase B — hold out a validation set

**For.** To tell whether retraining helped, we need structures the retrained head never saw. Held out
by **cluster**, never by structure: a near-duplicate split across the boundary would make the head
look good for the wrong reason.

**Why not 50 at random.** Cluster sizes are lopsided — cluster 171 alone holds 1,001 structures. A
random 50 either lands on 50 tiny clusters (about 1% of the data, too small to measure) or catches a
giant and validation becomes one system studied a thousand times: noisy, and informative only about
whichever antigen dominated.

**How instead.** Give each cluster the median deposition date of its structures; cut the clusters into
5 date bins of equal cluster count, so each bin is one era; draw 10 clusters per bin, 50 total, 20,000
times; reject any draw where one cluster exceeds a set share of the total; keep the draw landing
closest to 10% of the pool. Fixed seed, so it regenerates exactly.

**Out.** **50 clusters / 291 structures — 10.1% of the pool's 2,886 PDB entries.**

| era, by median deposition date | clusters | structures |
|---|---|---|
| 1993-09 to 2011-04 | 10 | 91 |
| 2013-05 to 2018-02 | 10 | 21 |
| 2018-03 to 2021-01 | 10 | 68 |
| 2021-07 to 2023-05 | 10 | 84 |
| 2023-11 to 2026-01 | 10 | 27 |

Largest validation cluster: 39 structures, 13.4% of validation — material, not dominant.

**A useful side effect.** Protenix v2's own training data stops at 2021-09-30. **17 of the 50 clusters
(78 structures) have a median date after that cutoff.** The date binning was not aimed at this, but it
gives a second, stronger check: improvement on structures Protenix itself never saw.

**What actually gets folded.** Not all 291 — folding is the expensive part and within-cluster
structures are near-duplicates. We ask for **2 representatives per cluster** and get them where they
exist:

| | clusters | structures |
|---|---|---|
| contributed 2 representatives | 29 | 58 |
| contributed 1 (only one available) | 21 | 21 |
| **total** | **50** | **79** |

The 21 are clusters whose instance count comes from repeats inside a single PDB entry; the
representative picker deduplicates to distinct PDB entries, so there is only one structure to take.
Where a cluster gives 2, its structures are sorted by size and two evenly spaced picks are taken, so
one is small and one large rather than both being tiny partial constructs.

**79 structures x 20 seeds = 1,580 folds**, roughly 16 h on two GPUs.

## How SAbDab2 clusters (reverse-engineered — no code ships with the record)

1. Antibodies clustered by CDR sequence → **1,914** `ab_cluster`
2. Antigens clustered by sequence → **631** antigen clusters
3. Any two entries sharing an antibody cluster **or** an antigen cluster are merged, transitively →
   **481** `ab_ag_cluster` (polypeptide-antigen subset)
4. Whole clusters are assigned to train or test, never split

Evidence for step 3: zero `ab_cluster`s and zero antigen clusters span more than one `ab_ag_cluster`,
yet one `ab_ag_cluster` holds up to 642 antibody clusters and 48 antigen clusters. Only transitive
merging produces that.

**Consequence — the pool is extremely unbalanced.** Cluster 171 contains 1,001 distinct PDB structures
that all carry CDRH3 `VRSIYYYGSSPFDF`: one antibody solved a thousand times. With cluster 2 (850
structures) it accounts for 1,851 of 3,754 structures. **Any training must sample per cluster, not per
structure, or those two dominate the gradient.**

A useful side effect: because an antigen cluster never spans `ab_ag_cluster`s, excluding by
`ab_ag_cluster` makes antigen-disjointness automatic. Verified — a stricter pool that additionally
forbids sharing an antigen cluster with any test complex returns identical counts.

## Building the pool

### Step 1 — exclude by cluster ID (188 of our 227 complexes)

Our benchmark complexes are themselves PDB entries, and SAbDab2 catalogues essentially every antibody
structure through 2026-07, so most are already rows with a cluster assigned. Matching on the 4-char
code (`9n6e` → `pdb_00009n6e`) places **188 of 227**, occupying **140 clusters**.

Read the cluster from **every** row a complex appears in, whatever its antigen type. Reading only the
polypeptide-antigen rows misses clusters whose rows for that complex are apo (`agtypes` blank) — those
rows do carry a cluster ID, and skipping them left 61 clusters unexcluded.

Remaining pool: **561 clusters / 2,924 structures / 4,598 instances** (rows whose `agtypes` contains
PROTEIN or PEPTIDE — the DockQ-scoreable ones).

Peptide antigens are **kept**: DockQ applies to any polypeptide chain, and our own benchmarks contain
peptide antigens (13 FoldBench, 3 PXM), so excluding them from training while testing on them would
make no sense.

### Step 2 — exclude the other 39 by sequence

39 of our complexes have **no row in SAbDab2 at all** — verified genuinely absent (zero substring hits
across all four CSVs, no cif either) and not a recency issue (the file spans depositions to
2026-07-22). They include **9n6e and 9x1h**, our two main case-study targets, and a nine-entry `8w5*`
series:

```
8f5n 8fxv 8r4u 8vzo 8w5d 8w5e 8w5i 8w5o 8w5p 8w5q 8w5t 8w5u 8w5v 8xnh 8zp2
9dm7 9e7k 9e7o 9eyd 9fjf 9g4f 9gry 9gs3 9h01 9hi4 9hi6 9ho5 9i20 9isf 9jo7
9k94 9kmp 9kms 9knx 9kur 9ms8 9n6e 9puo 9x1h
```

With no cluster to look up, they are excluded by sequence instead:

- **Structures come from RCSB**, not our local reference cifs. `chain_roles.csv` and our local copies
  disagree on chain naming — 9n6e's roles name chains E/F/G while its local cif holds A,B,C,D,H,I,L —
  so the two cannot be cross-referenced to isolate "the antigen".
- **Every polymer chain is taken as a candidate antigen** unless `hmmscan` types it as an antibody
  domain. This sidesteps the chain-mapping problem rather than solving it, and over-including can only
  drop *more* training clusters, which is the safe direction. 177 polymer chains → 96 antibody
  (H 60, K 28, L 8), 81 candidate antigens from all 39 complexes.
- **CDR-H3 via ANARCI, IMGT 105–117** — SAbDab2's own definition. Recovered 60/60 heavy chains across
  35 complexes; loop length 5–23, median 14.

Thresholds, searched with `mmseqs easy-search --seq-id-mode 2 -s 7.5` (identity over the shorter
sequence, so a 15-residue peptide against a 500-residue protein is not scored as trivially dissimilar):

| search | threshold | coverage | clusters hit |
|---|---|---|---|
| candidate antigen vs `PROTEIN` antigens | 40% | 0.50 | **4** |
| candidate antigen vs `PEPTIDE` antigens | 80% | 0.50 | 0 |
| **CDR-H3 vs `CDRH3` column** | **95%** | 0.90 | **0** |
| CDR-H3, brute force cross-check | 95% | — | 0 (agrees) |
| whole variable domain | 95% | 0.50 | 13 — *not used, see below* |

Targets include both `agresolvedseqs` and `agexpectedseqs`: resolved sequences have unmodelled gaps a
homolog could hide in, and searching both is strictly more conservative.

**The four real drops** — clusters 1496, 1497, 398, 499 — triggered by `8xnh_A` (96.6%), `9isf_A`
(97.3%), `9kur_A` (82.7%), `8zp2_A` (57.8%).

### Final pool

**561 → 557 clusters.**

## Two methodological points worth keeping

**The whole variable domain is the wrong antibody comparison.** At 95% it drops 13 clusters, and
splitting the queries by chain type shows why: **heavy chains hit 0, light chains hit all 13.** V-kappa
and V-lambda frameworks are germline and ~95% identical across unrelated antibodies, so the hits are
framework conservation, not shared specificity. CDR-H3 — the short hypervariable loop that does the
binding, and what SAbDab2's `cdrh3_cluster_95` is built on — is the criterion that matches their
method, and it finds nothing. Two antibodies can sit at 95% across VH with unrelated CDR-H3s, and
conversely share a near-identical CDR-H3 on different frameworks and fall below 95% on VH; that second
case is the leak a VH comparison misses. The 13 light-chain drops are **not** applied.

**ANARCI needs HMMER 3.1b2 here.** ANARCI parses `hmmscan`'s text output with Biopython's
`Hmmer3TextParser`, which cannot read HMMER 3.4's format — the HSP coordinates come back `None`, every
hit is dropped, and no domain is numbered (silently: `run_anarci` just returns no domains). The system
HMMER is 3.4. So `tools/hmmer-3.1b2-*` and a locally re-pressed `tools/anarci_hmms/ALL.hmm` are used
via `hmmerpath=` and a patched `HMM_path`, leaving the Protenix venv untouched. Verified against
SAbDab2's own annotation: CDR-H3 for their first row comes back `VRRAFDSDVGFAS`, matching their
`CDRH3` column exactly.

## Reproducing

```bash
cd diffusion_commit_clean_v2
/home/ebrahim/Protenix/.venv/bin/python extra_training/phases/phase_a_define_train_set.py \
    --out ../extra_training/train_val_splits/excluded_homolog_clusters.csv
```

Add `--no-fetch` to use the cached RCSB cifs rather than re-downloading. Thresholds are
`--prot-id 0.40 --pep-id 0.80 --cdr-id 0.95`.

## Dates and contamination

Our model is `protenix-v2` (464 M params), **wwPDB training cutoff 2021-09-30**
(`Protenix/README.md:65`, `docs/supported_models.md:17`; the training index is
`weightedPDB_indices_before_2021-09-30_...csv.gz`). Protenix also ships a 2025-06-30-cutoff model, but
it is the smaller 368 M base line, not v2.

The SAbDab2 split is **cluster-based, not temporal** — its test set spans 1984 onward, so it does not
by itself protect against Protenix having seen a structure. Date is an independent axis:

| | clusters | structures |
|---|---|---|
| post-cutoff by date, cluster clean of our benchmarks | 251 | 1,546 |
| …and no pre-cutoff structure anywhere in the cluster | **163** | **287** |

84% of post-cutoff structures sit in clusters that also hold pre-cutoff members, so Protenix trained on
a near-identical complex. Only 163 clusters are genuinely novel systems.

**For training the head this filter is probably unnecessary** — the point is novel *poses*, not novel
complexes, and the head has never seen a 200-step inference pose of any complex. **For evaluation it is
essential.** There is a real counter-argument: on pre-cutoff complexes the trunk is well fitted, so the
head would again learn "error when the trunk is confident and right", the condition we are trying to
escape. Worth resolving empirically — train with and without, weight pre-cutoff down, and compare on
the held-out novel clusters.

Note also that both our benchmarks are post-cutoff. Their difference is homology, not date: 36 of 98
matched FoldBench complexes sit in clusters with a pre-cutoff relative, against 7 of 90 for PXM-2025.

## Outstanding

- **CDR-L3 for four light-chain-only complexes.** `8r4u`, `9gry`, `9gs3`, `9i20` have no heavy chain in
  the deposited entry, so CDR-H3 is undefined and they are unchecked at the antibody level (their
  antigens were checked and did not hit). This also explains their absence from SAbDab2: an entry with
  no heavy chain cannot be numbered at IMGT 5–123, so it fails their ML-grade filter by construction.
- **Why the other 35 are absent.** SAbDab2's ML-grade filter is reported to require X-ray or cryo-EM,
  non-pathogenic antibodies, resolution ≤3.5 Å, and no residue anomalies at IMGT 5–123. Checking which
  filter each tripped is cheap and gives a methods sentence — and if resolution is the common cause,
  the training pool is systematically higher quality than our test benchmarks, which is a train/test
  distribution shift worth stating either way.
- **Thresholds are ours, not theirs.** SAbDab2 publishes no clustering code and documents no
  thresholds beyond what `cdrh3_cluster_95` implies. The 188 complexes excluded by cluster ID use their
  criterion; the 39 excluded by sequence use ours. That distinction belongs in the methods.
- **Cluster-balanced sampling** is not optional — see cluster 171 above.
