# Confidence-head recalibration on an out-of-distribution pose distribution — methods

Final procedure as implemented. Every count below was read back from the artifact that produces it,
not from notes. Steps not yet executed are marked **[pending]**.

---

## 1. Model

Protenix v2 (`protenix-v2`, 464 M parameters), checkpoint `protenix-v2.pt`. Its wwPDB training cutoff
is **2021-09-30** (`README.md:65`, `docs/supported_models.md:17`). All inference runs in `bfloat16`,
the documented default, with `N_cycle = 10` and `N_sample = 5`.

Protenix trains its confidence head in a dedicated final stage, FT3. `alpha_pae` is 0 in the Initial,
FT1 and FT2 stages and becomes 1.0 only in FT3, so ipTM — which is derived from PAE — is learned
almost entirely in that one stage, on a converged trunk. In FT3 the trunk is frozen three independent
ways: `loss_fns = {}` (`loss.py:1648`), the trunk modules are set to `.eval()`
(`protenix.py:283-287`), and `s_inputs`, `s_trunk` and `z_trunk` are each `.detach()`-ed under
`stop_gradient: True` (`confidence.py:179-182`, `configs_base.py:353`). A given seed therefore
produces bit-identical poses before and after FT3: recalibration can change the score but not the
pose.

Per training example, Protenix (like AlphaFold 3) runs a 20-step mini-rollout under `torch.no_grad()`,
detaches the resulting structure, and asks the head to predict its error against that same example's
ground truth. No held-out structure is ever scored during training. That is the gap this work targets.

## 2. Training and validation pool

**Source.** SAbDab2, Zenodo record 22019991, file `abag_split.csv` (antibody–antigen split).

**Unit of record.** A SAbDab2 *instance* is one antibody–antigen pairing within a PDB entry, so a PDB
entry may contribute several instances. Counts are given in both units throughout.

**Inclusion.** Rows whose `agtypes` contains a polypeptide antigen type. No date filter is applied:
restricting to post-cutoff entries that are also absent from both benchmarks leaves too few clusters
to train on. Because FT3 freezes the trunk, the trunk having seen these structures does not leak into
what is being fitted — only the head's calibration is changed.

**Exclusion against our benchmarks**, two steps (`phase_a_define_train_set.py`):

1. *By identifier.* 188 of our 227 benchmark complexes (FoldBench and PXM-2025) are listed in
   SAbDab2 and occupy **140 clusters**, all dropped. The cluster id is read from *every* row a
   complex appears in, whatever that row's antigen type — reading only the polypeptide-antigen rows
   misses clusters whose rows for that complex are apo, which left 61 clusters unexcluded.
2. *By homology*, for the 39 benchmark complexes SAbDab2 does not list. Their mmCIFs are fetched
   from RCSB, every non-antibody polymer chain is taken as a candidate antigen (chain typing by
   `hmmscan` against ANARCI's `ALL.hmm`), and antigens are compared with
   `mmseqs easy-search --seq-id-mode 2 -s 7.5`. **4 further clusters** are dropped at ≥40 % antigen
   identity (1496, 1497, 398, 499). CDR-H3 identity ≥95 % dropped **0**.

**Survivors: 557 clusters, 4,555 instances, 2,886 unique PDB entries.**

Clustering is SAbDab2's own `ab_ag_cluster`: antibodies clustered by CDR, antigens by sequence, and
the two transitively merged when they share either. Whole clusters are assigned to one side of the
split. The distribution is extremely uneven — the largest cluster holds 1,001 structures of a single
antibody — so cluster-balanced sampling is mandatory, as Protenix's own dataloader does via
`cluster_id`.

## 3. Validation split

Held out **by cluster** (`phase_b_define_val_clusters.py`). Clusters are binned into 5 bins of equal cluster
count by median deposition date; 10 clusters are drawn from each bin over 20,000 random draws, and
the draw whose structure count lands closest to 10 % of unique PDB entries is kept. Draws in which a
single cluster exceeds 15 % of the target are rejected — without that constraint one 183-structure
cluster reaches the 10 % target by itself.

**Result: 50 clusters, 291 structures = 10.1 % of the 2,886 unique PDB entries** (6.4 % of instances),
10 clusters per date bin, median deposition dates spanning 1993-09-11 to 2026-01-10, largest
validation cluster 39 structures.

**Folded subset.** 2 structures per cluster, chosen as evenly spaced picks after sorting the cluster's
structures by token count, ties broken on PDB id so the choice is deterministic — one rep is the
cluster median and two span its size range rather than both being small partial constructs. The
median validation cluster holds only 2 structures, so "5 per cluster" mostly does not exist.
**79 structures**, token counts min 193 / median 552 / max 1543.

Reference structures are the SAbDab2 instance cifs, one hard-linked per PDB id into `val_refs/`.

## 4. Interfaces scored

Predicted chains are labelled A, B, C… in the order of the job json's `sequences` array. This was
verified against a real fold rather than assumed: 8ror's json order (1457, 218, 222 residues) returns
as chains A, B, C with exactly those lengths.

`pool.csv` enumerates every (antibody chain, antigen chain) pair, alphabetically ordered — the same
convention the PXM-2025 pool uses, and it excludes the intra-antibody heavy–light interface.
**158 interfaces over the 79 structures.**

## 5. Multiple sequence alignments

MSAs are built per **unique chain sequence**, as Protenix's training layout requires:
`<root>/<sequence-id>/{pairing,non_pairing,hmmsearch}.a3m` plus `seq_to_pdb_index.json`.

Protenix does **not** use ColabFold's pairing. Its step-2 notebook searches each unique sequence once
with taxonomy information and pairs by species afterwards; `step3` writes NCBI TaxIDs into the UniRef
headers and `step4` splits each alignment on the `>UniRef100` prefix into `pairing.a3m` and
`non_pairing.a3m`.

**Query set, two batches.**

| batch | chains | n | why separate |
|---|---|---|---|
| main | all chains ≥20 residues: antibody heavy and light, plus every polypeptide antigen | **5,305** unique sequences (4,672 training + 633 benchmark, 0 shared) | — |
| short | antigen chains of 5–19 residues | **235** unique sequences (44 in validation clusters) | ColabFold keys its query database by position, so appending queries shifts every key and the kept `uniref.a3m` can no longer be resumed from |

The benchmark MSAs are rebuilt in the same pass because the originals were searched against
UniRef30 2103, which is no longer downloadable; rebuilding both sides on 2302 removes a
train/test database-version confound.

Short chains are searched rather than skipped or faked because **FoldBench contains them and searches
them normally**: of its 8 protein chains under 20 aa, 4 return query-only alignments but 4 carry real
depth (7, 4, 431 and 3,320 sequences). Dropping them would have cost 103 of 557 clusters and 13 of
the 50 validation clusters; substituting a query-only alignment would have misrepresented half of
them.

**Search.** `colabfold_search` with `--use-templates 0 --db-load-mode 0 --threads 64`, in two passes
per batch:

1. `--use-env 0` against **UniRef30 2302**, built from the prebuilt `uniref30_2302.db.tar.gz`
   (`FAST_PREBUILT_DATABASES=1`, which extracts straight to MMseqs format and never calls
   `tsv2exprofiledb`). An MMseqs index (301.8 GB) is built because `convertalis` resolves taxonomy
   only through `uniref30_2302_db.idx_mapping` / `.idx_taxonomy`; index-free search targets `_db_seq`,
   which has no taxonomy table, and the two have different key spaces (36,293,491 representatives vs
   350,950,053 members), so substituting one for the other is wrong. Taxonomy is emitted as
   `uniref_tax.m8` (**159,791,045 rows** for the main batch) by a `convertalis` call placed before
   `res_exp_realign_filter` is deleted.
2. `--use-env 1` against **colabfold_envdb_202108**, resuming from the kept `uniref.a3m`.

UniRef30 (232 GB + 302 GB index) and the environmental database (~250 GB) do not fit together in the
1,536 GB quota, so the order is forced: fetch the env tarball, run the short batch's UniRef pass, free
UniRef30, extract the env database, run the env pass for **both** batches, free the env database.
Peak usage is ~1,220 GB.

Two local patches to the ColabFold copy, both environment-gated, are required and recorded in
`phases/phase_d_colabfold_search_staging.patch`: the `convertalis` call that emits `uniref_tax.m8`
(`COLABFOLD_EMIT_TAXID=1`), and retention of `uniref.a3m` and `prof_res` plus dropping UniRef30 from
the precondition check once `uniref.a3m` exists (`COLABFOLD_KEEP_INTERMEDIATES=1`), which is what
makes the staged two-database build possible.

**Query-header adaptation.** Protenix's `step4` keys each output directory on the a3m's first header,
which in their pipeline is `<pdb_id>_<entity_id>` (their example `0.a3m` begins `>102m_1`). The
installed ColabFold instead writes a constant `>101` in all 5,305 files and carries the real query
name only in the filename; run unmodified it would resolve one key for every query and collapse all
output into a single directory, keeping the last. `phase_f_pair_and_split.py` gives each alignment its own
query header — both occurrences, because `step3` locates the UniRef/env boundary by the query header
repeating — and writes `step4`'s two index files from the query fastas. It first asserts that every
a3m's second line equals the fasta sequence for its filename (5,304 of 5,305; the single exception is
a 35-nt RNA antigen that MMseqs searched as protein, which is dropped rather than patched).

Protenix's `step1` is not used: it reads `_entity_poly.pdbx_seq_one_letter_code_can`, which the
SAbDab2 cifs do not carry (it skipped 4,550 of 4,555), and its sequences are the full deposited
constructs — signal peptides and tags included — rather than the SAbDab2 chain sequences the MSAs
were searched with.

## 6. Templates

`hmmsearch` against `pdb_seqres_2022_09_28.fasta`, via Protenix's own
`template_search.update_template_info`, writing `hmmsearch.a3m` beside each sequence's alignments.
Protenix calls that function only from `batch_inference.py`, never from the `inference.py` path used
here, so it is driven explicitly, once and single-process before any folding: MSA directories are
keyed by sequence, so two fold workers reaching a shared chain would otherwise run `hmmsearch` into
the same path concurrently. PXM-2025 folds use templates, so the validation set is built the same way.

## 7. Validation inference **[pending]**

79 structures × **20 seeds** (400000–400019) = **1,580 folds**, 5 poses each (7,900 poses), on 2 GPUs,
`bfloat16`, `N_cycle = 10`, `N_sample = 5`. Seeds were preferred over more structures per cluster
because ipTM is a seed property — 87 % of its variance lies between seeds — and because structures
within a cluster are near-duplicates by construction.

Re-using existing poses is impossible: FT3 freezes the trunk, so a seed's poses are bit-identical
before and after recalibration, but the trunk representations were never saved, and re-running does
not reproduce them — repeating one seed re-rolls the outcome (9n6e_GF, DockQ 0.022 → 0.644). The
trunk state must therefore be written in the same pass that generates the poses.

**Cached per (structure, seed)**, everything needed to re-run the confidence head offline:

| tensor | content |
|---|---|
| `z_trunk`, `s_trunk` | pair and single trunk representations, bf16 bits in a uint16 view |
| `ci_s_inputs`, `ci_s_trunk_f32` | the head's own inputs, float32 |
| `ci_coordinate` | the 5 poses in **model** atom order, float32 |
| `ci_contact_probs` | per-sample contact probabilities |
| `ci_feat_*` | `distogram_rep_atom_mask`, `atom_to_token_idx`, `atom_to_tokatom_idx`, `asym_id`, `has_frame`, `is_ligand` |
| `ci_N_cycle`, dtypes | recycle count and the source dtypes of z/s |
| RNG state | Python, torch and CUDA streams |

bf16, not fp16: the model runs in bf16, whose exponent range matches float32, and the head runs inside
autocast, so bf16 is what it would see. A float16 trunk dump once cost a good pose (0.495 → 0.056).

**ipTM is saved twice**: per sample as Protenix's `*_summary_confidence_sample_N.json`
(`iptm`, `chain_pair_iptm`, `chain_pair_iptm_global`, `chain_pair_pae_mean/min`, `plddt`, `ptm`,
`ranking_score`), and per interface in `results.csv` alongside DockQ. Per-interface `chain_pair_iptm`
is the reported quantity, never the global value.

One `.npz` per fold at `<dir>/<pdb>/<tag>_<seed>.npz`, written through `folder.py`'s `TRUNK_DUMP_DIR`,
which refuses to overwrite an existing file. Pointing Protenix's `PROTENIX_DUMP_TRUNK` at a directory
instead yields `<dir>/trunk.npz` — one file for every fold.

## 8. Scoring

OpenStructure (`ost`) for DockQ, which also performs the chain mapping — necessary because predicted
chains are A/B/C in job-json order while the references carry SAbDab2 chain names. Nothing
hand-rolls structure handling. An interface is *acceptable* at DockQ ≥ 0.23. Validation is a paired
before/after on the same poses with the same scorer, so any scorer offset cancels.

## 9. Scripts

| script | role |
|---|---|
| `extra_training/phases/phase_a_define_train_set.py` | benchmark exclusion, by id and by antigen homology |
| `extra_training/phases/phase_c_extract_msa_queries.py` | the unique-sequence query fastas |
| `extra_training/phases/phase_c_extract_msa_queries.py --short-vs` | the 5–19 residue antigen chains the first pass skipped |
| `extra_training/scripts/archive/phase1_uniref.sh`, `archive/phase1_index.sh`, `archive/phase1_search_idx.sh` | UniRef30 build, index and first search |
| `extra_training/scripts/archive/phase1b_short.sh` | second UniRef pass for the short antigen chains |
| `extra_training/scripts/phase2_envdb.sh` | environmental database, `download` / `build` modes |
| `extra_training/phases/phase_f_pair_and_split.py` | per-query a3m headers and `step4`'s index files |
| `extra_training/phases/phase_b_define_val_clusters.py` | the date-binned, cluster-held-out validation split |
| `extra_training/phases/phase_g_build_val_jobs.py` | job jsons, `pool.csv`, `chain_roles.csv`, staged references |
| `extra_training/phases/phase_h_search_templates.py` | templates for every validation chain |
| `extra_training/scripts/overnight.sh` | the gated preparation chain |
| `extra_training/phases/phase_i_fold_validation.sh` | smoke test and the validation fold |
| `extra_training/scripts/see_it_through.sh` | drives preparation → smoke → fold → hand back the GPUs |

Every stage gates on **content**, never on an exit code: `setup_databases.sh` touches its READY
marker and returns 0 even when it converted nothing, and `mmseqs dbtype` on a missing database also
returns 0.

## 10. Open items

- Validation inference and the confidence-head fit are not yet run.
- PXM-2025 folds use templates while FoldBench folds do not, so a head calibrated with template
  features would be evaluated without them on FoldBench. Unresolved.
- Ground-truth coordinates in model atom order are not cached. They are not needed to re-run the head
  or to compare ipTM before and after, but a training loss computed offline would require them.

## Decided 2026-10-11: the benchmarks stay in their released settings

**PXM-2025 and FoldBench are folded with the MSAs and templates they already have.** Their MSAs were
built on uniref30_2103 and their templates under Protenix's inference cutoff of 2021-09-30. Neither is
changed for the refold that applies the retrained confidence head.

**Why.** The refold exists to measure one thing — whether the retrained head selects better poses. If
it also changed the MSA database version or the template regime, any movement in DockQ or ipTM would
be attributable to the head *or* to the inputs, with no way to separate them. The same argument that
keeps the 2021-09-30 template cap in place applies to the MSAs: vary one thing.

It also preserves comparability with everything already measured on these benchmarks — the free-seed
baselines, the epitope decomposition, the PXM selection gap. Rebuilding their MSAs would invalidate
all of it.

**The cost, stated plainly.** The head is trained on poses built from uniref30_2302 MSAs and applied
to poses built from 2103 MSAs. That is a regime mismatch. It is judged the lesser risk because the
head predicts error from trunk representations, and MSA depth already varies enormously *within* any
dataset — Protenix's own confidence head was trained across all of it.

**What happens to the 2302 benchmark MSAs.** They are built and kept: 633 benchmark chain sequences
in the main search batch plus 11 short ones from the top-up. They become a **sensitivity check** —
fold a subset of benchmark complexes both ways and confirm the database version does not move the
baseline. That turns the paragraph above from an assumption into a measurement, cheaply, and it is
worth doing before the headline claim is written.

**What this does not change.** The training pool and the validation set keep their 2302 MSAs; they had
no 2103 alternative, since 2103 is no longer downloadable. The self-templating date filter applies to
training and validation folds, and leaves the benchmarks untouched because the 2021-09-30 cap already
binds for every PXM complex (all 90 dated ones are 2023-2025) and FoldBench folds use no templates.

## The pool is 555 clusters, and why val_clusters.csv predates it

**Decided 2026-10-11, second review pass.** Filter 2's coverage mode moved from 0 to 1.

`--cov-mode 0` requires the coverage fraction to hold for query *and* target, so a training antigen
contained inside one of our benchmark chains fails on coverage no matter how identity is computed.
Correcting `--seq-id-mode` from 2 (longer) to 1 (shorter) alone changed nothing for exactly that
reason. `--cov-mode 1`, coverage of the target, catches containment:

| cluster | matched | identity |
|---|---|---|
| 1035 | 9ms8 | **0.996** |
| 1147 | 9ms8 | **1.000** |
| 1496 | 8xnh | 0.990 |
| 1497 | 9kur | 1.000 |
| 398 | 9isf | 1.000 |
| 499 | 8zp2 | 0.601 |

Six clusters instead of four, so the pool is **561 → 555**. A training cluster whose antigen matches a
benchmark antigen at identity 1.000 is precisely what this filter exists to remove.

**`train_val_splits/val_clusters.csv` was drawn from the 557-cluster pool and is kept.** Re-running phase B
against 555 clusters changes 45 of the 50 — a different draw, not a better one — and re-drawing a
validation split after the fact is what phase B's own docstring warns against. The existing split
remains valid on the stronger test: **every one of its 50 clusters is still in the 555-cluster pool**,
and neither newly excluded cluster was ever in validation, so validation never contained leaked data.
`train_minus_val()` enforces that invariant on every call and fails if it is ever broken.

The consequence to state: phase B is not byte-reproducible against the current pool. That is a
deliberate pin, not an oversight.

**The searched query set stays at 5,305.** Phase C now derives 5,299 from the smaller pool, and the
six extra sequences already searched belong to the newly excluded clusters. The frozen
`query.searched.fasta` is authoritative — ColabFold keys by position, so the set that was searched is
the set phases E and F must use — and those six simply go unreferenced.
