# phases/ — one script per phase of the confidence-head finetune

Each file is a phase of the pipeline, named for what it does. Run them in order; each states its
inputs, outputs and gates in its own docstring. The plain-language account of what every phase is
*for* is in `extra_training/README.md`, under **Phase-by-phase walkthrough**.

| phase | script | writes |
|---|---|---|
| A | `phase_a_define_train_set.py` | `results/excluded_homolog_clusters.csv`; exports `training_clusters()` |
| B | `phase_b_define_val_clusters.py` | `results/val_clusters.csv` |
| C | `phase_c_extract_msa_queries.py` | `msa_work/query.fasta`, `msa_work/query_short.fasta`, `msa_queries/*` |

Phases D through I (UniRef search, env pass, taxids and pairing, job JSONs, templates, folding) still
live in `extra_training/scripts/` and `diffusion_commit_clean_v2/`. Moving them is pending.

## Why A exports the training set

`training_clusters()` is defined in phase A and imported by B, C and phase G (`build_val_jobs.py`).
Phase A *is* the definition of the training set, so that is where the derivation belongs. The
alternative — each phase re-deriving it, or reading a cached cluster list — gives either four copies
that can drift apart or a file that goes stale when the exclusion changes. Neither is acceptable in a
pipeline whose cluster count appears in a paper.

## Why `BENCHMARKS` is written out by hand

`paths.DATASET_NAMES` is `('pxm2025', 'foldbench', 'sabdab_val')`. `sabdab_val` is *derived* from this
pool, and it has a readable `pool.csv`, so iterating that tuple fails silently rather than loudly: in
phase A it would exclude the validation clusters from the pool they are drawn from, and in phase C it
would file the 79 validation structures as "benchmark" sequences. Both phases name the two evaluation
sets explicitly instead.

## Verification

Every phase was re-run after the move and its output diffed against the artifact the pre-move code
produced. All byte-identical:

| artifact | result |
|---|---|
| `results/excluded_homolog_clusters.csv` | identical |
| `results/val_clusters.csv` | identical |
| `msa_queries/{train,bench}.fasta`, `manifest.csv`, `complexes{_all.fasta,.csv}` | identical |
| `msa_work/query.fasta` | identical to the `cat`-produced original |
| `msa_work/query_short.fasta` | identical |

`msa_work/query.fasta` used to be produced by a bare `cat train.fasta bench.fasta` at a shell, which
left a step of the method with no code behind it. Phase C now writes it, and the check above is that
it writes the same bytes.

## Audit, 2026-10-11

The byte-identical checks above prove the *move* changed no behaviour. They do not prove the logic was
right beforehand — a faithful refactor reproduces a thinko faithfully. So phase A was read through
separately. Three silent-failure modes and one reproducibility defect were found and fixed; none
changed the result, all four could have changed it later without raising anything.

**1. Filter 3's "0 clusters removed" was unfalsifiable.** `cdrh3_of()` catches every exception per
chain, so a wrong HMM path or a missing hmmer-3.1b2 binary gives an empty query set; `search()` then
returned `{}` and the report read `CDR-H3 mmseqs >=95% id: 0 clusters hit`, identical to the genuine
answer. The recovery rate was printed (`60/60 heavy chains`) and never checked. That 0 is quoted as
evidence filters 1 and 2 were thorough, so it now has to earn it: `--min-cdr-frac` (default 0.95)
fails the run if ANARCI numbers fewer heavy chains than that. Negative-tested at 1.01, it fires.

**2. An empty search input counted as "no hits".** Same shape, across all five searches: a broken
antibody classifier would have reported a clean antigen filter. `search()` now raises on an empty
query or target fasta.

**3. Cluster ids were keyed by `str()`.** Correct only because `ab_ag_cluster` is `int64` today. One
NaN in that column makes pandas read it as `float64`, `str()` yields `"1496.0"`, every homology
exclusion stops matching, and the pool silently grows back from 557 to 561 — i.e. we would train on
benchmark-homologous clusters. Both sides now go through `_ckey()`, which collapses to a canonical
integer string, and `training_clusters()` raises if a cluster in the exclusion csv matches nothing in
the splits file.

**4. The artifact did not reproduce.** A complex often deposits several identical copies of its
antigen — 8xnh chains A, B and C are the same 438-residue sequence, 9kur A and B the same 116-residue
one — so they hit a cluster at exactly equal identity. Both hit selectors used a strict `>`, leaving
the winner to be whichever row mmseqs emitted first: the same inputs gave `8xnh_B` on one run and
`8xnh_C` on the next. Clusters and identities never moved, only the credited chain, but an artifact
that does not reproduce does not belong in a paper. Ties now go to the lowest query name, verified
identical across `--threads 32` and `--threads 8`. The stored csv was regenerated, so it reads
`8xnh_A` and `9kur_A`.

**5. Filter 3 scored contained loops by length, not by letters — FIXED.** `brute_cdr_hits()` took
the case where one loop sits inside another and scored it `min(len)/max(len)`, a length ratio, then
compared that against an identity threshold. A 13-mer sitting letter-perfect inside a 14-mer scored
0.93 and slipped under the 0.95 cutoff although it matched at every position — an exclusion filter
erring toward *missing* a leak, the unsafe direction. Replaced; see **Filter 3** below. Re-running
phase A gives the identical artifact and the same 561 → 557, so the correction costs no training
data.

## The two sequence filters, as implemented

Filters 2 and 3 exist only for the 39 benchmark complexes that have no row in SAbDab2 and therefore
no cluster to delete by id. Both ask whether a *training* cluster is too close to one of those 39.

### Filter 2 — antigen sequence  (removes 4 clusters)

The leak that matters most: if a training complex involves the same antigen as a held-out test
complex, the model has already seen that target's surface, which is exactly what we then measure.

- **Queries.** Every polymer chain of the 39, fetched fresh from RCSB, that `hmmscan` does *not* type
  as an antibody domain — 81 candidate antigens from 177 chains. Taking every non-antibody chain
  rather than the annotated antigen sidesteps a chain-naming disagreement between SAbDab2's roles and
  our local cifs (9n6e: roles say E/F/G, the cif holds A,B,C,D,H,I,L). Over-including can only drop
  more training clusters, the safe direction.
- **Targets.** Every antigen sequence of the candidate training pool, split by SAbDab2's own
  `agtypes` into PROTEIN and PEPTIDE, taking **both** `agresolvedseqs` and `agexpectedseqs` —
  resolved sequences have unmodelled gaps a homolog could hide in, so searching both is stricter.
- **Comparison.** `mmseqs easy-search -s 7.5 --seq-id-mode 2 --cov-mode 0 -c 0.50`. Identity is
  measured over the shorter sequence, so a 15-residue peptide against a 500-residue protein is not
  scored as trivially dissimilar. Sensitivity is raised from the default because a 40% match is
  remote.
- **Thresholds — ours, not SAbDab2's.** 40% identity for PROTEIN antigens, 80% for PEPTIDE. 40%
  sounds permissive but at that level two proteins are generally the same protein from another
  species or variant. Peptides need far higher identity before a match means anything, because short
  strings match by chance.
- **Result.** 4 clusters: 398 (9isf, 0.973), 1496 (8xnh, 0.966), 1497 (9kur, 0.827), 499 (8zp2,
  0.578).

### Filter 3 — antibody CDR-H3  (removes 0 clusters)

The mirror question: do we train on a benchmark *antibody*? The same antibody against a different
antigen still leaks.

- **Why CDR-H3 and not the whole variable domain.** The variable domain at 95% hits 13 clusters, all
  framework-driven — antibody frameworks are germline and cross 95% between unrelated antibodies.
  CDR-H3 is the loop that determines what an antibody binds, and 95% CDR-H3 identity is SAbDab2's own
  criterion (their `cdrh3_cluster_95` column). The variable-domain number is printed for reference
  and deliberately not used.
- **Queries.** CDR-H3 extracted by ANARCI at IMGT positions 105–117, SAbDab2's definition, from the
  60 heavy chains `hmmscan` found. Recovery is 60/60 and is now **asserted** (`--min-cdr-frac`),
  because ANARCI failing silently would make this filter report 0 hits whether or not any exist.
- **Comparison.** `cdr_hits()`: every query loop against every target loop, 60 × 4,598 pairs, no
  prefilter. For each pair the shorter loop is slid along the longer one ungapped and the best
  position-by-position match is taken; **identity** is counted over the shorter loop and **coverage**
  is the length ratio, reported and thresholded separately (0.95 identity, 0.90 coverage).
- **Why no search tool.** This was previously an `mmseqs` search cross-checked against a direct
  comparison, because `mmseqs` indexes short shared chunks and can discard a near-exact match on a
  14-letter string before aligning it. That left two sets of blind spots to reason about and a bug in
  one of them. At this scale the exhaustive comparison *is* the simple option — a few hundred
  thousand pairs, instant, nothing skipped, nothing to cross-check.
- **Why no gap handling.** At 95% a 14-residue loop must match 14 of 14, so an internal insertion is
  already out of reach and modelling one buys nothing. Note the consequence: Filter 3 as configured
  catches only essentially identical loops. Two 14-mers differing at one position score 0.929 and are
  not flagged. That is what 95% on a short string means; Filter 2, at 40%, is the permissive filter
  doing the real work.
- **Result.** 0 clusters — and because the recovery rate is now asserted and every pair is compared,
  that 0 is a finding rather than a possible silent failure.

Lower-severity, left as is: `fetch()` treats any cached file over 1000 bytes as complete, and
`train_tables()` zips `agtypes` against `agresolvedseqs` positionally, so a row whose two fields
disagree in length would drop its trailing antigen chains without comment.
