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

### Noted, not changed — needs a decision

`brute_cdr_hits()` scores the substring case as `min(len)/max(len)`, which is a length ratio rather
than an identity: a 13-mer sitting exactly inside a 14-mer scores 0.93 and is missed at the 0.95
threshold, although its aligned identity is 100%. That errs toward *missing* a leak, the unsafe
direction for an exclusion filter. Treating an exact substring as identity 1.0 and testing length
separately would be stricter and could drop further clusters — a change to the method, not a bug fix,
so it is left alone pending a decision.

Lower-severity, left as is: `fetch()` treats any cached file over 1000 bytes as complete, and
`train_tables()` zips `agtypes` against `agresolvedseqs` positionally, so a row whose two fields
disagree in length would drop its trailing antigen chains without comment.
