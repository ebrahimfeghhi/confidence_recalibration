# phases/ — one script per phase of the confidence-head finetune

Each file is a phase of the pipeline, named for what it does. Run them in order; each states its
inputs, outputs and gates in its own docstring. The plain-language account of what every phase is
*for* is in `extra_training/README.md`, under **Phase-by-phase walkthrough**.

| phase | script | writes |
|---|---|---|
| A | `phase_a_define_train_set.py` | `results/excluded_homolog_clusters.csv`; exports `training_clusters()` |
| B | `phase_b_define_val_clusters.py` | `results/val_clusters.csv`; exports `train_minus_val()` |
| C | `phase_c_extract_msa_queries.py` | `msa_work/query.fasta`, `msa_work/query_short.fasta`, `msa_queries/*` |
| D | `phase_d_search_uniref.sh` | `<out-dir>/*.a3m`, `uniref.a3m`, `uniref_tax.m8` per batch |

Phases E through I (env pass, taxids and pairing, job JSONs, templates, folding) still live in
`extra_training/scripts/` and `diffusion_commit_clean_v2/`. Moving them is pending.

**The end state** is one script per phase here, with `extra_training/scripts/` holding only what is
not a phase: the three watchers (`quota_guard.sh`, `gpu_guard.sh`, `health.sh`) and the shared
download helper `lib_fetch.sh`.

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

## Audit, second and third passes

Two further reviews of phases A–C. Fifteen issues; all fixed. The pool is unchanged throughout —
561 → 557 clusters, the same four drops at the same identities — so none of this moved a result. Every
one of them could have moved one later without raising anything.

### Phase A

**classify() kept one hit per chain, ranked by the wrong column.** Single-chain constructs carry two
variable domains — 8r4u_B is 236 residues with envelopes 1-127 and 126-236 — and only one was stored.
Worse, hits were ranked by column 7, the FULL-SEQUENCE E-value, which is a property of the (query,
target) pair and so identical for every domain of one chain: 8r4u_B's two halves both read 5.0e-64
against `human_L`. Ranking on it cannot distinguish a VH from a VL even in principle, and `human_H`
hit that chain at 1.9e-57. Four of the 39 complexes (8r4u, 9gry, 9gs3, 9i20) consist only of such
chains, so they contributed **no CDR-H3 query at all** and Filter 3 reported 0 for them whether or not
a leak existed. Now one entry per domain, ranked by the per-domain i-Evalue (column 13), with
overlapping envelopes merged: heavy domains 60 → **67**, complexes covered **35 → 39 of 39**. Filter 3
still finds 0, and that 0 now covers every complex.

**The CDR-H3 gate asserted chains, not complexes.** `--min-cdr-frac` checked recovery across chains
already classified heavy and said nothing about how many complexes contributed one — printed and never
checked, the exact pattern it was written to end. Asserted now, and it is what exposed the four above.

**Cluster ids were canonicalised on only one side of the comparison.** `train_tables()` built FASTA
headers with a raw f-string, `search()` and `cdr_hits()` parsed them back raw, and `main()` intersected
the result against `{_ckey(c) for c in train_cl}`. Under float64 the hit keys read `"1496.0"` and the
normalised side `"1496"`: the intersection empties, the run prints `union dropped: 0 clusters`, the csv
is written with only a header, `training_clusters()` reads an empty exclusion set, the orphan guard
finds no orphans to report, and the pool returns 561. The 557 → 561 failure the first pass identified,
reached through the one path its fix did not cover. Headers now go through `_ckey()` at construction.

**A missing exclusion csv counted as zero exclusions.** A mistyped `--excluded`, or any downstream
phase run before phase A, admitted benchmark-homologous clusters silently. Now fatal, with the
regeneration command in the message. A header-only file remains the way to record "none".

**Failed downloads still produced an exclusion artifact.** `main()` printed `bad` and searched whatever
had been fetched, so a network failure or partial cache under `--no-fetch` yielded a csv that looked
complete while covering only part of the benchmark. Now aborts before searching;
`--allow-missing-structures` to accept a knowingly partial exclusion.

**fetch() cached partial downloads forever.** `urlretrieve` wrote straight to the destination and the
"over 1000 bytes" test then treated a truncated file as complete. gemmi parses a truncated mmCIF
without complaint and returns a short chain set — fewer antigen queries, exclusions missed. Downloads
now land on `.part` and are renamed on success.

**ALLHMM hardcoded a venv path**, which breaks on a rebuild and silently changes which HMM set types
our chains. Resolved from the installed `anarci` package, with the literal as a last resort.

### Phase B

**Minimising the held-out structure count selects an extreme draw.** Of 20,000 draws, 11,296 pass the
dominant-cluster rule; their held-out share runs 2.4% to 10.1%, median 4.5%. `--select target` with
`--frac 0.10` therefore picks the single largest-holdout draw of the 11,296. It gets there because
holding out more structures requires larger clusters: median cluster size 2 against 1, and 29 of 50
clusters with two or more distinct PDB entries against 22. Only those can contribute two folded
representatives, so this draw yields 79 folded structures where a uniform draw yields a median of 72.

What is *not* established is the direction of the resulting bias. Large clusters here are **newer** —
median deposition 2021-07-30 for clusters of 6+ against 2019-10-09 for singletons, and only 39% of
their structures predate Protenix's 2021-09-30 cutoff against 64% — so they are if anything *less*
likely to sit inside the model's training data. Rank correlation between cluster size and date is 0.04.
Post-cutoff coverage is identical either way, 17 of 50 clusters, which the date binning guarantees.

`target` therefore remains the default and `results/val_clusters.csv` is unchanged: it is an extreme
draw on cluster size, in a direction not shown to flatter the result, and it buys 7 more folded
structures. `--select band` is implemented and documented so the alternative is available once
per-cluster DockQ exists to measure the two against each other.

**`best` could stay None**, then fail inside pandas naming neither cause nor remedy. Now an explicit
error naming both constraints and which knob to move.

**The remainder of `--n-val / --n-bins` was discarded.** `--n-val 51` with 5 bins wrote 50 clusters and
reported success. Distributed across bins, with a per-bin feasibility check.

**No `--splits` or `--excluded`**, so B silently read the module defaults while A and C could be
pointed elsewhere — three phases disagreeing about the pool. Both exposed.

**`train_minus_val()` added — the only place the train/val split is enforced.** Phase A derives the 557,
phase B holds out 50, and phase C deliberately builds MSAs for all 557, correctly, since the validation
complexes have to be folded. So nothing up to here subtracts validation from training, and nothing did:
the failure mode is training on your own validation set and reporting the result. Any training-index
builder must take its clusters from here (507), not from `training_clusters()` (557). A missing
`val_clusters.csv` is fatal, as is a validation cluster absent from the pool.

### Phase C

**An RNA antigen was searched as a protein.** `training_clusters()` keeps a row if *any* of its antigens
is PROTEIN or PEPTIDE, but `train_records()` and `train_complexes()` took every entry of
`agexpectedseqs` without consulting `agtypes`. The committed query set carried one: the RNA chain
`c228_6cf2_ag1_4666db`. ColabFold searched it anyway, and the a3m/fasta consistency check downstream
dropped it because the search had rewritten its `U` as `X` — so it cost a query slot and surfaced as a
mismatch far from its cause. `poly_antigens()` now pairs each antigen chain with its own type, and
errors if the two fields disagree in length rather than letting `zip()` drop the trailing chains.
Query set 5,305 → **5,304**.

**"verified 0 shared" was a comment.** Now checked, and it passes at 0. The reviewer described it as
testing all three filters at once; that overstates it and the code says so. Identity 1.0 is the
*easiest* case any filter faces — filter 2 sits at 0.40 for proteins, filter 3 at 0.95 — and filter 1
usually removes such a cluster by id before sequence comparison enters it. So it is a smoke test that
would fire loudly on a wiring mistake, not evidence the thresholds are calibrated. Worth having at two
lines; not worth citing as validation of the filters.

**Benchmark structures were skipped without a count.** Both bench functions `continue`d past a missing
reference cif or a gemmi failure, and the printed line reported only a sequence count. A complex
silently getting no 2302 MSA keeps its uniref30_2103 one at fold time, reintroducing the exact
train/test database-version difference this rebuild exists to remove. One `bench_chains()` now reads
them and asserts coverage: **227/227**.

**Batch 2 covered training chains only.** Batch 1 used `--min-len 20` for both sets, so the chains it
skipped exist on both sides; searching only the training ones left a benchmark complex with a 5–19
residue antigen treated differently from its training counterparts. `phase1b_short.sh` now runs batch 2
with `--which both`: 235 → **245** short queries.

### Also fixed outside the phases

**The index-mapping gate rejected a complete index.** The first audit claimed `.idx_mapping` and
`.idx_taxonomy` were hand-made leftovers and required them to postdate the index build.
`setup_databases.sh:168,171` creates them itself with `ln -sf`, and `createindex` does not — so the gate
failed a finished 301 GB index twice and the watcher gave up. `phase1b_short.sh` now creates the links
(we run `createindex` ourselves under `MMSEQS_NO_INDEX=1`, skipping the part of `setup_databases.sh`
that would have) and gates on what they resolve to.

## Reviewer edits not adopted, or adopted with reservations

**Not adopted: banding the held-out structure count (`--select band` as the default).** The diagnosis
was right — argmin does select an extreme draw — but the stated rationale was that the held-out
structure count "only affects how much training data you give up, not the validation set itself." It
affects the validation set directly: 55% of clusters are singletons and can only ever supply one
representative, so the held-out structure count is close to a proxy for how many clusters can supply
two. Switching cost 79 → 72 folded structures. And the bias has no established direction: large
clusters here are newer and less likely to be inside the model's training data. Implemented as an
option, not the default, with the measurements recorded above.

**Narrowing the band was tested and rejected.** [8%, 12%] leaves 25 of 20,000 draws and [9%, 13%]
leaves 2 — that is argmin again, with none of the bias removed. Held-out share is also a noisy proxy
for folded count: [8%, 12%] holds out more structures (236) yet folds fewer (71) than [5%, 12%]
(161 → 72). If this is revisited, the quantity to constrain is folded representatives, not structures.

**Adopted with a reservation: Filter 3's "0 clusters removed" as load-bearing evidence.** Both reviews
treat that 0 as meaningful. It is now honestly earned — 39 of 39 complexes queried, every pair
compared — but it was always close to expected. At 95% identity on a 14-residue loop an exact match is
required, and SAbDab2's `ab_ag_cluster` is a connected component over shared antibody *or* antigen
cluster, so any cluster sharing an antibody with one of the 188 listed complexes is already gone via
filter 1. Filter 3 can only ever speak to the 39 unlisted ones. Keep it; do not present it as
independent confirmation that the exclusion is thorough.

**Adopted with an escape hatch: asserting CDR-H3 complex coverage equals the number of structures.**
Correct today at 39/39, and hard-failing is the safe direction. But a future benchmark entry may
genuinely carry no antibody heavy chain, and the response to that should not be editing this file
while a pipeline is blocked — hence `--allow-missing-cdr-complexes`, off by default.

**`train_minus_val()` is insurance, not a guarantee.** It is the only enforcement of the train/val
split, and nothing calls it yet because the training stage does not exist. A function nobody is
obliged to call does not prevent the failure it was written for. The stronger form is for phase B to
emit a `train_clusters.csv` artifact that the training stage reads as its only input, so using the
full 557 requires deliberately going around the pipeline. Worth doing when that stage is written.

## Phase D

`phase_d_search_uniref.sh <query.fasta> <out-dir> [tag]` replaces four scripts that between them did
one job: `phase1_uniref.sh` (download and extract), `phase1_index.sh` (index), `phase1_search_idx.sh`
(search batch 1) and `phase1b_short.sh` (search batch 2). It is idempotent at every step — an existing
database, index or alignment is skipped — so the same code runs all three batches:

| query fasta | output | queries |
|---|---|---|
| `query.fasta` | `msa_initial_idx` | 5,304 chains of 20+ residues |
| `query_short.fasta` | `msa_short_idx` | 234 antigen chains of 5–19 residues |
| `query_short_TODO_bench.fasta` | `msa_bench_short_idx` | 11 benchmark chains of 5–19 residues |

`FREE_DB=1` deletes uniref30 and its index on success, so batches can run back to back without
re-downloading 100 GB each time. `ALLOW_THIN=1` accepts a batch that retrieves little or nothing,
which is a real outcome for short peptides — 4 of FoldBench's 8 chains under 20 aa come back
query-only — and a fault for a full batch.

### What phase D does not produce

Alignments, not MSAs Protenix can read. Phase E deepens them against the environmental database and
phase F writes the taxids into the headers and splits each one into `pairing.a3m` and
`non_pairing.a3m`. Phase D is the uniref30 half.

### The disk trick, and the patch it depends on

uniref30 (450 GB extracted plus a 301 GB index) and colabfold_envdb (291 GB) do not both fit in the
1,536 GB quota. So phase D searches uniref30, **keeps** its intermediates, and deletes the database;
phase E resumes from `uniref.a3m` with uniref30 gone. That resume only works because of
`patches/colabfold_search_staging.patch` — unpatched, `colabfold_search` refuses to start when
uniref30 is missing. Check it before trusting any of this:

```
envs/colabfold/bin/python -c "import colabfold.mmseqs.search as m; print(m._emit_taxid())"
```

### Why the 301 GB index is built

Not for speed — `setup_databases.sh` itself calls the index "not useful for colabfold_search in most
cases", and at this batch size it buys nothing. It is the only route to **taxonomy**: `convertalis`
resolves taxids through `.idx_mapping` and `.idx_taxonomy`, and an index-free search instead targets
`_db_seq`, whose 350,950,053 member keys are not in the same key space as the mapping table's
36,293,491 representatives. No index, no taxids; no taxids, no paired MSA; and phase F has nothing to
split on.

Earlier comments in this repo called the index unnecessary. That was true about speed and read as
though it were about correctness — the two claims sat in different scripts and contradicted each other
when read in sequence.

### Three defects carried over and fixed

**The success check was vacuous.** `phase1_uniref.sh` verified the build with
`mmseqs dbtype <db>`, which returns exit 0 for a database that does not exist — verified. Combined
with `setup_databases.sh` touching `UNIREF30_READY` *before* the steps that can fail, nothing in that
path could detect a failed build. Phase D gates on five data files being non-empty.

**No coverage gate on the search.** `phase1_search_idx.sh` ended with `exec colabfold_search`, so
nothing checked that 5,305 queries produced 5,305 alignments. Phase D compares the two and fails.

**Marker assumption.** Inherited from the fix in `c1cbe087`: the absence of `COLABDB_READY` makes
`setup_databases.sh` extract the env database alongside uniref30, which is how both landed on disk at
once. Phase D creates every marker whose database it wants left alone, and afterwards checks by data
file whether anything was built that it did not ask for.

### A deletion I got wrong

`phase1_search_idx.sh` and `phase1_index.sh` were removed in `c1cbe087` as dead code on the test
"nothing in the repo references it". That test is wrong for a script run once by hand: it is
referenced by its *artifact*, not by another script, and `phase1_search_idx.sh` is the provenance of
all 5,305 batch-1 alignments. Both restored, and their procedure is now recorded in
`phase_d_search_uniref.sh`. `phase1_search.sh` stays deleted — it produced the superseded
index-free `msa_initial/`, which had no taxonomy table and could not pair.
