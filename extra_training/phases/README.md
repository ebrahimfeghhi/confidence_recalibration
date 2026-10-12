# phases/ — one script per phase of the confidence-head finetune

Each file is a phase of the pipeline, named for what it does. Run them in order; each states its
inputs, outputs and gates in its own docstring. The plain-language account of what every phase is
*for* is in `extra_training/README.md`, under **Phase-by-phase walkthrough**.

| phase | script | writes |
|---|---|---|
| A | `phase_a_define_train_set.py` | `results/excluded_homolog_clusters.csv`; exports `training_clusters()` |
| B | `phase_b_define_val_clusters.py` | `results/val_clusters.csv`; exports `train_minus_val()` |
| C | `phase_c_extract_msa_queries.py` | `msa_work/query.fasta`, `msa_work/query_short.fasta`, `msa_queries/*` |
| D | `phase_d_search_uniref.sh` | `<out-dir>/*.a3m`, `uniref.a3m`, `uniref_tax.m8`, `query.searched.fasta` per batch |
| E | `phase_e_search_envdb.sh` | the same a3m, deepened against colabfold_envdb |
| F | `phase_f_pair_and_split.py` | `mmcif_msa/<chain>/{pairing,non_pairing}.a3m` — the first files Protenix can read |
| G | `phase_g_build_val_jobs.py` | `val_jobs/*.json`, `val_fold/pool.csv`, `val_fold/chain_roles.csv` |
| H | `phase_h_search_templates.py` | `mmcif_msa/<chain>/hmmsearch.a3m` |
| I | `phase_i_fold_validation.sh` | `val_fold/{poses,trunks,results.csv}` — the deliverable |

All nine phases are here. `extra_training/scripts/` now holds only what is not a phase: the three
watchers, the shared download helper, the two orchestrators, and the superseded scripts kept until
their replacements have each run once.

`extra_training/scripts/overnight.sh` is the orchestrator: it verifies phases A–D's artifacts and then
calls E, F, G and H in order. Each phase owns its own gates, so the orchestrator's job is order and
escalation, not re-checking — which took it from 230 lines to 129.

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

| query fasta | output | queries searched | state |
|---|---|---|---|
| `query.fasta` | `msa_initial_idx` | 5,305 chains of 20+ residues | done |
| `query_short.fasta` | `msa_short_idx` | 235 antigen chains of 5–19 residues | done |
| `query_short_TODO_bench.fasta` | `msa_bench_short_idx` | 11 benchmark chains of 5–19 residues | done 2026-10-11 15:33 |

The counts are what was **searched**, which is what each batch's frozen `query.searched.fasta` records.
Phase C's current output is one record shorter in the first two cases, because its antigen-type filter
now removes a nucleic-acid chain from each — see the frozen-fasta note below for why that difference
must never be handed to phase E.

### The 11-chain benchmark top-up, as an illustration of the depth gate

| chain | alignment depth |
|---|---|
| 8fym_P | 597 sequences |
| 8fbw_E | 130 |
| 8gho_C | 5 |
| the other 8 | 1 — query-only |

3 of 11 retrieved real alignments. That is the expected shape for 11–18 residue peptides and close to
FoldBench, where about half of the chains under 20 aa come back query-only. It is also exactly why
depth is gated on query length rather than on a per-batch flag: every one of these is under 20
residues, so the gate exempts them, and a 300-residue chain returning a single sequence would still
fail.

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

### Phase D, second review pass

**The patch is now a gate, not a comment.** Everything in phase D depends on
`patches/colabfold_search_staging.patch`: without it there is no `uniref_tax.m8` (so no paired MSA)
and no resume from `uniref.a3m` (so phase E cannot run once uniref30 is deleted). The header used to
tell the operator to verify it by hand, which meant an unpatched venv surfaced hours in, after a
100 GB download and the search. Checked at the top now — the two functions the patch adds must exist
on the imported module, and at least four `PATCHED (tts_abag)` markers must be present, which catches
a partially applied patch.

**`--threads` is derived from the taskset.** It was hardcoded at 64, which oversubscribes a narrower
`--cpu-list`: the threads exist but the scheduler has only the pinned cores to put them on. Now
`taskset -c "$CORES" nproc`, which reports exactly how many CPUs the range allows whatever its syntax
(`0-63`→64, `0-31`→32, `0,2,4`→3). The documented project cap is 64 cores, not 32 — `MSA_RUNBOOK.md:119`,
"more has crashed this server"; the 32 in `phase_a_define_train_set.py` is that script's own
conservative default.

**The taxid gate was measuring a column that cannot carry information.** It passed at "more than 10
usable taxids in the first 2,000 rows". Replacing it with per-query coverage exposed something worse:
**column 1 of the m8 is the constant `101` for every row**, because ColabFold writes a constant query
identifier — the same artifact that makes every a3m start `>101` and that `prep_step4_inputs.py`
exists to repair. So per-query attribution is impossible from this file, and a gate keyed on it
reports exactly one query forever.

What the file is actually for: `step3-uniref_add_taxid.py:253-255` reads it as a dictionary from
**column 2, the UniRef accession**, to **column 3, the taxid**. The query column is never touched. So
the measure that matters is: of the UniRef hits appearing in the alignments, how many carry a taxid?
That decides how much of each alignment can be paired, and it collapses to zero if `convertalis` or
the patch failed, while the alignments themselves still look normal.

Measured on the completed short batch:

| | |
|---|---|
| alignments | 235 |
| with at least one UniRef hit | **34** — the other 201 are query-only |
| distinct UniRef hits across them | 8,491 |
| carrying a taxid in the m8 | **8,491, 100.0%** |

So the mechanism is sound and what is low is *depth* — which is a property of 5–19 residue peptide
queries, not of the pipeline, and matches FoldBench, where about half of the chains under 20 aa come
back query-only. The gate therefore checks the mechanism (floor 95%, on a sample of 60 alignments
against a full m8 pass) and only **reports** depth.

### Verification status, stated plainly

`phase_d_search_uniref.sh` **has not been run end to end.** All three batches were produced by the
four scripts it replaces. What has been verified is every gate's logic against the completed
artifacts, and the script's syntax and usage handling:

| checked | result |
|---|---|
| by-name coverage gate | 0 of 5,304 and 0 of 234 queries without an alignment |
| taxid coverage of hits | 100% on the short batch (4,478 of 4,478) |
| depth by length | 200 of 200 for 20+ residues; 25 of 200 under 20 |
| intermediate completeness | `uniref.a3m` and `prof_res` entry counts equal the query count on all three batches |
| thread derivation | `0-63`→64, `0-31`→32, `0-7`→8, `0,2,4`→3 |
| patch gate inputs | 8 `PATCHED` markers, both functions present |

What is **untested** is the expensive path through the new code: download, extract, marker creation
and removal, index build, and the search invocation itself. Closing that needs either a uniref30 cycle
(~1.5 h) or a run against the 280 MB `mini_swissprot` database, which would also settle the two open
questions from the review — whether the 301 GB index is required at all, and whether the staged search
equals an unstaged one empirically as well as by code reading.

### Phase D verification, 2026-10-11 — both review experiments run

`scripts/verify_minidb.sh` runs both on the 280 MB `mini_swissprot` database, which
`DEBUG_MINI_DB=1` extracts twice, once as each of the two databases, so the whole staged search fits
in 2.5 GB of its own directory. 45 seconds end to end.

**Staged equals unstaged, byte for byte.** 6 alignments compared, 6 identical, 0 differing. Confirms
the code reading (`search.py:157/172/210/231`): filtering is applied per database pass with the same
parameters and `mergedbs` concatenates by key without filtering the union, so splitting the search
across two runs cannot change the result.

**The 301 GB index is not required for taxonomy.** `search.py:134-139` sets `dbSuffix1` to `_seq`
without an index and `.idx` with one, and `convertalis` wants the taxonomy sidecars for whichever
suffix it gets. With an index that is `.idx_mapping` / `.idx_taxonomy`; without one it is
`_db_seq_mapping` / `_db_seq_taxonomy`, which nothing creates — so an index-free run fails with
`exit 1` and `uniref_tax.m8 not produced`, and the index *looks* mandatory. Creating those two
symlinks makes it work:

| route | rows | with a taxid | targets | disagreeing assignments |
|---|---|---|---|---|
| index-free + `_db_seq` sidecars | 1,309 | 1,309 | 610 | — |
| index-based | 1,309 | 1,309 | 610 | **0** |

Identical target→taxid assignments, nothing unique to either side.

**So the two claims the review asked me to separate resolve differently.** "An index-free search
targets `_db_seq` rather than the representatives" is **true** and is the whole mechanism. "No index
means `convertalis` cannot resolve taxids" is **false** — it was a coincidental consequence of nobody
creating the `_db_seq` sidecars.

**`BUILD_INDEX=0` added; the default stays 1.** The mini database's `_db` and `_db_seq` differ 4.5×
(127,203 against 573,661 entries), so the key-space concern was exercised — but the real uniref30
differs nearly 10× (36,293,491 representatives against 350,950,053 members), and that scale is
untested. A wrong-but-nonzero taxid is far worse here than a missing one, so the verified-safe route
remains the default.

Worth flipping it, though: it saves 301 GB and ~5 minutes per batch, and it would let uniref30
(450 GB) and colabfold_envdb (291 GB) coexist — which removes the **entire staging arrangement**: the
two-pass split, the kept intermediates, and the reason `colabfold_search_staging.patch` exists. The
test that would justify it is one real-uniref30 batch run both ways with the m8 assignments compared,
which costs a single ~1.5 h cycle.

### Parameter parity with Protenix — checked, no differences

Diffed our effective flags against `Protenix/scripts/colabfold_msa.py` and ColabFold's own argparse
defaults. Protenix's `LocalColabFoldConfig` sets `use_env=1`, `filter=1`, `db_load_mode=0`, and never
passes `--use-templates`; note that `if config.db_load_mode:` is falsy at 0, so that flag is never
passed and the default applies.

| flag | ColabFold default | Protenix effective | ours |
|---|---|---|---|
| `--db1` / `--db2` / `--db3` | `uniref30_2302_db`, `""`, `colabfold_envdb_202108_db` | defaults | defaults |
| `--use-env` | 1 | 1 | phase D 0 + phase E 1 → net 1 |
| `--use-env-pairing` | 0 | 0 | 0 |
| `--use-templates` | 0 | 0 (never passed) | 0 |
| `--filter` | 1 | 1 | not passed → 1 |
| `--db-load-mode` | 0 | 0 (never passed) | 0 |
| `--pairing_strategy` | 0 | 0 | 0 |
| `--unpack` | 1 | 1 | 1 |
| `--threads` | 64 | 64 | derived from `CORES`, 64 |

Every effective parameter matches, so `filter_param` — the `qsc`/`--diff`/`--max-seq-id` list that
shapes each alignment — is identical to Protenix's. Combined with the staged-equals-unstaged result,
the premise the review was worried about holds: our MSAs are built the way Protenix's were.

## Phase I

`phase_i_fold_validation.sh smoke | run | verify`, absorbing `val_fold.sh` and the smoke assertions
that used to live in `see_it_through.sh`.

**What the phase is for.** Not the poses. The deliverable is a cache complete enough to re-run
Protenix's confidence head offline, repeatedly, without a GPU: `z_trunk` and `s_trunk`, the head's
other two inputs, the pose in *model* atom order, the contact probabilities the ipTM summary is
computed from, the six input features it reads, and the python/torch/cuda RNG state.

### Three things that are easy to get wrong, and how the phase pins them

**bf16, not fp16.** The model runs bf16 and the head runs inside autocast, so bf16 is what it would
see anyway — and bf16 has fp32's exponent range where fp16 does not. This is not cosmetic: a float16
trunk dump once cost a good pose outright, 8vqf seed 300048 going from DockQ 0.495 to 0.056. The
smoke assertion checks the stored dtype.

**`TRUNK_DUMP_DIR`, not `PROTENIX_DUMP_TRUNK`.** `folder.py:831` turns the former into a per-fold path
`<dir>/<pdb>/<tag>_<seed>.npz` and refuses to overwrite. Pointing the latter at a directory makes
`protenix.py:1362` write `<dir>/trunk.npz` for every fold — one file, overwritten 1,580 times — and a
smoke test that only counted files would pass. So the assertion opens the npz and checks the tensors.

**`pool.csv` must exist.** `folder.py`'s `interfaces_by_pdb()` reads it to decide what a `target=*`
unit is scored on. Without it the fold runs and scores *nothing*.

### Gates before any GPU is taken

Every job json is opened and every `pairedMsaPath`, `unpairedMsaPath` and `templatesPath` checked to
exist. A dangling path folds without the file and nothing reports it, which is the most expensive way
to find a missing file. `pool.csv`'s structure count must equal the json count.

**Templates are required, by decision.** pxm2025 folds use them and validation is the arm compared
against pxm2025, so it is built the same way. The consequence for the write-up: FoldBench folds do
*not* use templates, so a head calibrated here is evaluated there without them. Decided 2026-10-11 to
keep templates; the asymmetry is stated rather than resolved.

### Smoke is the phase's own precondition

`run` refuses to start without `SMOKE_OK`. 16 hours of GPU and a paused PXM sweep should not be
reachable by calling scripts in the wrong order, so the ordering is enforced by the phase rather than
by whichever orchestrator happens to call it.

### PXM

Paused by killing **by PID**, never by pattern — a pattern kill matches this script. `fx_watch` is
stopped too, since it would otherwise relaunch PXM straight onto the cards we just took. Afterwards
the phase confirms the workers actually came back rather than assuming the relaunch worked, and
checks that the number of trunk dumps equals the number of folded units, because if those diverge the
cache cannot re-run the head on every pose, which is the entire point.

### Phase E review, first pass — all four findings fixed

**Per-query env growth was not gated at all.** The by-name check required every query to have a
non-empty a3m — but phase D already left every query one, so it passed regardless of what the env
pass did. And `AFTER > BEFORE` is a *directory* total, satisfied by other queries' growth. A chain
that received zero environmental sequences while the rest grew was invisible.

Fixed by snapshotting per-query sizes before the pass and comparing after, with the same
length-shaped floor phase D uses: chains of 20+ residues must have gained something (floor 90%),
shorter ones are exempt because many legitimately gain nothing. Negative-tested by running the check
with before == after: 0 of 5,305 long queries grew, so it fails at 0% against the floor, where the
old gate passed.

**The query line was not re-asserted after the pass.** Verifying each a3m's line 2 against its fasta
record is the direct test that qdb keys did not shift, and the env pass is exactly where `mergedbs`
could misalign them — the input checksum prevents a changed fasta, not a misaligned merge. Phase D's
review ran that check by hand on the input; phase E now runs it on the output.

The threshold distinguishes the two causes rather than demanding zero: a key shift produces
*thousands* of mismatches (the worked example was 4,825 of 5,305), while a single mismatch is the
known nucleic-acid chain whose U the search transcribed to X. Measured now: 1 mismatch in each batch,
against a tolerance of 53 for the main one.

**`${CFDB}` was unguarded in both `rm` globs** while `${DB:?}` and `${m:?}` were guarded —
unreachable today, but inconsistent with the stated discipline. Now `"${CFDB:?}"`.

**`FREE_DB` defaulted to 1**, so anything phase F revealed about the env half would have cost a 120 GB
re-download and another 90 minutes. Default is now 0, and `overnight.sh` frees the database after
phase F has passed, which is the first point at which nothing in the pipeline still reads it.

## Self-templating, and the one rule that governs templates everywhere

The most serious finding of the phase E-I review, and the only one that could have invalidated the
validation signal rather than biased a number.

### The leak

Protenix filters template hits by release date, but the **per-query part of that rule is train-only**:
`template_featurizer.py:443` sets `query_release_date` only when `stage == "train"`, so at inference
the cutoff collapses to the global `max_template_date`, 2021-09-30. The template database
(`pdb_seqres_2022_09_28.fasta`) covers depositions to September 2022.

So any structure deposited on or before 2021-09-30 can be handed **its own deposited entry** as a
template at 100% identity:

| | exposed | total |
|---|---|---|
| validation structures | **48** | 79 |
| training-pool entries | **1,354** | 2,886 |

That matters most for the training pool: a head trained on self-templated poses learns to be confident
under conditions it will never meet at evaluation, which corrupts the thing being fitted rather than
the thing being measured.

### Why their design is reasonable and still fails for us

At inference you normally predict something *new*, so a single global cap is already a complete
self-exclusion rule — nothing in the database can be your answer. The per-query check is redundant,
and they drop it. Our validation structures are from 1992–2021, so that assumption does not hold.

### The rule

    cutoff = min(2021-09-30, own release date − 60 days)

which is Protenix's own training formula (`template_featurizer.py:358-361`), with their
`DAYS_BEFORE_QUERY_DATE` of 60 (`template_utils.py:72`). Where it binds:

| set | binding rule | effect |
|---|---|---|
| pxm2025, FoldBench | the 2021-09-30 cap | unchanged from the folds already on disk |
| 33 validation structures (post-2021) | the cap | identical to what pxm2025 gets |
| 46 validation structures (pre-2021) | own date | own entry blocked |
| 1,354 training entries (pre-2021) | own date | own entry blocked |

Dates come from Protenix's own `release_date_cache.json` (254,841 entries) and fall back to SAbDab2's
deposition date. Coverage is complete: every pool structure has one from at least one source. A hit
whose own date is unknown is **dropped**, matching Protenix's prefilter, which returns
`"{pdb_id} date unknown"` rather than admitting it.

### Why the global cap is kept rather than dropped

Protenix's *training* stage uses `max_template_date = "3000-01-01"` — no cap at all — so matching
their training conditions would mean dropping it. Three reasons not to, the last decisive:

- **pxm2025 is entirely post-2021**: 90 of 123 complexes have dates, spanning 2023-08-26 to
  2025-11-12, all after the cap. For pxm2025 the cap already *is* a complete self-exclusion rule.
- **33 of those 123 have no date in any source**, so a per-query rule could not be applied there
  uniformly even if we wanted it.
- **the benchmarks will be refolded to apply the retrained head.** Widening their templates at the
  same time would mean a change in DockQ or ipTM could be the head or the templates, with no way to
  tell which. The refold must vary one thing.

### Enforcement is structural

Phase H writes `<templatesPath>.filtered` beside each template file, recording the cutoff applied.
**Phase I refuses to fold any job whose `templatesPath` lacks that marker.** So a fold cannot
silently use unfiltered templates, whichever job set it is handed — validation or training. Measured
on a real template file: 30,092 hits in, 25,624 kept, 4,379 dropped as too recent, 89 dropped as
undated, and a second run is a no-op.

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

## Phase E-I review, round one: all findings addressed

### Phase F

**The emptiness of pairing.a3m was computed and printed but never gated.** This phase exists to
produce pairing, and its completion criterion did not test pairing. If step4 ran but parsed no
taxids, every `pairing.a3m` is empty, the directory count is correct, every name is one we indexed,
and the phase reports OK. Step 2 proves taxids are *in* the alignments; nothing proved step4 *used*
them. Now gated on the fraction of chains of 20+ residues with a non-empty `pairing.a3m`
(`--min-pairing-frac`, default 0.50), with shorter chains reported separately because a peptide that
retrieved nothing has nothing to pair.

**Within-batch sequence collisions were unchecked.** The clash test was cross-batch only;
`seq_to_index[seq] = name` overwrites silently, so two chains in one batch sharing a sequence yield
one directory and the other chain gets no MSA at all. `len(s2x) != len(rows)` catches it in two lines.

**`--min-frac 0.98` permitted 106 silent losses on batch 1** while the only documented legitimate
cause is one nucleic-acid chain per batch. Replaced by classification: a drop for any reason other
than that known sequence mismatch is fatal, the mismatches themselves are capped
(`--max-seq-mismatch`, default 5), and the index must then cover exactly `queries − mismatches`. Same
all-or-nothing discipline phase A uses, rather than a percentage sized to cover the known case.

**`os.path.exists` is False for a dangling symlink**, so a stale `uniref_tax.m8` link made
`os.symlink` raise `FileExistsError` instead of being replaced. Now `os.path.lexists`, which sees the
link itself.

### Phase I

**`--nsample` produces N poses per unit but the cache was gated as one npz.** `z_trunk` and `s_trunk`
are rightly per seed, but `ci_coordinate` and `ci_contact_probs` are per *pose*, and the assertion
only checked that the keys existed — an npz holding the last of five looks identical to one holding
all five. If it were one, four of every five poses would have no cached head inputs and the stated
deliverable would be 20% met, unrecoverably: repeating a seed does not reproduce its poses. The smoke
test now asserts the leading dimension equals `nsample`, after squeezing batch dimensions of 1.

**No `trap`, so a killed run never resumed PXM.** Every other phase restores what it changes. A run
killed between the pause and the resume left the sweep down and `fx_watch` dead indefinitely, because
the thing that would have restarted them was the script that died. INT, TERM and EXIT now route
through one `resume_pxm`, which is also the normal path, so there is a single code path.

**`run_workers` waited forever on a hung worker**, with PXM paused for the duration, and the
15-minute no-change line was logged and acted on by nothing. Two bounds now — total elapsed
(`MAX_WAIT_MIN`) and time since the last completed unit (`STALL_MIN`, default 90) — and on either the
workers are killed by PID so the round ends and the trap restores PXM.

**`results.csv` was read positionally** for DockQ and ipTM (`$7`, `$8`). `csv_numeric` resolves both
from the header row and exits non-zero if the column is absent, removing a silent dependency on
folder.py's column order. Verified against the live PXM results: 19,560 numeric rows for each, and a
missing column exits 2.

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

**`results/val_clusters.csv` was drawn from the 557-cluster pool and is kept.** Re-running phase B
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

## Analysis conventions, recorded before any numbers exist

**Equal cluster weighting.** Improvements are evaluated per cluster, not per pose or per interface.
The independent sample size is ~50 clusters; 157 interfaces and 1,580 folds are repeated measurements
within them, and weighting by interface would let one well-studied antigen dominate.

**Uncertainty by resampling clusters**, not poses or interfaces — the resampling unit has to match the
independence unit.

**Pre- and post-cutoff reported separately, by each folded structure's own deposition date.**
`val_jobs/manifest.csv` carries `deposited` and `post_cutoff` per structure. A cluster's median date
does not settle it: a cluster whose median is after 2021-09-30 can still have contributed a
representative from before, and freezing the trunk does not make that representative unseen by the
pretrained weights. Currently **30 of 79** folded structures are post-cutoff.

**The split is fixed before any DockQ is examined.** `phase_b_define_val_clusters.py` refuses to
overwrite an existing `val_clusters.csv` without `--force`, so a plain re-run cannot silently redraw
it. A re-run instead regenerates `results/train_clusters.csv` from the pinned split, which doubles as
the check that all 50 validation clusters are still in the pool.

## Final numbers, after the all-227 leakage check

**Decided 2026-10-11.** Filter 2 now queries the antigen chains of **all 227** benchmark complexes as
they are actually folded — `proteinChain.sequence` from the job jsons, classified by the same hmmscan
pass, keeping the 209 non-antibody chains — rather than only the 39 SAbDab2 does not list.

### Why the cluster-id exclusion was not enough

It is only as complete as SAbDab2's record of what each complex contains, and that record omits
chains the benchmark folds. `8vui`'s single SAbDab2 row reads `agtypes='ION'`, `agclusters=nan`: it is
catalogued as an antibody bound to an ion and its protein chain is never recorded as an antigen, so
its `ab_ag_cluster` was formed without that protein and there is no antigen-cluster link to follow to
the training cluster whose antigen is identical to it. Taking the 188 listed complexes' clusters on
trust left 18 such clusters in the pool, ten at identity 1.000.

This is not a stricter threshold than SAbDab2's. It is the same criterion applied to chains SAbDab2
never wrote down.

### The cascade

| | clusters | |
|---|---|---|
| after the id lookup | 561 | 140 clusters removed by name |
| +4 | 557 | antigen homology to the 39 unlisted complexes |
| +2 | 555 | `--cov-mode 1`, which exposed two 9ms8 matches the old setting hid |
| +18 | **537** | all 227 complexes, on the chains they fold |

| | before | after |
|---|---|---|
| training clusters | 555 | **537** |
| validation clusters | 50 | **50** (redrawn) |
| training clusters after removing validation | 505 | **487** |
| folded validation structures | 79 | **77** |
| validation interfaces | 157 | **157** |
| post-cutoff folded structures | 30 | **28** |

All 157 interfaces resolve against their staged reference, and all 197 chains the 77 structures need
are already in the searched query set — 537 is a subset of the 557-cluster pool that set was built
from, so no further MSA work is implied. The largest folded complex is now 2,289 tokens, up from
1,543; PXM folded 2,275-token complexes at about 37.5 GB on a 97 GB card, so it fits.

### The validation split was redrawn once, from the final pool, and is now frozen

Four pinned validation clusters — 125, 266, 278, 922 — were among the 18 and could not stay:
validating on a cluster that shares an antigen with a benchmark complex is validating on data that
leaks into the benchmark.

`--prune` removes exactly those and keeps the rest pinned, and that is the right repair **once folding
has begun**. It was not the right repair here. The rule against redrawing a validation split exists to
stop a split being chosen after performance has been seen — and no validation fold had run, so no
information had flowed from results to the choice. Pruning instead left 46 clusters with two date bins
holding 8 rather than the designed 10, and a split that could not be regenerated from the code,
because it was an artifact of a superseded pool plus a manual repair.

So the split was redrawn once from the final 537-cluster pool: **50 clusters, ten per date bin, 271
structures (9.7%)**. It regenerates byte-identically from the committed code and seed, verified. From
here it is frozen, and `phase_b_define_val_clusters.py` refuses to redraw without `--force` while
failing loudly — naming the clusters — if the pinned split ever stops being a subset of the pool.
