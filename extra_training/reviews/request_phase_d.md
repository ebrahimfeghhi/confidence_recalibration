# Review request: phase D

`extra_training/phases/phase_d_search_uniref.sh` (branch `epitope-recall-and-matrix-signposting`).
Context for phases A–C and the responses to your earlier passes are in this directory.

## What phase D is

It takes the sequences phase C listed and searches each against **uniref30_2302**, producing per
chain:

| | |
|---|---|
| `<name>.a3m` | the alignment |
| `uniref.a3m` | the raw search output, deliberately kept |
| `uniref_tax.m8` | the NCBI taxid of every hit, which phase F needs to pair chains |
| `prof_res` | the uniref profile, deliberately kept — phase E searches *from* this, not from the a3m |

Three batches have run or are running, all with identical parameters:

| query fasta | output | queries | state |
|---|---|---|---|
| `query.fasta` | `msa_initial_idx` | 5,304 | done |
| `query_short.fasta` | `msa_short_idx` | 234 | done |
| `query_short_TODO_bench.fasta` | `msa_bench_short_idx` | 11 | running |

This script replaces four that between them did the same job (`phase1_uniref.sh`,
`phase1_index.sh`, `phase1_search_idx.sh`, `phase1b_short.sh`). The first two batches were produced
by those; they are preserved in git at `c1cbe087^`.

## The design decision I most want checked

**uniref30 and colabfold_envdb cannot coexist on disk.** uniref30 is 450 GB extracted plus a 301 GB
index; the environmental database is 291 GB; the quota is 1,536 GB and the rest of the project uses
~550 GB. So the two searches are staged: search uniref30, keep its intermediates, **delete the
database**, then let phase E search the environmental database resuming from those intermediates.

That requires `phases/phase_d_colabfold_search_staging.patch` — 67 added lines to
`colabfold/mmseqs/search.py`, in four hunks:

1. `_emit_taxid()` / `_keep_intermediates()`, read from the environment.
2. Drop uniref30 from the precondition check when `uniref.a3m` already exists, so the resume path is
   reachable at all. Unpatched, `colabfold_search` raises before it gets there.
3. Emit `uniref_tax.m8` via `convertalis`, placed before the `rmdb` that destroys
   `res_exp_realign_filter`, and tolerant of failure.
4. Keep `uniref.a3m` and `prof_res` instead of deleting them, and tolerate an already-removed tmp dir
   on the second pass.

**The question: is the staged result identical to an unstaged single run?** My argument that it is:
the environmental search is driven from `prof_res` and the env database, both of which are present
when phase E runs, so its inputs are unchanged and only the wall-clock separation differs. I would
like that argument attacked, because if it is wrong our MSAs differ systematically from how
ColabFold normally builds them — and therefore from how Protenix's own training MSAs were built,
which is the one thing this whole exercise is trying not to perturb.

What I have verified, rather than assumed:

- `prof_res` survived both completed batches and covers **every** query: 5,305 index entries for
  5,304 queries and 235 for 234. The two extras are the nucleic-acid chains phase C's new
  antigen-type filter removed, so coverage is complete.
- `uniref.a3m` likewise, 5,305 and 235 entries.
- Every query in both fastas has its own non-empty `.a3m`: 0 missing of 5,304 and 0 of 234.

## The second claim I want checked

**Why a 301 GB index is built**, when `setup_databases.sh` itself calls it "not useful for
colabfold_search in most cases" and it buys nothing on search speed at this batch size.

It is built for taxonomy, not speed. `convertalis` resolves taxids through
`uniref30_2302_db.idx_mapping` and `.idx_taxonomy`, and an index-free search instead targets
`_db_seq`, whose 350,950,053 member keys are not in the same key space as the mapping table's
36,293,491 representatives. So: no index → no taxids → no paired MSA → phase F has nothing to split
on.

I believe this but have not proved it the hard way — I have not run an index-free search and
confirmed that `uniref_tax.m8` comes back empty or wrong. If that claim is false, we are spending
301 GB and ~6 minutes per batch for nothing, and the disk pressure that has caused most of this
project's failures largely goes away.

## Defects I already found and fixed, for calibration

- **The success check was vacuous.** The old build verified itself with `mmseqs dbtype <db>`, which
  returns exit 0 for a database that does not exist. Combined with `setup_databases.sh` touching
  `UNIREF30_READY` *before* the steps that can fail, nothing in that path could detect a failed
  build. Now five data files must be non-empty.
- **No coverage gate at all.** The old search script ended with `exec colabfold_search`. Now every
  query header must have its own non-empty a3m — by name, not by count, because a count satisfied
  with `-ge` cannot distinguish "this batch is covered" from "this directory happens to be full".
- **A false claim about resumability**, in the old script and in my first draft of the new one: that
  `colabfold_search` "skips a query whose `<name>.a3m` already exists". It does not. The skip is per
  *database search*, keyed on the intermediate — `search.py:191` logs "Skipping <uniref_db> search
  because uniref.a3m already exists". A re-run therefore skips the expensive search but rewrites all
  per-query files, so a partial output cannot be topped up query-by-query.
- **Contradictory index rationale** across two scripts, one calling it unnecessary and one required.

## Specific things I am unsure about

1. **`ALLOW_THIN=1` bypasses the taxid gate**, and the 11-chain benchmark batch is running with it.
   The justification is that short peptides legitimately retrieve nothing — 4 of FoldBench's 8 chains
   under 20 aa come back query-only — but it does mean a genuinely broken search for that batch
   would pass quietly. Is there a better discriminator than "the batch is small"?
2. **`MIN_TAX=10` over the first 2,000 rows of the m8** is arbitrary. The file is 77 GB for batch 1,
   so a full scan is expensive, but the first 2,000 rows are not a sample of anything.
3. **Flag choices**: `--use-env 0 --use-templates 0 --db-load-mode 0`, and `qsc`/`--diff`/
   `--max-seq-id` left at ColabFold defaults. These should match how Protenix's own MSAs were built
   (`scripts/colabfold_msa.py` defaults to `db_load_mode 0`), but I have not diffed our effective
   parameters against theirs end to end.
4. **The taxonomy tarball.** `setup_databases.sh` needs `uniref30_2302_newtaxonomy.tar.gz` for
   `_db_taxonomy`. We fetch it, but I have not checked whether the taxonomy it installs matches the
   version the uniref30 release expects, or what happens if they disagree.

## Not in scope

Phases E–I are not yet consolidated or reviewed. Phase E is the environmental pass that depends on
everything above.
