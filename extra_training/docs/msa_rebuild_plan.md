# Plan: rebuild the MSAs for the 155 corrected queries

**Revision 3, 2026-10-14.** Revision 2 (`e7de1264`, `a2d58b51`) was reviewed and four more defects were found. All four reproduce and are corrected below. One of them DISSOLVES the blocker revision 2 was built around. Nothing in this plan has run.

Revision history: revision 1 (`89b35118`) was corrected by revision 2 (`e7de1264`, `a2d58b51`) after
review round 1 found four execution defects; revision 2 is corrected here after review round 2 found
four more, one of which removes the blocker revision 2 was built around.

## Why this is needed

125 training and 16 validation structures were folded from queries that omit internal native
residues (`train_val_splits/refold_internal_omissions*.csv`). Their corrected queries are a few
residues longer, and an a3m is one match column per query residue, so the stored alignments no longer
describe the sequences. 155 unique changed sequences need new MSAs before those 2,850 folds can run.

## THE ENVDB BLOCKER IS DISSOLVED: the conversion was never invoked

Revision 2 called the mirror route blocked because the TSV form of envdb "produced 265 GB that
converted to nothing and still exited 0". Review round 2 predicted the cause and a direct test
confirms it: **`tsv2exprofiledb` was never called.**

`setup_databases.sh:20` defaults `FAST_PREBUILT_DATABASES=1`, and that one flag controls two
unrelated things:

| what it controls | at `=1` |
|---|---|
| which URL to download (`:78-85`) | the prebuilt `.db.tar.gz` |
| whether to convert after extracting (`:146`, `:188`) | **skip `tsv2exprofiledb`** |

`phase_e_search_envdb.sh` fetches the tarball itself and writes it to
`$DB/colabfold_envdb_202108.tar.gz`, then calls `setup_databases.sh`. That bypasses the download half
of the flag but not the conversion half. Put a TSV archive at that filename and the script extracts
it, sees the flag, skips the conversion, and exits 0.

Measured on a 223-byte stand-in archive with the real script:

| `FAST_PREBUILT_DATABASES` | `tsv2exprofiledb` invoked | exit |
|---|---|---|
| `1` (default) | **0 times** | 0 |
| `0` | **1 time** | 0 |

So nothing failed to convert. **Nothing was asked to convert.** The fix is to pass
`FAST_PREBUILT_DATABASES=0` alongside the `CFDB` and `MMSEQS_NO_INDEX=1` that phase E already passes.

**This means the mirror is a candidate complete substitute after all**, and the conclusion of
revision 2 -- that the mirror can only supply half -- is withdrawn. What remains unproven is whether
the conversion SUCCEEDS on the real 118 GB archive; only that it is now actually attempted. The
`mini_swissprot` harness cannot test that, because `DEBUG_MINI_DB=1` takes a `mvdb` path that never
touches `tsv2exprofiledb`, and the mini tarball is hosted on the server that is down.

## The source

`opendata.mmseqs.org` is 503 on every path. The GWDG mirror (`wwwuser.gwdg.de/~compbiol/colabfold/`,
the Söding lab's host) serves the same releases, TSV packaging only:

| file | mirror | status |
|---|---|---|
| `uniref30_2302.tar.gz` | 102,918,187,842 bytes | **200** |
| `colabfold_envdb_202108.tar.gz` | 117,965,643,010 bytes | **200** |
| either `.db.tar.gz` | — | **404** |

Versions are `uniref30_2302` and `colabfold_envdb_202108` — exactly what the existing 5,549 MSAs were
built from. `uniref30_2302.tar.gz` carries a bundled `md5sum` for six members; the envdb archive's
first member is `colabfold_envdb_202108_h.tsv`, 31.6 GB, dated 2021-09-13.

## The four review defects, corrected

### 1. Disk accounting was incoherent

Revision 1 said "~950 GB peak" while also retaining 1,042 GB of databases. Both cannot be true.
Properly enumerated, uniref30 from the TSV distribution:

| item | GB | present at peak? |
|---|---|---|
| archive `uniref30_2302.tar.gz` | 96 | yes, until extraction completes |
| extracted TSVs (`.tsv`, `_h.tsv`, `_aln.tsv`, `_seq.tsv`) | ~420 | yes |
| `tsv2exprofiledb` output (`_db_seq`, `_db_aln`, `_db_h`, …) | ~420 | yes, built alongside the TSVs |
| conversion intermediates | unmeasured | yes |
| **peak, uniref30 alone** | **~936 + intermediates** | |
| then: delete archive and TSVs, keep the db | ~420 | |
| envdb archive + extracted + db, same shape | ~700 | **cannot coexist with uniref30** |

Against 1,411 GB free, uniref30 alone fits with roughly 475 GB of headroom, and the unmeasured
conversion intermediates come out of that. Both databases at once do not fit, which is what phase E
already enforces.

**THE INDEX NEEDS `BUILD_INDEX=0`, NOT `MMSEQS_NO_INDEX=1`.** Revision 2 said the 301 GB index is not
built because `MMSEQS_NO_INDEX=1` is passed. That is only true of `setup_databases.sh`.
`phase_d_search_uniref.sh:216` builds its own index and defaults `BUILD_INDEX=1`:

    if [ "${BUILD_INDEX:-1}" = "0" ]; then   # skip the index, give _db_seq taxonomy sidecars instead
    ...
    $M createindex uniref30_2302_db tmp1 ...

So with the documented settings the peak would be ~936 + 301 = **~1,237 GB against 1,411 GB free**,
before intermediates — far tighter than revision 2 implied and possibly not viable. `BUILD_INDEX=0`
is required, and phase D's own comment says it is safe: it "saves 301 GB and ~5 minutes per batch"
and hands `_db_seq` the taxonomy sidecars instead, which `verify_minidb.sh` experiment 1 exists to
confirm.

### 2. `FREE_DB=0` throughout contradicts phase E

`phase_e_search_envdb.sh:112` refuses to start when uniref30 is still present:

    [ -s "$DB/uniref30_2302_db_seq" ] && die "uniref30 is still on disk; the two databases do not
     both fit in the quota. Free uniref30 first (phase D's FREE_DB=1 does it)."

So the setting is per phase, not global: **phase D runs with `FREE_DB=1`** so uniref30 is released
when its search completes, and **phase E runs with `FREE_DB=0`** so envdb survives for phase F to be
retried without a 118 GB re-download. Revision 1's blanket `FREE_DB=0` would have deadlocked at
phase E.

The consequence is that the uniref30 resume path matters after all: once phase D frees it, a failure
in phase E cannot be repaired by re-running phase D without another 96 GB download. That is what the
staging patch's `uniref.a3m` resume is for, and it has never been exercised twice.

### 3. Phase F and the job paths were missing entirely

Revision 1 went from the searches straight to folding. Two steps sit in between.

**Phase F** (`phase_f_pair_and_split.py`) is what turns raw search output into `pairing.a3m` and
`non_pairing.a3m` — taxonomy annotation and the paired/unpaired split. Without it there is no MSA in
the form a job json references.

**The job paths.** `phase_j_corrected_jobs.py` changed only `proteinChain.sequence`, leaving
`pairedMsaPath`, `unpairedMsaPath` and `templatesPath` pointing into the OLD store. A corrected
221-residue query would have been folded against a 214-column alignment built for the sequence it
replaced — a length contradiction that is either rejected or absorbed silently. **Fixed**: those three
keys are repointed into `mmcif_msa_corrected`, and a job whose corrected-role paths still resolve
into the old store is refused.

Verified over all 141 corrected jsons:

| | count |
|---|---|
| corrected-role paths into the corrected store | 486 |
| corrected-role paths still in the old store | **0** |
| unchanged-role paths left in the old store | 825 — correct, those sequences never changed |

That last row is the subtlety: only the corrected chains move. A complex has three chains and
typically one is corrected; the others' alignments still describe them exactly.

Fixing this exposed two bugs of my own, both recorded in the code: the gate ran AFTER the file was
written, so a rejected job was already on disk; and it compared paths with a bare `startswith`, which
matched `mmcif_msa_corrected` against `mmcif_msa` and rejected all 125 structures for pointing at a
store they had just been moved out of.

### 4. The unchanged-query control is now a required gate, and compares more than depth

**Gate, before step 6.** Take one structure that is NOT being corrected, rebuild its MSA through the
entire new path, and compare to its stored MSA on:

| compared | pass condition |
|---|---|
| hit identities | the set of homolog accessions, not just how many |
| depth | within a stated tolerance, with any loss attributed |
| alignment coverage | per-column occupancy profile across the query |
| taxonomy assignments | the taxid on each retained hit |
| match columns | every row exactly `len(query)` |

Same release names do not establish identical search results, and the existing uniref MSAs came from
the PREBUILT database while these would come from the TSV build. If the control does not pass, the
batch does not run.

**AND IT MUST COMPARE LIKE WITH LIKE.** Revision 2 had the control run phase D + F and compare to the
stored MSA. That comparison cannot pass: the stored `pairing.a3m` is the merge of the uniref AND
environmental passes, so a uniref-only rebuild differs from it by every envdb hit — a guaranteed
failure that says nothing about the TSV build. Either:

* compare the **UniRef component separately** — phase D's own `uniref.a3m` intermediate against the
  stored one, before phase F merges anything; or
* run the **full control after envdb is available**, D + E + F, against the stored merge.

The first is available now and is the one that tests the TSV conversion. The second is the real
equivalence check and waits on envdb.

## The plan

Every row reflects revision 3's corrections. Earlier revisions left the prose updated and this table
stale, which is the worse half of the mistake: the table is what someone executes from.

| # | step | settings that matter | cost | gate |
|---|---|---|---|---|
| 1 | fetch `uniref30_2302.tar.gz` from the mirror | — | 96 GB transfer | content-length ≥ 1 GB, then `tar -tzf` lists cleanly |
| 2 | extract, verify against the bundled `uniref30_2302.md5sum` | — | ~420 GB | every listed member matches; a mismatch stops the build |
| 3 | convert to a searchable db | **`FAST_PREBUILT_DATABASES=0`**, `MMSEQS_NO_INDEX=1` | hours CPU | `_db_seq`, `_db_aln`, `_db_h` exist and are non-empty — never a `READY` marker, and never the exit code: this step exits 0 when it does nothing |
| 4 | apply `uniref30_2302_newtaxonomy.tar.gz` (on disk, 1.97 GB) | — | minutes | taxonomy md5 equals the copy we already hold, `83105388…` |
| 5 | **CONTROL, conversion**: one unchanged query through steps 1-4 + phase D, comparing **phase D's own `uniref.a3m` against the stored one, BEFORE phase F merges anything** | `BUILD_INDEX=0` | hours | hit identities, depth with any loss attributed, per-column coverage, taxonomy assignments, every row exactly `len(query)` match columns |
| 6 | `phase_d_search_uniref.sh` on the 155 corrected queries | **`BUILD_INDEX=0`** (the 301 GB index is what the disk budget cannot afford), `FREE_DB=1` (phase E refuses to start while uniref30 is present) | hours | `MIN_LONG_HIT_PCT`, as the original used |
| 7 | fetch and convert envdb, then `phase_e_search_envdb.sh` | **`FAST_PREBUILT_DATABASES=0`** — this is the whole fix; the flag's default of 1 is why the earlier attempt converted nothing. `FREE_DB=0` so envdb survives a phase F retry | 118 GB + hours | `${CFDB}_db_seq`, `_db_aln`, `_db_h` non-empty. **UNPROVEN on the real archive**: the test so far shows only that the conversion is now invoked, not that it succeeds on 118 GB |
| 8 | `phase_f_pair_and_split.py` into `mmcif_msa_corrected` | — | CPU | every corrected role has `pairing.a3m` and `non_pairing.a3m` |
| 9 | **CONTROL, equivalence**: the same unchanged query through D + E + F, against the stored merged MSA | — | hours | the five comparisons of step 5, now on like-for-like inputs. **This is the real equivalence check** and it cannot run before step 7 |
| 10 | `phase_h_search_templates.py` for the corrected queries | — | CPU | date-filter marker, as before |
| 11 | re-run `phase_j_corrected_jobs.py` with `--expect 125` (and `--expect 16`), WITHOUT `--allow-missing-msa` | — | seconds | exits non-zero unless every expected job validates; anything it cannot validate is renamed `.notready` |
| 12 | smoke-test `11zw` end to end | — | one fold | 7 residues restored, masked in the label, flanks retained |
| 13 | queue 2,850 refolds into `training/fold_corrected/` | — | ~19 h GPU | separate tree; step 9 and step 11 must both have passed |

### THE STEP 5 ACCEPTANCE RULES

Revision 3 proposed 1% tolerances and a "no gained hits" rule. Both are withdrawn after review.

**"No gained hits" was unjustified.** Different database ordering can REPLACE retained hits, which
produces gains and losses together. A gain on its own does not establish corruption, so treating it
as a harder signal than a loss was wrong.

**And the percentage tolerances were screening rules presented as guarantees.** They can be passed by
an alignment that is simply wrong: shifted hit alignments with IDENTICAL hit ids, depth, taxonomy and
per-column occupancy were constructed during review. Occupancy counts how many rows occupy a column,
not whether the residues in it are the right ones.

So the rules are absolute, and comparison reaches the alignments themselves:

| # | rule |
|---|---|
| 1 | **Absolute correctness checks.** The query row equals the corrected sequence exactly. Every row carries exactly `len(query)` match columns. Taxonomy assignments are unchanged for shared hits. No tolerance on any of these |
| 2 | **Compare the actual alignments.** Match shared hits by accession AND segment, then compare each one's residue/gap string and its insertion count. Ignore row order and cosmetic metadata. This is what catches a shift that every aggregate statistic passes |
| 3 | **Start from exact normalised agreement** across the five unchanged queries. If differences appear, report GAINED hits, LOST hits and CHANGED alignments as three separate categories, and investigate before approving any fold. **Do not tune search parameters to force a pass** -- a parameter chosen to make the control agree makes the control meaningless |
| 4 | one `flock` lock across both routes, held by the process doing the build or search; the watcher reports recovery and does nothing while another route holds it |

Rule 3 is the one that matters most. The failure mode is not "the control fails" -- it is "the control
is made to pass". Differences are information to chase, not a number to get under.

### RUN THE CONTROL QUERIES IN THE SAME BATCH AS THE CORRECTED ONES

This is the change that avoids repeating a search, and it is a change to the ORDER of the plan, not
just its checks.

Phase D keeps per-batch intermediates -- `uniref.a3m` and `prof_res` -- and phase E resumes the
environmental pass from them once UniRef30 is deleted. Each batch gets its own output directory
precisely because the keys are positional and a second batch would shift them.

So if the five control queries are searched as a SEPARATE batch, their intermediates live in a
separate directory, and after UniRef30 is freed the environmental pass for them is no longer
possible without another 96 GB download. The full D + E + F equivalence check -- the only comparison
that can be made against the stored merged MSAs -- would be unreachable.

Therefore: **the five control queries go into `query.fasta` alongside the 155 corrected ones**, as one
batch of 160. Their intermediates are retained with the rest, the environmental pass covers them for
free, and the final comparison reuses a search that has already happened.

And before UniRef30 is deleted, `check_mmseqs_db` validates the data/index consistency of
`uniref.a3m` and `prof_res` for that batch -- not just that the files exist. Getting that wrong is
what costs the re-download.

### What is proven, and what is not

**Proven:** the envdb conversion was never invoked, and `FAST_PREBUILT_DATABASES=0` invokes it —
measured on the real script, 0 calls at the default against 1 call with the flag. Revision 2's claim
that envdb is unobtainable from the mirror is withdrawn, and step 7 is no longer a blocker.

**Not proven, and these are the two that decide whether the route works:**

1. that `tsv2exprofiledb` SUCCEEDS on the real archives. The 223-byte stand-in showed invocation, not
   success, and the `mini_swissprot` harness cannot test it — `DEBUG_MINI_DB=1` takes a `mvdb` path
   that never reaches the conversion, and the mini tarball is on the host that is down.
2. that a TSV-built database gives the same search results as the prebuilt one the existing 5,549
   MSAs came from. That is what step 5 tests for uniref and step 9 for the whole thing.

**So the mirror route is now plausible end to end, and steps 1-6 are worth starting.** They were not
an hour ago, when envdb looked unobtainable and the uniref half was therefore useless in both
branches. `await_msa_server.sh` keeps polling the canonical host every 5 minutes; if it returns, the
prebuilt databases are simpler and we switch to them, and whatever the mirror build has reached is
discarded without loss.

The 141 corrected job jsons already exist, written with `--allow-missing-msa`, so their paths are
correct and pending. **A fold must not be queued from them until step 11 passes without that flag.**

## What is not blocked

GPUs: FoldBench top-off (612), then 663 training folds of structures that are not being refolded.
Neither needs any of the above. `await_msa_server.sh` polls the canonical host every 5 minutes.

## Risks

| risk | mitigation | residual |
|---|---|---|
| the TSV conversion fails on the real archive | `FAST_PREBUILT_DATABASES=0` makes it run at all, which it never did before; the gate is non-empty `_db_seq`/`_db_aln`/`_db_h`, never the exit code | UNPROVEN at 118 GB. If it fails we are back to waiting on the canonical host, with gap-padding as the fallback |
| a TSV-built db searches differently from the prebuilt one | steps 5 and 9, comparing hit identities, coverage and taxonomy rather than depth alone | unknown until step 5 runs; the existing 5,549 MSAs came from the prebuilt db |
| `tsv2exprofiledb` differs from the prebuilt db | step 5's control | if it fails we learn the mirror route is unusable for uniref30 too |
| phase D frees uniref30, then phase E fails | the staging patch's `uniref.a3m` resume | never exercised twice; the fallback is a 96 GB re-download |
| conversion intermediates exhaust the disk | ~475 GB headroom; guard warns at 2,900 GB and will not kill the build | intermediates unmeasured |
| the mirror is a repackaged archive | md5sums verified against extracted source | matching names alone would not catch it |

## Rejected, recorded so they are not re-proposed

| approach | why not |
|---|---|
| gap-padding | verified reversible over 30,435 rows, but no aligned evidence at the restored positions. Only 0.2-7.4% of saved homologs carry residues at those loci, so it is near the recoverable ceiling. **Kept as the fallback if the server does not return** |
| mmseqs2 realignment against saved hits as a local database | failed its control: unchanged query lost 17.8% of depth (14,377 → 11,817, 0 gained), metadata stripped, rows with 215 match columns against a 214-residue query |
| uniref30 only, skipping envdb | systematically shallower than the 5,549 it would be compared against |
| a different database release | version consistency matters more than speed |
