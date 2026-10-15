# Plan: rebuild the MSAs for the 155 corrected queries

**Revision 2, 2026-10-14.** Revision 1 (`89b35118`) was reviewed and four execution defects were
found. All four are corrected below, and verifying them turned up a fifth that invalidates half of
revision 1's approach. Nothing in this plan has run.

## Why this is needed

125 training and 16 validation structures were folded from queries that omit internal native
residues (`train_val_splits/refold_internal_omissions*.csv`). Their corrected queries are a few
residues longer, and an a3m is one match column per query residue, so the stored alignments no longer
describe the sequences. 155 unique changed sequences need new MSAs before those 2,850 folds can run.

## THE BLOCKER REVISION 1 MISSED: the mirror has no prebuilt envdb

`opendata.mmseqs.org` is 503 on every path. The GWDG mirror
(`wwwuser.gwdg.de/~compbiol/colabfold/`, the Söding lab's host) serves the same releases, but only in
the TSV packaging:

| file | mirror | status |
|---|---|---|
| `uniref30_2302.tar.gz` | 102,918,187,842 bytes | **200** |
| `colabfold_envdb_202108.tar.gz` | 117,965,643,010 bytes | **200** |
| `uniref30_2302.db.tar.gz` | — | **404** |
| `colabfold_envdb_202108.db.tar.gz` | — | **404** |

`setup_databases.sh` distributes both packagings of each release and says so itself: `${DB}.db.tar.gz`
is the "new prebuilt ... don't require calling tsv2exprofiledb", `${DB}.tar.gz` the "old .tsv +
tsv2exprofiledb" form. Our phases fetch the prebuilt one.

**And the TSV form of envdb is a RECORDED FAILURE in this environment.**
`phase_e_search_envdb.sh:116-119` exists because of it:

> THE PREBUILT DATABASE. setup_databases.sh defaults FAST_PREBUILT_DATABASES=1, which extracts
> `${CFDB}.tar.gz` straight to MMseqs format and never calls tsv2exprofiledb. The URL above is the
> .db.tar.gz for that reason: **the plain .tar.gz is the tsv form, and extracting it once produced
> 265 GB that converted to nothing and still exited 0.**

So the mirror can plausibly supply the uniref30 half and **cannot** supply the envdb half by any path
this project has got working. Revision 1 presented the mirror as a complete substitute. It is not.

### What that leaves

| option | uniref30 | envdb | verdict |
|---|---|---|---|
| wait for `opendata.mmseqs.org` | prebuilt | prebuilt | **the only route to a complete MSA**; polled every 5 min |
| GWDG mirror | TSV + `tsv2exprofiledb`, untested here | TSV, known to produce nothing | incomplete |
| gap-padding (`phase_j_remap_msa.py`) | n/a | n/a | fallback; no aligned evidence at ~2% of positions |

A uniref30-only MSA is NOT a valid substitute: the existing 5,549 were built from both databases, so
dropping envdb for these 155 makes them systematically shallower than everything they are compared
against. The envdb pass is what the project's own phase E calls the source of "depth uniref30 cannot
reach".

### Do NOT start the mirror download yet — it is not useful in either branch

Revision 2 first said it was "worth starting for uniref30 alone, because it is free while the GPUs
work". That is wrong, and the correction belongs in the plan rather than in a conversation. Work
through both outcomes:

| if `opendata.mmseqs.org` … | what we would use for uniref30 | was the mirror build useful? |
|---|---|---|
| comes back | the prebuilt `.db` — simpler, and what the other 5,549 MSAs came from | **no**, the TSV build is discarded |
| stays down | the mirror TSV build | **no**, envdb is still missing so the MSA is incomplete |

It pays off only if the host returns for envdb but not uniref30 — same host, same outage, not a
realistic split. And "free" was wrong anyway: free of GPU is not free of 96 GB of transfer, ~420 GB
of disk and hours of CPU.

The mirror finding is still worth having. It establishes that a route EXISTS if the outage turns out
to be long-lived and we decide a uniref30-only rebuild beats gap-padding. That is a decision to take
with more information, not one to pre-commit 500 GB to.

### THE ENVDB FAILURE WAS NEVER DIAGNOSED, AND MIGHT BE FIXABLE

This is the open lead, and it reframes the blocker. The problem is NOT that the mirror holds less
data — it holds the same releases, verified sizes, with md5sums bundled. The problem is that the TSV
packaging has to be CONVERTED by `tsv2exprofiledb`, and for envdb that conversion produced nothing.

A missing-data problem has no fix. A conversion problem may well have one, and nobody looked: the
note in `phase_e_search_envdb.sh` reads like a workaround taken under time pressure while a working
prebuilt URL existed. That URL is gone now, so the workaround is no longer available and the
underlying failure is worth ten minutes.

What makes it suspicious rather than hopeless:

* **the conversion exited 0 while producing nothing.** Whatever went wrong was not reported, so the
  cause was never on screen.
* the archive's member list can be read without downloading all 118 GB, so whether the inputs
  `tsv2exprofiledb` expects are even present is a cheap check.
* plausible causes nobody ruled out: a member missing from the extraction, a flag the newer mmseqs
  requires, or `MMSEQS_NO_INDEX=1` interacting badly with the conversion.

**If that conversion can be made to work, the mirror becomes a complete substitute and the whole
blocker dissolves.** If it cannot, we are waiting on the canonical host either way and the download
should not be started. Diagnosing it is the highest-value next step and costs almost nothing.

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
already enforces. The 301 GB search index is NOT built — `MMSEQS_NO_INDEX=1`, as the original run
used.

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

## The plan

| # | step | cost | gate |
|---|---|---|---|
| 1 | fetch `uniref30_2302.tar.gz` from the mirror | 96 GB transfer | content-length ≥ 1 GB, then `tar -tzf` lists cleanly |
| 2 | extract, verify against the bundled `uniref30_2302.md5sum` | ~420 GB | every listed member matches; a mismatch stops the build |
| 3 | `tsv2exprofiledb`, `MMSEQS_NO_INDEX=1` | hours CPU | the data files exist, never a `READY` marker |
| 4 | apply `uniref30_2302_newtaxonomy.tar.gz` (on disk, 1.97 GB) | minutes | taxonomy md5 equals the copy we already hold, `83105388…` |
| 5 | **CONTROL**: one unchanged query through steps 1-4 + phase D + F, compared as above | hours | all five rows pass |
| 6 | `phase_d_search_uniref.sh` on the 155 corrected queries, `FREE_DB=1` | hours | `MIN_LONG_HIT_PCT`, as the original used |
| 7 | **BLOCKED**: envdb. The mirror cannot supply it; needs `opendata.mmseqs.org` | — | — |
| 8 | `phase_f_pair_and_split.py` into `mmcif_msa_corrected` | CPU | every corrected role has `pairing.a3m` and `non_pairing.a3m` |
| 9 | `phase_h_search_templates.py` for the corrected queries | CPU | date-filter marker, as before |
| 10 | re-run `phase_j_corrected_jobs.py` WITHOUT `--allow-missing-msa` | seconds | every corrected-role path must now resolve to a file that exists |
| 11 | smoke-test `11zw` end to end | one fold | 7 residues restored, masked in the label, flanks retained |
| 12 | queue 2,850 refolds into `training/fold_corrected/` | ~19 h GPU | separate tree |

Steps 1-6 are NOT started, for the reason above: they are discarded if the host returns and
insufficient if it does not. Step 7 is the hard blocker. The highest-value work right now is not any
step in this table -- it is diagnosing why `tsv2exprofiledb` produced nothing for envdb, because that
is what decides whether the mirror route exists at all. The 141 corrected job jsons already exist, written with `--allow-missing-msa`, so
their paths are correct and pending; **a fold must not be queued from them until step 10 passes
without that flag.**

## What is not blocked

GPUs: FoldBench top-off (612), then 663 training folds of structures that are not being refolded.
Neither needs any of the above. `await_msa_server.sh` polls the canonical host every 5 minutes.

## Risks

| risk | mitigation | residual |
|---|---|---|
| envdb unavailable | none yet; it is THE blocker. The lead is the undiagnosed `tsv2exprofiledb` failure, which exited 0 while producing nothing | a uniref30-only MSA is NOT an acceptable substitute |
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
