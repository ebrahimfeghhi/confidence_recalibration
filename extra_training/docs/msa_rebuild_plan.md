# Plan: rebuild the MSAs for the 155 corrected queries

**For review.** Nothing here has run. Written 2026-10-14 after `opendata.mmseqs.org` went to HTTP 503.

## Why this is needed

125 training and 16 validation structures were folded from queries that omit internal native
residues (`train_val_splits/refold_internal_omissions*.csv`). Their corrected queries are a few
residues longer, and an a3m is one match column per query residue, so the stored alignments no longer
describe the sequences. 155 unique changed sequences need new MSAs before those 2,850 folds can run.

## The source: GWDG mirror, not a substitute database

`opendata.mmseqs.org` is 503 on every path. ColabFold's own `setup_databases.sh` offers the same
releases from `wwwuser.gwdg.de/~compbiol/colabfold/` (the Söding lab's host, reached via two
redirects to `wwwuser.gwdguser.de`), and both files resolve:

| file | content-length | verified |
|---|---|---|
| `uniref30_2302.tar.gz` | 102,918,187,842 | HTTP 200; members `uniref30_2302.tsv` (2023-05-15) + `uniref30_2302.md5sum` |
| `colabfold_envdb_202108.tar.gz` | 117,965,643,010 | HTTP 200 |

Versions are `uniref30_2302` and `colabfold_envdb_202108` — **exactly** the releases the existing
5,549 MSAs were built from. That was the hard constraint: a different release would make these 155
alignments systematically different from everything they are compared against, and it would not
announce itself.

### One real difference: TSV, not prebuilt

`setup_databases.sh` distributes two packagings of the same release, and says so in its own comments:

    ${DB}.db.tar.gz   "new prebuilt GPU+CPU databases, that don't require calling tsv2exprofiledb"
    ${DB}.tar.gz      "old .tsv + tsv2exprofiledb databases"

Our `phase_d_search_uniref.sh` fetches the prebuilt one. The mirror has the TSV one, which needs
`mmseqs tsv2exprofiledb` to become searchable. That is an extra CPU step of hours, not a different
database — and it is a path this project has already taken: `building_msa/work/phase2_envdb.log`
reads `building envdb (tsv2exprofiledb, index-free)`, and `tsv2exprofiledb` is in
`quota_guard.sh`'s watch list.

### A checksum check that failed, and why it was the wrong check

The archive ships `uniref30_2302.md5sum` for six members, two of which survived the deletion on our
disk. Comparing them:

| file | ours | the archive's |
|---|---|---|
| `uniref30_2302_db_taxonomy` | `83105388…` | `561357ac…` |
| `uniref30_2302_db_mapping` | `f0b03428…` | `2b9d4e68…` |

Neither matched, which looked alarming for about a minute. Resolved:

* `_db_taxonomy` is byte-identical to the copy inside `uniref30_2302_newtaxonomy.tar.gz`, a separate
  download `setup_databases.sh` applies specifically to this release to REPLACE the taxonomy. So
  ours is the new taxonomy and the archive carries the old one. Expected.
* `_db_mapping` is 2.7 GiB where the tarball's is 5.4 GiB, and inspecting the bytes settles it: ours
  is BINARY (mmseqs-indexed, matching the `uniref30_2302_db.idx_mapping` symlink that points at it)
  while both archives ship TSV TEXT (`350950050\t2034991\n`).

So our files are build PRODUCTS and the archives ship SOURCE. Source checksums could never match a
processed artifact; the comparison was badly chosen rather than informative. **The md5sums are still
the right verification — applied to the extracted source files, which is where they belong.**

## The plan

| # | step | where | cost | gate |
|---|---|---|---|---|
| 1 | fetch `uniref30_2302.tar.gz` from the mirror | `tools/colabfold/db` | ~103 GB transfer | content-length ≥ 1 GB before accepting, then `tar -tzf` must list cleanly |
| 2 | extract and verify against the bundled `md5sum` | same | ~450 GB on disk | every listed member must match; a mismatch stops the build |
| 3 | `tsv2exprofiledb` to build the searchable db | same | hours CPU, working space | the data files must exist, not a `READY` marker |
| 4 | apply `uniref30_2302_newtaxonomy.tar.gz` (already on disk, 1.97 GB) | same | minutes | taxonomy md5 must equal the copy we already have, `83105388…` |
| 5 | `phase_d_search_uniref.sh` on the 155 corrected queries | `building_msa/work/msa_corrected_idx` | hours | `MIN_LONG_HIT_PCT`, as the original run used |
| 6 | fetch and build envdb, then `phase_e_search_envdb.sh` | same | ~118 GB + hours | same shape of gate |
| 7 | regenerate templates for the corrected queries (`phase_h_search_templates.py`) | `mmcif_msa_corrected` | CPU | date filter marker, as before |
| 8 | smoke-test ONE structure end to end before the batch | — | one fold | `11zw`: 7 residues restored, masked in the label, flanks retained |
| 9 | queue 2,850 refolds into `training/fold_corrected/` | GPUs | ~19 h | separate tree, so a same-seed refold cannot be joined to the old pool |

`FREE_DB=0` throughout: 1,042 GB of databases fits inside the 1,411 GB free, and keeping UniRef30
means a failure in the envdb half does not force a 103 GB re-download through a resume path that has
never been exercised twice.

## What runs in parallel, and what is not blocked

The GPUs are on the FoldBench top-off (612 folds) and then 663 training folds of structures that are
NOT being refolded. Neither waits on any of the above. Steps 1-7 are CPU and network only.

`shell_scripts/await_msa_server.sh` keeps checking the canonical host every 15 minutes regardless. If
`opendata.mmseqs.org` returns before step 3 finishes, the prebuilt database is the simpler path and we
take it; after that point the mirror build is already done and there is nothing to gain.

## Risks, stated

| risk | mitigation | residual |
|---|---|---|
| the mirror serves a different release | version strings match; md5sums verified against extracted source | a repackaged archive with matching names would not be caught by names alone |
| `tsv2exprofiledb` produces a db that differs from the prebuilt one | it is the path this project already used for envdb | unquantified: the existing uniref MSAs came from the PREBUILT db, so these 155 would come from a differently-built one |
| disk | 1,411 GB free against a ~950 GB peak; guard warns at 2,900 GB and will not kill the build | an unexpected intermediate could still exhaust it |
| the resume path if envdb fails | `FREE_DB=0` keeps uniref30, so no re-download | never exercised twice |

**The second row is the one I would most want a reviewer to push on.** The existing 5,549 uniref MSAs
were built against the PREBUILT database. Building these 155 from the TSV distribution should give the
same search results, because it is the same sequence data — but "should" is doing work there, and the
check that would settle it is to rebuild one UNCHANGED query through the new path and compare its MSA
to the stored one. That control is cheap and belongs before step 5, not after.

## Rejected alternatives, recorded so they are not re-proposed

| approach | why not |
|---|---|
| gap-padding (`phase_j_remap_msa.py`) | works and is verified reversible, but gives no aligned evidence at the restored positions. Measured: only 0.2-7.4% of saved homologs carry residues at those loci, so it is near the ceiling of what the saved files can give. Kept as a fallback if the rebuild fails |
| mmseqs2 realignment against the saved hits as a local database | failed its own control: rebuilding an UNCHANGED query from its own hits lost 17.8% of depth (14,377 → 11,817, 0 gained), stripped per-hit alignment metadata, and produced rows with 215 match columns against a 214-residue query — a malformed a3m |
| a different or smaller database | version consistency with the other 5,549 MSAs matters more than speed |
