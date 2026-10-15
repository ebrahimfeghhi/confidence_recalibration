# Connecting to the MSA database server

**Status 2026-10-14, 5:15 PM PT: BLOCKED.** `opendata.mmseqs.org` returns HTTP 503 on every path.
A retry loop (`shell_scripts/await_msa_server.sh`, tmux `msa_wait`) is checking every 15 minutes and
will start the search automatically when it recovers. Nothing else is blocked by this; the GPUs are
on the FoldBench top-off.

## What we need and why

155 corrected antibody/antigen chain sequences need multiple-sequence alignments before they can be
refolded. An MSA is built by searching two public databases for homologs:

| file | size | what it is |
|---|---|---|
| `uniref30_2302.db.tar.gz` | ~450 GB extracted, + 301 GB index | UniRef30, clustered UniProt, ~350M sequences |
| `colabfold_envdb_202108.db.tar.gz` | 291 GB | environmental/metagenomic sequences |

Both come from `https://opendata.mmseqs.org/colabfold/`, the dataset host run by the MMseqs2 and
ColabFold developers. This is the standard ColabFold pipeline, not a bespoke setup.

**The databases are not on disk.** They were deleted after the original search, by design: UniRef30
plus its index and envdb cannot all fit inside the quota at once, so the original build downloaded
one, searched, deleted it, and moved to the next. What remains in `tools/colabfold/db` is 5.2 GB of
taxonomy mapping files and empty `READY` markers.

## The observation

```
curl -sI https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz

HTTP/2 503
content-type: text/plain; charset=UTF-8
content-length: 17
```

Identical 503 on the bare root `https://opendata.mmseqs.org/` and on the envdb path. The 17-byte body
is an error page, not a file.

**Control:** other hosts answer normally from this machine — `files.rcsb.org` returned a valid HTTP
response — so outbound HTTPS works and the problem is specific to this host.

## Questions for anyone who can help

1. Does `curl -sI https://opendata.mmseqs.org/` return 503 from your network as well? That separates
   a global outage from something specific to this machine's route or IP.
2. If it works for you, what `content-length` does the uniref30 tarball report? It should be hundreds
   of GB. Anything close to 17 bytes means you are getting the same error page.
3. Is there a known-good mirror? It must serve the **same database versions** — `uniref30_2302` and
   `colabfold_envdb_202108` exactly. A different release would make these 155 alignments
   systematically different from the ~5,500 already built from these files, which is the kind of
   inconsistency that does not announce itself.

**Not useful:** substituting a different or smaller database. Version consistency matters more than
speed here. If the only option is a different release, say so and we will wait instead.

**Note on a positive answer:** "it works for me" would not by itself unblock us, since the download
has to run from this machine. But it would tell us the outage is route- or IP-specific, which is a
different and more tractable problem than waiting out a global one.

## Why the retry loop checks a file size, not a status code

A down server here returns a *successful delivery of a failure page*: 17 bytes, and some caches will
serve that as 200. The pipeline already fell for it once — the phase D log reads
`server reports 17 for uniref30_2302.tar.gz`, which is the script asking how big the file is, being
told 17, and believing it.

So the gate requires HTTP 200 **and** a content-length of at least 1 GB. The real tarball is hundreds
of GB, so the floor cannot pass an error page and cannot fail on the real file. Same pattern as the
other gates here: check the thing you need, not a proxy that can be true while the thing is absent.

## What happens when it comes back

`await_msa_server.sh` launches `phases/phase_d_search_uniref.sh` on
`building_msa/queries_corrected/query.fasta` (155 queries). `FREE_DB=0`, so UniRef30 stays on disk
afterwards: 1,042 GB of databases fits in the 1,411 GB free, and keeping it means a failure in the
envdb half does not force a 450 GB re-download through a resume path that has never been exercised
twice.

## Fallbacks, both tested

| approach | verdict |
|---|---|
| **Gap-padding** (`phases/phase_j_remap_msa.py`) — carry the existing alignment onto the corrected query by inserting pure-gap columns at one known point, re-aligning nothing | **works, verified reversible on 30,435 rows.** Costs aligned evidence at the restored positions, ~2% of each chain. Measured: only 0.2–7.4% of saved homologs carry residues at those loci anyway, so this is near the ceiling of what the saved files can give |
| **mmseqs2 rebuild** against the saved hits as a local database | **rejected by its own control.** Rebuilding an UNCHANGED query from its own saved hits lost 17.8% of depth (14,377 → 11,817 homologs, 0 gained), stripped per-hit alignment metadata, and produced rows with 215 match columns against a 214-residue query — a malformed a3m |

So the proper search is the real fix, and we are waiting on a server rather than on a decision.
