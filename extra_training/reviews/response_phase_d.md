# Response: phase D review

You were reading `b421b7f`; `11b6e886` had already landed the patch gate (your third "still open"
item). Everything else below is new work.

## Your qdb point was the serious one, and it was my bug

Confirmed and quantified. `search.py:517` runs `createdb <query_file> qdb --shuffle 0` unconditionally,
so phase E rebuilds the query database from whatever fasta it is handed, and ColabFold keys everything
by position.

Earlier today phase C's new antigen-type filter removed one RNA chain from `query.fasta`, 5,305 → 5,304.
That record sat at **key 479 of 5,305**, so rebuilding `qdb` from the shortened file would have shifted
**4,825 keys** and mis-assigned that many alignments to the wrong chains, silently. The fix was right
for future runs and wrong to apply retroactively to a fasta whose search had already happened.

Recovered the exact searched set from git and verified it three ways:

| check | result |
|---|---|
| record order vs `qdb.lookup` | identical, 5,305 entries |
| each sequence vs its own a3m query line | 5,305 of 5,305 match |
| the single mismatch | the RNA chain, where the search rewrote `U`→`X` — the known artifact |

Same for the short batch: 235 records, name set identical to the 235 a3m in both directions, 235 of
235 sequences matching, one mismatch being its nucleic-acid chain.

Both are now frozen beside their outputs as `<out-dir>/query.searched.fasta` with a `.sha256`, written
by phase D itself. **Phase E must read that file and verify the checksum, never `msa_work/query.fasta`,**
which is phase C's current output and may legitimately have moved on. Your "checksum gate rather than
a convention", with the convention removed entirely.

## Partial intermediates: gated now, not verified by hand

Right that the skip keys on `uniref.a3m` existing rather than being complete. Phase D now requires the
index entry counts of **both** `uniref.a3m` and `prof_res` to equal the query count exactly, and says
so in the failure message along with what to delete to force a re-search. Current state:

| batch | queries | `uniref.a3m` entries | `prof_res` entries |
|---|---|---|---|
| batch 1 | 5,305 | 5,305 | 5,305 |
| short | 235 | 235 | 235 |

## The staging question, settled by code rather than experiment

You identified the single premise my argument rested on — whether filtering applies per database pass
or to the merged alignment — and said it was a code question. It is, and the code answers it:

| `search.py` | |
|---|---|
| 157 | one `filter_param`: `--filter-msa`, `--filter-min-enable 1000`, `--diff`, `--qid 0.0,0.2,…,1.0`, `--qsc 0`, `--max-seq-id 0.95` |
| 172 | applied to `result2msa` for **uniref** → `uniref.a3m` |
| 210 | the same list applied to `result2msa` for **env** → `bfd.mgnify30.metaeuk30.smag30.a3m` |
| 231 | `mergedbs qdb final.a3m uniref.a3m bfd…a3m` |

So each pass is filtered independently with identical parameters, and the merge is a concatenation by
key with no filtering of the combined set. Staged and unstaged therefore produce the same `final.a3m`,
because the uniref half is byte-identical, the env half is produced from the same `prof_res` and the
same env database, and nothing acts on the union. The failure mode you named does not exist here.

I would still take the `DEBUG_MINI_DB=1` run as independent confirmation — it exists
(`MSA_RUNBOOK.md:234`) and is how the staging path was validated originally — but it is now
confirmation of a code-level result rather than the only evidence.

## The index: your two-claims point is correct, and the experiment is parked

You are right that I conflated two different mechanisms, and right that `--db-load-mode 0` means the
search is not using the index. Separated:

1. *`convertalis` resolves taxids only via `<db>.idx_mapping` / `.idx_taxonomy`.*
2. *An index-free search targets `_db_seq` rather than the representatives.*

If only (2) holds, the index is a coincidental fix for something addressable directly — and phase D
already creates both sidecar symlinks by hand, which is exactly your point.

**The experiment is parked, not refused.** It needs uniref30 on disk, and the 11-chain top-up running
now is the last uniref30 work in the pipeline; its final step frees the database. Running the test
afterwards costs a fresh ~1.5 h cycle. The procedure is recorded so it can be done whenever a cycle
is in progress: rename `uniref30_2302_db.idx` aside, keep the symlinks, run a handful of queries with
`COLABFOLD_EMIT_TAXID=1`, and inspect `uniref_tax.m8`. If taxids resolve, 301 GB per batch goes away
and most of this project's disk failures with it.

## Your four uncertainties

**1. ALLOW_THIN — adopted, and the flag is gone.** Gating on query length instead, which is what the
expectation is actually a property of. Measured on the completed batches, the split is clean:

| sampled queries | with UniRef hits |
|---|---|
| ≥20 residues | **200 of 200 — 100%** |
| <20 residues | 25 of 200 — 12% |

So phase D requires hits for queries of 20+ residues (floor 90%) and exempts shorter ones. The
235-chain short batch now passes *because its queries are short*, not because a flag was set, and a
mixed batch stays honest. No `ALLOW_THIN` remains in any code path.

**2. MIN_TAX — your proposed fix cannot work, and finding out why was useful.** Per-query coverage from
the m8 is impossible: **column 1 is the constant `101` on every row**, because ColabFold writes a
constant query identifier — the same artifact that makes every a3m begin `>101` and that
`prep_step4_inputs.py` exists to repair. A gate keyed on it reports one query forever, which is what my
first attempt did.

What the file is for: `step3-uniref_add_taxid.py:253-255` reads it as a dictionary from **column 2**,
the UniRef accession, to **column 3**, the taxid. The query column is never touched. So the gate now
asks the question that decides pairing — of the UniRef hits present in the alignments, how many carry
a taxid — sampling 200 alignments against a full m8 pass, floor 95%. Measured on the short batch:
4,478 distinct hits, 4,478 with a taxid, 100%.

**3. Parameter parity — yes please, take it.** It is the one that threatens the premise and I have not
diffed our effective flags against Protenix's end to end, including inherited defaults. For reference,
ours are `--use-env 0 --use-templates 0 --db-load-mode 0` with `filter_param` as listed above, and
`--threads` now derived from the taskset rather than hardcoded.

**4. The taxonomy tarball — procedure accepted.** Counting mapping entries with no corresponding
taxonomy entry is the right measure, and "newtaxonomy" implying a prior taxonomy was wrong for this
release is a good question. Parked with the index experiment, since both want uniref30 on disk.

## Your three still-open items

**Template READY markers — fixed.** Phase D now records exactly which markers it created and removes
those, restoring the prior state. It never removes one that was already present. You were right that
the discipline already existed in `phase2_envdb.sh`.

**ALLOW_THIN bypassing the `uniref.a3m` check — fixed, and the two failures are now separate.** The
completeness check has no length exemption at all: a batch without complete intermediates simply never
gets phase E, which is a different failure from a batch that legitimately retrieved little.

**The patch check — already a gate in `11b6e886`.** Two functions must exist on the imported module and
at least four `PATCHED (tts_abag)` markers must be present, which also catches a partial application.
It runs before the download.

## Also, from your previous pass

`--threads` is now `taskset -c "$CORES" nproc` rather than hardcoded 64. Verified: `0-63`→64,
`0-31`→32, `0-7`→8, `0,2,4`→3. One correction: the documented cap is **64** cores
(`MSA_RUNBOOK.md:119`, "more has crashed this server"), not 32 — the 32 you remembered is
`phase_a_define_train_set.py`'s own conservative default for hmmscan and mmseqs.
