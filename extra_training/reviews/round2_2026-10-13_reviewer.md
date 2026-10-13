# Review round 2 — external reviewer, 2026-10-13

Reviewer's pass over `extra_training` after round 1. Their text is quoted; the response follows each
item with the evidence and the commit.

Verdict summary:

| # | finding | verdict | fixed in |
|---|---|---|---|
| 1 | phase H drops the query record from every hmmsearch.a3m | **NOT CONFIRMED** | — |
| 2 | representative picker selects the largest structure at n=3 | **CONFIRMED** | `1cad97fc` |
| 3 | phase A zips agtypes against sequences with no length check | **CONFIRMED, latent** | `1cad97fc` |
| 4 | phase D's validator makes a crash look like a clean batch | **CONFIRMED** | this round |
| 5 | phase F `--min-pairing-frac` 0.50 against 90% floors in D and E | **CONFIRMED** | this round |
| 6 | phase F taxid stride 97 samples 1 of 11 files; gate skipped on an empty sample | **CONFIRMED** | this round |
| 7 | phase G writes artifacts before three of its own gates | **CONFIRMED** | this round |
| 8 | `list(rows[0])` raises 53 lines before its explanatory guard | **CONFIRMED** | this round |
| 9 | `agchains` indexed positionally without a check | **CONFIRMED, latent** | this round |
| 10 | READY markers not removed on die paths in D and E, no trap | **OPEN** | — |
| 11 | organisation: scripts vs phases, duplicated invariants, review docs as spec | **ADDRESSED** | this round |

---

## 1. "Phase H's date filter drops the query record from every hmmsearch.a3m. Silently corrupts."

> The loop treats every `>` line as a template hit and reads `line[1:5]` as a PDB id. An a3m's first
> record is the query, not a hit, and it gets dropped on any of three branches. [...] One check
> settles it before you change anything: `head -2` on any filtered hmmsearch.a3m.

**Not confirmed.** I ran the check they named.

- First header of 200 raw files, `msa_work/mmcif_msa/*/hmmsearch.raw.a3m`: **zero** contain "query".
  Every record is `>pdbid_chain/range [subseq from] mol:protein length:N  <description>` — the format
  phase H's docstring at `:190` asserts.
- Protenix parses these with `skip_first=False` (`template_featurizer.py:368` and `:737`) and passes
  the query separately as `query_seq`. It expects a hits-only file; one that led with a query record
  would be mis-parsed.

The cited evidence — `build_selftmpl_general.py:223` and `build_templates.py:248` opening with
`[">query", q]` — is from our self-templating template builders. Different artifact, different
consumer, and the convention does not transfer to hmmsearch output.

**The proposed fix would have introduced the bug**, by making phase H preserve a record that is not
there and that Protenix does not expect. No change made.

## 2. "The representative picker selects the largest structure for 3-structure clusters."

**Confirmed, and the comment was wrong in the way they describe.** Reproduced over lengths 1–9: at
n=3, k=2 the indices are `[0, 2]` — smallest and largest — because `(1+0.5)*2/2 = 1.5` and Python
rounds half to even. The comment claimed `[0, 1]`. 3 is the only length that selects the maximum.

Fixed by excluding the top index whenever there is a structure to spare (`k < len` and `len >= 3`);
every other length is unchanged. The live validation set was built with `--instances`, an explicit
list, so this never affected a fold.

## 3. "Phase A zips agtypes against the antigen sequences with no length check."

**Confirmed as a latent defect; measured as never having fired.** Over all 7,268 rows carrying a
polypeptide antigen: **zero** length mismatches between `agtypes` and either `agresolvedseqs` or
`agexpectedseqs`, so no chain was ever dropped and the exclusion filter was not weakened.

Fixed structurally rather than locally, which is also the reviewer's organisational point: phases C
and G each carried their own copy of this guard and A had none. `A.antigen_chains()` is now the one
validated pairing and all three call it.

## 4. "Phase D's heredoc validator returning empty on a traceback, and every `${n:-default}` making a crashed check look like a clean batch."

**Confirmed.** `COV=$(python … )` at `:319` is consumed with no exit-code check. On a traceback `COV`
is empty, `set -- $COV` leaves no positional parameters, `${1:-0}` … `${9:--1}` fill in zeros, and
both gates below are guarded by `[ "$LTOT" -gt 0 ]` — false at zero. The batch passes with no check
having run.

The reviewer is also right that phase E already has the fix (`:271-281`) with a comment naming this
exact failure. Phase D now mirrors it: capture `$?`, escalate on a non-zero exit or empty output, and
require exactly 9 fields.

## 5. "Phase F's `--min-pairing-frac` default of 0.50 against 90% floors in D and E."

**Confirmed.** Raised to 0.90. Measured on the completed run before changing it: **5,304 of 5,304**
chains of 20+ residues have a non-empty `pairing.a3m` (100.0%), and 245 of 245 short ones. A floor at
half would have passed a run where every second antibody chain lost its paired MSA.

## 6. "Phase F's `taxid_coverage` stride of 97 sampling 1 of 11 files for the small batch, and skipping the gate when the sample has no UniRef hits."

**Both confirmed.** `files[::97]` is a constant stride, so the sample size scaled with the batch:
5,305 files gave 55, 235 gave 3, and 11 gave **one** — a single file decided whether a batch passed.
And `if bs and bpct < …` treated an empty sample as nothing to check.

Fixed: the stride is now derived from the batch size to sample up to 200 files evenly (what phase D's
`TAX_SAMPLE` already did); a sample with no hits widens to the whole batch before concluding; and
zero hit records across every file in a batch is now a hard stop, which is what phase D already does.

## 7 and 8. "Phase G writing pool.csv, manifest.csv and the job JSONs before three of its own gates", and "`list(rows[0])` raising IndexError 53 lines before the guard meant to explain it."

**Both confirmed**, and the reviewer's framing — "gate-then-write, not write-then-gate" — is the fix.
Phase G built and wrote inside the loop, then ran five gates (`:469`, `:479`, `:500`, `:504`, `:517`).
Any gate firing left a rejected selection on disk: job jsons that phase I globs and would fold, a
`pool.csv` naming their interfaces, and a manifest describing a run that never happened.

Restructured: job payloads accumulate in `pending_jobs`, every gate runs, then everything is written.
The empty-`rows` guard moved to the top of the gate section, ahead of anything that indexes `rows`.

**Verified byte-identical**: re-running phase G on the live validation selection into a scratch
directory reproduces `manifest.csv`, `pool.csv`, `chain_roles.csv` and all **53** job jsons exactly.

## 9. "`agchains` indexed positionally without a check at `:292`."

**Confirmed as latent.** `ag<i>` must mean the same index in `agchains` as in `agtypes`, or an
antigen is given another antigen's auth chain name. Measured over all 10,395 rows carrying both
fields: zero disagreements. Guard added anyway, since the cost is one comparison.

## 10. READY markers not removed on die paths in D and E, no trap — **OPEN**

Not yet addressed. Severity: medium. A `die()` between creating a READY marker and finishing the
build leaves the marker behind, and `setup_databases.sh` decides what to rebuild from those markers,
so the next run can skip a database that is not actually complete. Phase D already removes
`UNIREF30_READY` deliberately at `:146` and `:426`, which shows the discipline exists but is applied
by hand rather than by a trap.

## 11. Organisation

> Scripts and phases have blurred. [...] Promote the split/selection scripts to named phases and
> delete or archive the superseded `phase1*` scripts rather than keeping them live.

Partly done. `scripts/archive/` now holds `phase1_uniref.sh`, `phase1_index.sh`,
`phase1_search_idx.sh`, `phase1b_short.sh` and `prep_step4_inputs.py`, with a README naming what
replaced each. `phase2_envdb.sh` stays in `scripts/`: it is the environmental-database download
helper and is still called live by `run_e_to_i.sh:119` and `overnight.sh:53`, which the README's end
state allows.

**Not done: promoting the selection scripts to phases.** `propose_val_split.py`,
`select_cache_structures.py`, `pick_second_reps.py` and `cluster_composition.py` do produce committed
artifacts and are pipeline steps by that test. Renaming them changes paths that the overnight chain,
the cron prompts and several docs refer to by name, so it is a rename worth doing deliberately rather
than in the middle of a review round. **OPEN**, severity low.

> The same invariant is validated in three places and missed in a fourth.

Done for both cases named: `A.antigen_chains()` is the single antigen pairing check, used by A, C and
G; and `propose_val_split.py` now uses `A._ckey()` instead of `str()` on the cluster id, closing the
float-vs-string hazard, and derives the pool from `A.training_clusters()` rather than reconstructing
it as `train_clusters | val_clusters` — which was circular, and is what kept the superseded
50-cluster validation file alive.

> The review documents are becoming the spec.

Agreed, and this directory is the answer. Round 1 moved here from `docs/`, this file records round 2,
and the index lists every round. Each round's **Open** section is the only place an unfixed finding
lives.

---

## Open after this round

| # | finding | severity |
|---|---|---|
| 10 | READY markers not removed on `die` paths in phases D and E; no trap | medium |
| 11 | selection scripts not promoted to named phases | low |
