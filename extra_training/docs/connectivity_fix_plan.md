# Proposed fix for the fused-fragment inputs. For review before anything runs.

Written 2026-10-15 11:40 AM ET. Addresses the defect in `hseq_connectivity.md`: 138 roles over 119
structures fold a query that splices resolved fragments across unresolved gaps, asserting peptide
bonds that do not exist.

## The fix in one line

Fold the FULL polymer sequence instead of the spliced one, and carry the existing MSA onto it by
inserting pure-gap columns -- no database rebuild.

## Why no rebuild is needed

An a3m has one match column per query residue, so a 221-residue query cannot use a 193-column
alignment. The obvious route is a fresh search, which needs UniRef30 re-fetched (96 GB, deleted) and
envdb built (~700 GB), one-shot, days.

`phases/phase_j_remap_msa.py` already does it without searching. It inserts `n` pure-gap columns at
ONE KNOWN insertion point, carries every existing column through byte for byte, and inserts the
restored residues into the query row at the same point. It refuses the file unless deleting the
inserted columns reproduces each original row exactly, the query equals the corrected sequence,
every row has the corrected match-column count, and headers and row order are unchanged. Two rounds
of hardening already, including `668b1606` checking the ABSOLUTE match-column count rather than the
delta.

## Why gap columns are acceptable HERE specifically

Padding's stated limitation is that restored positions get no aligned homolog evidence -- a gap means
"no evidence", which the model cannot distinguish from "relatives lack this residue". Normally that
degrades prediction at those positions.

It does not matter here, and the reason is specific rather than convenient: the restored residues are
UNRESOLVED, so the label carries no coordinates for them and the model is never graded on their
positions. What the fix has to deliver is chain TOPOLOGY -- that residue 134 is ten positions from
145 rather than bonded to it -- and gap columns deliver that exactly.

Two consequences worth stating rather than discovering:

| | |
|---|---|
| the `resolved` head | will now be supervised to predict "unresolved" at those positions, which is CORRECT supervision it previously never saw |
| per-interface ipTM | changes, because the chain has more tokens and ipTM is computed over all inter-chain PAE entries of the complete chains (`sample_confidence.py:536`). That is the point, not a side effect |

## What is missing before it can run

`phase_j_remap_msa.py` needs `query_insert_at` per role. The rebuilt refold list does not carry it:

    pdb, role, chain, qlen, span, n_missing, missing, reason, old_q, corrected_q

It is derivable from `missing`, which holds native-position runs, but it MUST be regenerated from the
native-position map rather than inferred by re-aligning `old_q` to `corrected_q`. A generic
re-alignment is exactly where a silent one-column shift comes from, and avoiding that is the whole
reason the tool is safe. `phase_j_job_cifs.py --native-gaps` emits it.

## Steps

| # | step | check that must pass |
|---|---|---|
| 1 | regenerate `query_insert_at` for the 138 fused roles from the native-position map | removing the insertions from `corrected_q` reproduces `old_q` byte for byte, all 138 |
| 2 | confirm each corrected query is the true contiguous polymer span | its length equals `span_hi - span_lo + 1`, and its residues match the entity sequence over that span |
| 3 | pad the MSAs into a SEPARATE store, `mmcif_msa_corrected` | the tool's own per-row reversibility and column-count checks |
| 4 | write corrected job jsons pointing at the padded store | no job may reference the old store; verified by path, not by assumption |
| 5 | refold those 119 structures into a fresh root | one fold per card; completion by `phase_j_stage_complete.py` |
| 6 | compare old vs new for a sample | DockQ and per-interface ipTM, same seeds, to measure what the splice cost |

Step 6 is the one that answers the question nobody has answered: whether a fused input measurably
harms the fold. It can be run on ONE structure after step 3, before committing the other 118.

## Cost

| | |
|---|---|
| MSA padding | CPU, minutes. No download |
| refold | 119 structures x 10 seeds = 1,190 units, ~325 GB |
| validation side | 13 structures x 100 seeds = 1,300 units, ~173 GB, if the same defect is present there |
| quota | 1,990 GB now; stop is 2,765 |

## The decision this forces

`training/fold_deferred` is currently refolding all 118 affected structures with the ORIGINAL
spliced queries, 215 of 1,180 done. It restores the cache to its prior state and is not
destructive, but it reproduces the defect. If the corrected-query refold happens, that work is
superseded. Options: let it finish and supersede later, stop at a fold boundary and hold, or stop and
go straight to the corrected path once step 3 passes.

## Risks and unknowns

| | |
|---|---|
| unmeasured | whether a spliced input actually degrades the fold. Step 6 settles it; until then the fix is justified on correctness, not on a measured gain |
| the 57 "chain not found" roles | `chain_roles.csv` names an auth chain absent from the cif for 57 roles. Not investigated. They are not in the fused count and may hide a separate mapping problem |
| validation not yet measured | the 138/119 figures are the TRAINING selection only. The same measurement has not been run on the 53 validation structures |
| gap columns | the restored positions get no evolutionary signal. Accepted above, but it is a real difference from a fresh search |

---

# CHECKED: the benchmarks splice too. This weakens the plan's justification.

Added 2026-10-15 11:55 AM ET, at the user's request, before any step was run.

Same test as on our own data: align each benchmark job query to its reference chain and ask whether
the native positions it maps onto form one contiguous run.

| | queries confidently matched to a chain | spliced across unresolved gaps |
|---|---|---|
| FoldBench | 135 | **15 (11%)** |
| PXM-2025 | 109 | **21 (19%)** |
| our training selection | 1,478 | 138 (9%) |

Examples: FoldBench `7yxu` chain H, a 228-residue query over 2 runs; `8ahn` chain H, 223 over 2.
PXM `8w5d` chain c, 131 over 2; `8w5e` chain c, 133 over 2.

LIMITATION OF THIS MEASUREMENT: only about 55% of benchmark queries could be confidently matched to a
reference chain -- 97 unmatched for FoldBench, 146 for PXM -- because their queries come from their own
FASTA rather than from these reference files. The percentages are over the matched subset and should
be read as a strong signal, not a precise rate.

## What it does to the argument

| | |
|---|---|
| physical correctness | unchanged. A spliced query still asserts a peptide bond that does not exist, in our inputs AND in theirs |
| train/eval consistency | **reversed.** Our spliced inputs currently MATCH benchmark practice. Correcting only ours would introduce the train/eval difference the correction was partly meant to remove |
| the case for running the plan | now rests on physical correctness alone, with a consistency COST rather than a consistency benefit |

## What would settle it

Step 6 of the plan, promoted from last to first: fold one affected structure both ways -- spliced and
contiguous -- and compare DockQ and per-interface ipTM at the same seeds. That measures whether the
splice costs anything at all. If it does not, the plan should not run, because it would only break
consistency with the benchmarks for no gain. If it does, the finding applies to the benchmarks too
and is worth reporting rather than quietly fixing on our side.

It needs one padded MSA, not the 96 GB rebuild, so it is cheap and should precede any decision about
the other 118 structures.
