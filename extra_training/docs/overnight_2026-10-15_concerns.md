# Overnight run of 2026-10-14/15: what completed, and four concerns

> **SUPERSEDED IN PART, 2026-10-15.** The input/label mismatch this file is built on DOES NOT
> EXIST. SAbDab2's `Hseq` is defined as the residues that are structurally RESOLVED, so our
> input already matched the label's coordinate set; 919 of the 935 "missing" residues have no
> coordinates in their own chain. See **`docs/hseq_correction.md`**. The refold still running
> is unaffected: it regenerates deleted trunks using the ORIGINAL queries.


Written 2026-10-15 7:05 AM ET. Numbers measured, not estimated. Interpretation is in the commit
message and the session, not here.

## What completed

| stage | units | result | size | result rows |
|---|---|---|---|---|
| 1 training cache | 723 | 723/723 complete | 264 GB | 7,365 |
| 2 validation | 300 | 300/300 complete | 23 GB | 3,001 |
| 3 FoldBench top-off | 612 | 58/612, running | -- | -- |

Every failure bucket is 0 on stages 1 and 2: no missing trunks, no trunk-without-poses, no duplicate
pose directories, nothing stuck mid-score. All three handoffs fired unattended on confirmed
completion; no worker was killed mid-fold. Quota 1,943 GB of 3,072, zero CUDA errors, zero OOM.

Order was changed overnight to training -> validation -> FoldBench. FoldBench is ~70% of the
remaining GPU time and is the one stage nothing downstream waits on. Stages 1 and 2 together took
~5.5 h; the earlier ordering would have delivered them Saturday.

---

## CONCERN 1 -- SUPERSEDED by `concern1_diagnosis.md`

Diagnosed 2026-10-15, revised 8:40 AM ET. 27 of the 31 never-scored pairs have ZERO reference
contacts at both 4.5 A and 5 A -- the pool is a cross product and those pairs are not interfaces
(review item J2). 4 ARE real interfaces, lost because OST leaves one target chain unmapped and
`scoring.py:1121` drops any interface whose chains are not both mapped.

A column-order bug in `eval/ost_dockq.py` was claimed at 8:10 AM and is RETRACTED: OST 2.8.0
documents `(trg_ch1, trg_ch2, mdl_ch1, mdl_ch2)` and its mapping is reference -> model, so the
parser is correct and no score is mis-attributed. Also corrected: the claim is RELEASED after
SCORING INCOMPLETE. Folds, poses and labels are unaffected throughout.

### original text, kept for the record: 200 of 723 training units have interfaces that never score

| | training | validation |
|---|---|---|
| units complete | 723 | 300 |
| units with >= 1 interface carrying NO DockQ row at all | **200 (28%)** | **0** |
| structures affected | 20 | 0 |

Per-unit record with the exact interfaces: `training/fold_regen/partial_scores.csv`.

It is whole structures, all 10 seeds each, and the same interfaces every seed:

| pdb | pool interfaces | ever scored | NEVER scored |
|---|---|---|---|
| 8vy9 | 4 | AC | AD, BC, BD |
| 9vb4 | 4 | AC, BC | AD, BD |
| 9vaw | 2 | *(none)* | AB, AC |
| 9x5w | 2 | CI | CJ |
| 9zb5 | 2 | AD | CD |
| 9q5m | 8 | BG, DG, DH, FG, FH | BH, EG, EH |
| 9q7b | 6 | DK, DL, FL | EK, EL, FK |

Distribution over all 200:

| pool interfaces | ever scored | units |
|---|---|---|
| 2 | 0 | 1 |
| 2 | 1 | 69 |
| 4 | 1 | 10 |
| 4 | 2 | 20 |
| 4 | 3 | 70 |
| 6 | 3 | 20 |
| 8 | 5 | 10 |

THE FOLDS ARE NOT AT FAULT. Every one of these units has its trunk and its poses on disk, and the
worker reports success for the inference: 8vy9 logs `N_asym 4, N_token 705`, a 40 s forward pass and
a written trunk. The worker then prints `SCORING INCOMPLETE -- 3 of 4 interface(s) short of 5
samples` and moves on, and never returns, because `folder.py` keeps the claim by design.

### Hypothesis tested and REFUTED: the prediction is missing those chains

The obvious reading is that the pool enumerates interfaces across every copy in the crystal while
each fold contains only one copy, so cross-copy interfaces cannot exist in the prediction. Measured,
that is not what is happening -- the pool references exactly the chains the fold produces:

| pdb | distinct chains in the pool's interfaces | chains the job folds |
|---|---|---|
| 8vy9 | 4 (ABCD) | 4 |
| 9vb4 | 4 (ABCD) | 4 |
| 9x5w | 3 (CIJ) | 3 |
| 9zb5 | 3 (ACD) | 3 |
| 9vaw | 3 (ABC) | 3 |
| 9q5m | 6 (BDEFGH) | 6 |

### Live hypothesis, and why it is already on the review list

What the pattern does show is that specific CHAINS never appear in any scored interface: 9vb4 scores
AC and BC and never AD or BD, so D never scores; 9x5w never scores J; 9zb5 never scores C. That is
consistent with those chain pairs not forming a real interface in the REFERENCE, which makes DockQ
undefined rather than zero.

If so this is open review item **J2** -- "check geometrically whether each selected antigen copy
contacts the selected antibody" -- not a new problem. A role resolved by sequence with an
alphabetical tie-break can take the antibody from one copy of the assembly and the antigen from
another: identical sequences, different assembly, no contact (9djw). Not yet confirmed; confirming
it means measuring reference contacts for the never-scored pairs.

### Why it is not blocking, and what it does affect

The confidence head trains on the trunk, the poses and the label. All three are intact for all 723
units. DockQ is this project's evaluation metric. So the limitation is on EVALUATION: about a third
of the training cache cannot be scored on every interface. Validation is unaffected at 0 of 300,
which is the arm the head is actually measured on.

It predates this run. The previous training fold shows the same shape -- `10op_FN` has zero samples
on every seed and `7sk4` is incomplete on 8 consecutive seeds, 30 of 200 sampled units there.

---

## CONCERN 2 (resolved by skipping, needs no action): 9mx1 cannot be folded

| | |
|---|---|
| residues | 2,941 |
| protenix-v2 limit | 2,560 (`runner/inference.py:399`) |
| units removed from the queue | 10 |
| attempts before removal | 35 across 3 workers |

Recorded in `training/fold_regen/skipped_unfoldable.csv`. Skipped with a reason, never folded at
reduced `N_sample`/`N_cycle`. It is the only structure over the limit in either queue. Left in the
queue it would have been retried indefinitely -- the claim releases on failure -- and the stage could
only ever have reached 723 of 733, so it would never have confirmed completion or handed off.

## CONCERN 3 (parked by decision): the refold's value is unproven

| location of the residues a refold would restore | residues | share |
|---|---|---|
| inside a variable domain | 15 | 1.4% |
| constant domain or termini | 1,044 | 98.6% |

Only 8zd6 H (118-127) and 9iut L (114-118) touch a variable domain, and both straddle its
C-terminal edge rather than a CDR. Cost of proceeding: 96 GB UniRef30, ~700 GB envdb, 2,480 folds.
The 136-role refold list and its 144 corrected queries are built and checked
(`train_val_splits/refold_audit_*.csv`, `building_msa/queries_corrected_v2/`), so the decision stays
reversible. The UniRef30 download was deleted at the user's instruction to reclaim 196 GB, so
restarting the refold now needs that 96 GB fetched again.

## CONCERN 4 (watch, no action): VRAM headroom on stage 3

| card | used | total | headroom |
|---|---|---|---|
| 0 | 90,553 MiB | 97,887 MiB | 7.5% |
| 3 | 90,475 MiB | 97,887 MiB | 7.6% |

FoldBench complexes run to 2,520 residues. Zero OOM so far. An OOM releases the claim and leaves the
unit retryable; the standing rule is to skip with a reason rather than reduce fold settings.

---

## Still open from earlier review rounds

Implementation status and validation status are different things.

| item | implemented | validated |
|---|---|---|
| `phase_j_build_cache.py` | yes, 593 lines | NO -- never run over the full cache |
| `phase_j_replay.py` | yes, 368 lines | NO -- it is what gates training on the cache |
| the confidence-head training loop | no | -- |
| `phase_i_check.py` | yes | NO -- reports COMPLETE when a job is absent from pool.csv; its cache validator accepts NaN coordinates and bad trunk shapes |
| phases D/E READY markers | yes | NO -- left on a die path, open since round 2 |
| J1/J3 entity+subchain rebuild | no | 9 split single-chain constructs, `reviews/round5_2026-10-13_phasej.md` |
| J2 antigen-copy contact check | no | the live hypothesis for CONCERN 1 |

## Defects found and fixed during the run

Each would have cost the night; all are committed with their evidence.

| commit | defect |
|---|---|
| `3f9fcda1` | VH/VL splice point guessed from G/S content: six refolds queued pure tether (`GGG`, `SGGG`, `TTPP`), and 8dpl's real `TVSS` excluded as tether |
| `ec4dab97` | stage 2/3 refolded into the old root, where `folder.py` rescores saved poses instead of folding and never writes a trunk -- 373 units would have hung |
| `8a2e7718` | completion counted trunks, which are written DURING inference, so the switch could kill workers mid-unit |
| `a8ac33f9` | a stage that gave up had its return code discarded and killed its workers anyway |
| `ecb12648` | the drain waited for the GPUs to go quiet, which never happens: a worker claims its next unit instantly |
| `7a90c948` | 9mx1 left in the queue, see CONCERN 2 |
| `541395b0` | completion required every interface scored, so CONCERN 1's 200 units would have blocked the handoff forever |
| `e6843f1b` | `partial_scores.csv` written a third of the way through named 82 units when the finished count is 200 |
