# Execution plan: what runs now, what waits on the MSA server

Written 2026-10-14 11:00 PM ET. Supersedes the queue sections of `msa_rebuild_plan.md`, which
remains canonical for the UniRef30/envdb build itself.

## The split

Every (structure, seed) unit with no trunk on disk is in exactly one group.

| group | test | training | validation | why |
|---|---|---|---|---|
| A, runnable now | structure NOT in the refold list | 733 units / 74 structures | 300 units / 3 structures | query unchanged, so the MSA on disk is still valid |
| B, deferred | structure IS in the refold list | 1,180 units / 118 structures | 1,300 units / 13 structures | query grows; stored MSA has the wrong column count |

Group A is in `training/fold/queue.tsv` and `validation/fold/queue_now.tsv`. Group B is in
`deferred_refold.csv` in each fold dir, deliberately NOT in any queue file. Verified overlap between
each runnable queue and its deferred list: 0.

Both built by `phases/phase_j_queue_now.py` (`--apply` writes; without it only counts print).

## Order of work

Driven by `shell_scripts/watch_train_val_then_foldbench.sh` (tmux `fbwatch`).
`watch_foldbench_then_training.sh` is SUPERSEDED and must not be run.

| stage | what | units | measured | consumer | writes to |
|---|---|---|---|---|---|
| 1 | training cache | 733 | 5.2 h | `tcache_0/1/3`, `sabdab_train` | `training/fold_regen/` |
| 2 | validation | 300 | 2.6 h | `vfold_0/1/3`, `sabdab_val2` | `validation/fold_regen/` |
| 3 | FoldBench top-off | 576 of 612 left | ~18.7 h | `fbtop_0/1/3`, `foldbench` | `foldbench_baseline_remaining_folds/` |
| deferred | the refolds | 2,480 | -- | blocked on the corrected MSAs | -- |

### Why this order, and what the first ETA got wrong

FoldBench is ~70% of the ~26.5 h remaining and is the one stage the confidence-head work is not
waiting on -- it is the benchmark top-off. The training cache and validation folds are what
everything downstream needs and together they are 7.8 h, so they run first. The total is unchanged.

The first estimate put FoldBench at 56.6 h. It was measured from a rate observed while the three
workers spent their entire first 3.5 h on the two LARGEST complexes in an unsorted queue -- `7zau`
at 2,520 residues and `7z2m` at 2,244, the only two folded in that window -- against a remaining
queue whose median is 1,230 residues. Rates and sizes, for the record:

| | n | median tokens | observed per-fold |
|---|---|---|---|
| FoldBench, folded so far | 36 | 2,520 | 1,202 s (29 of 34 within 1,195-1,208 s) |
| FoldBench, still queued | 576 | 1,230 residues | -- |
| training cache | 3,983 | 654 | 46 s |

A quadratic-in-residues model anchored on those two complexes predicts 11.5 GPU-h for the 36 already
done against 10.6 actual, 8% over, and gives ~18.7 h for the 576 remaining. It is an extrapolation
DOWNWARD into sizes nothing has folded yet, so treat it as plus or minus several hours until the
workers reach the smaller complexes.

### FoldBench is drained, never killed

`folder.py` has no stop file and no graceful exit, so a worker killed mid-fold loses that fold
outright and its poses do not reproduce. The switch waits for every claimed-but-unwritten unit --
a claim with no `run.json` is a fold on a GPU right now -- before stopping the workers, up to 2 h,
and if it times out it stops nothing. FoldBench is claim-based and resumes where it left off.

### Why stages 2 and 3 write to a fresh root

`folder.py` never reaches the folding path for a unit that already has poses or scores: it returns
`"skipped"` at `folder.py:931` when every interface is scored, and `"rescored"` at `:968` when
`saved_run_dir` finds a finished pose directory. Neither path runs inference, so neither writes a
trunk -- and the trunk is the artefact this phase exists to produce.

| queue | units with poses/scores already on disk | would hang at |
|---|---|---|
| training, 733 | 73 over 8 structures (60 with poses AND scores, 13 poses only) | 660/733 |
| validation, 300 | 300 over 3 structures (all with poses AND scores) | 0/300 |

Deleting the old poses would not have fixed it: 60 of the 73 also carry score rows, so they would
still return `"skipped"` on the csv alone. And a new trunk must not be paired with an old pose --
repeating a seed does not reproduce its poses, one has gone from DockQ 0.022 to 0.644 on a repeat --
so the regenerated poses and their trunks go into one fresh tree together, and nothing is deleted.
`phase_j_build_cache.py:84` already refuses a (pdb, seed) with two finished pose directories rather
than guessing, so a same-root refold would have failed the cache build regardless. It takes
`--fold`, so each root is built separately and each is internally consistent.

`shell_scripts/watch_foldbench_then_training.sh` drives all three stages unattended and remediates
each: it relaunches a dead worker after `reconcile_stale_claims.py` and flags a stall every 45 min.
Running as tmux `fbwatch`, log `experimental_data/.../foldbench_baseline_remaining_folds/watcher.log`.

### Completion, and why it is not the trunk count

`folder.py:1034` passes `PROTENIX_DUMP_TRUNK` into the fold, so the trunk is written DURING
inference while the poses are finalised and the interfaces scored afterwards -- and under
`--overlap-scoring` the scoring runs on a background thread after the fold returns. A trunk count
therefore reports COMPLETE while the last units are still being written, and the stage switch then
kills the workers mid-unit. Reproduced: a trunk with no poses passed the old check.

`phases/phase_j_stage_complete.py` requires, per queued unit, all three of:

| | requirement | matches |
|---|---|---|
| 1 | the trunk npz exists | `<fold>/trunks/<pdb>/<tag>_<seed>.npz` |
| 2 | EXACTLY ONE finished pose dir | what `phase_j_build_cache.py:84` demands |
| 3 | every interface, every expected sample scored | `folder.py`'s own `iface_scored` |

It reuses `folder.py`'s `iface_scored` and `interfaces_by_pdb` and reads the same `ds.pool_csv`
(`folder.py:595`), so the watcher cannot disagree with the worker about what "scored" means.

Controls: a fabricated trunk with no poses is reported `trunk but NO finished pose dir`, exit 1. On
200 units of the old root known to have trunks it reports 170 complete and 30 with incomplete score
rows -- a real pre-existing gap, e.g. `10op_FN` has zero samples on every seed and `7sk4` is
incomplete on 8 consecutive seeds.

### Nothing is torn down without confirmed completion

A stage that does not confirm completion **stops the watcher and leaves its workers running**. The
return code was previously discarded and `tmux kill-session` ran unconditionally, so a stage that
gave up killed workers that might each have been mid-fold -- and a killed fold's poses do not
reproduce, so that is unrecoverable work lost, strictly worse than waiting. Resuming is re-running
the script. This applies to all three stages, including FoldBench: its loop breaks on a POSE count,
and `folder.py:1074` writes `run.json` before `finish_unit` scores the unit, deferred to a
background thread under `--overlap-scoring`, so the count reaches 612 while the last folds are still
scoring. Stage 1's teardown is gated on `--no-trunk` completeness (FoldBench dumps no trunks).

A flat completed count is also not evidence of an idle worker. Three workers can each be part-way
through a long fold; measured, the slowest fold here is 1,208 s on FoldBench and 843 s on the
training cache, none over 2,700 s, but nothing guarantees a larger complex will not exceed the
window. So the give-up additionally requires that NO worker log has been written inside it -- a log
written means a fold is running. Claim count cannot serve as the liveness signal: claims persist
after success, 3,323 against 3,097 trunks.

The escape exists because some units never score completely: `10op_FN` has zero samples on every
seed, `7sk4` is incomplete on 8 consecutive seeds, and a partially scored unit keeps its claim by
design so it is never retried. Without an escape the handoff would hang on those -- but the escape
now only stops supervision, never kills anything.

## Disk

| point | GB used | note |
|---|---|---|
| now | 1,757 | of 3,072; guard stops at 3,000, warns at 2,900 |
| after stages 2+3 | ~2,000 | 733 x 280 MB + 300 x 136 MB |
| UniRef30 conversion peak | ~2,499 | TSVs and converted db coexist |
| peak if stage 2 overlaps the conversion | ~2,680 | 320 GB of headroom |

`BUILD_INDEX=0` is mandatory, not optional: `phase_d_search_uniref.sh:216` defaults it to 1 and the
index is 301 GB, which turns 320 GB of headroom into 19. UniRef30 and envdb never coexist -- phase E
refuses to start while `uniref30_2302_db_seq` exists.

FoldBench writes no trunks: 9 MB per fold measured, so its remaining 581 folds are ~5 GB.

## The refold list

| file | content |
|---|---|
| `train_val_splits/refold_audit_train.csv` | 136 roles / 118 structures / 935 residues, with `corrected_q` |
| `train_val_splits/refold_audit_val.csv` | 15 roles / 13 structures / 124 residues |
| `train_val_splits/skip_boundary_train.csv` | 10 roles not refolded, each with its measured reason |
| `train_val_splits/skip_boundary_val.csv` | 3 roles |
| `building_msa/queries_corrected_v2/` | 144 unique sequences over 151 roles, 157-234 aa, + 5 controls |

Derived by `phases/phase_j_domain_audit.py` then `phases/phase_j_refold_from_audit.py`. A role is
refolded only when the correction is unambiguous: one variable domain on the chain, residues missing
strictly inside a span the query already covers at >= 80%. Every corrected query is checked -- length
equals its span, old query is a subsequence of it -- and all 151 pass.

## Open question on the refold's value

Measured over all 554 structures, where the restored residues sit:

| location of the missing residues | residues | share |
|---|---|---|
| inside a variable domain | 15 | 1.4% |
| constant domain or termini | 1,044 | 98.6% |

Only 8zd6 H (118-127) and 9iut L (114-118) touch a variable domain, and both straddle its
C-terminal edge rather than a CDR. The head is calibrated on per-interface ipTM at the
antibody-antigen interface; these are CH1/CL loops. Cost of proceeding: 96 GB UniRef30, ~700 GB
envdb, 2,480 folds. Parked by decision on 2026-10-14 until the server returns; the 38 GB MSA store
is intact, so the decision stays reversible either way.

## Not done

Implementation status and validation status are different things and are separated here.

| item | implemented | validated |
|---|---|---|
| `phase_j_build_cache.py` | yes, 593 lines | not run over the full cache |
| `phase_j_replay.py` | yes, 368 lines | not run; it is what gates training on the cache |
| the confidence-head training loop | no | -- |
| `phase_i_check.py` | yes | NO: reports COMPLETE when a job is absent from pool.csv, and its cache validator accepts NaN coordinates and bad trunk shapes |
| phases D/E READY markers | yes | NO: left on a die path, open since review round 2 |
| J1/J3 entity+subchain rebuild | no | 9 split single-chain constructs, `reviews/round5_2026-10-13_phasej.md` |

CORRECTION, 11:40 PM ET: an earlier revision of this file claimed 7ubr, 9czd and 9wsv had no
antibody MSA. They do. All three resolve every chain through the path their job carries --
`c577_9wsx_H_506bac`, `c163_9cza_H_7280af`, `c163_7udh_H_9e04f9`. The claim came from globbing for
the pdb id in the MSA directory name, which is wrong because a directory is named after whichever
structure first contributed that sequence; the same mistake also reported seven group-A structures
as MSA-less. `phase_j_queue_now.py` now checks the job's own paths, and a proteinChain carrying
neither `pairedMsaPath` nor `unpairedMsaPath` counts as missing rather than passing through the
loop untested. No group-A structure fails that check: all 74 have every MSA path present.
