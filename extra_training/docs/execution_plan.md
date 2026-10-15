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

| stage | what | units | consumer | writes to |
|---|---|---|---|---|
| 1 | FoldBench top-off | 31 of 612 done | `fbtop_0/1/3` | `foldbench_baseline_remaining_folds/` |
| 2 | training cache | 733 | `tcache_0/1/3`, `sabdab_train` | `training/fold_regen/` |
| 3 | validation | 300 | `vfold_0/1/3`, `sabdab_val2` | `validation/fold_regen/` |
| deferred | the refolds | 2,480 | blocked on the corrected MSAs | -- |

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

Because of those, the check cannot be a strict AND with no escape: a partially scored unit keeps its
claim by design and is never retried, so the handoff would hang on it. A stage therefore also ends
when the completed count has not moved for `STALL_POLLS` (9 polls, 45 min) while all three workers
are alive, and the stragglers are then NAMED in the log. Claim count is not a liveness signal --
claims persist after success, 3,323 against 3,097 trunks -- so progress is the completed count.

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
