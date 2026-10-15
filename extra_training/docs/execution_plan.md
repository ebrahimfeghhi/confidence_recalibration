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

| stage | what | units | consumer |
|---|---|---|---|
| 1 | FoldBench top-off | 31 of 612 done | `fbtop_0/1/3` |
| 2 | training cache | 733 | `tcache_0/1/3`, dataset `sabdab_train` |
| 3 | validation | 300 | `vfold_0/1/3`, dataset `sabdab_val2` |
| deferred | the refolds | 2,480 | blocked on the corrected MSAs |

`shell_scripts/watch_foldbench_then_training.sh` drives all three stages unattended and remediates
each: it relaunches a dead worker after `reconcile_stale_claims.py` and flags a stall every 45 min.
Running as tmux `fbwatch`, log `experimental_data/.../foldbench_baseline_remaining_folds/watcher.log`.

Completion per stage is the count of trunk npz for that queue's own rows, never a pose count and
never the absence of workers. Poses for the group-A structures already exist -- their trunks were
deleted, not their poses -- so a pose count reports stage 2 finished before one trunk is written.
Positive control for the counter: on the pre-split queue it returns 3,097 of 3,760, the known
trunk count.

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

| item | state |
|---|---|
| `phase_j_build_cache.py`, `phase_j_replay.py` | not written; the replay gates training on the cache |
| the confidence-head training loop | does not exist |
| `phase_i_check.py` | reports COMPLETE when a job is absent from pool.csv; accepts NaN coords and bad trunk shapes |
| phases D/E | leave READY markers on a die path |
| J1/J3 entity+subchain rebuild | 9 split single-chain constructs, `reviews/round5_2026-10-13_phasej.md` |
| 7ubr H, 9czd, 9wsv H/L | no antibody MSA was ever built; drop or build before they can fold |
