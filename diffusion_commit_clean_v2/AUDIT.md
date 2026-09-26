# Audit — self-template, fixed-seed arm

Started 2026-09-25. Scope: `experimental_data/self_template_fixed_50_seed/` — templates, folding,
bookkeeping, and the folds carried over from earlier runs.

## Findings

| # | finding | status |
|:---|:---|:---|
| 1 | Templates are correct: the model receives the donor pose itself | verified, 1 job |
| 2 | Folds were overwriting each other's poses; scores unaffected | fixed `2fd17b8` |
| 3 | 2,585 carried-over folds match their source exactly | verified, one gap open |
| 4 | Two leftover `watch.sh` auto-restarters from 2026-09-24 still running | harmless; awaiting stop |

### 1. Templates

For `8ror_AB_A00`, the template features Protenix builds (CPU, no model) were compared against the
donor cif read independently with gemmi:

| check | result |
|:---|:---|
| template residue types = query sequence | 1,897 / 1,897 tokens |
| template atoms on donor atoms | worst offset 0.0000 Å |
| template distogram = donor Cβ distances | 100% of 3.6 M pairs, incl. 1.38 M cross-chain |

Cross-chain template geometry reaches the model (`PROTENIX_MULTICHAIN_MASK=0`) — **intended**
(confirmed by the user). The a3m hit line carries two deliberate point mutations per hit
(`build_templates.py:246`); the features come from the cif, so they do not reach the model.

Script: `archive/audits/audit_template_features.py`.

### 2. Overwritten poses

`runmeta.run_id()` drew from the global `random`, which Protenix re-seeds with the fold seed before
every fold. Every templated fold uses seed 430000, so directory names repeated.

| run | folds | distinct run dirs |
|:---|---:|---:|
| free | 11,028 | 11,028 |
| free_compute_matched | 1,432 | 1,432 |
| self_template_fixed_50_seed | 181 | 27 |
| reffree_50 | 79 | 9 |

- Scores are intact: each fold is scored from its own job-named subdirectory before its poses move.
  7,341 (interface, seed, tag) groups, each with exactly samples 0–4; donor matches the queue on
  every row; no DockQ/ipTM outside [0, 1].
- 154 / 181 fixed-arm folds lost their poses — listed in
  `experimental_data/self_template_fixed_50_seed/poses_overwritten_2026-09-25.tsv`. Not refolded
  (poses not needed).
- Fix: `run_id()` uses `SystemRandom`; the run directory is created exclusively and pose moves refuse
  to overwrite, so any future collision fails loudly. `tests/test_no_overwrite.py` passes on the fix
  and fails all three checks on the old code.
- Both fixed-arm workers restarted on the fix at 18:07 / 18:09; folds since then each get their own
  directory.

### 3. Carried-over folds

| source | folds | how verified |
|:---|---:|:---|
| folded by `reffree_50` | 856 | rows identical; its logs survive ("Found 1 templates") |
| inherited by `reffree_50` from `anchor_50` | 1,729 | rows identical; run records deleted |

- 33,290 carried rows identical to `reffree_50` (DockQ, ipTM per interface and sample); 0 duplicates;
  no fold both carried and refolded; every arm tag matches the queue.
- Chain set of every carried fold equals what today's `build_templates.py` picks: 2,585 / 2,585.
- Template flags and hit construction unchanged in git since at least 2026-09-18.
- **Open:** `anchor_50` was deleted 2026-09-25 with its logs and `run.json`, so for its 1,729 folds
  only the code in git — not a record of the run — shows the settings. A direct check (refold a
  sample and compare) is hard because folds do not reproduce exactly in reduced precision; revisit
  later.

### 4. Leftover watchers

`watch.sh --exp reffree_50 --gpus 1,5,6` (PID 1429227) and `watch.sh --exp free_matched --gpus 1,5,6`
(PID 1476361). Neither can launch now — `reffree_50` has no unclaimed folds, `free_matched/` no longer
exists — but both are auto-restarters on our GPUs. Stopping them was blocked by the permission
system; to stop: `kill 1429227 1476361 3314649 3321606`.

## Tidy-ups (`2e8c961`)

- `run.json` `git_sha` now records the code the worker loaded, not HEAD when the fold finished.
- `folder.py`: removed the `SELFTMPL_*` env vars (read by nothing) and the unused `CLEAN_CODE`.
- `paths.SELFTMPL_DIR` / `TEMPLATES_DIR` removed (pointed at the deleted `self_template/`).

## Not yet checked

- Template features on ~20 more jobs, including complexes with more chains.
- Selection rule: every donor seed in 300000–300049, legacy fold directory, no duplicate donors.
- The arm comparison recomputed independently of `compare_arms.py`.
