# Deleted: `self_template_from_trunks/` — 2026-10-14

Deleted on user instruction ("you can delete poses for the self templated runs thats ok"). Recorded
here because a deletion no one can see is indistinguishable from an oversight.

## What was there

| | |
|---|---|
| total | 959 MB, 97,193 files |
| json (per-pose confidence summaries) | 85,886 files, 544 MB |
| a3m | 3,223 files, 1 MB — `.no_templates.a3m` marker files, not real MSAs |
| csv / tsv / md / log | 1,185 files |
| cif / pdb / npz (poses and trunks) | **0** |

**There were no poses left to delete.** They had already gone on 2026-10-02, per the tree's own
`DELETED_BULK_2026-10-02.log`: *"deleted all .npz (trunks) and .cif (poses) ... on user instruction
(1 TB quota). Text records (csv/md/py/sh/json/logs) kept."* What remained was the text record of
folds whose structures were already gone.

## What was kept

- **The experiment's results**, copied to `experimental_results/`:
  `self_template_from_trunks_summary_all.csv` and `..._summary_classes.csv` — 43 interfaces each,
  with `pool_best_dockq`, `free_pick_dockq`, `ctl_pick_dockq`, `arm_pick_dockq` and the per-arm
  acceptable flags. This is the data behind the finding that mask-ON self-templating loses to free.
- **114 tracked files** (README.md, PLAN.md, `code/`, and 107 under its own `archive/`) remain in git
  history and can be restored with `git checkout <commit>^ -- self_template_from_trunks/`.

The ~97,000 untracked files are gone for good. Nothing in `reports/`, `experimental_results/` or
`extra_training/` cited them.
