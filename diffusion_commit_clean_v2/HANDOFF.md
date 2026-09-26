# Handoff — self-templating vs free sampling

Written 2026-09-25 21:35. Read `README.md` for the pipeline and `AUDIT.md` for every change made on
2026-09-25 and why. This file is the live state.

## The question

Does self-templating on the model's own poses rescue confident decoys better than spending the same
compute on more free seeds? Arms, all selected the same way (top ipTM, DockQ read off the selected pose,
acceptable at DockQ >= 0.23), compared per class by `compare_arms.py`:

| arm | directory | what it is |
|:---|:---|:---|
| free 50 | `experimental_data/free/` | seeds 300000-300049; also the donor pool. **Never rename** (absolute paths everywhere) |
| free + matched | `experimental_data/free_compute_matched/` | extra free seeds, per complex >= the method's donor folds |
| fixed seed | `experimental_data/self_template_fixed_50_seed/` | every donor templated, all on seed 430000 |
| variable seed | `experimental_data/self_template_variable_50_seed/` | the same donors and templates, one seed each from 300100-300149 |

Superseded, kept for provenance: `self_template_fixed_50_seed_old_crystal_iptm/` (its still-valid
folds were carried into the new arm) and `self_template_variable_50_seed_old_crystal_iptm/` (never
folded).

## Live state (21:35)

- Selection rebuilt crystal-free: 3,524 donors over 123 complexes. 2,144 folds carried from the old arm
  (25,060 rows, verified identical); **1,380 to fold** (`queue_tofold.tsv`).
- Folding: fixed arm on GPUs 0, 1, 2, 5 (tmux `self_template_fixed_50_seedw<gpu>`). Measured on the
  first 20 folds: 135 s per fold on 8uzp; per-fold cost varies 20 s - 20 min by complex, so no ETA
  from that alone.
- `tmux stagearms` — `diffusion_commit_clean_v2/stage_arms.sh`, per GPU: fixed arm -> the 12-fold free
  top-up (`free_compute_matched/queue_free_topup_arm_a.tsv`) -> variable-seed arm (3,524 folds, with
  `--overlap-scoring`: scoring runs in the background while the next fold runs, ~13% faster). Log
  `experimental_data/stage_arms.log`.
- `tmux watcharms` — `diffusion_commit_clean_v2/watch_arms.sh`, every 30 min, claims and scored rows
  checked separately. Log `experimental_data/watch_arms.log`.
- Our GPUs are 0, 1, 2, 5. GPU 4 is another user's. **Never more than 4 without asking.** An idle card
  is held with `hold_gpus.py` (tmux `hold<gpu>`); the stager releases it at launch.

## When the fixed arm finishes

1. `compare_arms.py --exp experimental_data/self_template_fixed_50_seed` — free 50 vs free + matched vs
   fixed seed. It reports complexes not yet compute-matched and excludes them; after the top-up there
   should be none.
2. The same with `--exp experimental_data/self_template_variable_50_seed` once that arm finishes.
3. Commit the arms' `results.csv` and `index.tsv` at that milestone (not while workers write them).

## Open items

- **The 33 unannotated chains** (22 complexes): the benchmark labels only the chains in its scored
  crystal interfaces, so which chains may form interfaces is still slightly shaped by the crystal.
  Close it by labelling every chain from 1-D metadata, checked by hand, and measuring the change.
- **Anchor sensitivity:** the anchor only names chains, but for duplicate copies the naming can split or
  merge binding modes across interface lists. Test: rerun selection with a different anchor and compare
  donors. `analysis/crystal_frame_selection.md` is partial evidence (B names chains by the crystal).
- **Template replacement:** `point_templates_at()` replaces Protenix's 3-4 PDB templates with the one
  self-template rather than adding it, so a rescue could come from the self-template or from removing
  the PDB templates. Separating them needs a third condition.

## Working rules for this user

Lead with the answer, keep it short, one idea at a time when explaining. Report n on every row; per-
interface `chain_pair_iptm`, never global ipTM. Markdown tables. Measure before claiming; say "cannot
measure that yet" rather than estimate. Report, don't interpret. Never launch, kill or exceed 4 GPUs
without approval; kill by PID, never `pkill -f` (it matches its own command line). Never delete poses.
Everything lives in `tts_abag`. Code in `diffusion_commit_clean_v2/`, data in `experimental_data/`.
Docstrings follow SUMMARY / INPUTS / OUTPUTS / EXAMPLE / FUNCTIONS, functions `in: ... out: ...`.
