# experimental_data

The arms, the inputs the pipeline reads, the paper analyses, and an archive. Most of this is not in
git: poses (`.cif`), DockQ matrices (`.npz`), MSAs and per-pose confidence json are excluded by size.
Queues, selections, scores and diffs are committed — enough to read an experiment without rerunning it.
Code lives in `../diffusion_commit_clean_v2/`; see its `README.md` and `HANDOFF.md`.

| directory | size | what it is |
|:---|---:|:---|
| `free/` | 26 G | the free pose pool, seeds 300000–300049 (+ deeper seeds on some complexes); every template's donor. **Never rename** |
| `free_compute_matched/` | 6.6 G | ARM: extra free seeds, per complex at least the method's donor folds |
| `self_template_fixed_50_seed/` | 3.0 G | ARM: the crystal-free selection (3,524 donors), every donor templated on seed 430000 |
| `self_template_variable_50_seed/` | — | ARM: the same donors and templates, one seed each from 300100–300149 |
| `self_template_fixed_50_seed_old_crystal_iptm/` | 11 G | superseded 2026-09-25: the previous selection; source of the 2,144 carried folds |
| `pxm2025/` | 8.0 G | inputs: job jsons, MSAs, crystal cifs, the crystal-scored pool csv, chain maps and roles |
| `analysis/` | 52 M | intermediates of `diffusion_commit_clean_v2/analysis/` (e.g. the crystal-frame selection B) |
| `pose_dockq_matrices/` | 166 M | pose-to-pose DockQ over the free-50 pool, one npz per interface, one subdirectory per (dataset, frame) — **read its `README.md` first**. `pxm2025_crystal` (292, symlink into `analysis/`) and `foldbench_crystal` (160) are COMPLETE for their pools; get them as `paths.DATASETS[name].matrices` and never rebuild them. `pxm2025/`, `foldbench/` are the anchor-frame sets. One other 292-npz set elsewhere looks identical and is WRONG — the README names it |
| `archive/` | 180 G | earlier eras; `archive/superseded_2026-09-25/` holds the crystal-framed pipeline's outputs (`anchor_frame/`, `matrices*/`) and `reffree_50/` |

Top-level logs: `stage_arms.log` (GPU handoffs), `watch_arms.log` (30-minute checks).

## Inside an arm

| name | what it holds |
|:---|:---|
| `poses.csv` | the pool the selection drew from |
| `chain_frame/` | per-pose contacts and ipTM, the anchor per complex, chain maps onto it |
| `interfaces.csv` | the interfaces, in anchor letters |
| `pose_dockq_matrices/` | a symlink into the top-level `pose_dockq_matrices/<dataset>/` (since 2026-10-02 the npz live there, not per arm). Names are in the arm's ANCHOR chain letters (`chain_frame/`, `interfaces.csv`) — a crystal-frame name looked up here silently misses |
| `queue.tsv` | every selected fold: kind, pdb, interface, tag, seed, donor pose |
| `queue_tofold.tsv` | the folds not carried over from an earlier build |
| `carry_over_folds.tsv`, `diff_vs_*.md` | what was carried, by fold identity (donor, seed, template chains) |
| `templates/` | one namespace per fold: job json, chosen chains, a3m hits; template mmcifs |
| `poses/<pdb>/<seed>-<run id>/` | one directory per fold: `predictions/` (5 poses + confidence) and `run.json` |
| `claims/` | one directory per claimed fold; the workers' locking |
| `results.csv` | one row per scored pose and crystal interface — also the to-do list |
| `w<gpu>.log`, `build.log` | worker and build logs |
