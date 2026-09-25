# experimental_data

Four arms, an archive, and the inputs the pipeline reads. Most of this is NOT in git: poses
(`.cif`), pose-to-pose DockQ matrices (`.npz`), MSAs (`.a3m`) and per-pose confidence json are
excluded by size. Committed instead: queues, selections, scores, roles and diffs — enough to
read the experiment without rerunning it.

| directory | size | files | poses | matrices | what it is |
|:---|---:|---:|---:|---:|:---|
| `anchor_frame/` | 3 MB | 7 | 0 | 0 | chain maps and roles shared across runs |
| `archive/` | 177.2 GB | 1,283,559 | 133,239 | 4,986 | superseded era |
| `free/` | 25.2 GB | 85,133 | 34,805 | 0 | shared donor pool — every arm templates on these poses |
| `free_compute_matched/` | 5.3 GB | 21,826 | 6,416 | 0 | ARM: extra free seeds, compute-matched to the method |
| `matrices/` | 94 MB | 298 | 0 | 292 | pose-to-pose DockQ, crystal frame |
| `matrices_anchor_all/` | 76 MB | 254 | 0 | 242 | pose-to-pose DockQ, anchor frame |
| `matrices_spur/` | 7 MB | 57 | 0 | 53 | pose-to-pose DockQ, predicted-only interfaces |
| `pxm2025/` | 7.9 GB | 36,385 | 131 | 0 | input pool: job jsons, ground-truth cifs, the pool index |
| `reffree_50/` | 1.0 GB | 8,979 | 1,281 | 0 | provenance for the 2,585 carried scores |
| `self_template_fixed_50_seed/` | 2.6 GB | 21,818 | 3,357 | 328 | ARM: every templated fold on seed 430000 |
| `self_template_variable_50_seed/` | 1.6 GB | 14,416 | 2,425 | 0 | ARM: one seed per donor, seeds 300100-300149 |

## inside an arm

| name | what it holds |
|:---|:---|
| `templates/` | the template namespace: one namespace per fold, plus the template mmcifs |
| `poses/` | fold output — the predicted structures |
| `pose_dockq_matrices/` | N x N symmetric pose-to-pose DockQ, one npz per interface |
| `predicted_only_interfaces/` | pair ipTM for interfaces the predictions declare and the reference lacks |
| `chain_frame/` | chain maps and roles in the anchor frame |
| `queue.tsv` | every fold the selection asks for |
| `queue_tofold.tsv` | the subset still needing GPU |
| `carry_over_folds.tsv` | folds reused from an earlier run rather than refolded |
| `results.csv` | one row per scored interface per pose |
| `claims/` | atomic mkdir claims so workers share one queue |
