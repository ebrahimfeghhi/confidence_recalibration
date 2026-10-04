# Pose-to-pose DockQ matrices

How similar every pose of an interface is to every other pose, as DockQ. Built by
`diffusion_commit_clean_v2/pose_to_pose_dockq.py` (GPU, ~15 s per interface).

**These already exist for every interface of both pools. Read them; do not rebuild them.**
Reach them from code as `paths.DATASETS[name].matrices`, never by a literal path.

## Which directory

| directory | frame | interfaces | complete? | use it? |
|:---|:---|---:|:---|:---|
| `pxm2025_crystal` -> `../analysis/crystal_frame_selection/pose_dockq_matrices` | crystal | 292 | yes, all of pxm2025 | **YES** |
| `foldbench_crystal` | crystal | 160 | yes, all of FoldBench | **YES** |
| `pxm2025` | anchor | 414 | — | only for anchor-frame work |
| `foldbench` | anchor | 465 | — | only for anchor-frame work |

**Frame is not cosmetic.** Crystal-frame files are keyed by the interface name as
`chain_roles.csv` spells it (`9n6e_EF`), anchor-frame files by anchor letters (`8ror_BA`), so the
two name the same interface differently and a name lookup in the wrong directory silently misses.
Of this project's 89 pxm2025 decoy interfaces, `pxm2025` matches 40 by name and `pxm2025_crystal`
matches all 89. `pxm2025_crystal` is a symlink, so it costs no extra disk.

## Do NOT use

`../self_template_fixed_50_seed_old_crystal_iptm/pose_dockq_matrices` also holds 292 crystal-framed
npz and covers all 89 decoy interfaces, so a file listing makes it look equally good. It is wrong:
on `9n6e_EF` its `dockq` differs from a fresh build by **max |diff| 0.83** and its `model_ab` names a
different antibody chain. Its directory name is the only warning it carries.

`archive/superseded_2026-09-25/matrices`, `matrices_anchor_all` and `matrices_spur` are EMPTY.
`paths.py` used to define `XTAL_MATRICES` / `ANCHOR_MATRICES` / `SPUR_MATRICES` pointing at them;
those names were removed, because a constant that resolves to an empty directory returns zero files
instead of an error.

`seed_scatter_50/matrices` (1771 npz) is a different thing — keyed per fold, not per interface.

## What is in an npz

| key | shape | meaning |
|:---|:---|:---|
| `dockq` | (N, N) float32 | symmetrised DockQ; 1.0 = same arrangement, ~0 = unrelated |
| `seed`, `sample` | (N,) | which pose each row is |
| `model_ab`, `model_ag` | (N,) | the chains compared, **per pose** — not one pair for the set |
| `pair_iptm` | (N,) | the model's confidence at those chains |

- **Index by `(seed, sample)`, never by row order.** Rows are labelled precisely so that a row order
  drifting from its labels cannot pass unnoticed.
- `dockq` is symmetrised (the mean of both directions) because raw DockQ is directional: fnat divides
  by the reference's contact count.
- **N < 250.** A pose with no antibody-antigen contact at that chain pair is dropped before the matrix
  (median N = 247 on pxm2025; `9jt1_BE` drops 70). Its similarity to everything would be 0 by
  construction, which is the absence of an interface, not diversity.
- On a complex with several antigen copies, different poses bind different copies, which is why the
  compared chains are stored per pose.

## Verified

2026-10-04: `pxm2025_crystal/9n6e_EF.npz` and `foldbench_crystal/8fbw_AC.npz` reproduce a fresh
`pose_to_pose_dockq.py` build exactly — same pose set, same chains, max |diff| 1e-6 and 0.

Measurements built on these: `../../experimental_results/decoy_pose_redundancy.csv`.
