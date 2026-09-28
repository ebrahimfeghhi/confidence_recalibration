# Self-templating from the model's own poses: summary (2026-09-28)

**Question.** Does refolding Protenix with its own free poses as templates (up to 50 structurally different
donor poses per complex, one fold each) rescue confident-decoy antibody-antigen interfaces better than
spending the same compute on more free seeds?

**Conclusion (user, 2026-09-28).** Promising on pxm2025, net harmful on FoldBench: there, templating raised
commitment and ipTM more for bad poses than for good ones in several cases and broke multiple solved
interfaces. Little room is left to iterate on templating as a lever. A useful negative result: the gain does
not generalize across benchmarks.

Selection everywhere is the pose with the highest ipTM at the interface; acceptable = DockQ >= 0.23 against
the crystal. Classes come from the first 50 free seeds: **decoy** = an acceptable pose exists but the top-ipTM
pose is not, **solved** = the top-ipTM pose is acceptable, **nocov** = no acceptable pose. Arm names:
**free + matched** = the 50 free seeds plus as many extra free seeds as the method spent donor folds, per
complex; **self-template** = the fixed-seed (430000) templated folds.

## 1. pxm2025 (292 interfaces, 123 complexes; complete)

| class | interfaces | free 50 | free + matched | self-template | self-template only / free + matched only | McNemar p |
|:---|---:|---:|---:|---:|---:|---:|
| decoy | 89 | 0/89 | 6/89 | **20/89** | 17 / 3 | 0.0026 |
| solved | 142 | 142/142 | 141/142 | 137/142 | 1 / 5 | 0.22 |
| nocov | 61 | 0/61 | 0/61 | 1/61 | 1 / 0 | 1 |

Per complex (cluster bootstrap, 10,000 resamples): decoy acceptable-rate difference +0.15, 95% CI
[+0.015, +0.281]; solved -0.04, [-0.110, +0.014].

**Templating did not raise pose quality; it changed which pose was picked.**

| class | best available pose acceptable: free + matched | self-template | share of all poses acceptable: free + matched | self-template |
|:---|---:|---:|---:|---:|
| decoy (89) | 89 | 83 | 10.5% | 9.0% |
| solved (142) | 142 | 142 | 70.4% | 64.2% |
| nocov (61) | 7 | 7 | 0.0% | 0.3% |

## 2. FoldBench (160 interfaces; fixed arm complete, compute-matched baseline partial)

Against the 50 free seeds (complete):

| class | interfaces | free 50 | self-template | rescued | broken |
|:---|---:|---:|---:|---:|---:|
| decoy | 28 | 0/28 | 5/28 | 5 | 0 |
| solved | 119 | 119/119 | 109/119 | 0 | **10** |
| nocov | 13 | 0/13 | 0/13 | 0 | 0 |

- Rescued: 7z2m_AJ, 8av2_AC, 8slb_AB, 8slb_AC, 9jbq_AC.
- Broken: 7yxu_AB, 7yxu_AC, 7zoz_AB, 7zoz_AC, 8be3_AB, 8fxv_AB, 8pn0_AE, 8r4u_AB, 8ux6_DF, 9fww_AB. Pooled with
  the free poses, 8 stay broken.

Against free + matched: decoy and nocov are fully matched (extra free seeds 0-100 were already on disk);
solved is matched on 17 of 119 interfaces so far (top-up running).

| class | interfaces matched | free + matched | self-template | McNemar p |
|:---|---:|---:|---:|---:|
| decoy | 28/28 | 0/28 | 5/28 | 0.0625 |
| nocov | 13/13 | 1/13 | 0/13 | 1 |
| solved | 17/119 (partial) | 16/17 | 12/17 | 0.125 |

**Commitment rose more for the wrong pose.** Within-fold agreement = mean pairwise DockQ among a fold's 5
samples (OST), for the donor that produced the winning wrong fold vs the best correct donor:

| interface | wrong donor: free -> templated | correct donor: free -> templated | top ipTM after templating, wrong vs correct |
|:---|:---|:---|:---|
| 7yxu_AB | 0.50 -> 0.84 | 0.44 -> 0.73 | 0.697 vs 0.666 |
| 7yxu_AC | 0.65 -> 0.91 | 0.69 -> 0.82 | 0.706 vs 0.654 |
| 7zoz_AB | 0.30 -> 0.72 | 0.36 -> 0.43 | 0.517 vs 0.419 |
| 7zoz_AC | 0.26 -> 0.71 | 0.44 -> 0.46 | 0.510 vs 0.409 |
| 8be3_AB | 0.47 -> 0.75 | 0.72 -> 0.36 | 0.859 vs 0.328 |
| 8fxv_AB | 0.77 -> 0.91 | 0.38 -> 0.81 | 0.943 vs 0.670 |

The wrong fold ends tighter in 6/6 and gains more agreement in 5/6 (8fxv is the exception). 8pn0_AE has no
wrong donor. On the 5 rescues the opposite held: correct donors gained more agreement than wrong ones
(median +0.33 to +0.88 vs +0.10 to +0.54, dockq_torch; 8slb not yet cross-checked against OST).

## 3. Follow-ups run on the FoldBench breaks

- **7zoz** (the main case study; 7zoz_AB): the wrong donor A05 refolds to the same wrong pose on its own free
  seed as on seed 430000 (top ipTM 0.519 vs 0.517); its three wrong siblings from that free fold refold to
  0.40-0.44; all 24 donors as templates in one fold give ipTM about 0.30. Data: `self_template_donor_seed_7zoz/`,
  `self_template_siblings_7zoz/`, `self_template_all24_7zoz/` under `experimental_data/protenix/foldbench/`.
- **Query-only MSA** (MSA cut to the query row, same template and seed; 7 broken interfaces): selection flips
  to the good donor on 7yxu_AB and 7yxu_AC (the latter by 0.0001); 7zoz, 8be3 and 8fxv still pick the decoy.

## 4. Caveats

- FoldBench solved vs free + matched is partial until the free top-up finishes (1,972 folds, ~10% done at
  the time of writing).
- pxm2025's variable-seed arm is incomplete (paused at 1,510 claims); only the fixed-seed arm is reported.
- 60% of pxm2025's templated folds (the 2,144 carried from the old arm) have scores but no poses on disk, so
  the scatter analysis is FoldBench-only.

## Sources

Every table is regenerated by the script named; per-interface data are in the csvs.

| table | script | output |
|:---|:---|:---|
| pxm2025 arms, McNemar, bootstrap | `compare_arms.py`, `analysis/arm_statistics.py` | `arms_compare.md`, `arm_statistics_pxm2025.md` |
| pxm2025 best available pose | `compare_arms.py --select oracle` | `arms_compare_oracle.md` |
| pxm2025 share of acceptable poses | `analysis/pose_quality.py` | `pose_quality_pxm2025.csv` |
| FoldBench vs free 50 | `score_experiment.py --dataset foldbench` | `foldbench/vs_free50.md` |
| FoldBench vs free + matched | `analysis/arm_statistics.py --dataset foldbench` | `foldbench/arm_statistics_partial.md` |
| FoldBench agreement, breaks | `analysis/fold_scatter.py --method ost` | `experimental_data/protenix/foldbench/self_template_fixed_50_seed/fold_scatter_broken.csv` |
| FoldBench agreement, rescues | `analysis/fold_scatter.py --method torch` | `.../fold_scatter_rescued_torch.csv` |
| query-only MSA | `analysis/query_only_msa.py`, `analysis/fold_scatter.py` | `experimental_data/protenix/foldbench/self_template_msa1_broken/` |

Scripts are in `diffusion_commit_clean_v2/`; outputs in `experimental_results/` unless a path is given.
