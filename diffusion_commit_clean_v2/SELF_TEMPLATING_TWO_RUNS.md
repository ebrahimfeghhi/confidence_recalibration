# Self-templating: the earlier run (10 recycles from scratch) and the trunk-resume run

Written 2026-09-29 23:36. Every table below was generated from the files named in the last section, not typed by hand. **The trunk-resume run is still going: 41 of 160 benchmark interfaces (28 of 104 complexes) are finished, so its numbers are provisional.** The queue runs decoy complexes first, so the finished set is not a random sample (it holds 17 of the 119 solved interfaces).

Class of an interface (used throughout unless stated): from the original FoldBench free pool (seeds 101-150, 250 poses). **solved** = the top-ipTM pose is acceptable (DockQ >= 0.23); **decoy** = it is not, but an acceptable pose exists in the pool; **nocov** = no acceptable pose in the pool. FoldBench has 119 solved, 28 decoy, 13 nocov interfaces.

## 1. The two designs

| | earlier run (`self_template_fixed_50_seed`) | trunk-resume run (`self_template_from_trunks`) |
|---|---|---|
| free folds | the original 50 seeds (101-150), stock settings | re-run: 50 seeds (101-150), stock settings, bf16 trunks (z, s) saved for every fold |
| donors | up to 50 structurally different poses per complex, crystal-free (`build_experiment.py`) | the same selection code, run on the re-run free folds |
| templated fold | fold from scratch with the donor as template, **10 recycles**, one fixed seed (430000) for every donor, stock MC-dropout | resume the donor's own saved trunk for **3 extra recycles** with the donor as template, the donor's own seed, MC-dropout **off** |
| control | none in this run (compute-matched free seeds were a separate arm) | same trunks, 3 extra recycles, **no template**, same number of folds (donors' seeds first, then more seeds from the 50, ascending) |
| selection / scoring | top-ipTM pose over the arm's folds; DockQ vs crystal (evaluation only) | the same |
| MSA | stock (rows subsampled per cycle) | stock |

## 2. Earlier run, all 160 interfaces (acceptable = top-ipTM pick, DockQ >= 0.23)

| class | interfaces | free pick acceptable | earlier run acceptable | broken (solved -> wrong) | rescued (decoy -> acceptable) |
|---|---|---|---|---|---|
| solved | 119 | 119 | 109 | 10 | - |
| decoy | 28 | 0 | 5 | - | 5 |
| nocov | 13 | 0 | 0 | - | - |
| **all** | **160** | **119** | **114** | **10** | **5** |

Dropout in the earlier run (measured 2026-09-29: first donor of each complex re-run exactly as the original arm, decision logged; `self_template_from_trunks/archive/dropout_check_original_arm/decisions.csv`): of the 10 broken solved interfaces, the complexes of 7 had MC-dropout **off** at seed 430000 (7yxu, 7zoz, 8fxv, 8ux6, 9fww) and 3 had it **on** (8be3, 8pn0, 8r4u). The original run did not log the decision.

## 3. Trunk-resume run, finished interfaces (class and pick from this run's own free folds)

| class | interfaces | free pick acceptable | control | templated | mean DockQ free / control / templated |
|---|---|---|---|---|---|
| solved | 14 | 14 | 14 | 14 | 0.712 / 0.696 / 0.704 |
| decoy | 24 | 0 | 0 | 2 | 0.081 / 0.061 / 0.086 |
| no_acceptable | 3 | 0 | 0 | 1 | 0.072 / 0.073 / 0.306 |
| all | 41 | 14 | 14 | 17 | 0.296 / 0.279 / 0.313 |

Interfaces where an arm is acceptable and the free pick is not: 9jbq_AC [no_acceptable] (free 0.04, control 0.04, templated 0.73), 8ux6_AF [decoy] (free 0.10, control 0.09, templated 0.24), 8ux6_DF [decoy] (free 0.14, control 0.16, templated 0.30).

## 4. Matched comparison on the same finished interfaces (old-pool classes)

| class (old pool) | interfaces | old free pick | earlier run | new free pick | new control | new templated |
|---|---|---|---|---|---|---|
| solved | 17 | 17 | 15 | 14 | 14 | 15 |
| decoy | 22 | 0 | 4 | 0 | 0 | 2 |
| nocov | 2 | 0 | 0 | 0 | 0 | 0 |
| **all** | **41** | **17** | **19** | **14** | **14** | **17** |

The new free run is a fresh set of folds, so its classes differ from the old pool for: 8tmz_AC (old solved, new decoy), 8tmz_BC (old solved, new decoy), 7ucx_AC (old decoy, new no_acceptable), 9jbq_AC (old decoy, new no_acceptable), 8oxx_AB (old nocov, new decoy), 8ux6_DF (old solved, new decoy).

## 5. Caveats

- Provisional: the trunk-resume run has finished only the interfaces listed above. The 119-solved "how much does it break" question is barely tested so far.
- The two runs differ in more than one way (recycles from scratch vs a 3-recycle resume, fixed vs own seed, stock vs off dropout), so differences between them are not attributable to one factor.
- A rescue near the 0.23 line (DockQ 0.24 to 0.30 for the two 8ux6 interfaces) is marginal.
- Class definitions come from one 50-seed pool and shift between pools.
- Nothing here has been tested for a mechanism; these are counts.

## 6. Where everything is

| what | path |
|---|---|
| earlier run: queue, poses, results, donor summary | `experimental_data/protenix/foldbench/self_template_fixed_50_seed/` (`results.csv`, `donor_summary.md`) |
| free pool, chain roles | `experimental_data/protenix/foldbench/pool.csv`, `chain_roles.csv` |
| trunk-resume run: code, plan, per-complex data, summary | `self_template_from_trunks/` (`code/pipeline.py`, `code/report.py`, `PLAN.md`, `data/<pdb>/`, `summary_all.csv`, `summary_classes.csv`, `controller.log`, `deletions.log`) |
| v2 scripts it calls | `diffusion_commit_clean_v2/` (`folder.py`, `build_experiment.py`, `pose_to_pose_dockq.py`, `structural_diversity_filter.py`, `build_templates.py`, `preflight.py`, `hold_gpus.py`) |
| Protenix edits (trunk save/load, dropout rate, template gain) | `Protenix` branch `wnoise-hooks`, commit `bbc26ef` |
| dropout check of the original arm | `self_template_from_trunks/archive/dropout_check_original_arm/` |
