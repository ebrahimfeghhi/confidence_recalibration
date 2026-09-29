# 9fww template gain sweep (launched 2026-09-28)

**Why:** FoldBench interface where self-templating broke a solved interface (9fww_AB, one of the 10). Test whether
scaling the template contribution to z by g=2 makes every donor's fold commit (low DockQ scatter across its 5 samples).
9fww chosen because it is fastest: 243 tokens, 18 donors, ~8 s/fold in the existing arm.

**Method**
- Code change: `Protenix/protenix/model/modules/pairformer.py` TemplateEmbedder.forward, env `PROTENIX_TMPL_GAIN=<g>`
  multiplies the template contribution u (added into z by protenix.py) by g. Unset/1.0 = unchanged. Backup of the file
  before the change: `../pairformer.py.before_gain`. Logs a WARNING once per process when g != 1.
- Same 18 donors, templates and seed (430000) as the existing arm `experimental_data/protenix/foldbench/self_template_fixed_50_seed`
  (namespace reused read-only, tags A00-A17). queue.tsv = its 9fww rows.
- Arms here: `g2` (PROTENIX_TMPL_GAIN=2) and `g1` (PROTENIX_TMPL_GAIN=1.0, control through the same code path).
  Folds do not reproduce run to run, so g1 gives the run-to-run noise; the original arm's rows are a third reference.
- folder.py --dataset foldbench --dtype bf16 (as the original arm) --ncycle 10 --nsample 5; outputs under `g2/` and `g1/`.
- Measured per fold: DockQ to crystal (results.csv), within-fold agreement (pairwise DockQ of the 5 samples), DockQ to donor.

## Result (2026-09-28)
Both arms complete: 18 folds, 90 rows each. Within-fold agreement from `diffusion_commit_clean_v2/analysis/fold_scatter.py
--method ost --poses <arm dir>` -> `fold_scatter_g1.csv`, `fold_scatter_g2.csv`. Selection = highest-ipTM pose over all 18 donors x 5 poses
(same rule as the arm comparison). Check: g1 reproduces the original arm donor-by-donor (median DockQ and max ipTM equal to 2 decimals on
most donors; same top-ipTM pose), so the hook is inert at g=1.
- Within-fold agreement (mean pairwise DockQ of the 5 samples; 1 = no scatter), median over 18 donors: free 0.146, g=1 0.474, g=2 0.243.
  No fold reaches >=0.95 at either gain (g=1 best 0.936 = A01, g=2 best 0.609 = A00). 15 of 18 folds have LOWER agreement at g=2 than g=1.
- Selection (highest ipTM over 18x5 poses): original arm and g=1 pick A01 sample 1, ipTM 0.530, DockQ 0.096 (decoy; interface broken).
  g=2 picks A00 sample 1, ipTM 0.548, DockQ 0.449 (acceptable). The g=2 top-5 ipTM poses are all A00 (DockQ 0.45-0.56); the best
  wrong-donor ipTM at g=2 is 0.435 (A04). A01's max ipTM fell 0.530 -> 0.360; A00's rose 0.436 -> 0.548.
- Share of all 90 poses with DockQ>=0.23: original 0.18, g=1 0.17, g=2 0.29.
- LIMITS: one interface, one seed per donor (430000), one g value, no repeat of the g=2 arm.
