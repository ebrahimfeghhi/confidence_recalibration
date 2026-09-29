# 8fxv template gain test (launched 2026-09-28)

**Why:** second test of the template gain (see ../9fww_tmpl_gain/README.md). 8fxv_AB is one of the 10 FoldBench interfaces where
self-templating broke a solved interface; 517 tokens, 11 donors, ~25 s/fold in the existing arm.

**Method:** identical to the 9fww test. `PROTENIX_TMPL_GAIN` (hook in Protenix pairformer.py, backup at ../pairformer.py.before_gain)
scales the template embedder's contribution to z; arms `g2` (2.0) and `g1` (1.0, control through the same code). Same 11 donors,
templates (namespace reused read-only) and seed 430000 as `experimental_data/protenix/foldbench/self_template_fixed_50_seed`.
bf16, ncycle 10, nsample 5, --dataset foldbench. GPUs 3 and 6 (g2), 7 (g1) -- the only free ones; 0,1,2,5 run the FoldBench
free top-up and 4 is held by another process.
Analysis: analysis/fold_scatter.py --method ost --poses <arm dir>, then per-donor ipTM and top-ipTM selection as for 9fww.

## g=3 and g=5 (2026-09-28)
Same setup as g=1/g=2 above (same 11 donors, templates, seed 430000, bf16, ncycle 10, nsample 5; PROTENIX_TMPL_GAIN=3.0 and 5.0),
arms `g3` (GPUs 0,1) and `g5` (GPUs 2,5). GPUs 0,1,2,5 belong to the FoldBench free top-up; on the user's instruction it was paused with
SIGSTOP (`pause_sweep.sh`) for the duration and resumed with SIGCONT (`resume_sweep.sh`).
g=8 arm added on request: `g8` (PROTENIX_TMPL_GAIN=8.0), started per GPU as each g3/g5 worker ends (`chain_g8_then_resume.sh`); the sweep stays
paused until all g8 workers end, then `resume_sweep.sh` (SIGCONT) runs automatically.
