# Gain sweep on 8 of the 10 broken FoldBench interfaces (launched 2026-09-28)

**Why:** repeat the template-gain test (PROTENIX_TMPL_GAIN, hook in Protenix pairformer.py) at g=2,3,4,5 on the interfaces where self-templating broke
a solved interface: 7yxu_AB, 7yxu_AC, 7zoz_AB, 7zoz_AC, 8be3_AB, 8fxv_AB, 8ux6_DF, 9fww_AB. 8r4u_AB and 8pn0_AE are excluded (fold times ~1250 s
and ~257 s; user chose to skip them).

**Method:** every donor of those 6 complexes (203 folds per gain, queue.tsv = the existing arm's rows), same templates (namespace reused read-only),
seed 430000, bf16, ncycle 10, nsample 5, --dataset foldbench, exactly as `../8fxv_tmpl_gain` and `../9fww_tmpl_gain`. One worker per gain:
g2 on GPU 0, g3 on GPU 1, g4 on GPU 2, g5 on GPU 5. Estimated ~2.4 h (scoring not included).

**GPUs:** 0,1,2,5 belong to the FoldBench free top-up; it is paused with SIGSTOP (`pause_sweep.sh`) for the run and resumed with SIGCONT
(`resume_sweep.sh`) automatically by `wait_then_resume.sh` when all four workers end (guarded, runs even if a worker fails).
Previous g=1 (default MSA sampling and dropout) reference arms exist only for 8fxv and 9fww; the original arm's results.csv is the baseline for all.
