# batch2: the other 6 interfaces (7yxu_AB, 7yxu_AC, 7zoz_AB, 7zoz_AC, 8be3_AB, 8ux6_DF), started 2026-09-29

Same analysis as trunk_pilot (9fww, 8fxv): (A) 50 free seeds (101-150) per complex with z_trunk/s_trunk saved in bf16 (`trunks/`), (B) crystal-free donor
selection on those new folds (shadow dataset `dataset/`, `build_shadow.py`, `stageB.sh`; templates in `select/templates`), (C) every donor refolded with its
own seed's trunk loaded and the donor pose as template: {MSA on = msa_c*, single MSA row = row1_c*} x {1,3,5,10} recycles (`resume/<arm>/`).
Driver: `run_batch2.sh` in tmux session `b2_driver`, log `driver.log`. It fully stops the FoldBench sweep (4 workers, commands saved in
../trunk_pilot/sweep_launch/) and relaunches it at the end (trap, runs even on failure). User approved a ~3 h continuous stop on 2026-09-29.
Stage C jobs are balanced over GPUs 0,1,2,5 by estimated cost (`make_stageC.py`, `stageC_plan.log`). `verify.py A|C` checks completeness; each stage is retried once.
