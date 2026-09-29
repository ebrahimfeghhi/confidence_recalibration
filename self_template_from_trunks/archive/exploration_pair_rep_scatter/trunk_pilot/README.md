# Trunk pilot: 50 free seeds with z_trunk/s_trunk saved in bf16 (staged 2026-09-29, not yet launched)

**Plan (user's design):**
1. Fold 50 free seeds (101-150, the same seed values as the FoldBench free pool) for 9fww and 8fxv, saving each fold's z_trunk and
   s_trunk (bf16 bits, uint16 view, `dtype="bf16"` in the npz, plus RNG state). Poses and DockQ to crystal are kept too.
2. Select donor poses from these NEW folds with the existing crystal-free algorithm.
3. Re-fold each donor with its OWN seed's saved trunk loaded (PROTENIX_LOAD_TRUNK), the donor pose as the template, 10 recycles,
   no MSA module (PROTENIX_MSA_FROM_CYCLE=10). Measure DockQ scatter across the 5 samples and vs the donor.

**This step (1):** folder.py kind `free`, dataset foldbench (free units get an empty template file, as the pool did), bf16, ncycle 10, nsample 5,
default MSA sampling and MC dropout (stock behaviour). 100 folds. `TRUNK_DUMP_DIR` -> `trunks/<pdb>/free_<seed>.npz`.
Code added (all env-gated, default off; backups in `../backups/`): Protenix `PROTENIX_DUMP_TRUNK_BF16` (dump + per-dump lossless check in the log:
bf16 round-trip error, and what fp16 would clip/underflow), `PROTENIX_LOAD_TRUNK` reads bf16 files; folder.py `TRUNK_DUMP_DIR`.
**Check on the first fold:** log line `DUMP_TRUNK_BF16 z ... roundtrip max|diff| ... n_diff ...`. n_diff=0 means bf16 storage is exact.
Storage: ~177 MB per seed pair (9fww 40 MB + 8fxv 137 MB, z in 16 bit), ~8.9 GB for 50 seeds; gitignored.

## Step 3: resumed-trunk templated folds (staged 2026-09-29)
Donors: 22 (9fww 12, 8fxv 10) selected crystal-free from the new free folds (`select/`, log `select.log`; shadow dataset `dataset/`, `build_shadow.py`).
Templates: `select/templates` (22 built, preflight passed).
Each donor is refolded with the donor pose as its template and its OWN seed's saved trunk loaded (`TRUNK_LOAD_DIR` -> PROTENIX_LOAD_TRUNK, seed read
from the donor pose path), seed 430000, gain 1, bf16, nsample 5. 8 arms = {MSA on, MSA off} x {1,3,5,10} resumed recycles, all 22 donors each
(176 folds). MSA off = PROTENIX_MSA_FROM_CYCLE=<n recycles> (MSA module skipped on every resumed cycle; logged per cycle).
MSA on = stock (MSA module runs each resumed cycle, default row subsampling and MC dropout). RNG state is restored from the trunk file.
Arms: resume/{msa,nomsa}_c{1,3,5,10}. Two arms per GPU in sequence (GPUs 0,1,2,5); sweep paused with SIGSTOP and resumed by `wait_resume2.sh`.
Code added (default off, backups in ../backups/): folder.py TRUNK_LOAD_DIR; protenix.py MSA-skip log line.
Launch layout (user chose to pause the sweep and use GPUs 0,1,2,5 although the sweep held 75-91 GB each): 9fww on GPU 5 (all 8 arms in turn),
8fxv on GPUs 0,1,2 (3,3,2 arms). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Per-complex queues select/queue_{9fww,8fxv}.tsv share each arm's out/csv.

## Correction (2026-09-29, user): "MSA off" = a single MSA row, not a skipped module
The skip-the-module arms (nomsa_c*, PROTENIX_MSA_FROM_CYCLE) were stopped mid-run on the user's instruction (partial results left in resume/nomsa_*; not to be
used as the intended condition). Replaced by arms `row1_c{1,3,5,10}`: MSA module runs on ONE row (PROTENIX_MSA_NROWS=1, PROTENIX_MSA_STRATEGY=topk = the query
sequence only), logged per draw (PROTENIX_LOG_MSA=1). `kill_nomsa.sh` ends only workers whose path contains /resume/nomsa_.
Chains ended on the user's instruction (2026-09-29 ~01:47) to run ONLY the single-row 10-recycle arm first (row1_c10, all 22 donors: 8fxv on GPUs 0,1,2 sharing one
claim dir, 9fww on GPU 5). Complete: msa_c10 (110/110 rows). Partial/abandoned: msa_c3, msa_c5, msa_c1?, nomsa_* (skip-module, superseded), row1_c1/c3/c5 (never ran).

## Step 4: 1, 3 and 5 recycles, MSA on and single-row (2026-09-29)
User request: both setups (MSA on = msa_c*, single MSA row = row1_c*) at 1, 3, 5 recycles, all 22 donors, donor's own trunk. 6 arms, 132 folds.
8fxv: GPUs 0,1,2; 9fww: GPU 5. folder.py skips folds already scored in an arm's csv, so the earlier partial msa_c3/msa_c5 are resumed, not repeated.
Pause mode (user, 2026-09-29): "always pause the sweep during analyses" = FULL STOP of the 4 sweep workers with their exact commands saved in sweep_launch/*.cmd,
relaunched by relaunch_sweep.sh after (SIGSTOP freezing keeps the GPU memory and caused OOM: the sweep had grown to 75-92 GB per card).
