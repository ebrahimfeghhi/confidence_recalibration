# RESUME from epoch 2, WITHOUT ACTIVATION CHECKPOINTING.
#
# blocks_per_ckpt=1 recomputed all four pairformer blocks' forwards during every backward. Over
# 4,860 units the largest crop was 801 tokens against a 3,400 cap -- p50 332, p99 678, nothing
# above 1,000 -- so there was no large-crop case for it to protect.
#
# MEASURED, not assumed, on two structures at the median crop (analysis/oneoff/09):
#   gpu_backward      1.3 min -> 1.0 min   (-23%)
#   s/pose            1.29   -> 1.00
#   peak allocated    6.02 GB -> 13.02 GB  (+116%)
# On epoch 2's 20.2 min of gpu_backward that is roughly 4.6 min/epoch. The cost is peak memory
# 2.2x: scaling from the 24.97 GB that checkpointing-ON peaked at on the p100 crop gives ~53 GB
# against ~96 GB usable per card. Expected to fit; the per-epoch peak lines say by how much.
#
# IF IT OOMs, the revert is --blocks-per-ckpt 1 and nothing else.
#
# ALSO NEW SINCE jhgfxy: the fingerprint cache now persists beside the fold, so this process does
# NOT re-pay the ~9.5 min of hashing that epoch 2 paid as a fresh process; empty_cache() runs at
# the boundary so the val hooks get the ~28 GB of cached-but-free back; and the timers are
# exclusive with the residual printed, after review found an ~8 min double-count.
#
# EPOCH 2'S METRICS ARE SAFE: train PAE 2.6623, held-out PAE 2.6779, and its DockQ pass completed
# before this run was launched. The series is x3k2m4+jhgfxy+this, one experiment in three parts.
#
# WHAT THIS RESUMES FROM. jhgfxy/resume_ep2.pt -- epoch 2's bundle, written at its boundary
# before the val hooks ran. Epochs 1 and 2 are not repeated.
#
# THE SERIES IS x3k2m4 + jhgfxy + THIS RUN, one experiment in THREE parts, and any curve must join
# all three the way 335psh+2u8tf4 had to be joined. x3k2m4 ran epoch 1 and died on a NameError in
# the held-out monitor; jhgfxy resumed and ran epoch 2; this runs epoch 3 onward. The segments do
# not share an implementation -- this one has checkpointing off, the persistent fingerprint cache,
# empty_cache at the boundary and exclusive timers -- so THROUGHPUT AND MEMORY ARE NOT COMPARABLE
# ACROSS SEGMENTS even though the metrics are. docs/full_train_results.md states which segment
# produced each epoch for exactly that reason.
#
# EPOCH 1 AND 2'S NUMBERS, all recorded before this launch:
#   ep1  train PAE 2.7384  held-out PAE 2.6953  DockQ 0.2999 / 0.5556
#   ep2  train PAE 2.6623  held-out PAE 2.6779  DockQ 0.2543 / 0.3333
# against a shipped head of held-out PAE 2.9756 and DockQ 0.2909 / 0.5556 at budget 30.
#
# THE FULL TRAINING SET, with every metric that matters measured every epoch.
#
# WHY THIS RUN EXISTS. Measured 2026-10-20 on the 30-structure overfit anchor (335psh+2u8tf4):
#   training PAE   2.3248 (ep1) -> 1.4324 (ep142)   -38%
#   val10 PAE      2.8290 (ep1) -> 4.2639 (ep142)   +51%, and 2.9756 for the SHIPPED head
#   training DockQ 0.0420 -> 0.1843,  frac_acc 0.000 -> 0.2987   (77 interfaces, budget 10)
#   val10 DockQ    0.2626 -> 0.2251,  frac_acc 0.500 -> 0.4444   (18 interfaces, budget 100)
# So the objective DOES reorder poses -- 4.4x on the data it trained on -- and does not transfer
# from 30 structures. Held-out PAE was best at EPOCH 1 and 43% worse than the untrained head by
# 142. The fix is data, which is what this run tests: 488 structures instead of 30.
#
# AND NOTHING IN THE RUN COULD SEE ANY OF THAT. The DockQ metric sat inside its noise band the
# whole time, because ipTM barely orders poses within an interface -- Spearman 0.20 for the shipped
# head, 0.10 for the trained one (analysis/oneoff/05_val10_rank_correlation.py). Held-out PAE was
# not computable at all: validation/fold had no labels until today. Both are now wired to the epoch
# boundary, which is the whole point of this script.
#
# THE FOUR METRICS, every epoch:
#   1. train PAE      -- trajectory.csv, pose-weighted per epoch, as always
#   2. val10 PAE      -- --val-loss-every 1, a --loss-only subprocess on the held-out fold
#   3. val10 mean DockQ \ --train-eval-pool at budget 30, sharded over all three cards
#   4. val10 frac_acc /
#
# BUDGET 30, AT THE USER'S DIRECTION, so the baselines are NOT the budget-100 ones. Derived from
# results.csv 2026-10-20 (phase_k_eval now computes this from the pool it scored instead of the
# hardcoded full-pool string it used to print):
#   budget  30:  baseline 0.2909 / 0.5556    oracle 0.5968 / 1.0000
#   budget 100:  baseline 0.2626 / 0.5000    oracle 0.6662 / 1.0000
# Never quote one against the other.
#
# HPARAMS ARE THE OVERFIT RUNS' HPARAMS, at the user's direction: lr 1e-2, warmup 20, accum 85,
# nsample 5, crop 10/5 at 3400, clip 25.0, alpha 1/1, masked PAE-only, shuffle by structure,
# seeds 500000-500009. One thing necessarily differs: --save-every is 224 rather than 18, because
# an epoch here is 224 optimiser steps rather than 18, so that is still one checkpoint per epoch.
#
# 500 EPOCHS, NO EARLY STOP. The user stops it when the val metrics plateau; stopping on the
# TRAINING loss is exactly the mistake the numbers above diagnose.
#
# COST. 488 structures x 10 seeds x 5 poses = 24,400 poses/epoch at 84/step = ~290 steps/epoch.
# The per-epoch wall clock is NOT measured for this configuration -- the ~37 min figure in
# launch_lr_probe.sh was for 377 structures and 224 steps, and it was an estimate there too. The
# monitors add ~6.5 min (val PAE, 1500 poses at the 0.21 s/pose measured today) plus ~2 min
# (DockQ at budget 30, scaled from 7.4 min at budget 100). Read the first epoch's real numbers
# before trusting any projection of 500 of them.
#
# DISK. A resume bundle is 226 MB and one is written every epoch: 500 epochs is ~110 GB, of which
# two thirds is AdamW's moments and only the 79 MB head is needed to score or measure. The user
# has accepted this and will prune later.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "FULL TRAINING SET (488 structures, 977 interfaces) with four per-epoch metrics: train PAE, val10 PAE, val10 mean DockQ, val10 frac-acceptable. Overfit-run hparams -- lr 1e-2, warmup 20, accum 85, masked PAE-only. 500 epochs, no early stop, stopped by hand on a val plateau. Tests whether the 30-structure result -- 4.4x better selection on train, held-out PAE 51% WORSE, best at epoch 1 -- was a data-quantity failure." \
  -- --resume training/runs/jhgfxy/resume_ep2.pt \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight.json \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 500 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 224 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/val10/structures.csv \
  --val-loss-pool training/val10/pool.csv \
  --val-loss-preflight training/val10/alignment_preflight.json \
  --val-loss-seeds 400000-400029 \
  --val-loss-prep validation/prep \
  --train-eval-pool training/val10/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 30 \
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
