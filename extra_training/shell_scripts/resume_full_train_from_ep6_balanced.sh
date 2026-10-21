#!/bin/bash
# RESUME from epoch 6 with the two throughput fixes: rank-balanced sharding and a prefetch that
# actually prefetches. Segment 4 of the full-set run: x3k2m4 + jhgfxy + 2w5to5 + this.
#
# THE METRICS ARE COMPARABLE ACROSS SEGMENTS. THE THROUGHPUT IS NOT. Nothing here changes the
# loss, the data or the head -- only which poses are grouped into a step and when the CPU crop
# runs -- but the wall clock and the memory peaks belong to this segment alone.
#
# FIX 1: EVERY RANK DOES IDENTICAL WORK. The step used to be ten CONSECUTIVE seeds of ONE
# structure per rank, because --shuffle structure grouped a structure's units so one built
# dataset item served all of them -- and --minimal-items never builds that item, so the grouping
# had been protecting nothing since 2026-10-17 while still pinning each rank's per-step cost to a
# single structure's size. Now a step is K whole structures and each rank takes ONE seed of each.
# Every seed of a structure crops to an IDENTICAL token count -- the interface crop is defined on
# the ground-truth interface, not the predicted pose, and 0 of 488 structures disagree across
# their ten seeds -- so the work is equal BY CONSTRUCTION, not on average.
#
#   order                            per-step spread   straggler wait
#   --shuffle structure (before)     111% median        38%
#   full unit shuffle                 51%               22%
#   K structures, 1 seed per rank      0.0%              0.0%
#
# Epoch 6 of 2w5to5 spent 22.6 / 19.8 / 25.3 min per rank in loss_stats_collective_plus_wait out
# of 65.4 -- the rank with the most tokens waited least and the one with the fewest waited most.
#
# FIX 2: THE PREFETCH NOW FIRES. crop_slice was 0.70 s/unit with --prefetch-units on and 0.71
# without; _pf_take used get_nowait(), so the worker stayed exactly one unit behind all epoch and
# every take missed. A miss falls back to the inline crop and ticks the same counter, which is
# why a 100% miss rate looked identical to no prefetch. It blocks now, and the per-rank hit rate
# is printed every epoch so this cannot hide again.
#
# K=6 AND THE BATCH IS 90, NOT 84. K comes from --accum as the nearest whole number of
# structures: 85 over 3 ranks at nsample 5 -> K=6 -> 90 poses/step. The user approved the change
# of effective batch. An epoch is 488 structures x floor(10/3) groups = 1,464 groups / 6 = 244
# steps EXACTLY, with no partial step. 1,464 units/rank, 21,960 poses/epoch against 24,300: the
# tenth seed of each structure sits out, and the per-epoch rng means a DIFFERENT one sits out
# each epoch rather than a fixed tenth never being trained on.
#
# THE GRADIENT SPANS MORE STRUCTURES THAN BEFORE, NOT FEWER: 6 per step against the 3 that
# one-structure-per-rank gave.
#
# THE RESUME GUARD PASSES BUT ITS PREMISE IS BENT, SAID OUT LOUD. The guard refuses a bundle whose
# accum differs because "changes to world/seed/accum/nsample move which poses land in which
# batch". --accum is still 85 so it passes, yet this change does move which poses land in which
# batch. That is benign -- AdamW's moments are running gradient averages and a reordering does
# not invalidate them, and 84 -> 90 at a fixed lr is a 7% change -- but it is a real deviation
# from what the guard thinks it is checking.
#
# MEASURED ON A 43-UNIT SMOKE TEST (runs 92pa3m and 6sru9u), resuming from this same bundle:
#   epoch 6, before     0.48 s/pose   crop_slice 16-19 min/epoch   waits 19.8-25.3 min
#   rank balance only   0.30 s/pose   all three ranks identical in every timer
#   + prefetch fixed    0.23 s/pose   crop_slice 0.0 min, 43/43 hit on every rank
# DO NOT PROJECT AN EPOCH FROM THIS. 43 units ran against warm fingerprint and page caches; a
# real epoch reads 1,464 distinct trunks. Read epoch 7's own line.
#
# EPOCHS 1-6, all recorded before this launch (budget 30; shipped head 2.9756 / 0.2909 / 0.5556):
#   ep1  train 2.7384  held-out 2.6953  DockQ 0.2999 / 0.5556
#   ep2  train 2.6623  held-out 2.6779  DockQ 0.2543 / 0.3333
#   ep3  train 2.6061  held-out 2.7359  DockQ 0.2864 / 0.4444
#   ep4  train 2.5784  held-out 2.6931  DockQ 0.2908 / 0.4444
#   ep5  train 2.5509  held-out 2.6576  DockQ 0.3219 / 0.5556   <- best on all four
#   ep6  train 2.5301  held-out 2.6717  DockQ 0.3134 / 0.5556
#
# --save-every 244 is one checkpoint per epoch at the new step count (was 224 at 290 steps/epoch).
# 500 EPOCHS, NO EARLY STOP: the user stops it by hand on a val plateau. Stopping on the TRAINING
# loss is the exact mistake the 30-structure anchor diagnosed.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "FULL SET segment 4, resumed from 2w5to5/resume_ep6.pt. Two throughput fixes and nothing else: K=6 structures per step with one seed per rank (per-step work equal by construction, straggler wait 38% -> 0%), and a prefetch take that blocks so the CPU crop actually overlaps the GPU (crop_slice 0.70 -> 0.00 s/unit, 100% hit). Effective batch 84 -> 90, 244 steps/epoch. Metrics comparable to epochs 1-6; throughput and memory are not." \
  -- --resume training/runs/2w5to5/resume_ep6.pt \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight.json \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 500 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 244 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
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
