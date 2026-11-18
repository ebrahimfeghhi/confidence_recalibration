#!/usr/bin/env bash
# ARM B, RESUMED FROM ITS OWN EPOCH 10, WITH A COSINE DECAY TO 1e-4 BY EPOCH 20.
#
# WHY A RESUME AND NOT A FRESH ARM: epochs 1-10 are identical between this and w57hjo by
# construction -- same data order, same seed, same flat 1e-3 -- so re-running them would spend
# ~70 minutes of GPU reproducing a curve we already have, and any difference in the first ten
# epochs would be noise that muddies the one comparison this arm exists to make. Resuming from
# head_ep10 means the ONLY difference from w57hjo's epochs 11-20 is the learning rate, which is
# what makes the epoch-20 numbers a clean A/B of the schedule.
#
# WHAT IT TESTS: all four six-site arms bottom out on good-pose PAE around epoch 11-12 and then
# drift UP through epoch 20 while bad-pose PAE keeps falling, and held-out mean DockQ peaks at
# epoch 10 and gives some back by 20 (Arm B: 0.5940 -> 0.5908). Every arm so far ran at a
# constant 1e-3 -- lr_final defaults to 0, which sends lr_at_step down its "hold lr flat" branch.
# If the turn-up is the optimiser bouncing at too large a step, decaying to 1e-4 should let the
# good-pose curve keep falling and hold the epoch-10 DockQ; if it still turns up, the cause is
# capacity contention between the two pose groups and the schedule is not the lever.
#
# THE OVERRIDE NAMES EXACTLY THE THREE FIELDS THAT DIFFER. lr_final, lr_decay_start_epoch and
# lr_decay_end_epoch are guard fields; lr stays 1e-3 and lr_decay_shape is already "cosine", so
# naming either of those would be a stale override, which phase_k_train.py:1482 calls an error in
# its own right because it weakens the guard on the next resume that reuses the command line.
set -euo pipefail
cd "$(dirname "$0")/.."

B_RID=$(cat training/runs/pose6_paeln_rid)
CKPT="training/runs/${B_RID}/resume_ep10.pt"
[ -f "$CKPT" ] || { echo "STOP: no epoch-10 checkpoint at $CKPT"; exit 1; }
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm B decay: RESUME $B_RID @ epoch 10,"
echo "    epochs 11-20 with lr 1e-3 -> 1e-4 (cosine); val + slice at epoch 20"
# EVALS AT EPOCH 20 ONLY. Epochs 1-10 are identical to the original Arm B by construction -- same
# data order, same seed, same flat 1e-3 -- so an epoch-10 pass here would re-measure a number we
# already have (val mean 0.5940 / 73.0%), at ~40 min of GPU for the pair.

env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-20}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-20}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    LR_FINAL="${LR_FINAL:-1e-4}" \
    LR_DECAY_START="${LR_DECAY_START:-10}" \
    LR_DECAY_END="${LR_DECAY_END:-20}" \
    RESUME="$CKPT" \
    RESUME_OVERRIDE="lr_final,lr_decay_start_epoch,lr_decay_end_epoch" \
    RESUME_NOTE="cosine 1e-3->1e-4 over epochs 10-20; resume_ep10.pt bundle; schedule is the only change" \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=20}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
