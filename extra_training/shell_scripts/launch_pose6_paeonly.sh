#!/bin/bash
# SIX INJECTION SITES, FROM SCRATCH: the 5-site PAE-only arm with the post-LayerNorm site added.
#
# IDENTICAL TO 4yu3s4 EXCEPT THE SIXTH SITE. Same PAE-only loss (no ALPHA_RANK, so no
# --rank-loss), same lr 1e-3, same pair-dropout 0.25, same PAE balance OFF with split 0.23 for the
# tally, same 20 epochs with val and slice at 10 and 20. From scratch rather than forked, so the
# comparison is two arms from the shipped weights rather than a shared trajectory.
#
# THE SIXTH SITE IS ZERO-INIT AND ADDS NOTHING AT STEP 0. --pose-readout-post-zero injects AFTER
# pae_ln through its own zero matrices. The fifth site is PRE-norm, where the layernorm divides the
# injection's magnitude back out so it can only ROTATE the pair feature; the sixth is POST-norm and
# can SCALE the readout. Two independent sets of weights, so neither is the other at a scale.
#
# 18 TRAINABLE, NOT 15. pose_readout.py registers pose_readout_post_d,
# pose_readout_post_d_wo_onehot and pose_readout_post_frame_dir; --expect-trainable is a
# fail-closed gate, so a wrong count stops the run before the first step rather than training a
# different architecture than intended.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') 6-site PAE-only, lr ${LR:-1e-3}, ${EPOCHS:-20} epochs"
env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-18}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
