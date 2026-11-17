#!/bin/bash
# FIVE POSE INJECTION SITES WITH ORIENTATION, ALL COPY-INITIALISED, AND NOTHING ELSE TRAINABLE.
# Mean-centred regression on the true iTM (--rank-form rel) at lr 1e-4, pair-dropout 0.25.
#
# THE FIVE SITES. launch_pae_crop_arm.sh hardcodes --pose-perblock --pose-perblock-copy
# --frame-direction, which is the four pairformer blocks each with its own copy of block 0's
# injection plus the orientation arm. --pose-readout is documented as "THE FIFTH INJECTION SITE";
# --pose-readout-prenorm moves it BEFORE pae_ln, and --pose-readout-copy-distance starts its two
# distance arms from block 0's pretrained weights rather than zero, which is the "all copy" part.
#
# --pose-readout-post-zero IS DELIBERATELY OFF. It adds a SECOND readout injection after the
# layer norm, which would make six sites, not five with the fifth before the norm.
#
# ONLY THOSE FIVE TRAIN. TRAIN_ONLY is the project's own canonical list of the pose tensors --
# the same names launch_pae_crop_arm.sh's LORA_KEEP default uses to hold them out of the adapters
# -- plus the readout's. Everything else, the pairformer ops the default TRAIN_ONLY re-includes
# included, is frozen. EXPECT_TRAINABLE is a fail-closed assertion rather than a setting: if the
# count is wrong the trainer refuses and prints the count it found.
#
# ALPHA_RANK IS NOT GUESSED. It comes from a --grad-probe measurement on THIS architecture, which
# is what shell_scripts/probe_pose5_alpha.sh runs; 75 was picked against the mid-range of five
# measurements that had each frozen a different fraction of the head.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
POSE_TENSORS='frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_'
export TRAIN_ONLY="${TRAIN_ONLY:-$POSE_TENSORS}"
export POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1
export EXPECT_TRAINABLE="${EXPECT_TRAINABLE:?set it from the DRY run reported count}"
export ALPHA_RANK="${ALPHA_RANK:?set it from the grad-probe parity measurement}"
# CROPPED LABELS BECAUSE THE ARM IS CROPPED: rel trains the head to reproduce a pose's true iTM
# and the trainer refuses whole-structure labels under --crop-pool, since the head's prediction
# would average a subset of the interface's pairs while the target covered all of them.
export REL_LABELS="${REL_LABELS:-training/fold/itm_labels_crop}"
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') 5-site pose injection, rel, lr ${LR:-1e-4},"
echo "    trainable: $TRAIN_ONLY  (expect $EXPECT_TRAINABLE)  alpha_rank $ALPHA_RANK"
env CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-60}" LR="${LR:-1e-4}" PAIR_DROPOUT=0.25 \
        EVAL_EVERY="${EVAL_EVERY:-2}" \
        PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
