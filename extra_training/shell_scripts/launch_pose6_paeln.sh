#!/bin/bash
# ARM B: THE SIX-SITE ARM WITH pae_ln UNFROZEN. One change against Arm A.
#
# WHY THIS NORM AND NOT ANOTHER. The readout is linear_no_bias_pae(pae_ln(z_pair)). The fifth
# injection site adds BEFORE pae_ln, so the layernorm divides its magnitude back out and the site
# can only ROTATE the pair feature -- that is the stated reason the post-norm sixth site exists.
# Making pae_ln.weight and pae_ln.bias trainable lets the model SCALE what comes out of that
# normalisation, which is a cheaper and more direct version of the same capability: 512 parameters
# against the sixth site's 11,008.
#
# 20 TRAINABLE, NOT 18. The pattern gains `pae_ln`, which matches exactly pae_ln.weight and
# pae_ln.bias -- NOT pde_ln, plddt_ln, resolved_ln or input_strunk_ln, which are different names.
# --expect-trainable is fail-closed, so a pattern that caught more or fewer stops the run before
# the first optimiser step.
#
# EVERYTHING ELSE IS ARM A, so a difference is pae_ln and nothing else.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') 6-site + pae_ln, lr ${LR:-1e-3}, ${EPOCHS:-20} epochs"
env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-20}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
