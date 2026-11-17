#!/bin/bash
# PAE-ONLY ABLATION OF THE 5-SITE ARM: is the ranking loss doing anything?
#
# IDENTICAL TO jyz3g2 EXCEPT THE RANKING TERM. Same five injection sites with orientation, same
# fifteen trainable tensors and everything else frozen, same lr 1e-4, same pair-dropout 0.25, same
# unbalanced PAE loss. ALPHA_RANK is unset, so launch_pae_crop_arm.sh passes no --rank-loss,
# no --rank-form and no --rel-labels: one forward per pose instead of two, which also makes the
# epoch roughly half the wall clock.
#
# BALANCING OFF, DELIBERATELY. The standing rule is that --pae-balance-accept stays on, and this
# arm breaks it on purpose so the comparison against jyz3g2 (which has it off) isolates the
# ranking term and nothing else. PAE_SPLIT_ACCEPT stays at 0.23 because it is a TALLY, not the
# objective -- phase_k_train.py:6445 uses it only for _pae_split_tally -- so it buys the
# good-vs-bad pose loss panels for free without changing what is optimised.
#
# 20 EPOCHS, EVERYTHING EVERY 10. Val at 10 and 20, the train slice at 10 and 20.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') 5-site PAE-only, lr ${LR:-1e-4}, 20 epochs"
env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-15}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-4}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
