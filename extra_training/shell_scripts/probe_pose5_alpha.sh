#!/bin/bash
# MEASURE alpha PARITY FOR THE 5-SITE POSE-INJECTION rel ARM, on that exact architecture.
#
# WHY ON THIS ARCHITECTURE AND NOT REUSED. The five alpha measurements on record span ratio
# 0.0041 to 0.4728, and most of that spread is that each froze a different fraction of the head
# rather than anything about the losses. This arm freezes MORE than any of them -- only the pose
# injection tensors train -- so none of those numbers transfers. 75 came from the mid-range of
# that spread and is arbitrary here.
#
# ONE EPOCH, K=1, SO EVERY OPTIMISER STEP IS MEASURED. queue_rank_gradprobe_1ep.sh writes
# grad_split.r<rank>.csv with an alpha_for_parity column that divides the applied alpha back out,
# so the answer does not depend on the alpha this probe runs at. Read it with
#   python -I run_logging/grad_split_report.py <rid>
#
# THE ARCHITECTURE ENV IS THE ARM'S, not retyped: the same TRAIN_ONLY and POSE_READOUT* that
# launch_pose5_rel_lr1e4.sh exports, so the gradient split is measured on the parameters that
# will actually receive it.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
export TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_}"
export POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1
export EXPECT_TRAINABLE="${EXPECT_TRAINABLE:?pass the count; a wrong one is refused with the count found}"
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') alpha-parity probe: 5 sites, rel, lr 1e-4,"
echo "    trainable $TRAIN_ONLY (expect $EXPECT_TRAINABLE)"
LR=1e-4 PAIR_DROPOUT=0.25 GRAD_PROBE_EVERY=1 ALPHA_RANK="${ALPHA_RANK:-75}" \
    DRY="${DRY:-1}" CARDS="${CARDS:-4,5,7}" \
    ./shell_scripts/queue_rank_gradprobe_1ep.sh
