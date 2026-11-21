#!/usr/bin/env bash
# ARM F: ARM E + THE NONLINEAR POSE-CONDITIONING MLP BRANCH, FROM SCRATCH, 20 EPOCHS.
#
# BASE CHOSEN BY THE USER, not by the decision rule. On val mean DockQ the two bases are
# 0.5940 (Arm B ep10) against 0.5937 (Arm E ep10) -- 0.0003, far inside one interface changing
# hands on 141, so the rule could not separate them. Arm E is the base on the user's instruction.
# What distinguishes it on the measurements: held-out rho 0.2232 against 0.2166, the best any arm
# has posted, and a good-pose training floor 0.061 lower (3.0397 vs 3.1010).
#
# WHAT IS ADDED over Arm E's 53 tensors: the 15 tensors of pose_mlp -- a one-hot layer (64x39), a
# continuous layer (64x8 + 64) and six zero-init output layers (128x64 + 128 each), one per
# injection site. 52,992 parameters, which is a 58% increase over Arm E's 91,136.
#
#   --train-only gains ONE pattern, `pose_mlp`, and --expect-trainable goes 53 -> 68.
#   Both are required: none of Arm E's patterns matches a pose_mlp_* name, so --pose-mlp without
#   the pattern would create 52,992 parameters, freeze them at their initialisation and leave
#   --expect-trainable at 53 with every guard silent. phase_k_train.py refuses that outright.
#
# FROM SCRATCH, NOT RESUMED, on the user's instruction. The branch adds an input the earlier
# weights never saw, so an epoch-10 Arm E checkpoint is not a warm start for it -- and a
# 20-epoch run is what makes the epoch-10/20 numbers comparable with every other arm.
#
# THE ONE THING NO GPU HAS MEASURED: the branch caches an [N,N,64] activation from the pre-stack
# site through backward, ~1.5 GiB in bf16 at the 3,400-token cap, and the two readout sites each
# materialise an [N,N,c_z] in bf16 and again in fp32. Arm E peaked well under the card, but this
# is the first arm to hold that cache, so epoch 1 is where an OOM would appear.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm F: Arm E + the nonlinear pose MLP branch"
echo "    68 tensors, 144,128 params; 20 epochs, lr 1e-3, val + slice every 10"

env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae,blocks.0.pair_transition.layernorm1,blocks.1.tri_mul_in.layer_norm_in,blocks.1.tri_mul_in.layer_norm_out,blocks.1.tri_mul_out.layer_norm_in,blocks.1.tri_mul_out.layer_norm_out,blocks.2.pair_transition.layernorm1,blocks.2.tri_mul_in.layer_norm_in,blocks.2.tri_mul_in.layer_norm_out,blocks.2.tri_mul_out.layer_norm_in,blocks.2.tri_mul_out.layer_norm_out,blocks.3.pair_transition.layernorm1,blocks.3.tri_att_start.layer_norm,blocks.3.tri_mul_in.layer_norm_in,blocks.3.tri_mul_in.layer_norm_out,blocks.3.tri_mul_out.layer_norm_in,blocks.3.tri_mul_out.layer_norm_out,pose_mlp}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 POSE_MLP=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-68}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
