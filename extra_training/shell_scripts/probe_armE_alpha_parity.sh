#!/usr/bin/env bash
# ALPHA PARITY ON ARM E's ARCHITECTURE: how big is the mean-centred iTM regression's gradient
# against the PAE loss's, on the 53 tensors Arm E actually trains?
#
# NOT AN EXPERIMENT. One epoch, no eval, --grad-norm-every 1 so every optimiser step is measured.
# It writes grad_split.r<rank>.csv with g_pae, g_rank, ratio, cos and alpha_for_parity per step.
#
# WHY IT HAS TO BE RE-MEASURED HERE. Every alpha_rank this project has used came from a probe on
# a DIFFERENT frozen set, and the five on record span ratio 0.0041 to 0.4728 -- most of that
# spread is which fraction of the head was trainable, not anything about the losses. 75 was
# picked against the mid-range of those, on a 5-site arm with 15 trainable tensors. Arm E has 53
# across the pair-track LayerNorms and linear_no_bias_pae, which is a different denominator, so
# the old number does not transfer.
#
# ALPHA 75 IS PASSED DELIBERATELY even though the point is to question it: the gradient the
# optimiser actually takes is the one with alpha applied. alpha_for_parity divides it back out
# and is the alpha-independent answer -- that is the column to read.
#
# READ THE ACCUMULATED NUMBERS, NOT PER-POSE ONES: the rel coefficients sum to zero inside an
# interface group, so poses cancel against each other by up to a factor of ~7.6.
#
# CROPPED LABELS, BECAUSE THE ARM IS CROPPED. itm_labels_crop was built against this exact pool;
# the full-structure itm_labels_853 is refused under --crop-pool, because the head's prediction
# would average a SUBSET of the interface's pairs while the target stayed whole-structure.
#
# THE TRAINABLE SET IS COPIED FROM ARM E's LAUNCHER BY SCRIPT, not retyped -- the whole point is
# that the denominator matches Arm E exactly.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') alpha parity probe on Arm E's 53 tensors"
echo "    1 epoch, no eval, K=1, rel (mean-centred iTM), alpha 75 -- read alpha_for_parity"

env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae,blocks.0.pair_transition.layernorm1,blocks.1.tri_mul_in.layer_norm_in,blocks.1.tri_mul_in.layer_norm_out,blocks.1.tri_mul_out.layer_norm_in,blocks.1.tri_mul_out.layer_norm_out,blocks.2.pair_transition.layernorm1,blocks.2.tri_mul_in.layer_norm_in,blocks.2.tri_mul_in.layer_norm_out,blocks.2.tri_mul_out.layer_norm_in,blocks.2.tri_mul_out.layer_norm_out,blocks.3.pair_transition.layernorm1,blocks.3.tri_att_start.layer_norm,blocks.3.tri_mul_in.layer_norm_in,blocks.3.tri_mul_in.layer_norm_out,blocks.3.tri_mul_out.layer_norm_in,blocks.3.tri_mul_out.layer_norm_out}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-53}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS=1 \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    ALPHA_RANK="${ALPHA_RANK:-75}" RANK_FORM=rel RANK_S="${RANK_S:-3}" \
    REL_LABELS="${REL_LABELS:-training/fold/itm_labels_crop}" \
    GRAD_PROBE_EVERY="${GRAD_PROBE_EVERY:-1}" \
    EVAL_EVERY=1000 \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
