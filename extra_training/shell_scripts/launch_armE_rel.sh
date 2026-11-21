#!/usr/bin/env bash
# ARM E + THE MEAN-CENTRED iTM RANKING TERM, 10 EPOCHS, AT A GIVEN ALPHA.
#
#   ALPHA_RANK=100 ./shell_scripts/launch_armE_rel.sh      # and 200, and 50
#
# WHY THESE THREE ALPHAS. The one-epoch parity probe t58nah measured the two terms' gradient
# norms at EVERY optimiser step on this exact trainable set: alpha_for_parity median 98.7,
# geometric mean 114.2, interquartile range 49.5 to 266.2. 100 is parity. 200 and 50 are the
# half-decade either side, and they are not decoration -- the interquartile range is 5x wide, so
# no single alpha balances the two terms at more than a minority of steps, and the sweep is the
# only way to find out whether that spread matters.
#
# ALPHA IS NOT DERIVED FROM LOSS VALUES. A ranking loss of 0.01-0.25 against a PAE loss of 2.3
# says nothing about gradient magnitudes on the weights; every number above came from norms.
#
# THE TRAINABLE SET IS ARM E's 53 TENSORS, copied from its launcher by script. The parity
# measurement is only valid for this denominator -- the five older alpha measurements on record
# span ratio 0.0041 to 0.4728, and most of that spread is which fraction of the head was frozen.
#
# CROPPED LABELS, BECAUSE THE ARM IS CROPPED. itm_labels_crop keys (crop_index, seed, sample,
# iface) and pins d0 at the UNCROPPED scored chain-pair count, which is what the predicted side
# does too -- all four iptm_pooled call sites pass n_pair_tokens. The full-structure labels are
# refused outright under --crop-pool.
set -euo pipefail
cd "$(dirname "$0")/.."

ALPHA_RANK="${ALPHA_RANK:?set ALPHA_RANK -- 100 is parity, measured on run t58nah}"
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm E + rel ranking at alpha $ALPHA_RANK"
echo "    53 tensors, 91,136 params; 10 epochs, lr 1e-3, val + slice at epoch 10"

env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae,blocks.0.pair_transition.layernorm1,blocks.1.tri_mul_in.layer_norm_in,blocks.1.tri_mul_in.layer_norm_out,blocks.1.tri_mul_out.layer_norm_in,blocks.1.tri_mul_out.layer_norm_out,blocks.2.pair_transition.layernorm1,blocks.2.tri_mul_in.layer_norm_in,blocks.2.tri_mul_in.layer_norm_out,blocks.2.tri_mul_out.layer_norm_in,blocks.2.tri_mul_out.layer_norm_out,blocks.3.pair_transition.layernorm1,blocks.3.tri_att_start.layer_norm,blocks.3.tri_mul_in.layer_norm_in,blocks.3.tri_mul_in.layer_norm_out,blocks.3.tri_mul_out.layer_norm_in,blocks.3.tri_mul_out.layer_norm_out}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-53}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-10}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    ALPHA_RANK="$ALPHA_RANK" RANK_FORM=rel RANK_S="${RANK_S:-3}" \
    REL_LABELS="${REL_LABELS:-training/fold/itm_labels_crop}" \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
