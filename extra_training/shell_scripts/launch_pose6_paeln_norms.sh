#!/usr/bin/env bash
# ARM E: Arm B + the PAIR-TRACK LayerNorms of the four pairformer blocks + linear_no_bias_pae.
#
# WHAT IS ADDED over Arm B's 20 tensors:
#   * 32 LayerNorm tensors -- gain and bias of the pair-track norms in the 16 modules that are
#     measured LIVE. The 7 pair-track norms per block would be 56 tensors, but 10 of the 20
#     pairformer modules emit exactly zero (blocks.1.pair_transition has max|w| 4.9e-37), so
#     their norms cannot receive a gradient; the modules are named explicitly rather than
#     pattern-matched, so a dead one cannot creep back in.
#   * linear_no_bias_pae.weight, the 256->64 map into the error bins. PDE, pLDDT and resolved
#     stay frozen.
# 53 tensors, 91,136 params, 0.460% of the 19.8M head.
#
# THREE NORMS IN EACH BLOCK ARE DELIBERATELY NOT TRAINED: single_transition.layernorm1,
# attention_pair_bias.layernorm_a and attention_pair_bias.layernorm_z. The first two are on the
# single track. The third READS the pair representation -- it normalises z to build the attention
# bias -- but its output goes into the single track, and the pair track never reads back from
# single, so under a PAE-only loss its gradient is zero exactly like the other two. Training it
# would add 8 tensors that cannot move and inflate --expect-trainable into a weaker check.
#
# THE DEAD MODULES' NORMS ARE EXCLUDED, on the user's instruction. 24 of the 56 pair-track norm
# tensors sit in the ten measured-dead ops, where nothing reads the output and no gradient can
# reach the gain or bias. Including them would inflate --expect-trainable into a weaker check and
# allocate AdamW moments for tensors that cannot move.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm E: Arm B + pair-track LayerNorms + linear_no_bias_pae"
echo "    53 tensors, 91,136 params; 20 epochs, lr 1e-3, val + slice every 10"

env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae,blocks.0.pair_transition.layernorm1,blocks.1.tri_mul_in.layer_norm_in,blocks.1.tri_mul_in.layer_norm_out,blocks.1.tri_mul_out.layer_norm_in,blocks.1.tri_mul_out.layer_norm_out,blocks.2.pair_transition.layernorm1,blocks.2.tri_mul_in.layer_norm_in,blocks.2.tri_mul_in.layer_norm_out,blocks.2.tri_mul_out.layer_norm_in,blocks.2.tri_mul_out.layer_norm_out,blocks.3.pair_transition.layernorm1,blocks.3.tri_att_start.layer_norm,blocks.3.tri_mul_in.layer_norm_in,blocks.3.tri_mul_in.layer_norm_out,blocks.3.tri_mul_out.layer_norm_in,blocks.3.tri_mul_out.layer_norm_out}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-53}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
