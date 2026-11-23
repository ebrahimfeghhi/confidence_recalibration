#!/usr/bin/env bash
# ARM G: ARM E + RANK-8 LoRA ON THE TRIANGLE MULTIPLICATIONS OF THE LIVE BLOCKS.
#
# WHERE THE ADAPTERS GO, and why these three layers:
#   linear_a_p, linear_b_p  -- the two input projections that make the a and b copies
#   linear_z                -- the output projection back to the pair width
# The three GATES (linear_a_g, linear_b_g, linear_g) are deliberately NOT adapted.
#
# BLOCKS 1, 2 AND 3 ONLY. Both triangle multiplications in block 0 are measured DEAD: every
# tensor in them sits in a constant band around 4.9e-37, the product of two such factors
# underflows fp32, and the op emits exactly zero. An adapter there could never receive a
# gradient. 6 live modules x 3 layers = 18 adapted, 73,728 adapter parameters at rank 8.
#
# lora_B IS ZERO AND lora_A IS RANDOM, so the head starts bit-identical to Arm E. Both zero
# would be a dead gradient (dL/dA flows through B and dL/dB through A); both random would break
# the identical start. This is the only pairing that gives neither.
#
# --lora-keep-trainable IS NOT OPTIONAL HERE. apply_lora freezes the entire head and the trainer
# then sets `trainable` to the adapter names ALONE, so --train-only does not survive LoRA. Arm E's
# 53 tensors would be frozen at their init -- and frame_dir_proj (256,3) and
# pose_perblock_linear_no_bias_d_wo_onehot (256,1) are below rank 8, so they would be skipped for
# rank-too-high and adapted by nothing either. The pose pathway would be present and inert, which
# is the same as not having it. KEEP therefore carries Arm E's full pattern list.
#
# THE TARGETS ARE GATED BEFORE LAUNCH by run_logging/test_lora_targets.py, which reads these exact
# env values rather than the launcher defaults and refuses if an adapter lands on a dead op, on a
# dead readout or on a pose tensor, or if a pose tensor is missing from KEEP.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm G: Arm E + rank-8 LoRA on tri_mul (blocks 1-3)"
echo "    18 modules adapted, 73,728 adapter params; Arm E's 53 tensors stay dense; 20 epochs, val + slice at 10 and 20"

env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae,blocks.0.pair_transition.layernorm1,blocks.1.tri_mul_in.layer_norm_in,blocks.1.tri_mul_in.layer_norm_out,blocks.1.tri_mul_out.layer_norm_in,blocks.1.tri_mul_out.layer_norm_out,blocks.2.pair_transition.layernorm1,blocks.2.tri_mul_in.layer_norm_in,blocks.2.tri_mul_in.layer_norm_out,blocks.2.tri_mul_out.layer_norm_in,blocks.2.tri_mul_out.layer_norm_out,blocks.3.pair_transition.layernorm1,blocks.3.tri_att_start.layer_norm,blocks.3.tri_mul_in.layer_norm_in,blocks.3.tri_mul_in.layer_norm_out,blocks.3.tri_mul_out.layer_norm_in,blocks.3.tri_mul_out.layer_norm_out}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-53}" \
    LORA_RANK="${LORA_RANK:-8}" LORA_EXPECT_ADAPTED="${LORA_EXPECT_ADAPTED:-18}" \
    LORA_GATE_REF="${LORA_GATE_REF:-training/runs/73rf4v/head_ep20_step2900.pt}" \
    LORA_TARGETS="${LORA_TARGETS:-blocks\\.[123]\\.tri_mul_(in|out)\\.linear_(a_p|b_p|z)$}" \
    LORA_KEEP="${LORA_KEEP:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae,blocks.0.pair_transition.layernorm1,blocks.1.tri_mul_in.layer_norm_in,blocks.1.tri_mul_in.layer_norm_out,blocks.1.tri_mul_out.layer_norm_in,blocks.1.tri_mul_out.layer_norm_out,blocks.2.pair_transition.layernorm1,blocks.2.tri_mul_in.layer_norm_in,blocks.2.tri_mul_in.layer_norm_out,blocks.2.tri_mul_out.layer_norm_in,blocks.2.tri_mul_out.layer_norm_out,blocks.3.pair_transition.layernorm1,blocks.3.tri_att_start.layer_norm,blocks.3.tri_mul_in.layer_norm_in,blocks.3.tri_mul_in.layer_norm_out,blocks.3.tri_mul_out.layer_norm_in,blocks.3.tri_mul_out.layer_norm_out}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
