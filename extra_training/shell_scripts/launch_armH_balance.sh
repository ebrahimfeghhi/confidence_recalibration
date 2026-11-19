#!/usr/bin/env bash
# ARM H: ARM E WITH THE PER-LIST PAE CLASS BALANCE SWITCHED ON. No LoRA, no adapters -- Arm E.
#
# THE ONE TRAINING CHANGE: PAE_BALANCE_ACCEPT 0 -> 0.23. Each interface's ACCEPTABLE poses now
# share a combined weight of 0.5 and its unacceptable poses share the other 0.5, so a list with
# three good poses and twelve bad ones stops being twelve-to-three in the gradient.
#
# WHY IT IS WORTH A RUN. Measured on val83 uncropped against the shipped head, every arm so far
# moves the two pose groups in OPPOSITE directions: good-pose PAE loss RISES (Arm E +0.273 after
# ONE epoch, +0.355 by ten; Arm G +0.505; the alpha-250 ranking arm +0.637) while bad-pose loss
# FALLS (-0.055, -0.102, -0.108). Selection needs good poses scored BELOW bad ones, so the arms
# are degrading the half that decides the metric -- which is why DockQ has never moved despite
# large training-loss gains. That most of the damage lands in the FIRST epoch is the signature of
# a weighting effect rather than slow memorisation: good poses are the minority in most lists, so
# an unweighted mean lets the bad ones set the scale. This arm tests exactly that explanation.
#
# THE BALANCE IS A TRAINING-SIDE CHANGE ONLY. It multiplies the OBJECTIVE
# (_obj = cum * _pb_w / n_poses); the pae_loss written to the trajectory and the good/bad tally
# stay unweighted, so both the training curve and the held-out number remain comparable with Arm E.
# run_logging/valpae_flags.py likewise refuses to copy --pae-balance-accept into a measurement.
#
# ALL FOUR METRICS, EVERY EPOCH. --eval-pae-loss makes each eval pass report PAE loss beside mean
# DockQ, frac acceptable and rho, computed from the 64-bin logits that pass already produced -- no
# extra forward. Verified bit-identical to the standalone --loss-only measurement on 45 poses
# (max |diff| 0.00000000). Val every epoch, the hard subset every 5.
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

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm H: Arm E + per-list PAE class balance at 0.23"
echo "    53 tensors, 91,136 params; 20 epochs, lr 1e-3; ALL FOUR metrics every epoch, hard subset every 5"

env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae,blocks.0.pair_transition.layernorm1,blocks.1.tri_mul_in.layer_norm_in,blocks.1.tri_mul_in.layer_norm_out,blocks.1.tri_mul_out.layer_norm_in,blocks.1.tri_mul_out.layer_norm_out,blocks.2.pair_transition.layernorm1,blocks.2.tri_mul_in.layer_norm_in,blocks.2.tri_mul_in.layer_norm_out,blocks.2.tri_mul_out.layer_norm_in,blocks.2.tri_mul_out.layer_norm_out,blocks.3.pair_transition.layernorm1,blocks.3.tri_att_start.layer_norm,blocks.3.tri_mul_in.layer_norm_in,blocks.3.tri_mul_in.layer_norm_out,blocks.3.tri_mul_out.layer_norm_in,blocks.3.tri_mul_out.layer_norm_out}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-53}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0.23}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-1}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=5}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
