#!/bin/bash
# ARM C: ARM B PLUS RANK-4 LoRA ON THE FINAL PAIR TRANSITION.
#
# ONE CHANGE AGAINST ARM B: rank-4 adapters on pairformer_stack.blocks.3.pair_transition and
# nothing else. Block 3 is the FINAL pair transition and it is ALIVE (max|w| 1.72); block 1's is
# measured dead -- output exactly 0.0 -- so adapters there would never see a gradient.
#
# 26 TRAINABLE, NOT 20. apply_lora freezes the whole head and sets trainable to the ADAPTER names
# alone, so everything that must stay dense has to be named in --lora-keep-trainable. Block 3's
# pair transition carries three adaptable 2-D weights (linear_no_bias_a, _b, linear_no_bias), so
# 3 adapters = 6 tensors, plus the 20 dense from Arm B = 26.
#
# LORA_KEEP IS NOT THE DEFAULT. The launcher's default predates the readout sites and omits
# pose_readout_ and pae_ln; under LoRA that means those 8 tensors would be adapted or frozen
# instead of trained dense, silently turning this into a different arm.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') 6-site + pae_ln + rank-4 LoRA on blocks.3.pair_transition"
env TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    LORA_RANK=4 \
    LORA_TARGETS='blocks\.3\.pair_transition\.' \
    LORA_KEEP='frame_dir_proj,pose_perblock_,^linear_no_bias_d,pose_readout_,pae_ln' \
    LORA_GATE_REF="${LORA_GATE_REF:-}" \
    EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-26}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0 \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-20}" EVAL_EVERY="${EVAL_EVERY:-10}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT=0.25 \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=10}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
