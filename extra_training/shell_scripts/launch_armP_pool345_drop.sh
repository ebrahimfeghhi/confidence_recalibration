#!/usr/bin/env bash
# ARM P: ARM O PLUS PAIR DROPOUT. One change from Arm O: --pair-dropout 0.25 on the pairformer
# blocks. Pool, objective, head, lr, crop setting and manifest are all Arm O's.
#
# WHY. Every arm so far peaks on val around epoch 6 and decays by 12 while in-sample keeps
# climbing -- Arm L, Arm M, Arm N and (on the loss curve) Arm O all show it. That is overfitting a
# 180-structure pool with 5.3M trainable parameters, and dropout is the cheapest thing that
# attacks it directly rather than through the data mix.
#
# 0.25 IS THE PROJECT'S OWN PRECEDENT, from run 2k68zu, not a fresh guess. The flag sets p_drop on
# every pairformer block and the trainer refuses if the assignment does not take
# (phase_k_train.py:2345), so a silently-ignored dropout is not a failure mode here.
#
# 12 EPOCHS, NOT 18: the decay sets in by 12 on every arm measured, so epochs 13-18 have cost GPU
# and told us nothing new three times running. Passes at 6 and 12 bracket the peak.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm L: NATIVE shipped head -- no injected layers, no dropout"
# EPOCHS DEFAULTS TO 18, NOT 3: both eval specs are every=6, and (ep+1) % 6 never fires inside
# three epochs, so at the old default the arm produced no val and no in-sample number at all --
# only loss curves. Raised in review.
#
# THIS NOTE IS OUTSIDE THE env BLOCK ON PURPOSE. Put inside it, the '#' ends the backslash
# continuation, env runs with no command, and the rest of the block becomes a separate invocation
# that silently falls back to every base default -- which is exactly what happened on the first
# launch attempt.
echo "    topped-up 345 + pair dropout 0.25, uncropped, 3 epochs, NO in-sample pass; dead ops stubbed as always"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-103}" \
    POSE_PERBLOCK="" POSE_PERBLOCK_COPY="" FRAME_DIRECTION="" \
    POSE_READOUT="" POSE_READOUT_PRENORM="" POSE_READOUT_COPY_DISTANCE="" \
    POSE_READOUT_POST_ZERO="" \
    MANIFEST="${MANIFEST:-training/slice345/structures.csv}" \
    SKIP="${SKIP-}" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/slice345/pool.csv}" \
    PREFLIGHT="${PREFLIGHT:-training/slice345/alignment_preflight.json}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-12}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}" \
    POOL_DIR="${POOL_DIR-}" \
    EXTRA_EVAL="${EXTRA_EVAL-name=slice345,pool=training/slice345/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=6;name=valfull83,pool=training/valfull83/pool.csv,fold=validation/fold,seed_base=400000,budget=30,every=6}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
