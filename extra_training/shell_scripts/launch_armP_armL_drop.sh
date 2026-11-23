#!/usr/bin/env bash
# ARM P = ARM L's EXACT SETUP PLUS PAIR DROPOUT. Nothing else differs.
#
# WHY NOT THE 345 POOL. The first Arm P was Arm O (345 = hard 280 + 65 easy) plus dropout, which
# moved two things at once against Arm L and so could not attribute a flat result. Worse, the 345
# pool is now known to cost us where the headroom is: on val83's 21 hard interfaces at matched
# epochs, Arm L reached 9.5% frac acceptable and 19.0% at top-5 while Arm O reached 0% and 14.3%.
# Shipped scores 0% on those 21, so they are the only subset where either arm does something
# shipped cannot, and the easy top-ups damaged exactly that. Arm L's 280 is therefore the base.
#
# DROPOUT RATE 0.25 is the project's own earlier value (run 2k68zu), not a fresh guess.
#
# 12 EPOCHS, passes at 6 and 12: every arm measured peaks on val near epoch 6 and has decayed by
# 12 while in-sample keeps climbing, so 13+ has cost GPU and taught nothing four times running.
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
    MANIFEST="${MANIFEST:-training/slice280/structures.csv}" \
    SKIP="${SKIP-}" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/slice280/pool.csv}" \
    PREFLIGHT="${PREFLIGHT:-training/slice280/alignment_preflight.json}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-12}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}" \
    POOL_DIR="${POOL_DIR-}" \
    EXTRA_EVAL="${EXTRA_EVAL-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=6;name=valfull83,pool=training/valfull83/pool.csv,fold=validation/fold,seed_base=400000,budget=30,every=6}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
