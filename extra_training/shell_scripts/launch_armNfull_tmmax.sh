#!/usr/bin/env bash
# ARM N-FULL = ARM S PLUS THE TM-WEIGHTED PAE LOSS, max(s_true, s_pred).
#
# THE ONE DIFFERENCE FROM ARM S IS THE WEIGHT. Same 842 structures, all seeds, uncropped, lr 1e-3,
# dropout 0.25, split 0.23, balance 0, dead ops skipped, 103 trainable tensors. --pae-tm-weight
# with mode `max` weights each scored pair by max(s_true, s_pred): full weight when the pair really
# is a contact OR when the head believes it is and it is not -- the confident decoys -- and drops
# only pairs that are truly far AND already predicted far. That matters because 64.2% of an
# unacceptable pose's pairs sit in the last bin, so "everything is far apart" is most of the
# unweighted objective and is cheap to fit.
#
# VAL EVERY EPOCH FOR 1-10, which is the point: Arm S was only scored at 4,5,6,10,12 and its top-1
# peak sat at or before epoch 4, so the shape of the first few epochs was never measured. Each val
# pass is ~30 min on three cards, so the eval cost here is comparable to the training cost. The
# in-sample pass stays at every 5 for the same reason it always has -- it answers a different
# question and doubling it would double the bill.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm N-full: native head + TM-weighted PAE, mode max"
echo "    full training set (842 structures, all seeds -- Arm S's exact manifest), uncropped, dropout 0.25, per-pair TM weight"
echo "    max(s_true,s_pred); val83 EVERY epoch, in-sample every 5; dead ops stubbed"

# SKIP APPEARS ONCE IN THE env BLOCK. The launcher this was copied from already had
# SKIP="${SKIP-}", and the new list was first inserted ABOVE it, so the empty one won: the run
# trained all 853 structures, its manifest digest stopped matching
# alignment_preflight_sharded.json, and the trainer refused with a digest mismatch. The 12
# names are exactly Arm S's resolved manifest_skip, which is what makes the two arms
# comparable and the preflight marker valid.
#
# THE COMMENT LIVES HERE, NOT INSIDE THE env BLOCK. A # on a backslash-continued line does not
# comment -- it ENDS the command, which split `env` into an env-with-no-command plus a second
# invocation missing EXPECT_TRAINABLE, MANIFEST and the POSE_* vars, so the launcher would have
# fallen back to its hard-280 defaults. `bash -n` accepts that, because it is valid syntax.

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-103}" \
    POSE_PERBLOCK="" POSE_PERBLOCK_COPY="" FRAME_DIRECTION="" \
    POSE_READOUT="" POSE_READOUT_PRENORM="" POSE_READOUT_COPY_DISTANCE="" \
    POSE_READOUT_POST_ZERO="" \
    MANIFEST="${MANIFEST:-train_val_splits/cache_train_structures_extended.csv}" \
    SKIP="${SKIP-4zto,7m7i,8vba,9b9j,9bbn,9dzf,9dzg,9e7x,9mx1,9q5m,9q5v,9q7b}" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/combined/pool.csv}" \
    PREFLIGHT="${PREFLIGHT:-training/alignment_preflight_sharded.json}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-10}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}" \
    POOL_DIR="${POOL_DIR-}" \
    EXTRA_EVAL="${EXTRA_EVAL-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=valfull83,pool=training/valfull83/pool.csv,fold=validation/fold,seed_base=400000,budget=30,every=1}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    PAE_TM_WEIGHT="${PAE_TM_WEIGHT:-1}" \
    PAE_TM_WEIGHT_MODE="${PAE_TM_WEIGHT_MODE:-max}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
