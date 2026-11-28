#!/usr/bin/env bash
# ARM Q = ARM P PLUS THE FIVE NEW POSE INJECTION SITES, ALL FIVE ZERO-INITIALISED.
#
# SIX SITES, ONE PRETRAINED AND FIVE NEW:
#   site 1  the native pre-stack injection (linear_no_bias_d / _wo_onehot) -- PRETRAINED, already
#           trainable in Arm P, untouched here.
#   sites 2-4  --pose-perblock: one zero-init LinearNoBias(num_bins -> c_z) per injected block,
#           blocks 1..3 (block 0 keeps the pretrained pre-stack injection, so wrapping it would
#           duplicate rather than extend). 3 tensors, pose_perblock_proj.{0,1,2}.
#   site 5  --pose-readout --pose-readout-prenorm: BEFORE pae_ln, so the layernorm renormalises
#           the sum and the injection can turn the pair feature but not scale it.
#   site 6  --pose-readout-post-zero: a SECOND, independent zero-init set AFTER pae_ln, which can
#           scale the readout. 4 readout tensors in total with --frame-direction off.
#
# ZERO-INIT IS A FLAG DIFFERENCE, NOT A DEFAULT. POSE_PERBLOCK_COPY defaults to 1 in
# launch_pae_crop_arm.sh (line 74, colonless) and seeds every depth from block 0's PRETRAINED
# weights -- the opposite of what this arm wants -- so it is explicitly cleared below. The same
# goes for POSE_READOUT_COPY_DISTANCE, which the earlier six-site arm (launch_pose6_paeln.sh) set
# and this one must not. With both off, every one of the five new sites returns exactly 0.0, so the
# FUNCTION at step 0 is identical to Arm P's -- the sites add nothing until they are trained.
#
# THAT IS A STATEMENT ABOUT THE FUNCTION, NOT ABOUT A REALISED STEP. Nothing in the trainer seeds
# torch, so pair-dropout masks differ run to run in Arm P as well; two single steps will not match
# bit for bit and the attribution is statistical over the arm, not exact at step 0.
#
# THE THREE NEW BLOCK SITES ARE NARROWER THAN THE OTHER THREE. In zero-init untied mode each
# injected block gets ONE LinearNoBias(num_bins -> c_z) and so carries only the BINNED one-hot,
# while the native pre-stack site and both readout sites add a raw-distance term as well
# (pose_perblock.py:145-147 calls the one-hot-only form a different operator). The term-faithful
# alternative is --pose-perblock-copy, which is pretrained-init and therefore not this arm.
#
# EXPECT_TRAINABLE 110 = Arm P's 103 native head tensors + 3 perblock + 4 readout. The new tensors
# attach to the HEAD, not to pairformer_stack, so Arm P's own --train-only already matches them
# through its leading ^(?!pairformer_stack\.) -- the pattern is therefore IDENTICAL to Arm P's and
# the count is the guard that it reached the new sites. If the startup refuses with a different
# number, that number is the truth and this comment is the thing that was wrong.
#
# EVERYTHING ELSE IS ARM P: the hard 280 pool, pair dropout 0.25, lr 1e-3, no TM weight, no crop,
# balance 0 / split 0.23, 12 epochs with in-sample and val passes at 6 and 12. The evaluator is
# handed --pose-perblock, --pose-readout, --pose-readout-prenorm and --pose-readout-post-zero by
# the trainer itself (phase_k_train.py:8469-8513); without that the held-out numbers would be
# missing or measured on a different architecture, which has happened before (commit 7d9d4d10).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm Q: six pose sites -- site 1 pretrained, sites 2-6 ZERO-init"
# EPOCHS DEFAULTS TO 12, NOT 3: both eval specs are every=6, and (ep+1) % 6 never fires inside
# three epochs, so at the old default the arm produced no val and no in-sample number at all --
# only loss curves. Raised in review.
#
# THIS NOTE IS OUTSIDE THE env BLOCK ON PURPOSE. Put inside it, the '#' ends the backslash
# continuation, env runs with no command, and the rest of the block becomes a separate invocation
# that silently falls back to every base default -- which is exactly what happened on the first
# launch attempt.
echo "    hard 280 + pair dropout 0.25, uncropped, 12 epochs, in-sample + val at 6 and 12; dead ops stubbed as always"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-110}" \
    POSE_PERBLOCK=1 POSE_PERBLOCK_COPY="" FRAME_DIRECTION="" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE="" \
    POSE_READOUT_POST_ZERO=1 \
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
