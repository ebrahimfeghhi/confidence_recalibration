#!/usr/bin/env bash
# ARM R = THE SIX POSE SITES ARE THE ONLY TRAINABLE THING, WITH ORIENTATION, ALL NEW WEIGHTS ZERO.
#
# WHY THIS AND NOT ARM Q. Arm Q trained the 103 native head tensors AND the sites together, so a
# flat result could not be attributed: the native tensors can absorb or cancel whatever the sites
# contribute. Freezing everything else asks the question directly -- can the pose pathway express
# something the head cannot already express? Arm Q is withdrawn; this replaces it.
#
# SIX SITES, EACH CARRYING DISTANCE *AND* ORIENTATION (15 trainable tensors):
#   site 1  pre-stack: linear_no_bias_d + _wo_onehot (PRETRAINED, the only non-zero start) and
#           frame_dir_proj (NEW, zero)                                              = 3 tensors
#   sites 2-4  blocks 1..3: onehot + raw distance + orientation, all zero           = 9 tensors
#           (the orientation copies take frame_dir_proj's weights, which pose_direction zero-inits
#            and VERIFIES exactly zero, so they are zero too -- pose_perblock.py:180-184,214-219)
#   site 5  before pae_ln: pose_readout_d, _d_wo_onehot, _frame_dir                 = 3 tensors
#   site 6  after pae_ln: pose_readout_post_d, _post_d_wo_onehot, _post_frame_dir   = 3 tensors
#
# THE READOUT ITSELF IS ALSO TRAINABLE: pae_ln (weight, bias) and linear_no_bias_pae (weight),
# 3 tensors on top of the 15, for 18. Sites 5 and 6 inject either side of that layernorm, and the
# head has to be able to USE what they inject -- a frozen readout can only reweight the 64 bins the
# way the shipped head already does, so the sites would be forced to speak in the shipped basis.
# The cost is that pae_ln's scale can partly stand in for the sites' magnitude, so a win is "the
# pose pathway plus a retrained readout", not the pathway alone. That is the intended trade.
#
# EVERYTHING ELSE IS FROZEN: all 4 pairformer blocks, every other head tensor, 230 of the 248.
#
# EVERY PATTERN WAS CHECKED AGAINST THE REAL NAMES in head_ep9_step513.pt, not guessed:
#   linear_no_bias_d    -> linear_no_bias_d.weight, linear_no_bias_d_wo_onehot.weight   (2, and
#                          NOT linear_no_bias_pae, which does not contain '_d')
#   pose_perblock_      -> the 3 distance projections and the 3 orientation copies       (6)
#   frame_dir_proj      -> frame_dir_proj.weight                                         (1)
#   pose_readout_       -> the 6 readout tensors across sites 5 and 6                    (6)
#   pae_ln              -> pae_ln.weight, pae_ln.bias                                    (2)
#   linear_no_bias_pae  -> linear_no_bias_pae.weight                                     (1)
# 18 in total, which is what EXPECT_TRAINABLE asserts. Cross-check: the prior six-site arm's 20
# was these 18 minus linear_no_bias_pae, plus the 3 extra tensors copy-mode adds.
#
# ZERO-INIT IS STILL A FLAG DIFFERENCE: POSE_PERBLOCK_COPY defaults to 1 in the base launcher and
# would seed each depth from block 0's PRETRAINED distance weights, so it stays explicitly empty,
# as does POSE_READOUT_COPY_DISTANCE. Every new tensor returns exactly 0.0 at step 0.
#
# THE REST IS UNCHANGED FROM ARM P: hard 280 pool, pair dropout 0.25, dead ops skipped, lr 1e-3,
# balance 0 / split 0.23, uncropped, 12 epochs with in-sample and val passes at 6 and 12.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm R: six pose sites ONLY trainable, distance+orientation, new weights ZERO"
# EPOCHS DEFAULTS TO 12, NOT 3: both eval specs are every=6, and (ep+1) % 6 never fires inside
# three epochs, so at the old default the arm produced no val and no in-sample number at all --
# only loss curves. Raised in review.
#
# THIS NOTE IS OUTSIDE THE env BLOCK ON PURPOSE. Put inside it, the '#' ends the backslash
# continuation, env runs with no command, and the rest of the block becomes a separate invocation
# that silently falls back to every base default -- which is exactly what happened on the first
# launch attempt.
echo "    hard 280 + pair dropout 0.25, uncropped, 12 epochs, in-sample + val at 6 and 12; dead ops stubbed as always"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-21}" \
    TRAIN_ONLY="${TRAIN_ONLY:-frame_dir_proj,pose_perblock_,linear_no_bias_d,pose_readout_,pae_ln,linear_no_bias_pae}" \
    FRAME_DIRECTION=1 \
    POSE_BLOCK_TERMS=onehot,dist,orient POSE_BLOCK_INIT=zero \
    POSE_READOUT_SITES=pre,post POSE_READOUT_TERMS=onehot,dist,orient POSE_READOUT_INIT=zero \
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
