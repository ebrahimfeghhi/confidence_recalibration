#!/usr/bin/env bash
# ARM K: ARM I WITH THE PER-PAIR CENTRED TM ERROR INSTEAD OF THE PAE CROSS-ENTROPY.
#
# IDENTICAL TO ARM I IN EVERY OTHER RESPECT -- same hard 280, same uncropped poses, same live
# tensors with the dead ops stubbed, same lr, same seed, same 3 epochs, same in-sample pass at
# epoch 3. The loss is the ONLY difference, which is what makes the epoch-3 comparison mean
# something.
#
# THE LOSS. Per pose and per scored cross-chain pair, on the same mask the PAE loss uses:
#     s_pred = sum_bins p_bin * f(bin_center)      f(e) = 1 / (1 + (e/d0)^2)
#     s_true = f(true aligned error)
#     e      = s_pred - s_true                     (signed)
#     c      = e - e_mean(pair)                    e_mean over the STRUCTURE's poses
#     loss   = mean(c^2)
# No PAE cross-entropy, no other term. The TM transform and d0 are IMPORTED from protenix
# sample_confidence, never reimplemented: s_pred is bit-identical to its token_token_ptm, and d0 is
# per CHAIN PAIR from the pair token count _set_pae_mask already stores.
#
# WHY. On the hard 280 every arm gains frac-acceptable at TOP-5 while top-1 barely moves (Arm G
# +8.6 points), and +2.5 points of that survives on the 81 CONTESTABLE val interfaces. So the head
# is learning a per-structure LEVEL it is not rewarded for, and is weak on the within-structure
# ORDERING that selection reads. Centring removes the level from the objective entirely: a
# level-shifted head scores EXACTLY 0 on this loss (asserted in phases/test_tm_centered.py) while
# an ordering-shifted one scores the variance of the shift.
#
# --rank-cross-seed IS REQUIRED AND IS NOT A RANKING ARM. rank_loss stays off; the flag is only the
# STEP GROUPING, which puts all of a structure's seeds in one step so e_mean can be taken over its
# poses. _xs_prepare then gathers the per-pair sums across ranks -- without that each pose was
# centred against a third of its structure's poses and the objective depended on the world size.
#
# UNBALANCED, AND THE TRAINER REFUSES OTHERWISE: --pae-balance-accept and --pae-row-frac both
# reweight the cross-entropy this arm does not train, so passing either would change nothing while
# the launch line claimed it did.
#
# THE STEP SIZE NEEDS NO ALPHA. The centred gradient is ~240x smaller than the cross-entropy's
# (3.8e-5 vs 9.2e-3 L2 on identical logits), but AdamW's update is m/sqrt(v) and so is invariant to
# a constant scale on the loss. The two things that could break that do not: weight_decay is 0.0,
# and eps at its 1e-8 default sits 3,800x below a 3.8e-5 gradient (~0.03% of the denominator).
# --clip-norm 25.0 never fires for either arm. Watch the reported gradient norm only if it falls
# below ~1e-6, where eps would start to matter.
# UNCROPPED, WHICH IS NOW THE POINT OF THE ARM. POOL_DIR="" passes no --crop-pool, so every pose
# is scored WHOLE.
#
# WHY THIS CHANGED. Measured over 60 random training structures, the 256-token crop keeps a MEDIAN
# 19.8% of a structure's scored cross-chain pairs (p10 6.0%, worst 3.1% on 10gh at 1,474 tokens)
# while keeping 44% of its TOKENS -- the pair count is quadratic and a cross-chain pair needs both
# partners inside the crop. So every arm run so far has been fit on about a fifth of each
# interface and then selected on all of it, and d0 (calculate_normalization of the pair's token
# count) differs between the two as well. A cropped version of this arm would answer "can a full
# head fit a fifth of the hard set", which is not the question.
#
# IT ALSO REMOVES THE d0 MISMATCH that the ranking loss had to patch around
# (n_pair_tokens=_uncrop_npt, added after pass A was found scoring with a cropped d0); the PAE
# loss has no equivalent fix, and uncropped it needs none.
#
# THE HARD 280 FITS: median 589 tokens, max 1,745, against --crop-size 3400. The cost is a
# full-size forward per pose instead of a 256-token one.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') Arm K: Arm I + CENTRED TM ERROR instead of PAE cross-entropy"
echo "    identical to Arm I except the loss; uncropped, unbalanced, 3 epochs, in-sample at 3"

env EXPECT_TRAINABLE="${EXPECT_TRAINABLE:-119}" \
    POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
    POSE_READOUT_POST_ZERO=1 \
    MANIFEST="${MANIFEST:-training/slice280/structures.csv}" \
    SKIP="${SKIP:-}" \
    SCORED_PAIRS="${SCORED_PAIRS:-training/slice280/pool.csv}" \
    PREFLIGHT="${PREFLIGHT:-training/slice280/alignment_preflight.json}" \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT="${PAE_BALANCE_ACCEPT:-0}" \
    CARDS="${CARDS:-4,5,7}" EPOCHS="${EPOCHS:-3}" EVAL_EVERY="${EVAL_EVERY:-0}" \
    LR="${LR:-1e-3}" PAIR_DROPOUT="${PAIR_DROPOUT:-0.25}" \
    EXTRA_EVAL="${EXTRA_EVAL:-name=slice280,pool=training/slice280/pool.csv,fold=training/fold,seed_base=500000,budget=15,every=3}" \
    EVAL_PAE_LOSS="${EVAL_PAE_LOSS:-1}" \
    TM_CENTERED="${TM_CENTERED:-1}" \
    POOL_DIR="${POOL_DIR-}" \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY="${DRY:-1}" \
    ./shell_scripts/launch_pae_crop_arm.sh
