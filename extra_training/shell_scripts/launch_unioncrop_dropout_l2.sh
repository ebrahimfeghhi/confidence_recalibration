#!/bin/bash
# UNION-CROP + PAIR DROPOUT + L2-SP. A fresh 60-epoch arm from the pretrained head.
#
# THE TWO MOTIVATIONS ARE SEPARATE AND SHOULD STAY SEPARATE IN ANY WRITE-UP:
#
#   THE CROP IS FOR SPEED.  Not regularization. The uncropped arm runs 143-155 min/epoch, which
#   makes a 60-epoch arm a week and makes testing hypotheses the bottleneck. The cropped arm
#   (o2qczq, 53.3% of tokens) ran 28-30 min. This crop keeps ~60% of tokens at margin 5, so it
#   should land well inside the 40-50 min/epoch this arm needs to be worth running. Pair compute
#   is quadratic in tokens, so 60% of tokens is ~36% of the pair stack.
#
#   DROPOUT AND L2-SP ARE FOR OVERFITTING.  In-sample selection at epoch 27 is +0.0321 mean DockQ
#   and +0.0542 frac acceptable over shipped and still climbing (epoch 19 was +0.0150/+0.0216),
#   while held-out selection has sat at or just below shipped for 27 epochs and held-out
#   within-interface Spearman has not trended since epoch 3. That gap is what these two attack.
#
# WHY THIS CROP AND NOT --crop-margin. o2qczq's crop was ONE crop built from the ground-truth
# interface, identical every epoch (0 of 488 structures disagree across their ten seeds), and its
# gain needed the same crop at inference: read uncropped, that checkpoint scores 0.5387 in-sample
# against shipped's 0.5474. Section 13 then showed a label-free crop recovers none of it. This
# mask is the UNION over all 50 of a structure's poses of each pose's own predicted interface,
# with the ground-truth contacts forced in.
#
# WHY THE UNION AND NOT A RANDOM ANCHOR. A single random anchor would be a real augmentation --
# the crop would vary per epoch and could not be memorised -- but it breaks the objective: 81% of
# these structures have 2+ scored interfaces (247 have two, 32 three, 18 four, 1 six), and a crop
# anchored on one can cut another while that other's DockQ label, a GLOBAL number computed on the
# full structure, stays in the loss. The head would be asked to score an interface absent from its
# input, against an unreachable target, producing consistent and non-averaging gradient noise on
# most of the pool, silently. The union has no such hole. It is also deterministic, so it is
# computed once and costs nothing per epoch -- and it buys no augmentation, which is why dropout
# is here doing that job instead.
#
# WHY --force-gt. The pose union alone gives interface coverage 1.0000 at the MEDIAN but p10
# 0.9500 on TRAIN at margin 2.5 -- the bottom decile loses 5%+ of the very contacts the loss
# scores, which is the hole above in milder form. Forcing the bare ground-truth contact tokens in
# makes coverage 1.0 BY CONSTRUCTION and costs almost nothing: median 0 tokens added, worst 6
# over the structures measured. It makes the mask label-dependent, which is fine for a TRAINING
# crop with uncropped inference and is NOT a deployment claim.
#
# WHY MARGIN 5 AND NOT 2.5. Measured on TRAIN: 2.5 -> 41.4% of tokens, recall of the old crop's
# context 0.5875, coverage p10 0.9500; 5.0 -> 59.7%, recall 0.8069, coverage p10 1.0000. At 5 the
# coverage guarantee holds even WITHOUT forcing, so forcing becomes a belt-and-braces check rather
# than load-bearing, and 81% of the previous arm's context is retained so the input distribution
# is not wildly new. Below ~5 the margin is a NO-OP on the union anyway: margins 0, 1 and 2.5 all
# gave an identical 44.6% on the sample measured, because pose diversity has already swept up that
# shell -- so 2.5 buys no context over 0, it just loses coverage.
#
# ONE SHARED MASK PER STRUCTURE, which is load-bearing twice over. The relative loss grades the
# GAPS between a structure's poses, so poses cropped differently would have their score
# differences measure the crop. And build_epoch_cross_seed relies on every seed of a structure
# cropping to an IDENTICAL token count to make per-rank work equal by construction -- unequal
# per-rank unit counts do not degrade, they deadlock the first all-reduce.
#
# PAIR DROPOUT 0.25. ConfidenceHead builds its pairformer stack with dropout=0.0, overriding
# Protenix's own 0.25 default for pair stacks, so this project has trained 19.8M parameters with
# NO dropout and no masking of any kind. 0.25 is Protenix's number for this module, not a tuned
# one. It is ROW-WISE on z -- whole rows of the pair representation, structured noise over token
# pairs rather than per-element -- which is the AF3 design and is plausibly well matched to "do
# not memorise this complex's pair pattern". run_logging/test_pair_dropout.py asserts it is live
# (two train-mode forwards differ, eval-mode forwards match) rather than merely configured.
#
# L2-SP 1e-4, DELIBERATELY WEAK. tv6hue is the same 19.8M copy-pose config plus L2-SP 1e-3 and it
# finished at -0.0084 +- 0.0054 held-out selection; jer222 ran 1e-2. So this is a tested lever
# that has not moved this, and 1e-4 is an order below the one that did not work -- it is here to
# anchor the pretrained weights lightly, not to carry the arm. The pose-injection tensors are
# EXCLUDED from the anchor: frame_dir_proj is zero-init and the nine pose_perblock_* tensors are
# new, so anchoring them would price movement away from a value that is not a pretrained prior.
#
# LR IS A STAIRCASE, NOT A COSINE: 1e-2 through epoch 19, 1e-3 from 20, 1e-4 from 40, held to 60.
# The decay shapes interpolate continuously and cannot express a hold-then-drop; --lr-steps does.
# run_logging/test_lr_schedule.py pins it.
#
# --lean-readouts IS ON. pde, pLDDT and resolved are not COMPUTED, only stubbed by --pae-only
# today, and test_lean_readouts.py proves the pae logits are bit-identical with it installed. It
# buys 2.7 GB at the 855-token boundary (73.51 GB against 76.18) for no change in mathematics, so
# there is no reason to leave it off. Recommended in review.
#
# CHECKPOINTING BY SIZE. --blocks-per-ckpt 1 with --ckpt-token-threshold 855 means any unit at or
# under 855 tokens trains UNCHECKPOINTED. The threshold compares the POST-CROP token count, and at
# ~60% of a 619-token median that is nearly every unit, so this is effectively "no recompute"
# while still protecting the few large ones. 855 was measured 2026-11-02: uncheckpointed peak
# 73.51 GB at 855 tokens, so the epoch peak falls below the 80.22 GB an epoch is known to survive.
#
# VALIDATION IS UNCROPPED, which is how deployment works and is already the default -- the
# evaluator's --crop-margin is 0 and section 13 confirms t4jtjr's valfull pass was the uncropped
# reference. A matching eval mask exists at training/valfull/union_crop_m5_gt.npz if we ever want
# the cropped reading, but it is --force-gt too, so it is a diagnostic and not a deployment number.
#
# THE FAILURE TO WATCH, and it is measurable per epoch rather than at epoch 30: in-sample on
# CROPPED inputs climbing while in-sample on UNCROPPED inputs stalls or falls. That is exactly
# what o2qczq did -- 0.6305 cropped against 0.5387 uncropped -- and it is the signature of
# learning to rank crops rather than to rank poses. run_logging/insample_eval.py refuses a cropped
# bundle, so the uncropped pass is the one to run periodically.
set -u
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
cd "$(dirname "$0")/.."

MASK=training/full_train/union_crop_m5_gt_ost.npz
BUILD_CMD="$PY_BIN -I run_logging/build_union_crop_masks.py \
  --pool training/full_train/pool_winnable.csv --roles training/fold/chain_roles.csv \
  --trunks training/fold/trunks --labels training/fold/labels --fold training/fold \
  --out $MASK --contact 5.0 --margin 5.0 --force-gt --ost-pairs \
  --seed-base 500000 --budget 10 --nsample 5"
if [ ! -f "$MASK" ]; then
  echo "REFUSING: $MASK is missing. Build it with:"; echo "  $BUILD_CMD"; exit 1
fi
# THE FILE EXISTING IS NOT THE GUARANTEE. phase_k_train's content digest proves the mask has not
# CHANGED across a resume; it says nothing about whether this mask was correct to begin with. A
# static-only mask (no --ost-pairs) hashes just as stably as the right one while covering a chain
# pair the trainer may not be supervising. So the mask's own metadata is checked here. Raised in
# review of e0ee07b5.
$PY_BIN -I run_logging/check_crop_mask.py "$MASK" \
  --require-force-gt --require-ost-pairs --structures 368 --poses 50 \
  --margin 5.0 --contact 5.0 \
  || { echo "REFUSING: $MASK does not establish the required configuration. Rebuild with:"; \
       echo "  $BUILD_CMD"; exit 1; }

for g in test_lr_schedule test_guard_fields test_pair_dropout test_crop_tokens \
         test_two_pass_dropout test_ckpt_threshold test_lean_readouts \
         test_lean_readouts_arch; do
  $PY_BIN -I run_logging/$g.py >/dev/null 2>&1 || { echo "REFUSING: $g failed"; exit 1; }
done
echo "eight gates and the mask provenance check pass; launching"

LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 \
  --note "UNION-CROP + PAIR DROPOUT 0.25 + L2-SP 1e-4, fresh 60-epoch arm from the pretrained head. THE CROP IS FOR SPEED (uncropped runs 143-155 min/epoch; ~60% of tokens at margin 5 should land inside 40-50, and pair compute is quadratic so 60% of tokens is ~36% of the pair stack). DROPOUT AND L2-SP ARE FOR OVERFITTING: in-sample selection at epoch 27 is +0.0321 mean DockQ / +0.0542 frac acceptable over shipped and still climbing, while held-out selection has sat at or below shipped for 27 epochs and held-out within-interface Spearman has not trended since epoch 3. THE CROP is the union over all 50 of a structure's poses of each pose's own PREDICTED interface at a 5 A margin, with the bare ground-truth contact tokens forced in so interface coverage is 1.0 by construction rather than 1.0 at the median (the pose union alone gives p10 0.9500, and because the DockQ label is a GLOBAL number computed on the full structure it cannot be cropped to match a partially-kept interface -- the head would be scoring an interface absent from its input). A single random anchor would regularize better but breaks exactly that: 81% of these structures have 2+ scored interfaces. ONE SHARED MASK PER STRUCTURE because the relative loss grades the gaps between poses, and because build_epoch_cross_seed needs identical token counts across a structure's seeds or the first all-reduce deadlocks. PAIR DROPOUT 0.25 is Protenix's own default for pair stacks, which ConfidenceHead overrides to 0.0 -- so this project has trained 19.8M parameters with no dropout and no masking at all; it is row-wise on z. L2-SP 1e-4 is deliberately an order below tv6hue's 1e-3, which finished at -0.0084 +- 0.0054, with the zero-init frame_dir_proj and the nine new pose_perblock_* tensors excluded from the anchor. LR IS A STAIRCASE: 1e-2 through 19, 1e-3 from 20, 1e-4 from 40. VALIDATION IS UNCROPPED, as deployment is. THE FAILURE TO WATCH is in-sample cropped climbing while in-sample UNCROPPED stalls, which is what o2qczq did (0.6305 cropped vs 0.5387 uncropped) and is the signature of learning to rank crops rather than poses." \
  -- --rank-loss --rank-form rel --rank-cross-seed --alpha-rank 75 \
  --rank-pool mean --rel-labels training/fold/itm_labels \
  --rank-structs-per-step 3 \
  --pose-perblock --pose-perblock-copy --frame-direction \
  --fold training/fold \
  --crop-tokens "$MASK" \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
  --manifest-selection training/full_train/structures_winnable.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 60 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --lr-steps '20:1e-3,40:1e-4' \
  --pair-dropout 0.25 \
  --l2-sp 1e-4 --l2-sp-exclude 'frame_dir_proj,pose_perblock_' \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only --lean-readouts \
  --shuffle structure --seeds 500000-500009 \
  --eval-pool training/valfull/pool.csv \
  --eval-pool-fold validation/fold --eval-pool-seed-base 400000 \
  --eval-pool-budget 30 --eval-pool-every 1 \
  --prefetch-units 2 --prefetch-workers 4 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 855 \
  --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "launch_unioncrop_dropout_l2 returned $rc"
exit $rc
