#!/bin/bash
# BIASED TOKEN DROPOUT + PAIR DROPOUT + L2-SP. A fresh 60-epoch arm from the pretrained head.
#
# ONE CHANGE FROM ffj8m4, which is the arm this must beat: the union mask stops being a FIXED crop
# and becomes a PRIOR on a fresh draw each epoch. A token inside the union survives with p=0.70,
# one outside with p=0.20, drawn once per structure per epoch.
#
# WHY. ffj8m4 trains on one fixed 59.8% of tokens, which is memorisable, and the head NEVER sees
# the periphery -- which is exactly why o2qczq scored 0.6305 on cropped inputs and 0.5387 on
# uncropped, below shipped's 0.5474. A varying subset cannot be memorised, and because periphery
# tokens appear 20% of the time, uncropped inference stops being a regime the head has never seen.
# That is an attack on the failure mode rather than a hedge against it.
#
# AND IT IS FASTER. A dropped token has its ROW AND COLUMN removed from z, not masked to zero, so
# the tensors genuinely shrink: 0.70/0.20 keeps 0.499 of tokens against ffj8m4's 0.598, which is
# 0.249 of the original pair work against 0.357 -- about 0.69x ffj8m4's pair cost.
#
#   keep union  keep other   tokens   pair work   of a scored chain
#         1.00        0.00    0.598       0.357               0.620   <- ffj8m4, the fixed crop
#         0.70        0.20    0.499       0.249               0.510   <- this arm
#         0.50        0.50    0.500       0.250               0.500   <- uniform dropout
#
# This family contains both endpoints, so the setting names the point rather than the mechanism.
#
# THE TWO INVARIANTS THE DRAW MUST HOLD, both easy to break silently:
#
#   ONE DRAW PER STRUCTURE PER EPOCH, shared by all 50 of its poses. The relative loss grades the
#   GAPS between those poses, so poses on different token sets would have their score differences
#   measure the draw. Per-epoch is exactly per-step, because build_epoch_cross_seed puts every
#   seed of a structure in ONE step -- and per-epoch is also what the prefetch worker can compute,
#   since it crops units before the step is known.
#
#   EVERY RANK DRAWS THE SAME MASK. Under --rank-cross-seed the ranks hold different SEEDS of the
#   same structure inside one step, so they must agree exactly. The seed is sha256 of
#   (--seed, epoch, pdb); the first version used hash(), which Python SALTS PER PROCESS and would
#   have given every rank a different draw. run_logging/test_token_dropout.py pins both, and
#   demonstrates hash() really does vary across processes so the pin is not decorative.
#
# THE COST BEING ACCEPTED, stated rather than waved through. ipTM pools over the scored pair's
# token pairs and normalises by a d0 set from their count, so a varying subset varies the
# ACHIEVABLE score while the label stays at its full-structure value -- and d0 enters as
# 1/(1+(d/d0)^2), so it does not cancel under the per-interface mean-centring. ffj8m4 already
# carries that mismatch at a fixed 62% of each scored chain; this makes it vary between ~45% and
# ~62% instead. The bet is that the regularisation is worth more than the added target noise,
# which is the same bet the user's own masking result on neural data won.
#
# A FLOOR IS ASSERTED, not hoped for: the draw refuses if fewer than two chains or fewer than 16
# tokens survive, because ipTM is undefined for a pair with one chain missing.
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
# L2-SP 1e-5, DELIBERATELY VERY WEAK. tv6hue is the same 19.8M copy-pose config plus L2-SP 1e-3
# and it finished at -0.0084 +- 0.0054 held-out selection; jer222 ran 1e-2. So this is a tested
# lever that has not moved this, and 1e-5 is TWO orders below the one that did not work -- it is
# here to anchor the pretrained weights lightly, not to carry the arm.
#
# AND THE COEFFICIENT ALONE DOES NOT ESTABLISH "WEAK", which is why it was lowered from 1e-4. The
# penalty is lambda * ||W-W0||^2, so it depends on how far the weights actually travel. At the
# 67,052 the unconstrained full head reached by epoch 30, lambda 1e-4 would be a penalty of 6.7
# against a loss near 2.9 -- 231% of the objective. At 1e-5 the same displacement costs 0.67, or
# ~23%. The epoch line now prints the penalty as a share of that epoch's own mean loss, so this is
# checkable from the log rather than assumed. Raised in review.
#
# The pose-injection tensors are
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
# EVAL IS ALWAYS UNCROPPED, and there are only two passes. The previous arm also ran a CROPPED
# validation every 2 epochs; this one does not, because a cropped reading is only ever diagnostic
# -- the mask carries forced ground-truth contacts, so no number from it is a deployment number --
# and because under token dropout there is no single crop to read it on. So:
#
#   every 5 epochs    valfull, UNCROPPED      -> metric_valfull_ep<N>.json
#   every 10 epochs   the 738 winnable TRAINING interfaces, UNCROPPED, budget 10 = all 50 poses
#                                             -> metric_insample_ep<N>.json
#
# THE BUDGET THAT SETS THAT CADENCE: 6 valfull passes at ~7 min and 3 in-sample at ~33 min is
# 141 min of eval, leaving 1299 of a 24 h day for 30 epochs = 43.3 min/epoch. In-sample is 70% of
# the eval cost because it scores 738 interfaces x 50 poses against valfull's 92 x 150; if it is
# ever wanted more often, subsampling to ~200 interfaces costs ~9 min instead of 33.
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
         test_lean_readouts_arch test_extra_eval test_token_dropout; do
  $PY_BIN -I run_logging/$g.py >/dev/null 2>&1 || { echo "REFUSING: $g failed"; exit 1; }
done
echo "nine gates and both mask provenance checks pass; launching"

LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 \
  --note "BIASED TOKEN DROPOUT 0.70/0.20 + PAIR DROPOUT 0.25 + L2-SP 1e-5, fresh 60-epoch arm from the pretrained head. ONE CHANGE FROM ffj8m4, the arm it must beat: the union mask stops being a FIXED crop and becomes a PRIOR on a fresh draw each epoch -- a token inside the union survives with p=0.70, outside with p=0.20, drawn once per structure per epoch. WHY: ffj8m4 trains on one fixed 59.8% of tokens, which is memorisable, and never shows the head the periphery -- which is why o2qczq scored 0.6305 cropped and 0.5387 uncropped, below shipped 0.5474. A varying subset cannot be memorised and periphery tokens appear 20% of the time, so uncropped inference stops being a regime the head has never seen. AND IT IS FASTER: a dropped token loses its ROW AND COLUMN from z rather than being zeroed, so 0.499 of tokens is 0.249 of the original pair work against ffj8m4 0.598/0.357. TWO INVARIANTS: one draw per structure per EPOCH shared by all 50 poses, because the relative loss grades the gaps between them and per-epoch is exactly per-step under cross-seed; and every rank draws the SAME mask, seeded sha256(--seed, epoch, pdb) -- hash() is salted per process and would have given each rank a different draw. THE COST ACCEPTED: ipTM pools over the scored pair and normalises by a d0 from its token count, so a varying subset varies the achievable score while the label stays full-structure, and d0 enters nonlinearly so it does not cancel under mean-centring. ffj8m4 already carries that at a fixed 62% of each scored chain; this varies it ~45-62%. EVAL IS ALWAYS UNCROPPED: valfull every 5 epochs, the 738 winnable training interfaces every 10, both uncropped, because a cropped reading carries forced ground-truth contacts and is never a deployment number." \
  -- --rank-loss --rank-form rel --rank-cross-seed --alpha-rank 75 \
  --rank-pool mean --rel-labels training/fold/itm_labels \
  --rank-structs-per-step 3 \
  --pose-perblock --pose-perblock-copy --frame-direction \
  --fold training/fold \
  --crop-tokens "$MASK" \
  --token-keep-union 0.70 --token-keep-other 0.20 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
  --manifest-selection training/full_train/structures_winnable.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 60 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --lr-steps '20:1e-3,40:1e-4' \
  --pair-dropout 0.25 \
  --l2-sp 1e-5 --l2-sp-exclude 'frame_dir_proj,pose_perblock_' \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only --lean-readouts \
  --shuffle structure --seeds 500000-500009 \
  --eval-pool training/valfull/pool.csv \
  --eval-pool-fold validation/fold --eval-pool-seed-base 400000 \
  --eval-pool-budget 30 --eval-pool-every 5 \
  --extra-eval "name=insample,pool=training/full_train/pool_winnable.csv,fold=training/fold,seed_base=500000,budget=10,every=10" \
  --prefetch-units 2 --prefetch-workers 4 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 855 \
  --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "launch_unioncrop_dropout_l2 returned $rc"
exit $rc
