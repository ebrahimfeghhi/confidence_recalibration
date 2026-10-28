#!/bin/bash
# FULL CONFIDENCE HEAD + POSE COPY, L2-SP 1e-4 ON THE NON-POSE TENSORS, 40 EPOCHS.
#
#   loss = masked PAE + 1e-4 * ||W - W_shipped||^2   over the 246 NON-POSE head tensors
#   the 8 pose tensors train FREE, with no anchor at all
#
# At the user's direction. TWO VARIABLES against tv6hue, both named: lambda 1e-3 -> 1e-4, and
# 20 -> 40 epochs. Every other flag is copied verbatim from launch_fullhead_l2sp_1e2.sh.
#
# WHERE 1e-4 SITS. On the 7,950-pose full val set: lambda 1e-2 (jer222) 2.7501 at epoch 7;
# lambda 1e-3 (tv6hue) 2.7174 at epoch 5, the best measured in this project. The unanchored end
# (wpg274, lambda 0) reached 2.6912 at epoch 22 but only on the 1,500-pose subset, which is NOT
# comparable. So 1e-4 is the rung between tv6hue and no anchor at all.
#
# WHAT THE PENALTY SHOULD WEIGH. tv6hue measured ||W-W0||^2 settling near 49, so at 1e-4 the
# penalty prices around 0.005 against a PAE loss near 2.6 -- roughly 0.2% of the objective. That
# is a weak constraint, much closer to the unanchored arm than to tv6hue. Read the measured
# "l2-sp: ||W-W0||^2 = ... penalty = ..." line each epoch against train and val PAE rather than
# predicting which regime it lands in; that prediction is what failed at 1e-2.
#
# LAMBDA IS NOT DERIVED. l2sp_lambda.py's transfer formula is deliberately unused: on wpg274 it
# produced a penalty under 2% of the objective by sizing lambda for a drift 18x larger than the
# one that occurred.
#
# THE LEARNING RATE IS FLAT HERE. The user specified the lambda and the epoch count only, so no
# decay flags are set; the decaying-lr arm is queued separately as
# shell_scripts/launch_fullhead_l2sp_1e3_lrdecay.sh.
#
# 40 EPOCHS AGAINST A KNOWN PLATEAU, stated before launch so it cannot be reframed after: tv6hue
# hit its floor at epoch 5 and sat between 2.717 and 2.745 for fifteen more epochs, and wpg274 at
# lambda 0 ran val PAE 2.7463 -> 3.0990 across 50 epochs while train fell to 1.7854. A weaker
# anchor over twice the epochs is the direction wpg274 already explored. The epochs after the
# floor are therefore the measurement, not the hope.
#
# THE LAUNCH LOG MUST PRINT "8 tensor(s) stay TRAINABLE but are OUT of the anchor, 246 anchored".
# If it says 2, the pose copies are being anchored and the arm is not what it claims.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching FULL HEAD + pose-COPY, L2-SP 1e-4, full_train -> valfull, 40 epochs"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "FULL CONFIDENCE HEAD + POSE COPY, L2-SP 1e-4 ANCHORING EVERYTHING EXCEPT THE POSE PARAMETERS, 40 EPOCHS. At the user's direction. Two variables against tv6hue, both named: lambda 1e-3 -> 1e-4, and 20 -> 40 epochs. Everything else is copied verbatim from launch_fullhead_l2sp_1e2.sh -- same fold, selection, skip list, crop 3400, margin 10, contact 5, lr 1e-2 flat after 20 warmup steps, accum 85, nsample 5, seeds 500000-500009, pose-copy injection, --pae-only, --l2-sp-exclude linear_no_bias_d, valfull held-out at budget 30, --blocks-per-ckpt 0. THE SWEEP SO FAR, all on the 7,950-pose full val set: lambda 1e-2 (jer222) 2.7501 at epoch 7; lambda 1e-3 (tv6hue) 2.7174 at epoch 5, the best this project has measured; lambda 0 unanchored (wpg274) 2.6912 at epoch 22 but on the 1,500-pose subset, NOT comparable. 1e-4 is the rung between tv6hue and unanchored. LAMBDA IS NOT DERIVED -- l2sp_lambda.py's transfer formula is deliberately unused, since on wpg274 it produced a penalty under 2% of the objective by sizing lambda for a drift 18x larger than the one that occurred. Read the measured 'l2-sp: ||W-W0||^2 = ... penalty = ...' line each epoch against train and val PAE: tv6hue settled near 49, so 1e-4 should price it near 0.005 against a PAE loss of ~2.6, about 0.2% of the objective -- a weak constraint, closer to the unanchored arm than to tv6hue. THE LEARNING RATE IS FLAT, not decayed, because the user specified the lambda and the epoch count only; the decaying-lr arm is queued separately as launch_fullhead_l2sp_1e3_lrdecay.sh. NOTE 40 EPOCHS AGAINST A PLATEAU: tv6hue reached its floor at epoch 5 and sat between 2.717 and 2.745 for fifteen more epochs, and wpg274 at lambda 0 ran val PAE 2.7463 -> 3.0990 over 50 epochs while train fell to 1.7854. A weaker anchor over twice the epochs is the direction wpg274 already explored, so the epochs after the floor are the measurement, not the hope. THE LAUNCH LOG MUST PRINT '8 tensor(s) stay TRAINABLE but are OUT of the anchor, 246 anchored' -- if it says 2, the pose copies are being anchored and the arm is not what it claims. FRESH launch, so the anchor is built from the shipped weights rather than inherited. VAL10 AND VALFULL ARE NOT COMPARABLE." \
  -- --pose-perblock --pose-perblock-copy \
  --l2-sp 1e-4 --l2-sp-exclude 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 40 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/valfull/structures.csv \
  --val-loss-pool training/valfull/pool.csv \
  --val-loss-preflight training/valfull/alignment_preflight.json \
  --val-loss-seeds 400000-400029 \
  --val-loss-prep validation/prep \
  --train-eval-pool training/valfull/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 30 \
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "launch_fullhead_l2sp_1e4_40ep returned $rc"
exit $rc
