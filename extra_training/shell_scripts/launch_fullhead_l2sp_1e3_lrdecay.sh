#!/bin/bash
# tv6hue's ARM WITH A DECAYING LEARNING RATE. ONE VARIABLE: the lr schedule.
#
#   lr 1e-2 flat through epoch 4, then exponential decay to 1e-4 by epoch 20
#   everything else copied from launch_fullhead_l2sp_1e2.sh with --l2-sp back to 1e-3
#
# WHY. tv6hue holds the best full-val PAE this project has measured -- 2.7174 at epoch 5 over the
# 7,950-pose valfull set -- and then sat between 2.717 and 2.745 for fifteen more epochs while its
# training loss kept falling. A flat 1e-2 is the obvious suspect: a step that found the basin by
# epoch 4 is too big to settle into it.
#
# THE SCHEDULE, on tv6hue's measured geometry of 244 steps/epoch and 4,880 steps:
#
#     epoch  0.1     1.000e-02   (warmup over 20 steps finishes here)
#     epoch  4       1.000e-02   (hold ends; decay begins)
#     epoch  8       3.162e-03
#     epoch 12       1.000e-03
#     epoch 16       3.162e-04
#     epoch 20       1.000e-04
#
# EXPONENTIAL RATHER THAN COSINE because the span is 100x over 16 epochs. Cosine would be at
# 9.6e-3 at epoch 6 and 5.0e-3 at epoch 12 -- the first half of the window barely decays -- while
# exponential is geometric at 0.7499 per epoch and spends equal time in each decade.
#
# WHAT WOULD MAKE IT WORTH KEEPING: full-val PAE below 2.7174 at any epoch after 4.
#
# WHAT WOULD FALSIFY THE PREMISE: the curve tracking tv6hue within noise to epoch 20. That would
# say the plateau is this objective's floor on 487 structures rather than a step-size artefact,
# which is where the ABAG-Rank arms landed by a different route. A decay cannot find signal that
# is not there; it can only stop overshooting signal that is.
#
# THE QUEUE, not a direct launch: shell_scripts/queue_lrdecay_when_free.sh polls cards_busy.py and
# runs this once cards 4, 5 and 7 are all free. This script refuses on its own if they are not, so
# neither path can land on an occupied card, and NEITHER CAN KILL ANYTHING -- the queue can only
# ever fail to launch.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching tv6hue CONFIG + LR DECAY (hold 1e-2 to epoch 4, exponential to 1e-4 by epoch 20)"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "tv6hue's CONFIG WITH A DECAYING LEARNING RATE -- the one variable is the lr schedule. tv6hue is the best full-val PAE this project has measured, 2.7174 at epoch 5 on the 7,950-pose valfull set, and it then sat flat between 2.717 and 2.745 for fifteen more epochs while its training loss kept falling. A flat 1e-2 for twenty epochs is the obvious suspect for that plateau: the step size that found the basin by epoch 4 is too large to settle into it. So --lr 0.01 is held through epoch 4, the epoch tv6hue had already reached its floor, and then decays to 1e-4 by epoch 20. EXPONENTIAL, not cosine: the span is 100x over 16 epochs, and cosine would still be at 5.0e-3 at epoch 12 -- barely decayed through the first half of the window -- where exponential is at 1.0e-3, a clean 0.7499 factor per epoch. EVERY OTHER FLAG IS COPIED FROM launch_fullhead_l2sp_1e2.sh WITH l2-sp BACK TO 1e-3: same fold, selection, skip list, crop 3400, margin 10, contact 5, accum 85, nsample 5, seeds 500000-500009, pose-copy injection, --pae-only, --l2-sp-exclude linear_no_bias_d, valfull held-out at budget 30, 20 epochs, --blocks-per-ckpt 0. WHAT WOULD MAKE THIS WORTH KEEPING: full-val PAE below 2.7174 at any epoch after 4. WHAT WOULD FALSIFY THE PREMISE: the curve tracks tv6hue's within noise through epoch 20, which would say the plateau is the objective's floor on 487 structures rather than a step-size artefact -- the same conclusion the ABAG-Rank arms reached by a different route. STATED BEFORE LAUNCH so it cannot be reframed after: a decay cannot find signal that is not there; it can only stop overshooting signal that is. THE LAUNCH LOG MUST PRINT '8 tensor(s) stay TRAINABLE but are OUT of the anchor, 246 anchored' -- if it says 2, the pose copies are being anchored and the arm is not what it claims. VAL10 AND VALFULL ARE NOT COMPARABLE." \
  -- --pose-perblock --pose-perblock-copy \
  --l2-sp 1e-3 --l2-sp-exclude 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 20 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --lr-final 1e-4 --lr-decay-start-epoch 4 --lr-decay-end-epoch 20 \
  --lr-decay-shape exponential \
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
echo "launch_fullhead_l2sp_1e3_lrdecay returned $rc"
exit $rc
