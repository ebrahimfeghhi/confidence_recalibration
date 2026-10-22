#!/bin/bash
# L2-SP: THE WHOLE HEAD TRAINS, BUT MOVING AWAY FROM THE SHIPPED WEIGHTS COSTS SOMETHING.
#
#   loss = masked PAE  +  LAMBDA * ||W - W_shipped||^2         LAMBDA = 1e-5
#
# WHY THIS AND NOT WEIGHT DECAY. weight_decay has been 0.0 in every arm and always was, so
# "restriction helped" and "any regularisation helped" are still not distinguishable. But plain
# weight decay pulls toward ZERO, which is not a prior anybody holds about this head -- the
# SHIPPED weights are. L2-SP prices distance from them instead. It is the continuous form of the
# two arms already run: --train-only forbids 245 of 248 tensors from moving, --lora-rank forbids
# any move outside a rank-8 subspace, and this one lets every parameter move while charging for it.
#
# LAMBDA COMES FROM THE MEASURED DRIFT, not from a default. The unconstrained full head (mdv9x7)
# ended 30 epochs at ||W - W_shipped||^2 = 67,051.81 over 19,752,270 parameters (RMS 0.0583 per
# parameter). LAMBDA 1e-5 prices that drift at 0.67 against a PAE loss near 2.0 -- about a third
# of the objective, enough to bend the solution without freezing it.
#   1e-6 -> 0.067 at that drift (~3% of the loss; probably too weak to matter)
#   1e-5 -> 0.671            <- this arm
#   1e-4 -> 6.705 (3x the loss; probably pins the head near shipped)
# LAMBDA IS NOT TUNED. It is the hyperparameter of the experiment and one value is one sample of
# it. Every arm so far plateaued or turned by epoch 13-21, so 100 epochs at one LAMBDA spends most
# of its time in the tail; three arms of 30 epochs at 1e-6/1e-5/1e-4 would cost the same wall
# clock and locate LAMBDA instead. Run at 100 epochs at the user's explicit direction.
#
# WHAT IT IS TESTING. The three capacity arms gave: 0.135% never overfit and never reached a good
# optimum; 100% reached the best optimum (val10 2.6881 at ep12) then broke; 1.45% reached 2.7057
# at ep13 and began drifting EARLIEST of the three. Capacity therefore does NOT order the onset,
# which is the result that motivates this arm: if the problem is HOW FAR the weights travel rather
# than HOW MANY of them may travel, pricing distance should beat forbidding motion.
#
# READ THE DRIFT, which is printed every epoch as "l2-sp: ||W-W0||^2 = ... penalty = ...". If it
# sits near 67,052 the penalty is too weak to be doing anything; if it sits near 0 the head is
# pinned and this is just the shipped head with extra steps. Either outcome names the next LAMBDA.
#
# THE ANCHOR IS THE SHIPPED HEAD and is snapshotted at init to <out>/l2sp_anchor.pt, because a
# RESUMED run's weights are not the shipped ones -- resuming without that file is refused rather
# than silently anchoring to the wrong point.
#
# This arm does NOT hit the held-out-monitor bug that cost ggfhj5 and m8rote their early points:
# the trainable set is the plain 248 head tensors, so the monitor's own default set matches the
# bundle and no inheritance is needed. The penalty is a training term only and is absent from the
# held-out measurement by design -- val10 PAE stays comparable across every arm.
#
# ONE VARIABLE against mdv9x7: identical 30 structures, seeds, crop 10/5 at 3400, accum 85,
# nsample 5, K=6, lr 1e-2, warmup 20, masked PAE-only, per-epoch val10 + DockQ at budget 30, fresh
# from the shipped head, checkpointing off, single-worker prefetch. Only --l2-sp is added, and
# --epochs is 100 rather than 30.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "L2-SP ARM: full head trains, with LAMBDA 1e-5 * ||W - W_shipped||^2 added to the loss. 100 epochs, fresh from the shipped head. The continuous form of the two restriction arms -- every parameter may move, but moving from the shipped weights is priced rather than forbidden. LAMBDA chosen from the measured drift: mdv9x7 ended 30 epochs at ||dW||^2 = 67,052, which 1e-5 prices at 0.67 against a PAE loss near 2.0. LAMBDA IS NOT TUNED. Also the first arm with any regularisation at all -- weight_decay has been 0.0 throughout -- so it is the control the capacity arms have been missing. ONE VARIABLE against mdv9x7 apart from epoch count." \
  -- --l2-sp 1e-5 \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 100 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/val10/structures.csv \
  --val-loss-pool training/val10/pool.csv \
  --val-loss-preflight training/val10/alignment_preflight.json \
  --val-loss-seeds 400000-400029 \
  --val-loss-prep validation/prep \
  --train-eval-pool training/val10/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 30 \
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
# PROPAGATE IT. The chain runner stops on a non-zero arm, and a trailing bare `echo` always
# returns 0 -- which is why a broken arm reported success and the chain marched straight on to
# the next one.
echo "launcher returned $rc"
exit $rc
