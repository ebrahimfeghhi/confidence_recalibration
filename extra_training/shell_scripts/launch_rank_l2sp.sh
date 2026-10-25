#!/bin/bash
# RANKING LOSS ALONE + L2-SP ANCHOR, POSE INJECTION AT ITS ORIGINAL SINGLE SITE.
#
# WHAT THIS ARM IS. 4y3ve6's objective and trainable set, with TWO changes, both requested:
#   1. NO --pose-perblock. The pose embedding enters the head exactly ONCE, where Protenix put
#      it, instead of being re-applied at blocks 1-3. Not an architecture change at all, so
#      step 0 IS the shipped head bit-for-bit -- which --pose-perblock-share could never be,
#      because sharing a pretrained matrix into three blocks that never had it changes epoch 0.
#   2. --l2-sp 1e-3. lambda * ||W - W_shipped||^2 added to the loss, anchored on the two pose
#      projections (the only trainable tensors), snapshotted at init to l2sp_anchor.pt.
#
# It therefore differs from 4y3ve6 in two ways, not one. Stated plainly because no run so far has
# been rank-only WITHOUT --pose-perblock-share, so a difference cannot be attributed to the
# anchor alone. The honest controls are 4y3ve6 (same objective, same trainable set, perblock
# share, no anchor) and 8eiek8 (same trainable set, no ranking term).
#
# WHY AN ANCHOR, MEASURED NOT ASSUMED. The two pose projections have ||W_shipped||^2 = 18.18.
# 4y3ve6 moved them, against that shipped point:
#       ep1    3.86     ep3    49.23     ep9   327.97     ep15   910.66     ep19  1379.21
# So by epoch 3 the UPDATE is 2.7x the squared norm of the weights being updated, and by epoch 19
# it is 76x. Epoch 3 is also where 4y3ve6's held-out ranking loss bottomed (0.03773) before
# rising to 0.04012 by epoch 12. The arm walks a long way from the shipped point and generalises
# worse the further it goes, which is the thing an anchor prices.
#
# WHY LAMBDA 1e-3. The penalty gradient is 2*lambda*||dW||, against |g_rank| = 7.96e-02 measured
# on this exact configuration by probe ev3d6u. At epoch 3's drift (||dW|| = 7.02) that is
# 1.4e-02, 18% of the task gradient -- permissive where held-out performance peaked. At epoch
# 19's drift (||dW|| = 37.1) it is 7.4e-02, 93% -- an equal and opposing force exactly where the
# unregularised run ran away. In loss terms it is 0.049 at epoch 3 against a ranking loss near
# 0.04. 1e-4 was rejected: 1.8% of the task gradient at epoch 3, and the one prior null result
# (4x8t44) sat at 2% of the loss. LAMBDA IS NOT TUNED -- it is the hyperparameter of this arm,
# chosen against the drift above and recorded here so a second value is a second experiment.
#
# L2-SP's RECORD HERE IS 0-FOR-2: 4x8t44 was null at 2.0% of the loss, wpg274 did not stop the
# ordering collapse. Neither was this configuration -- both carried a different objective -- but
# the prior is not favourable and should not be forgotten when this is read.
#
# TRAINABLE: linear_no_bias_d (256x39 = 9,984) + linear_no_bias_d_wo_onehot (256x1 = 256)
# = 10,240 parameters, 0.052% of the head. pae_ln and linear_no_bias_pae stay FROZEN. The
# startup log must report BOTH tensors twice -- once from --train-only and once as the l2-sp
# anchor's tensor count -- because an anchor over zero tensors would apply no penalty silently.
#
# WHAT WOULD FALSIFY IT: ||dW||^2 tracking 4y3ve6's curve anyway (the anchor priced the drift and
# the drift was worth paying for), or the drift held down while held-out rank loss still turns at
# epoch 3 (the departure was never the mechanism).
#
# EVERYTHING ELSE MATCHES 4y3ve6: 30 structures, seeds 500000-500009, crop 10/5 at 3400,
# accum 85, nsample 5, lr 1e-2, warmup 20, masked PAE-only, per-epoch val10 + DockQ at budget 30,
# checkpointing off, single-worker prefetch, 100 poses/step, 50 epochs.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "RANKING LOSS ALONE + L2-SP ANCHOR, POSE INJECTION AT ITS SINGLE ORIGINAL SITE. --alpha-pae 0, so the ONLY objective is pose ordering; --rank-form softplus at tau 0.0305, cross-seed, 2 structures x 10 seeds x 5 poses = 100 poses per step. Trainable set is the two pose projections -- linear_no_bias_d (256x39) and linear_no_bias_d_wo_onehot (256x1), 10,240 parameters; pae_ln and linear_no_bias_pae stay FROZEN. NO --pose-perblock: the pose enters the head exactly once, where Protenix put it, so step 0 is the shipped head bit-for-bit -- which --pose-perblock-share can never be, since sharing a pretrained matrix into three blocks that never had it changes epoch 0. WHY THE ANCHOR, AND WHY 1e-3. The two pose projections have ||W_shipped||^2 = 18.18. Measured from 4y3ve6's own checkpoints, the unregularised arm moved them to ||dW||^2 = 3.86 at ep1, 49.23 at ep3, 327.97 at ep9, 910.66 at ep15, 1379.21 at ep19 -- so by epoch 3 the update is 2.7x the squared norm of the weights being updated and by epoch 19 it is 76x, while held-out rank loss bottomed at ep3 (0.03773) and rose to 0.04012 by ep12. The penalty gradient 2*lambda*||dW|| against |g_rank| 7.96e-02 (probe ev3d6u, this exact configuration) is 18% of the task gradient at the ep3 drift and 93% at the ep19 drift: permissive where generalisation peaked, an equal opposing force where the run ran away. In loss terms 0.049 at ep3 against a ranking loss near 0.04. 1e-4 was rejected as 1.8% of the task gradient at ep3, and the one prior null (4x8t44) sat at 2% of the loss. LAMBDA IS NOT TUNED; it is this arm's hyperparameter. L2-SP IS 0-FOR-2 IN THIS PROJECT: 4x8t44 null at 2.0% of the loss, wpg274 did not stop the ordering collapse -- neither carried this objective, but the prior is not favourable. TWO VARIABLES AGAINST 4y3ve6, NOT ONE: the anchor and the removal of perblock sharing. No rank-only arm has previously run without --pose-perblock-share, so a difference cannot be attributed to the anchor alone. CONTROLS: 4y3ve6 (same objective and trainable set, perblock share, no anchor, overfit from ep3) and 8eiek8 (same trainable set, no ranking term). WHAT WOULD FALSIFY IT: drift tracking 4y3ve6's curve anyway, or drift held down while held-out rank loss still turns at epoch 3. THE HELD-OUT MONITOR MEASURES AT alpha_pae 1 or val10 PAE would read 0.0 every epoch and the one metric comparable across all arms would vanish." \
  -- --l2-sp 1e-3 \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 0.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 3 \
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
