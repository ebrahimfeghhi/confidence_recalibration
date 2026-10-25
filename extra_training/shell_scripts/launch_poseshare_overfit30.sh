#!/bin/bash
# POSE INJECTION SHARED ACROSS ALL FOUR PAIRFORMER BLOCKS.
#
# WHAT THE ARCHITECTURE DOES. The confidence head embeds the predicted pose ONCE -- a one-hot of
# cdist over the distogram representative atom through linear_no_bias_d, PLUS the raw distance
# through linear_no_bias_d_wo_onehot -- adds both into z_pair, then runs 4 pairformer blocks that
# know nothing about the pose. This re-applies block 0's OWN injection, both terms verbatim, at
# blocks 1, 2 and 3. One matrix pair, four depths, ZERO new parameters.
#
# VERIFIED ON CPU BEFORE LAUNCH, because this mode had never executed: install adds 0 parameters,
# block 0 stays unwrapped, and a wrapped block equals the unwrapped block fed pre-injected z to
# max|diff| 0.00e+00 -- so the injection is exactly z + d(onehot) + d_wo_onehot(dist).
#
# WHAT IT TESTS, AND WHAT IT CANNOT. It tests whether the pose signal DEGRADES crossing the stack,
# by handing every block the identical embedding. It adds ACCESS, not information: the one-hot is
# the same tensor each time and still carries NO ORIENTATION -- pose enters only as a distance
# between representative atoms, never as a backbone frame. If the head's deficit is orientation
# rather than attenuation, this arm cannot find it, and --frame-direction (pathway C) is the one
# that would. With only 4 blocks the attenuation story is thin on its face.
#
# TRAINABLE: linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 parameters,
# 0.052% of the head's 19.75M. Everything else frozen, as in the pose-focused arm.
#
# STEP 0 IS NOT BIT-IDENTICAL TO THE SHIPPED HEAD. Blocks 1-3 gain an injection they never had,
# using weights that are already non-zero, so epoch 0 of this arm is NOT the shipped head. That is
# inherent to sharing a pretrained matrix rather than adding a zero-init one, and it means the arm
# is a different ARCHITECTURE evaluated from the shipped weights, not a one-variable change to
# them. The --pose-perblock and --pose-perblock-tied modes keep bit-identity; this one cannot.

#
# ONE VARIABLE between the two pose arms: only --l2-sp differs. Everything else matches the three
# capacity arms -- 30 structures, seeds 500000-500009, crop 10/5 at 3400, accum 85, nsample 5,
# K=6, lr 1e-2, warmup 20, masked PAE-only, per-epoch val10 + DockQ at budget 30, checkpointing
# off, single-worker prefetch, 15 steps/epoch, 90 units/rank, 1350 poses -- except 100 epochs.
#
# --pose-perblock ADDS NO TENSORS IN THIS MODE but still changes the forward, and the val-loss
# subprocess and the evaluator are both passed the flags (phase_k_train.py forwards them), which
# is the thing that cost ggfhj5 and m8rote their early held-out points when it was missed.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "POSE-SHARE ARM: block 0's own pretrained pose injection re-applied at every pairformer block (4 depths, ZERO new parameters), training ONLY that shared injection -- linear_no_bias_d and linear_no_bias_d_wo_onehot, 10,240 parameters at c_z 256. 100 epochs, fresh from the shipped head. Tests whether the pose signal degrades crossing the 4-block stack, by giving every block the identical embedding. Adds ACCESS, not information: the one-hot is the same tensor at each depth and still carries no orientation. Step 0 is NOT bit-identical to the shipped head -- blocks 1-3 gain an injection they never had from already-non-zero weights -- so this is a different architecture evaluated from the shipped weights." \
  -- --pose-perblock --pose-perblock-share \
  --train-only 'linear_no_bias_d,linear_no_bias_d_wo_onehot' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
