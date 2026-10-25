#!/bin/bash
# LoRA ARM on the 30-structure set, against val10, 30 epochs. THE THIRD POINT ON THE CAPACITY AXIS.
#
#   pose-focused  ggfhj5+zcuiab     26,624 params   0.135% of the head   3 tensors, hard freeze
#   LoRA r=8      this arm         567,112 params   4.42%                141 modules, rank-8 updates
#   full head     mdv9x7            19,752,270      100%                 248 tensors, unconstrained
#
# WHY IT IS NOT JUST AN INTERPOLATION, which is the honest reason to bother. LoRA constrains the
# DIRECTION of every weight's update to rank r rather than freezing most weights outright, so it
# touches the four pairformer blocks that the pose-focused arm froze solid -- everything can move
# a little, instead of three tensors moving freely. If the two behave the same at matched
# parameter counts, capacity alone explains both; if they differ, WHICH computations may change
# matters beyond HOW MANY parameters do.
#
# WHAT THE OTHER TWO ARMS ALREADY SETTLED, so this one is not asked to re-answer it:
#   val10 PAE: pose-focused converged to a 2.751-2.764 plateau and never broke, 21 epochs.
#              full head reached 2.6881 at ep12, the best of any run, then drifted up -- the last
#              five of its 30 epochs all above 2.80, with a 3.1125 spike at ep28 that crossed the
#              shipped head's 2.9756. Overfit onset ~ep21-28, NOT ~ep9 as 2u8tf4 implied.
#   val10 DockQ: BELOW the 0.2909 baseline in essentially every epoch of BOTH arms. The single
#              best held-out-PAE head ever recorded (mdv9x7 ep12) scored DockQ 0.2689 with 8 of
#              18 acceptable, against 0.2909 and 10 of 18 UNTRAINED.
# So the prediction for this arm is a val10 best between 2.69 and 2.75 with a later onset than
# ep21, and DockQ still under baseline. SAY SO IF THAT IS WHAT HAPPENS -- a confirmed prediction
# is a result, not a disappointment, and it closes the capacity axis.
#
# WEIGHT DECAY IS STILL 0.0 AND ALWAYS HAS BEEN (phase_k_train.py, the AdamW construction). That
# is the cheaper regulariser and it is not switched on, so "LoRA helped" and "any regularisation
# would have helped" remain indistinguishable. A weight-decay arm is the control this one wants
# and it does not exist yet.
#
# WHERE THE ADAPTERS GO: every submodule of confidence_head holding a 2-D nn.Parameter named
# `weight` -- 141 modules at rank 8. Selection is by name and shape, NOT by class: only 54 are
# Linear/LinearNoBias and the other ~96 matrices inside tri_mul_* / tri_att_* are a different
# class carrying 1.57M parameters. The trunk is untouched, as in every arm.
#
# STEP 0 IS BIT-IDENTICAL TO THE SHIPPED HEAD. B is zero-init (verified max|diff| 0.000e+00), A is
# Kaiming-normal, so A gets no gradient at step 0 and training starts at step 1. Biases are
# neither adapted nor unfrozen.
#
# CHECKPOINTS STAY ORDINARY: adapters are merged on save, so head_step*.pt carries the same 248
# tensor names and shapes a full finetune writes and phase_k_eval needs no change. A LoRA resume
# bundle, by contrast, holds adapter tensors no other arm has -- so this is FRESH FROM THE SHIPPED
# HEAD and cannot be resumed from any existing bundle.
#
# ONE VARIABLE against mdv9x7: identical 30 structures, seeds, crop 10/5 at 3400, accum 85,
# nsample 5, K=6, lr 1e-2, warmup 20, masked PAE-only, per-epoch val10 + DockQ at budget 30,
# checkpointing off, single-worker prefetch, 30 epochs. Only --lora-* is added.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "LoRA ARM r=8 on the 30-structure set against val10, 30 epochs, fresh from the shipped head. 567,112 adapter parameters, 4.42% of the head, over 141 modules -- the third point on the capacity axis between the pose-focused arm (26,624, 0.135%) and the full head mdv9x7 (19.8M). Unlike the pose-focused arm it constrains the DIRECTION of every update to rank 8 rather than freezing most weights, so the pairformer blocks can move a little. ONE VARIABLE against mdv9x7; only --lora-* added. Prediction on record: val10 best between 2.69 and 2.75, overfit onset later than ep21, DockQ still below the 0.2909 baseline." \
  -- --lora-rank 8 --lora-alpha 16 --lora-dropout 0.0 \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 30 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
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
echo "launcher returned $?"
