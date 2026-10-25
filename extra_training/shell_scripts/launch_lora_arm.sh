#!/bin/bash
# LoRA ARM: the same experiment as the full-set run, with the head's capacity constrained.
#
# FRESH FROM THE SHIPPED HEAD, NOT A RESUME. LoRA changes the PARAMETER SET -- 282 adapter
# tensors instead of 248 head tensors -- so no existing bundle can supply them and the guard
# refuses the resume on the missing-tensor check. This starts where every arm started.
#
# WHY. Full FT3 moves 19.8M parameters. The 30-structure anchor showed what that capacity does
# with too little data: training PAE -38% while held-out PAE rose 51%, best held-out at epoch 1.
# The full set is better behaved but the gap is still opening -- epochs 1-6 gave train -7.6%
# against held-out -0.9%. LoRA constrains each weight's UPDATE to rank r, which limits what the
# finetune can express rather than penalising it after the fact.
#
# RANK 8 OVER THE WHOLE HEAD, measured on the head itself:
#   rank  modules  adapter params  % of head
#      2      149         143,890      1.12%
#      4      149         287,780      2.24%
#      8      141         567,112      4.42%
#     16      141       1,134,224      8.85%
#     32      137       2,250,016     17.55%
# (a few modules are skipped at higher rank because r exceeds a dimension -- the 39-bin distance
# input and the 16x128 pair-bias projection -- and that is reported at launch, not silently.)
#
# STEP 0 IS BIT-IDENTICAL TO THE SHIPPED HEAD. B is zero-initialised, verified max|diff| 0.000e+00
# on an adapted leaf, so LoRA-on against LoRA-off is a clean single variable at the start. A is
# Kaiming-normal: both-zero would be a dead gradient forever, both-random would lose the identical
# start. A receives no gradient at step 0 and starts training at step 1, which is LoRA's intended
# cold start and was checked rather than assumed.
#
# THE CHECKPOINTS ARE ORDINARY. Adapters are merged into the base weights on save, so
# head_step*.pt carries the same 248 tensor names and shapes a full-finetune checkpoint does and
# phase_k_eval needs no changes. Verified: loads into a plain ConfidenceHead with 0 missing and 0
# unexpected, and the merged forward matches the adapted forward to 2.4e-06.
#
# BEFORE READING MUCH INTO THIS ARM: weight_decay is 0.0 in the AdamW construction and always has
# been. That is the cheaper regulariser and it is not switched on, so "LoRA helped" and "any
# regularisation would have helped" are not yet distinguishable. A weight-decay arm is the
# control this one wants.
#
# EVERYTHING ELSE MATCHES THE RUNNING ARM so the two curves can be read against each other:
# 488 structures, lr 1e-2, warmup 20, accum 85 (K=6, 90 poses/step), nsample 5, crop 10/5 at
# 3400, clip 25.0, alpha 1/1, masked PAE-only, seeds 500000-500009, the four per-epoch metrics at
# budget 30, checkpointing off, prefetch on.
#
# DO NOT LAUNCH WHILE ocqtyt HOLDS CARDS 4,5,7. This needs the same three cards.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "LoRA ARM: rank 8 alpha 16 over every adaptable module of the confidence head, 0.567M adapter parameters against 19.8M for full FT3. Fresh from the shipped head because LoRA changes the parameter set. Tests whether constraining the UPDATE's rank closes the train/held-out gap that full FT3 opens. One variable against the full-set arm; weight_decay is still 0.0 in both, so this does not yet separate LoRA from regularisation in general." \
  -- --lora-rank 8 --lora-alpha 16 --lora-dropout 0.0 \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight.json \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 500 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 244 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
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
