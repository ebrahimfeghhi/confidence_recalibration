#!/bin/bash
# POSE-FOCUSED ADAPTATION: train only how geometry ENTERS the head and how PAE is READ OUT.
#
# THE IDEA, in the user's words: LoRA limits how much the weights can change; selective
# finetuning also controls WHICH COMPUTATIONS can change. The pretrained pairformer already
# learned useful structural representations, so rather than letting training rewrite them and
# overfit individual complexes, adapt only the pose pathway and the PAE readout.
#
# THE ARCHITECTURAL PREMISE WAS CHECKED, NOT ASSUMED (protenix/model/modules/confidence.py):
#   pose enters ONCE, at line 281:  distance_pred = cdist(x_pred_rep_coords, x_pred_rep_coords)
#   and reaches z_pair through BOTH of these, added together, not one or the other:
#       z_pair += linear_no_bias_d(one_hot(distance_pred, lower_bins, upper_bins))   (128, 39)
#       z_pair += linear_no_bias_d_wo_onehot(distance_pred.unsqueeze(-1))            (128, 1)
#   and leaves at line 331:  pae_pred = linear_no_bias_pae(pae_ln(z_pair))           (64, 128)
# Nothing else in the head sees coordinates. The four pairformer blocks, both s_inputs
# embeddings, the pde head and the plddt/resolved weights are all frozen.
#
#   13,312 parameters -- 0.104% of the head -- against 19.8M for full FT3.
#
# pae_ln IS NOT IN THIS ARM, and arguably should be. It is a LayerNorm sitting directly between
# z_pair and the PAE projection, so it is part of "how PAE is predicted" by any reading, and it
# is 256 more parameters (0.106% total). Left out here to run the experiment as specified; add
# it by appending ',pae_ln' to --train-only.
#
# THE HYPOTHESIS: training PAE improves less, held-out PAE improves more and stays stable
# longer. Read against the full-head arm epochs 1-6, which gave train -7.6% and held-out -0.9%.
#
# THE RISK, SAID PLAINLY: 13,312 parameters may simply be too few to move ipTM at all, and the
# frozen pairformer was trained jointly with the distance embedding it is now being fed a
# modified version of. A flat held-out curve here is as consistent with underfitting as with a
# genuine ceiling, so the training-PAE curve is what distinguishes them: if train PAE also
# barely moves, the arm underfit rather than generalised.
#
# LEARNING RATE IS INHERITED, NOT TUNED. lr 1e-2 was chosen for 19.8M parameters; 13,312 is a
# different optimisation problem and the right rate for it has not been measured. Kept the same
# so this is ONE variable against the full-head arm -- but if the curve is flat, the lr is the
# first thing to vary before concluding anything about the method.
#
# FRESH FROM THE SHIPPED HEAD, as the hypothesis requires. Everything else matches the running
# arm: 488 structures, warmup 20, accum 85 (K=6, 90 poses/step), nsample 5, crop 10/5 at 3400,
# clip 25.0, alpha 1/1, masked PAE-only, seeds 500000-500009, four per-epoch metrics at budget
# 30, checkpointing off, prefetch on.
#
# CHECKPOINTS STAY COMPLETE. Only three tensors train, but head_step*.pt is written with ALL 248
# head tensors, because phase_k_eval requires SET EQUALITY between a checkpoint's keys and the
# model's head parameters -- a three-tensor file would be refused at the first eval.
#
# DO NOT LAUNCH WHILE ocqtyt HOLDS CARDS 4,5,7.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "POSE-FOCUSED ADAPTATION: train only linear_no_bias_d, linear_no_bias_d_wo_onehot and linear_no_bias_pae -- the complete pose pathway plus the PAE readout, 13,312 parameters, 0.104% of the head. The four pairformer blocks and everything else are frozen, so the pretrained structural representation is preserved and only how geometry enters and how PAE is read out can adapt. Fresh from the shipped head. One variable against the full-head arm; lr 1e-2 is inherited from a 19.8M-parameter problem and is NOT tuned for this one." \
  -- --train-only 'linear_no_bias_d,linear_no_bias_d_wo_onehot,linear_no_bias_pae' \
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
echo "launcher returned $?"
