#!/bin/bash
# GRADIENT PROBE for the new trainable set: the 12 orientation+distance tensors PLUS block 3's
# pair_transition. One accumulation window, no checkpoint written, exits immediately after.
#
# WHY IT RUNS BEFORE THE ARM. L2-SP's lambda only carries across a change of trainable set if the
# ratio 2*lambda*||dW|| / |g_task| is preserved, and both factors move: the parameter count goes
# 44,032 -> 830,976 (18.9x) and |g_rank| is unknown at the new set. run_logging/l2sp_lambda.py
# needs the measured |g_rank| to do the transfer, so this measures it.
#
# alpha_pae 1 and NO --l2-sp, deliberately: measuring both terms is free, |g_rank| does not
# depend on alpha_pae, and omitting the anchor makes the reported gradient the PURE TASK
# gradient, which is the quantity the transfer needs.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "STOP: $BUSY process(es) on cards 4,5,7"; exit 1; }
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "GRAD PROBE for the SIMPLE ranking + PAE arm at the pair_transition-block-3 trainable set, NOT A TRAINING RUN. Measures |g_rank| at --train-only 'linear_no_bias_d,pose_perblock,frame_dir,pairformer_stack\\.blocks\\.3\\.pair_transition' at --rank-pairs cross and DEFAULT_TAU, plus |g_PAE| at the same weights so --alpha-rank can be set for GRADIENT PARITY rather than guessed, and so l2sp_lambda.py can transfer lambda from 24u34d's 2.5e-4 at 40,960 anchored parameters to this set. Exits after one accumulation window and writes no head_*.pt. alpha_pae 1 and no --l2-sp so the reported gradient is the PURE TASK gradient." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs cross \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir,pairformer_stack\.blocks\.3\.pair_transition' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --grad-probe --epochs 1 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 100000 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
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
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "probe returned $?"
