#!/bin/bash
# GRAD PROBE for the STRADDLE arm on the full fold. NOT A TRAINING RUN.
#
# WHY. lambda cannot be carried from zg5fnw unchanged even though the ANCHORED SET IS IDENTICAL
# (8 tensors / 40,960 parameters, since dropping --frame-direction also removes the only tensors
# zg5fnw excluded). l2sp_lambda.py's size factor is therefore exactly 1 and the whole transfer
# rests on the task gradient -- which this measures, because the objective changed twice over:
# --rank-straddle 0.23 keeps 143,680 of 1,056,035 pairs (13.6%) and tau rises 0.0305 -> 0.1429.
# Fewer pairs and a wider tau both move |g_rank|, and in opposite directions, so guessing is
# not available.
#
# alpha_pae 0 and NO --l2-sp: the reported gradient is the PURE ranking gradient, which is the
# only task term this arm has.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY process(es) occupy cards 4,5,7."; exit 1; }
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "GRAD PROBE for the STRADDLE arm on the FULL FOLD, NOT A TRAINING RUN. Measures |g_rank| at --train-only 'linear_no_bias_d,pose_perblock' under --rank-pairs all --rank-straddle 0.23 --rank-tau 0.1429 so l2sp_lambda.py can transfer lambda from zg5fnw's 1e-05 at |g_rank| 2.4260e-03 and 40,960 anchored parameters on this same fold. The anchored set is IDENTICAL in size here -- dropping --frame-direction removes the only tensors zg5fnw excluded -- so the size factor is 1 and the transfer is the gradient ratio alone. Exits after one accumulation window and writes no head_*.pt. alpha_pae 0 and no --l2-sp so the number is the PURE ranking gradient." \
  -- --pose-perblock --pose-perblock-copy \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs all --rank-straddle 0.23 --rank-tau 0.1429 \
  --train-only 'linear_no_bias_d,pose_perblock' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --grad-probe --epochs 1 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 0.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 100000 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 --prefetch-units 2 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/valfull/structures.csv \
  --val-loss-pool training/valfull/pool.csv \
  --val-loss-preflight training/valfull/alignment_preflight.json \
  --val-loss-seeds 400000-400029 --val-loss-prep validation/prep \
  --train-eval-pool training/valfull/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 --train-eval-budget 30 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "probe returned $rc"
exit $rc
