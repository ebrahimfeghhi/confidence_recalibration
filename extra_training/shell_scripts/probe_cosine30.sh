#!/bin/bash
# GRAD PROBE for the cosine-decay arm. Not a training run: one accumulation window, no head_*.pt.
#
# Measures |g_rank| for zg5fnw's objective (cross-seed softplus at DEFAULT_TAU, PAE on, orientation
# on) at zg5fnw's trainable set, but on the overfit30 pool, so l2sp_lambda.py can transfer lambda
# rather than have it guessed. --alpha-pae 1.0 and NO --l2-sp, so the number reported is the PURE
# TASK gradient -- and because --grad-probe prints |g_rank| before evaluating its own fail list,
# the probe must be allowed to exit 0 for the chain to trust it (ibpz8g/t7dvob, both relaunched).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "GRAD PROBE for the COSINE-DECAY arm, NOT A TRAINING RUN. Measures |g_rank| for zg5fnw's exact objective -- cross-seed softplus pairs at DEFAULT_TAU 0.0305, --rank-weight-by-gap, alpha_pae 1, orientation on -- at zg5fnw's trainable set linear_no_bias_d,pose_perblock,frame_dir (12 tensors / 44,032 params, 8 tensors / 40,960 anchored once frame_dir is excluded), but on the OVERFIT30 pool instead of full_train, so l2sp_lambda.py can transfer lambda from a measured gradient instead of inheriting zg5fnw's 1e-05 across a 16x change in pool size. No --l2-sp here so the printed gradient is the pure task gradient. Exits after one accumulation window and writes no checkpoint." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs cross --rank-weight-by-gap \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --grad-probe --epochs 1 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 100000 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 --prefetch-units 2 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "probe_cosine30 returned $rc"
exit $rc
