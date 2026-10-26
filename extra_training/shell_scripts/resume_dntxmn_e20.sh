#!/bin/bash
# RESUME dntxmn (copy x PAE) AT EPOCH 15 AND FINISH THE 20-EPOCH HORIZON.
#
# The arm was paused after epoch 14 at the user's direction (2026-10-26) to free cards 4,5,7 for
# the two L2-SP lambda sweep arms on overfit30/val10. Nothing about the model changes: same
# --pose-perblock --pose-perblock-copy, same --train-only, same full_train fold, same valfull
# held-out set, same lr 0.01 flat after warmup, same seeds, same 20-epoch horizon. The pause is
# wall clock only.
#
# EVERY ARG BELOW IS COPIED FROM launch_pae_copy_full.sh UNCHANGED except --resume, so the
# resume guard should report NO drift and need no --resume-override. If it reports drift, STOP
# and read what differs rather than overriding it -- a resume that silently changes the arm makes
# epochs 1-14 and 15-20 two different experiments plotted as one curve.
#
# WHERE IT STOOD AT EPOCH 14: see training/runs/dntxmn/run.log. Through epoch 13 the arm ran
# paired dDockQ +0.0016 mean against shipped on valfull's 92 interfaces, rho 0.2478 against
# shipped 0.2218, val PAE 2.7580 (its low) and train PAE 2.6418.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
BUNDLE=training/runs/dntxmn/resume_ep14.pt
[ -f "$BUNDLE" ] || { echo "REFUSING: $BUNDLE does not exist -- epoch 14 never checkpointed"; exit 1; }
echo "RESUMING dntxmn (copy x PAE) from epoch 14 to the 20-epoch horizon"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "RESUME OF dntxmn AT EPOCH 15 TO FINISH THE 20-EPOCH HORIZON. Paused after epoch 14 at the user's direction to run two L2-SP lambda sweep arms (1e-3 then 1e-4, full confidence head, 15 epochs each, overfit30 -> val10) on cards 4,5,7. NOTHING ABOUT THIS ARM CHANGES: every argument is copied from launch_pae_copy_full.sh except --resume, so the guard should report no drift. copy x PAE remains the empty cell this arm was built to fill -- 4 untied weight sets, 40,960 parameters across 8 tensors, step 0 bit-identical to the share arm. WHERE IT STOOD AT EPOCH 13 on valfull 92 interfaces against shipped derived from this run own shards: paired dDockQ mean +0.0016 over epochs 1-13, rho 0.2478 vs shipped 0.2218, val PAE 2.7580 and train PAE 2.6418. The share control (ip8ga4 chain) finished 20 epochs at paired +0.0074 +- 0.0040 and rho 0.2550, and its val PAE bottomed at epoch 13 then rose to 2.7923 by epoch 20 -- so whether copy val PAE also turns after epoch 13 is the thing epochs 15-20 answer. VAL10 AND VALFULL ARE NOT COMPARABLE." \
  -- --pose-perblock --pose-perblock-copy \
  --resume "$BUNDLE" \
  --train-only 'linear_no_bias_d,linear_no_bias_d_wo_onehot' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 20 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
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
echo "resume_dntxmn_e20 returned $rc"
exit $rc
