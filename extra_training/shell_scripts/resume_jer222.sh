#!/bin/bash
# RESUME jer222 (FULL HEAD + POSE COPY, L2-SP 1e-2) FROM EPOCH 7 TO THE 20-EPOCH HORIZON.
#
# WHY IT WAS PAUSED: to score the three completed arms on the TRAINING pool, which needs all three
# cards. The pause was at the user's explicit direction ("You can pause this sweep to do that").
#
# EVERY FLAG IS COPIED FROM launch_fullhead_l2sp_1e2.sh UNCHANGED except --resume. The L2-SP
# anchor is NOT rebuilt: phase_k_train copies l2sp_anchor.pt from beside the resume bundle into
# the new run directory, because a resumed run's weights are not the shipped ones and snapshotting
# them would silently re-anchor the penalty to a drifted head. The log must say
# "anchor INHERITED from .../jer222/l2sp_anchor.pt".
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
BUNDLE=training/runs/jer222/resume_ep7.pt
[ -f "$BUNDLE" ] || { echo "REFUSING: $BUNDLE does not exist"; exit 1; }
[ -f training/runs/jer222/l2sp_anchor.pt ] || { echo "REFUSING: no l2sp_anchor.pt beside the bundle; a resume would re-anchor to drifted weights"; exit 1; }
echo "RESUMING jer222 from epoch 7 (full head, L2-SP 1e-2, pose copy)"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "RESUME OF jer222 AT EPOCH 8 TO FINISH THE 20-EPOCH HORIZON. Paused after epoch 7 at the user's direction so the three completed arms could be scored on the TRAINING pool, which needs all three cards. Every other flag is identical to launch_fullhead_l2sp_1e2.sh; the L2-SP anchor is inherited from jer222 rather than re-snapshotted, because a resumed run's weights are not the shipped ones. VAL10 AND VALFULL ARE NOT COMPARABLE." \
  -- --pose-perblock --pose-perblock-copy \
  --resume "$BUNDLE" \
  --l2-sp 1e-2 --l2-sp-exclude 'linear_no_bias_d' \
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
  --prefetch-units 2 --prefetch-workers 4 \
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
echo "resume_jer222 returned $rc"
exit $rc
