#!/bin/bash
# RESUME ij24wr (full head + pose copy, L2-SP 1e-3 on the 246 non-pose tensors) AFTER ITS EPOCH 1.
#
# Usage: ./shell_scripts/resume_ij24wr.sh
#
# WHY THERE IS A PAUSE AT ALL: epoch 1 was stopped at the user's direction so the ipTM join
# verification could run on genuinely free cards, which is the only way its per-pose ipTM
# comparison means anything. Nothing about this arm changes -- every argument is copied from
# launch_fullhead_l2sp_posefree.sh except --resume, so the resume guard should report NO drift and
# need no --resume-override. If it reports drift, STOP and read what differs: a resume that
# quietly changes the arm makes epoch 1 and epochs 2-20 two experiments on one curve.
#
# THE ANCHOR IS INHERITED. A resumed --l2-sp run looks for l2sp_anchor.pt in its own fresh
# directory, and launch.py refuses a non-empty one, so there was no way to put it there by hand --
# which made an --l2-sp arm unresumable. The trainer now inherits it from beside the --resume
# bundle and copies it in, so the record travels with the run. The launch log must say
# "anchor INHERITED from training/runs/ij24wr/l2sp_anchor.pt".
#
# THE SPEED FLAGS STAY ON: --numa-map 0:0,1:0,2:1 and --prefetch-workers 4. Epoch 1 already
# measured them on free cards; epochs 2-20 simply keep them.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
BUNDLE=training/runs/ij24wr/resume_ep1.pt
[ -f "$BUNDLE" ] || { echo "REFUSING: $BUNDLE does not exist -- epoch 1 never checkpointed"; exit 1; }
echo "RESUMING ij24wr from epoch 1 to the 20-epoch horizon (full head, L2-SP 1e-3, pose copy)"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "RESUME OF ij24wr AT EPOCH 2 TO FINISH THE 20-EPOCH HORIZON. Epoch 1 was stopped at the user's direction so the ipTM join verification could run on genuinely free cards -- its --compare checks PER-POSE ipTM against the eval's own rows, which only means anything if no other job is sharing a GPU. NOTHING ABOUT THIS ARM CHANGES: every argument is copied from launch_fullhead_l2sp_posefree.sh except --resume, so the guard should report no drift. Full confidence head trains, the 8 pose tensors stay OUT of the anchor and the other 246 are anchored at 1e-3 * ||W - W_shipped||^2; the launch log must print 8 excluded / 246 anchored and must say the anchor was INHERITED from ij24wr. Speed flags stay on: --numa-map 0:0,1:0,2:1 and --prefetch-workers 4, both measured on epoch 1 with the cards to themselves." \
  -- --pose-perblock --pose-perblock-copy \
  --resume "$BUNDLE" \
  --l2-sp 1e-3 --l2-sp-exclude 'linear_no_bias_d' \
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
echo "resume_ij24wr returned $rc"
exit $rc
