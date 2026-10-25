#!/bin/bash
# 8eiek8's CONFIGURATION ON THE FULL TRAINING FOLD. One variable: 30 structures -> 487.
#
# Usage:  ./shell_scripts/launch_pae_share_full.sh
#
# WHAT TRAINS: 2 tensors, 10,240 parameters -- linear_no_bias_d (256x39) and
# linear_no_bias_d_wo_onehot (256x1) -- applied at ALL FOUR pairformer blocks via
# --pose-perblock-share, which re-applies block 0's own pretrained pair verbatim and adds ZERO new
# parameters. Its own help calls this "the literal form of 'tie the weights so they are the same
# across all 4'". No orientation, no L2-SP, no ranking term.
#
# WHY PAE AND NOT THE RANKING LOSS. order_accuracy.py measures ordering with the SIGN only, so it
# cannot be faked by shrinking margins -- which is what the ranking arms were doing. On held-out
# val10 against shipped 0.5765 correct / 0.7469 DockQ-weighted on cross-seed pairs:
#
#   8eiek8  PAE, pose-share       ep1 0.5867/0.7485   peak ep48 0.6075/0.7665   ABOVE shipped
#   4y3ve6  ranking, pose-share   never beats shipped
#   x5frz6  ranking within-seed   ep11 0.5755/0.7239  BELOW shipped on all four figures
#
# PAE is the only objective measured here that improves pose ordering. The ranking arms compress
# spread instead: x5frz6 drove within-seed std 0.00593 -> 0.00147 and the top1-top2 gap 0.00205 ->
# 0.00015 while its held-out loss fell, which is margin collapse, not learning. PAE INFLATES
# spread (0.00591 -> 0.01030) and orders better.
#
# WHY MORE DATA AND NOT MORE CAPACITY. 8eiek8's held-out PAE fell at -0.00712/epoch for ten epochs,
# then 13x slower (-0.00053), then turned POSITIVE after epoch 30 (+0.00380), while training PAE
# kept falling 2.633 -> 2.326. Final train-val gap 0.479 and widening. That is overfitting at
# 10,240 parameters on 30 structures, and it is why DockQ peaked at epoch 10 and decayed after 28.
# Adding parameters would overfit sooner; adding data is the change that addresses it.
#
# PREDICTION: at 16x the data with the same capacity the epoch-10 ceiling largely lifts.
# FALSIFIER: held-out PAE flattens at the same rate on 487 structures, which would say the tied
# 10,240-parameter injection has simply learned all it can express and data was never the limit.
#
# SCORED ON valfull, 92 interfaces. val10 AND valfull ARE NOT COMPARABLE -- no number from any
# overfit30 arm transfers. The reporting derives shipped from this run's own eval shards.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching PAE pose-share on the FULL training fold, 15 epochs"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "8eiek8's CONFIGURATION ON THE FULL TRAINING FOLD, 15 epochs. ONE VARIABLE: 30 structures -> 487. WHAT TRAINS: 2 tensors / 10,240 parameters, linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1), applied at ALL FOUR pairformer blocks through --pose-perblock-share, which re-applies block 0's OWN PRETRAINED pair verbatim and adds ZERO new parameters -- the literal tying of one matrix pair across four depths. No orientation, no L2-SP, no ranking term, PAE only. WHY PAE: order_accuracy.py measures ordering by SIGN alone, so it cannot be faked by shrinking margins, and on held-out val10 against shipped 0.5765 correct / 0.7469 DockQ-weighted cross-seed, 8eiek8 (PAE, pose-share) runs 0.5867/0.7485 at epoch 1 to 0.6075/0.7665 at epoch 48 -- ABOVE shipped throughout -- while 4y3ve6 (ranking, same config) never beats shipped and x5frz6 (within-seed ranking) ends BELOW shipped on all four figures. The ranking arms were compressing spread rather than learning: x5frz6 drove within-seed std 0.00593 -> 0.00147 and the top1-top2 gap 0.00205 -> 0.00015 while its loss fell, which is margin collapse. PAE INFLATES spread (0.00591 -> 0.01030) and orders better. WHY DATA AND NOT CAPACITY: 8eiek8's held-out PAE fell -0.00712/epoch for ten epochs, then 13x slower at -0.00053, then turned POSITIVE after epoch 30 at +0.00380, while training PAE kept falling 2.633 -> 2.326 for a final train-val gap of 0.479 and widening -- overfitting at 10,240 parameters on 30 structures, and the reason DockQ peaked at epoch 10 and decayed after 28. More parameters would overfit sooner. PREDICTION: at 16x the data with the same capacity the epoch-10 ceiling largely lifts. FALSIFIER: held-out PAE flattening at the same rate on 487 structures, which would say the tied injection has learned all it can express and data was never the limit. CRITERION: selected DockQ and fraction above 0.23 over the 92 valfull interfaces against shipped DERIVED FROM THIS RUN'S OWN SHARDS; held-out PAE is the secondary read and the quantity the prediction is about. VAL10 AND VALFULL ARE NOT COMPARABLE -- no number from any overfit30 arm transfers, and shipped 0.2909 / 0.5556 are val10 figures that do NOT apply here. CONTROLS: 8eiek8 (this arm on 30 structures, 50 epochs) and zg5fnw (full fold, per-block COPIES with orientation and an anchor, ranking loss). TEN-EPOCH MINIMUM before any trend claim." \
  -- --pose-perblock --pose-perblock-share \
  --train-only 'linear_no_bias_d,linear_no_bias_d_wo_onehot' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 15 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 3 \
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
echo "launch_pae_share_full returned $rc"
exit $rc
