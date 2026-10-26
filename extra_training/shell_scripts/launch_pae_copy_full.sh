#!/bin/bash
# THE EMPTY CELL: four UNTIED injection weight sets under PAE alone.
#
# Usage:  ./shell_scripts/launch_pae_copy_full.sh
#
# The grid is {share, copy} x {ranking, PAE}. share x ranking is 4y3ve6, share x PAE is 8eiek8 and
# j583i8/uhyypy, copy x ranking is thirty runs. copy x PAE has never been run.
#
# --pose-perblock-copy gives each of the four sites its own weight set, every one initialised FROM
# BLOCK 0 and asserted equal to it bit for bit, so step 0 matches the share arm exactly and the
# single variable is capacity: 40,960 parameters instead of 10,240.
#
# --train-only is UNCHANGED on purpose. It matches by unanchored regex search, so the pattern
# linear_no_bias_d already reaches pose_perblock_linear_no_bias_d.*. Check the launch line: it
# must say 8 tensor(s) / 40,960 parameters. If it says 2 / 10,240 the copies are frozen and the
# run is the share arm with extra untrained weights.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching PAE pose-COPY (4 sites, 4 untied weight sets) on the FULL training fold, 20 epochs"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "ONE VARIABLE AGAINST j583i8/uhyypy: THE FOUR INJECTION SITES GET FOUR INDEPENDENT WEIGHT SETS INSTEAD OF ONE. --pose-perblock-copy initialises every copy from block 0 own pretrained linear_no_bias_d and linear_no_bias_d_wo_onehot, and install() ASSERTS the copies equal block 0 to the bit -- so STEP 0 IS IDENTICAL to the share arm and the only thing that differs is capacity, 40,960 parameters across 8 tensors against 10,240 across 2. Everything else is held: PAE only, full training fold, same selection and skip list, same crop, same lr 0.01 flat after warmup (lr_final 0.0 is NO DECAY), same seeds, same valfull held-out set. --train-only linear_no_bias_d,linear_no_bias_d_wo_onehot needs no change: the match is an unanchored regex search, so it reaches pose_perblock_linear_no_bias_d.* as well; the launch log must print 8 tensor(s) / 40,960 parameters and NOT 2 / 10,240. WHY THIS CELL IS EMPTY AND WORTH FILLING: all 30 runs that have ever used --pose-perblock-copy carried the RANKING loss (24u342, 8fkrsu, zg5fnw, ykhvzs, hronsr, oe6sru and 24 others), and every PAE-only arm has been share. The ranking arms were later shown to be lowering their loss by MARGIN COLLAPSE rather than by ordering -- x5frz6 drove within-seed std 0.00593 -> 0.00147 and the top1-top2 gap 0.00205 -> 0.00015 while its loss fell -- so copy has never been tried under the one objective measured to improve sign-only ordering. copy x PAE is the empty cell. WHAT THE CONTROL SAYS: uhyypy (share x PAE, this arm at 1x capacity) sits LEVEL with shipped on valfull through epoch 9 -- paired dDockQ +0.0000 +- 0.0021 at epoch 6 and +0.0007 +- 0.0025 at epoch 8 -- while reshuffling 37 of 92 interface selections. Its held-out PAE also flattened early, -0.01425/epoch over epochs 1-5 then -0.00088 over 5-8, 16x slower. PREDICTION: if the flattening is a CAPACITY limit, 4x the weight sets should extend the descent past epoch 5; if it is a DATA or OBJECTIVE limit, copy flattens at the same epoch and merely overfits sooner, and the tied injection was never the bottleneck. That is the falsifier and it is readable by epoch 10. CRITERION: selected DockQ and fraction above 0.23 over valfull 92 interfaces against shipped DERIVED FROM THIS RUN OWN SHARDS. VAL10 AND VALFULL ARE NOT COMPARABLE. TEN-EPOCH MINIMUM before any trend claim." \
  -- --pose-perblock --pose-perblock-copy \
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
echo "launch_pae_copy_full returned $rc"
exit $rc
