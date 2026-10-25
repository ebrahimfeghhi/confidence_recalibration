#!/bin/bash
# THE BASELINE ARM: full-head FT3 on the same 30 structures, against val10, 30 epochs.
#
# ONE VARIABLE AGAINST THE POSE-FOCUSED ARM (ggfhj5+zcuiab). Everything is identical -- same 30
# structures, same seeds, same crop 10/5 at 3400, same accum 85 / nsample 5 / K=6, same lr 1e-2,
# same warmup 20, same masked PAE-only loss, same per-epoch val10 PAE and DockQ, same fresh start
# from the shipped head, checkpointing off, single-worker prefetch -- EXCEPT that --train-only is
# absent, so all 19.8M head parameters train instead of 26,624 (0.135%).
#
# WHY NOT JUST USE 2u8tf4, WHICH IS ALREADY THIS ARM. Because it ran with NO validation during
# training. Its val10 PAE exists at a handful of epochs only, backfilled afterwards from resume
# bundles (ep1 2.8290, ep5 2.7231 best, ep142 4.2639), and its DockQ at one. You cannot plot that
# against a per-epoch curve point for point, and the interesting region -- epochs 1-30, where the
# pose-focused arm converged and 2u8tf4 turned -- is exactly where 2u8tf4 has almost no points.
# 30 epochs reproduces it properly at ~8 min/epoch, about 4 hours.
#
# WHAT THE COMPARISON IS FOR. The pose-focused arm, 21 epochs:
#   val10 PAE  2.8063 (ep2) -> 2.7521 (ep21), converged by ep12, ten flat epochs at 2.751-2.764
#   train PAE  2.4282 -> 2.2329, -8.0%
#   val10 DockQ below the 0.2909 baseline in 16 of the last 16 epochs; frac_acc 9/18 against 10/18
#     untrained for most of the last seven
# So restriction prevented the overfit collapse and left selection slightly WORSE than untrained.
# This arm answers what the unrestricted head does over the same 30 epochs with the same
# instrumentation -- and whether its brief val10 advantage (2.7231 at ep5) is real and reproducible
# or was an artifact of a sparsely sampled curve.
#
# EXPECT IT TO OVERFIT, AND SAY SO IN ADVANCE: 2u8tf4 bottomed at epoch 5 and rose from epoch 9.
# If this reproduces that, the pose-focused arm's flat curve is a genuine stability win. If it does
# NOT -- if the full head also plateaus -- then 2u8tf4's collapse was about its 142-epoch length
# and the restriction bought nothing at all, including stability.
#
# Fresh from the shipped head (no --resume). 30 structures x floor(10/3) = 90 group(s) / K=6 = 15
# step(s)/epoch, 90 unit(s)/rank, 1,350 poses. Resume bundles are 226 MB here, not 325 KB, because
# 19.8M parameters and their AdamW moments all change: ~6.8 GB over 30 epochs.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "BASELINE ARM: full-head FT3 on the 30-structure set against val10, 30 epochs, fresh from the shipped head. All 19.8M head parameters train. ONE VARIABLE against the pose-focused arm ggfhj5+zcuiab (26,624 params, 0.135%) -- every other setting identical. Re-run rather than reusing 2u8tf4 because that arm had no validation during training and its val10 curve is only a handful of backfilled points, none of them in the epoch 1-30 region that matters." \
  -- --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 30 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
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
