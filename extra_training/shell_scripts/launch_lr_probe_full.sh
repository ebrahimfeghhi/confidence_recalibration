#!/bin/bash
# The first arm of the baseline sweep: how high can the learning rate go and stay stable.
#
# TRAINING SET: training/full_train/structures.csv -- the FULL training fold. 491 structures after
# the historical skip list, minus 7stz, 8gpu and 9blx, which have no interface in results.csv: a
# structure with no scored chain pair has nothing for --pae-scored-pairs to mask, so its poses
# would train the UNMASKED PAE term, which is the one thing the mask exists to prevent. 488
# structures, 977 interfaces.
#
# WHY NOT THE SOLVED-CUT SUBSET. An earlier version of this arm trained on the 377 structures that
# are not already solved, dropping the 111 whose every interface has margin < 0.05 and a selected
# DockQ >= 0.23. That cut does not survive scrutiny: margin is a property of the RANKING, and the
# loss here is PAE cross-entropy. A structure whose ipTM happens to order poses correctly can
# still have badly calibrated PAE, so there is real signal in those 111 -- and it is the signal
# that keeps the head from drifting on the cases it already gets right. The cut bought no speed
# either, because cost per optimiser step is nearly flat in set size: at a fixed step budget the
# full fold is the same wall clock. It also removed the obligation to score an excluded set.
#
# WHAT IS BEING MEASURED: stability at lr 5e-2, which is 5x the highest rate run so far. The rule
# the user set is to take the highest learning rate at which training is stable, dropping to 1e-2
# if 5e-2 is not. Watch for grad_norm reaching the 25.0 clip, nonfinite_grad rows, step_ok == 0,
# or the pose-weighted epoch PAE rising rather than falling.
#
# WARMUP IS NON-ZERO HERE, unlike every earlier arm. 200 steps is about one epoch. A fresh run at
# 5x the lr is exactly where an un-warmed AdamW blows up on its first few steps, and that failure
# would say nothing about whether 5e-2 is a usable rate. It does change what "stable" means --
# stable in the steady state rather than from step 1 -- so if you want the stricter test, set this
# to 0.
#
# CADENCE: 377 structures x 10 seeds x 5 poses = 18,850 poses/epoch at accum 84 = 224 steps/epoch.
# So --eval-every 1452 is validation every 5 epochs and --save-every 290 is a checkpoint every
# epoch. Resume bundles are written every epoch regardless of --save-every.
#
# BUDGET: 50 epochs = 11,220 optimiser steps, ~31 h at the 360 steps/h measured on cached data,
# plus 10 validation passes at ~25 min sharded over the three cards = ~35 h. THE I/O RISK IS
# UNPRICED: 377 structures x 10 seeds x 205 MB is ~770 GB of trunks against an 80 GB/rank cache,
# so every epoch re-reads from disk where overfit30's 60 GB did not. If s/pose comes in above the
# 0.30 measured on overfit30, that 35 h estimate grows in proportion.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "BASELINE SWEEP arm 1: lr 5e-2 on the 488-structure full train set, masked PAE-only, 50 epochs, early stop off. Validation every 5 epochs at budget 100, checkpoint every epoch. Testing the highest stable learning rate; drop to 1e-2 if unstable. Warmup 200 steps, unlike earlier arms." \
  -- --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight.json \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.05 --warmup-steps 200 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 290 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --eval-every 1452 --eval-budget 100 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
