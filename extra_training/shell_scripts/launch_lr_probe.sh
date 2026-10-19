#!/bin/bash
# The first arm of the baseline sweep: how high can the learning rate go and stay stable.
#
# TRAINING SET: training/solved_cut/structures.csv -- the 385 of 496 structures that are NOT
# already solved, minus the 8 of the historical skip list that fall inside it, so 377 structures
# and 814 interfaces. "Solved" means every interface of that structure has margin < 0.05 AND a
# selected DockQ >= 0.23: ipTM already picks a near-best, acceptable pose, so there is nothing for
# the head to learn there. 111 structures / 197 interfaces dropped, a 22.4% cut.
#
# NOTE THE ASYMMETRY THIS CREATES. Those 111 structures are exactly where a trained head could
# REGRESS while improving elsewhere -- the epoch-6 head once gained 0.0014 mean DockQ while losing
# 0.0041 frac_acc. They are out of training and must therefore be scored at least once before any
# result from this arm is believed.
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
# So --eval-every 1122 is validation every 5 epochs and --save-every 224 is a checkpoint every
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
  --cards 4,5,7 --note "BASELINE SWEEP arm 1: lr 5e-2 on the 377-structure solved-cut set, masked PAE-only, 50 epochs, early stop off. Validation every 5 epochs at budget 100, checkpoint every epoch. Testing the highest stable learning rate; drop to 1e-2 if unstable. Warmup 200 steps, unlike earlier arms." \
  -- --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/solved_cut/alignment_preflight.json \
  --manifest-selection training/solved_cut/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.05 --warmup-steps 200 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 224 \
  --pae-scored-pairs training/solved_cut/pool.csv --pae-only \
  --eval-every 1122 --eval-budget 100 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
