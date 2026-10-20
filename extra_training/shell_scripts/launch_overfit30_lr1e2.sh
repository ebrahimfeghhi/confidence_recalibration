#!/bin/bash
# THE CLEAN ANCHOR: overfit-30 probe at lr 1e-2 under the current protocol.
#
# WHY THIS EXISTS when lr 1e-2 has been run several times before: none of those runs is usable as
# a reference. They had warmup 0 rather than 20; every masked arm was cut short by the early-stop
# bug fixed in 6cc69d30; they were scored on the pool they TRAIN on rather than on held-out
# structures; and dapghi and krah8z are byte-for-byte the same experiment, which made lr 1e-2 look
# like it had two independent supports when it has one. So there is nothing to compare a new arm
# against. This run is that reference.
#
# IDENTICAL TO THE lr 1e-1 PROBE (e4tvxe+f8rqck) IN EVERY RESPECT BUT THE RATE: same 30
# structures, same masked PAE-only loss, same accum 85, same crop, same seeds, warmup 20, fresh
# from the shipped head, early stop off, val10 eval every 2 epochs at budget 100. One variable.
#
# WHAT lr 1e-1 PRODUCED, for comparison at matched epochs: val10 0.2151/0.4444, 0.2420/0.3889,
# 0.2753/0.5556, 0.2222/0.3889, 0.2006/0.3333 at ep2..10 against a shipped head of 0.2626/0.5000 --
# one point above baseline that did not hold, then three declines. Pose-weighted PAE flat around
# 2.35-2.42 from ep4, against 2.19-2.21 for the old lr 1e-2 arms by ep7.
#
# THE EVALS ARE NOW CHEAPER: 8.9 min before the trainer passed --join-cache / --pair-iptm-only /
# --prefetch (c1338812), 6.7 while building the cache, 5.7 reading it. The cache lives beside the
# pool at training/val10/join_cache_shard{0,1,2}.json, so this run reads it from its first eval.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "CLEAN ANCHOR: overfit-30 at lr 1e-2, warmup 20, 50 epochs, masked PAE-only, early stop off, val10 eval every 2 epochs at budget 100, fresh from the shipped head. Identical to the lr 1e-1 probe but for the rate. Exists because no historical 1e-2 arm is comparable: warmup 0, cut short by the early-stop bug, scored on the training pool, and dapghi/krah8z are the same experiment." \
  -- --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 18 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/val10/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 100 \
  --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
