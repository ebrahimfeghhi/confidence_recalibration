#!/bin/bash
# OVERFIT-30 SPEED PROBE at lr 1e-1, scored on HELD-OUT validation structures.
#
# TRAINS on the 30-structure overfit30 set. SCORES every 2 epochs on training/val10/pool.csv --
# the 10 validation structures with the most total headroom, 18 interfaces. That is the change
# from q7nyu6, which scored the training pool and so could only show memorisation.
#
# WHY THESE 10. Ranked by oracle-minus-selected summed over each structure's interfaces. Shipped
# head 0.2626 / 0.5000, oracle 0.6662 / 1.0000, headroom 0.4036 -- comparable to overfit30's
# 0.4678, so an improvement is as visible here as on the training probe. Every interface has an
# acceptable pose available and the shipped head finds 9 of 18, so 9 are winnable.
#
# READ THE RESULT WITH THIS IN MIND: 9qrv and 9bx7 are two of the three interfaces that carried
# 85-88% of every earlier arm's validation gain. Ranking by headroom necessarily selects them, so
# a gain here is partly those same interfaces again. Decompose per interface before believing it.
#
# WARMUP 20, NOT 200. q7nyu6 was launched with 200, which at 18 steps/epoch is ELEVEN EPOCHS of
# ramp -- a fifth of the run before the learning rate even reaches 1e-1. 200 was carried over from
# the full-fold arm where 290 steps/epoch made it 0.7 epochs. 20 is the trainer's own default and
# its help text says why: "SHORT BECAUSE THE RUN IS SHORT ... 20 is 6.8%".
#
# THE EVAL COSTS ~4.6 min SHARDED (10 structures x 100 seeds = 1,000 units at 0.82 s/unit, split
# over three cards) against ~10.8 min for two epochs. If that 42% overhead is too much,
# --train-eval-budget 50 halves it -- at the cost of changing the task, since selecting among 250
# poses is not the same problem as selecting among 500.
#
# --train-eval-fold and --train-eval-seed-base are new. Without them the train-eval path derives
# --fold, --results, --roles and --seed-base from the run's own --fold, so it could only ever
# score the training fold; and --train-eval-budget defaults to the count of THIS run's --seeds,
# which is 10, and would have silently scored a tenth of a 100-seed fold.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "OVERFIT-30 at lr 1e-1, warmup 20, 50 epochs, masked PAE-only, early stop off. Scored every 2 epochs on the 10 HIGHEST-HEADROOM VALIDATION structures (18 interfaces, shipped 0.2626/0.5000, oracle 0.6662/1.0000) at budget 100 instead of on the training pool. Fresh from the shipped head. Replaces q7nyu6, whose 200-step warmup was 11 epochs at 18 steps/epoch." \
  -- --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.1 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
