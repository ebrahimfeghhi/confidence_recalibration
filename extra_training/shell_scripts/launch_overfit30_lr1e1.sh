#!/bin/bash
# OVERFIT-30 SPEED PROBE: how fast can 30 structures be fitted, starting at lr 1e-1.
#
# WHAT THIS IS FOR. Not for picking the baseline's learning rate -- the recipe that memorises 30
# structures fastest is not necessarily the one that trains 488 well, because memorisation rewards
# aggressive settings that need not transfer. This measures the SPEED CEILING on the probe.
#
# lr 1e-1 is 10x the 1e-2 the existing lineage ran at, and 2x the 5e-2 that ovqdv8 showed stable
# on the full fold (max grad_norm 7.77 against the 25.0 clip over 128 steps). Watch for grad_norm
# reaching the clip, nonfinite_grad rows, step_ok == 0, or epoch PAE rising.
#
# EVERYTHING ELSE MATCHES THE EXISTING LINEAGE so the curves are comparable point for point:
# same 30 structures, same pool, same accum 85, same crop, same masked PAE-only loss, same seeds.
# The two deliberate differences are lr and the 200-step warmup, which the lineage ran at 0 --
# at 1e-1 an un-warmed first step moves every weight by ~0.1 against weights of order 0.01-0.1.
#
# FRESH FROM THE SHIPPED HEAD, not resumed: the question is time-to-fit from the start.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "OVERFIT-30 SPEED PROBE at lr 1e-1: 30 structures, masked PAE-only, 60 epochs, warmup 200, early stop off, train-eval every 2 epochs at budget 10. Fresh from the shipped head. Measuring how fast the probe can be fitted, not picking the baseline lr." \
  -- --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 60 --lr 0.1 --warmup-steps 200 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 18 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/overfit30/pool.csv --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
