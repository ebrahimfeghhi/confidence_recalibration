#!/bin/bash
# Resume the orange lineage (pykssn -> x3888d -> tgfiqx -> 89uwjx -> owdyyp) from its LAST state,
# ep57, at double the learning rate and with the early-stop rule OFF.
#
# FROM ep57, NOT the best-DockQ ep46. The DockQ series over ep38-56 reads .1526 .1512 .1312 .1243
# .1769 .1067 .1174 .1487 .1490 .1514 -- the .1769 at ep46 is one point above a band sitting at
# ~.14, not a peak the run climbed to and fell from, so treating it as the better head is reading
# noise. ep57 has the lower training loss and is the actual continuation.
#
# lr 5e-3 -> 1e-2 is real drift, so --resume-allow-drift is required and the run is recorded as a
# FORK of the bundle. --early-stop-patience is left at 0 (disabled) deliberately: the rule ended
# this lineage at ep57 while its loss was still falling, and the point of this arm is to find where
# it actually tops out.
#
# CANCEL: ps -eo pid,cmd | grep phase_k_train | grep -v " grep ", then kill the literal pid of the
# torchrun parent -- NEVER pkill a pattern, which matches the shell doing the killing.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "FORK of the orange lineage at ep57: lr 1e-2 (was 5e-3), early stop OFF, 100 epochs. Masked PAE-only. Tests where the arm tops out now that the stop rule is not cutting it short, and whether 2x lr escapes the ~1.99 plateau." \
  -- --resume training/runs/owdyyp/resume_ep57.pt --resume-allow-drift \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 100 --lr 0.01 --warmup-steps 0 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 17 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/overfit30/pool.csv --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
