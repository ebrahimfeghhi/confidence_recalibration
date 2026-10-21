#!/bin/bash
# Continue the lr 1e-2 anchor from ep21 to ep200, WITH NO VALIDATION DURING TRAINING.
#
# WHY NO EVAL. val10's ten evals on this arm ran 0.2098 to 0.3045 with the shipped head's 0.2626
# sitting inside that spread -- on 18 interfaces the metric carries roughly +/-0.05 of noise and
# the arm never left it. At 5.1 min each, every-2-epochs was spending 2.1 h per 50 epochs to
# resolve nothing. A resume bundle is written EVERY epoch and carries all 248 head tensors, so
# any epoch can be scored afterwards for the same 5.1 min -- and only the epochs worth scoring.
#
# A CONTINUATION, not a fork: same lr, warmup, accum, nsample, seed, mask, loss and manifest, so
# the resume guard passes without --resume-allow-drift and the curve joins point for point.
#
# The val10 plot will show only the PAE panel from here, because nothing writes "epoch N TRAIN:"
# lines without --train-eval-pool. That is expected, not a failure.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "CONTINUATION of the lr 1e-2 anchor 335psh from ep21 to ep200, NO validation during training. The user asked for 200 epochs with scoring deferred to the end, after val10's ten evals spanned 0.2098-0.3045 around a 0.2626 baseline. Per-epoch resume bundles make any epoch scorable later." \
  -- --resume training/runs/335psh/resume_ep21.pt \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 200 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 18 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
