#!/bin/bash
# Continue the lineage from jy6sah's ep137 bundle to ep200.
#
# WHY ep137 AND NOT ep150. The user asked to pause at 2:12 PM PT so results.csv could be
# deduplicated before the next run's train evals read it, rather than waiting for ep150 and
# letting the armed queue do it. jy6sah was stopped mid-epoch-138, so ep137 is the last complete
# epoch and its bundle is the resume point; epoch 138's partial work is lost, about 5 minutes.
#
# results.csv WAS DEDUPLICATED IMMEDIATELY BEFORE THIS RUN. phase_j_drop_superseded_rows.py
# --in-place resolved 8,645 of 8,655 duplicated rows left by the 2026-10-15 refold, backing the
# original up to results.csv.bak.20261019-141255. 4 of the 77 train-eval interfaces are affected
# -- 35uj_AC, 35uj_BC, 8dfh_AB, 8dfh_AC -- so this run's evals on those four may differ from
# jy6sah's. That is a correction: selecting on the resolved file reproduces ceiling.json's
# 0.0420 / 0.0000 over the 77 exactly, which the duplicated file did not.
#
# A CONTINUATION, not a fork: same lr 1e-2, same mask, same PAE-only loss, same seed/accum/nsample,
# so the resume guard passes without --resume-allow-drift and the curve joins point for point.
# Early stop stays OFF.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "CONTINUATION of jy6sah from ep137 to ep200: lr 1e-2, masked PAE-only, early stop off. jy6sah was paused at ep138 so results.csv could be deduplicated first; 4 of the 77 train-eval interfaces may read differently from here on. ep136 set the lineage best at 0.2386 / 0.3896 (30 of 77) and the ep122-129 PAE shelf broke to a 1.5032 low at ep131." \
  -- --resume training/runs/jy6sah/resume_ep137.pt \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 200 --lr 0.01 --warmup-steps 0 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 17 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/overfit30/pool.csv --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "launcher returned $?"
