#!/bin/bash
# RESUME p5yoza (ocqtyt 27->50, full head, no L2, no pose injection) FROM WHERE IT WAS PAUSED.
#
# It was stopped at epoch 32 so the ABAG-Rank PAE dump could use all three cards. Every flag is
# copied from resume_ocqtyt_26_to_50.sh; only --resume moves to the newer bundle.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
BUNDLE=$(ls -t training/runs/p5yoza/resume_ep*.pt 2>/dev/null | head -1)
[ -n "$BUNDLE" ] || { echo "REFUSING: no resume bundle in training/runs/p5yoza"; exit 1; }
echo "RESUMING p5yoza from $(basename $BUNDLE) to the 50-epoch horizon"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "RESUME OF p5yoza (itself ocqtyt 27->50) AFTER THE ABAG-RANK PAE DUMP TOOK THE CARDS. Full confidence head, NO L2-SP, NO pose injection, full training set. Training settings unchanged from resume_ocqtyt_26_to_50.sh; only --resume moves to the newer bundle. The DockQ eval stays on valfull so every epoch is comparable to shipped 0.5830. TEN-EPOCH MINIMUM does not apply -- the user retracted it." \
  -- --resume "$BUNDLE" --resume-allow-drift \
  --resume-note "continue p5yoza after the ABAG-Rank dump; no training setting changes" \
  --fold training/fold --labels training/fold/labels \
  --crop-margin 10 --crop-contact 5 --crop-size 3400 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight.json \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --indices training/prep/job_indices.csv \
  --bio-dir training/prep/job_bioassembly \
  --mmcif-dir training/prep/job_cifs \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 \
  --save-every 244 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/val10/structures.csv \
  --val-loss-pool training/val10/pool.csv \
  --val-loss-preflight training/val10/alignment_preflight.json \
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
echo "resume_p5yoza returned $rc"
exit $rc
