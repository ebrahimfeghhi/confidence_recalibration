#!/bin/bash
# EXERCISE THE ROW-WEIGHT GLUE ON REAL POSES before committing 60 epochs to it.
#
# WHY. test_pae_row_weight.py covers the pure functions in phases/pae_row_weight.py, which is the
# part that was already correct. The bug that review caught -- weights cast to the bool dtype of
# PAELoss's pair_mask, making the whole term a silent no-op -- lived in the GLUE: the forward
# wrapper, the mask hook, the per-pair block construction. None of that is importable, so none of
# it is covered, and a no-op does not crash. It logs itself as working for sixty epochs.
#
# WHAT IT ASSERTS, all from the artifacts of a few real optimiser steps:
#   1. the banner reports the builder's steps/epoch, not len(flat)//world
#   2. the row-weight banner fires at all (the flag reached the hook)
#   3. pae_rows_rank*.csv has rows with a finite ESS strictly below the row count
#   4. pae_loss and pae_loss_unweighted DIFFER -- the no-op check, on real data
#   5. nothing in the log says the diagnostic failed
#
# A SMALL REAL MANIFEST, NOT --limit-units. The first version truncated the unit list with
# --limit-units 12, which changes the manifest DIGEST -- it is computed after the limit is applied
# -- so the alignment preflight marker for the 368-structure manifest no longer matched and the
# preflight guard refused the run. The guard was right and the test was wrong. structures_smoke.csv
# is the four smallest winnable structures with their own derived marker, so every guard sees a
# population it has evidence for.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
OUT=training/runs/_smoke_rowweight
rm -rf "$OUT"; mkdir -p "$OUT"
LOG="$OUT/smoke.log"

echo "smoke: a few steps with --pae-row-frac 0.2, logging every pose"
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES=4 RANK=0 WORLD_SIZE=1 \
$PY_BIN -I phases/phase_k_train.py \
  --out "$OUT" \
  --pose-perblock --pose-perblock-copy --frame-direction \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_smoke.json \
  --manifest-selection training/full_train/structures_smoke.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 1 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --lr-final 1e-4 --lr-decay-shape cosine \
  --lr-decay-start-epoch 20 --lr-decay-end-epoch 50 \
  --pae-row-frac 0.2 --pae-row-mass 0.8 --pae-row-floor 10 --pae-row-log-every 1 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --pae-scored-pairs training/full_train/pool_smoke.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 1 --prefetch-workers 1 \
  --blocks-per-ckpt 0 --max-rss-gb 80 --preload-min-free-gb 200 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift \
  > "$LOG" 2>&1
rc=$?
echo "  trainer exited $rc; checking the artifacts"
$PY_BIN -I run_logging/check_smoke_rowweight.py "$OUT" "$rc"
