#!/bin/bash
# Is activation checkpointing numerically inert, and what does it cost?
#
# WHY. blocks_per_ckpt=1 recomputes all four pairformer blocks' forwards during backward. Epoch 1
# of x3k2m4 measured gpu_backward 20.7 min against gpu_forward 6.7 -- consistent with recompute
# plus true backward -- while the largest crop over 4,860 units was 801 tokens against a 3,400
# cap. So there is no large-crop case for checkpointing to protect and the recompute looks like
# pure waste. "Looks like" is not a measurement, hence this.
#
# TWO STRUCTURES: 8ut2 (801 kept tokens, the p100 of the whole training set) and 8h8q (332, the
# median). The memory question has to be asked at the LARGEST crop, not the mean.
#
# IDENTICAL SEED, IDENTICAL MANIFEST, IDENTICAL EVERYTHING but --blocks-per-ckpt. Checkpointing is
# supposed to be a memory-for-compute trade with no effect on the math, so per-pose losses should
# agree to floating-point reassociation and grad_norm should track. A disagreement beyond that
# means it is NOT inert and the flag cannot be flipped on a live experiment.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
for BPC in 1 0; do
  OUT=training/ckpt_probe/bpc$BPC
  rm -rf "$OUT"
  echo "== blocks_per_ckpt=$BPC =="
  CUDA_VISIBLE_DEVICES=7 LAYERNORM_TYPE=openfold $PY -I phases/phase_k_train.py \
    --fold training/fold \
    --crop-margin 10 --crop-contact 5 --crop-size 3400 \
    --minimal-items --alignment-preflight training/ckpt_probe/alignment_preflight.json \
    --manifest-selection training/ckpt_probe/structures.csv \
    --epochs 1 --lr 0.01 --warmup-steps 0 --seed 0 --accum 20 --nsample 5 \
    --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 \
    --pae-scored-pairs training/ckpt_probe/pool.csv --pae-only \
    --shuffle structure --seeds 500000-500009 \
    --blocks-per-ckpt $BPC --save-every 0 --mem-log \
    --require-replay training/ft3_pilot_replay.log --allow-code-drift \
    --out "$OUT" > "$OUT.log" 2>&1
  echo "  exit $?"
done
