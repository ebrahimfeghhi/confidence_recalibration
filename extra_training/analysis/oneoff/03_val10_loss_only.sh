#!/bin/bash
# Pose-weighted PAE loss of a saved head on the HELD-OUT val10 structures.
# Identical crop, mask, loss stub and weights to the anchor's training loss -- same code, --loss-only.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUNDLE=$1; TAG=$2; CARD=${3:-7}
CUDA_VISIBLE_DEVICES=$CARD LAYERNORM_TYPE=openfold \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_train.py \
  --loss-only \
  --resume "$BUNDLE" --resume-allow-drift \
  --fold validation/fold --labels validation/fold/labels \
  --crop-margin 10 --crop-contact 5 --crop-size 3400 \
  --minimal-items --alignment-preflight training/val10/alignment_preflight.json \
  --manifest-selection training/val10/structures.csv \
  --epochs 1 --accum 85 --nsample 5 --save-every 0 --warmup-steps 0 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 \
  --pae-scored-pairs training/val10/pool.csv --pae-only \
  --shuffle structure --seeds 400000-400029 \
  --blocks-per-ckpt 1 \
  --indices validation/prep/job_indices.csv \
  --bio-dir validation/prep/job_bioassembly \
  --mmcif-dir validation/prep/job_cifs \
  --out "training/val10/lossonly_$TAG"
echo "exit $?"
