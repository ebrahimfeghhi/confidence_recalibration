#!/bin/bash
# The SHIPPED head's val10 PAE: epoch 0, the reference both trained points are measured against.
# Without it, "ep1 2.8290 -> ep142 4.2639" cannot say whether training ever helped held-out PAE or
# only ever hurt it. No --resume, so --loss-only measures the checkpoint as built.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
CUDA_VISIBLE_DEVICES=${1:-7} LAYERNORM_TYPE=openfold \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_train.py \
  --loss-only \
  --fold validation/fold --labels validation/fold/labels \
  --crop-margin 10 --crop-contact 5 --crop-size 3400 \
  --minimal-items --alignment-preflight training/val10/alignment_preflight.json \
  --manifest-selection training/val10/structures.csv \
  --epochs 1 --accum 85 --nsample 5 --save-every 0 --warmup-steps 0 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 \
  --pae-scored-pairs training/val10/pool.csv --pae-only \
  --shuffle structure --seeds 400000-400009 \
  --blocks-per-ckpt 1 \
  --indices validation/prep/job_indices.csv \
  --bio-dir validation/prep/job_bioassembly \
  --mmcif-dir validation/prep/job_cifs \
  --out training/val10/lossonly_shipped
