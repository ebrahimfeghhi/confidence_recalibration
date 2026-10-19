#!/bin/bash
# Build the alignment marker for training/solved_cut/structures.csv. CPU only, ~2.3 h at the
# 22.4 s/structure measured on two structures. Wrapped in a script because direct backgrounded
# python kept dying before its first structure, where `bash script.sh` under setsid survives.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
export LAYERNORM_TYPE=openfold
exec /home/ebrahim/Protenix/.venv/bin/python -I -u phases/phase_k_alignment_preflight.py \
  --out training/solved_cut/alignment_preflight.json \
  --manifest-selection training/solved_cut/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --seeds 500000-500009
