#!/bin/bash
# check_alignment for the 10 val10 structures: the pose and the label describe the same molecule.
#
# NEEDED, not ceremonial. --crop-margin requires --minimal-items, --minimal-items requires this
# marker, and without the crop the held-out loss covers a different atom set than the training loss
# and the comparison is fake. These 10 structures had never been checked -- this is not a re-check.
# 6.4 s/structure, CPU only, so it runs beside a training job.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_alignment_preflight.py \
  --fold validation/fold \
  --manifest-selection training/val10/structures.csv \
  --seeds 400000-400029 \
  --indices validation/prep/job_indices.csv \
  --bio-dir validation/prep/job_bioassembly \
  --mmcif-dir validation/prep/job_cifs \
  --crop-size 3400 \
  --out training/val10/alignment_preflight.json
