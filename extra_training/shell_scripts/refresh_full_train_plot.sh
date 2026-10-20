#!/bin/bash
# Regenerate the 4-metric plot for the full training run. One place, so the watch cannot drift
# from what is actually being read.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
/home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_full_train.py \
  "x3k2m4+jhgfxy=full set, lr 1e-2" \
  --out training/runs/full_train_plot.png
