#!/bin/bash
# SIX HELD-OUT PANELS for the ranking arm alone. The dashed lines are the SHIPPED head, derived
# from this run's own stored_iptm column -- a baseline, not another arm. Do not add other arms.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
RANK=$(cat training/runs/rank_rid 2>/dev/null || echo zbbci6)
/home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_val_panels.py \
  "$RANK=PAE + ranking, alpha 50" \
  --out training/runs/val_panels.png
