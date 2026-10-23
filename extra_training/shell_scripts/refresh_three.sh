#!/bin/bash
# THREE HELD-OUT PANELS for the live ranking-only arm: mean DockQ, frac acceptable, and
# Spearman(ipTM, DockQ), each with the standard error ACROSS the 18 val10 interfaces.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:-$(cat training/runs/armB_rid 2>/dev/null)}"
[ -n "$RID" ] || { echo "no rid: pass it or write training/runs/rank_only_rid"; exit 1; }
$PY -I run_logging/plot_three.py "$RID" --budget 30 --out training/runs/three_panels.png
