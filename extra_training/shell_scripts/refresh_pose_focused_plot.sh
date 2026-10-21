#!/bin/bash
# Regenerate the 4-metric plot AND the results table for the pose-focused overfit30 arm.
#
# SAME TWO READERS AS THE FULL-SET PLOT, so the two arms cannot drift apart in how a metric is
# computed -- which is the whole point of the comparison. Reference lines are budget 30, the same
# budget the full-set curve uses: baseline 0.2909 / 0.5556, oracle 0.5968 / 1.0000, shipped head's
# val10 PAE 2.9756. NEVER mix these with the budget-100 numbers.
#
# RUN ID: first argument, else $RID, else training/runs/pose_focused_rid, else the newest run dir.
# Pinned in a file rather than hardcoded because the id is only known at launch.
#
# --shipped-train-pae is the SHIPPED head's pose-weighted PAE over the TRAINING set. That number
# is measured over the 488-structure set (training/runs/shipped_train_pae), NOT over these 30, so
# it is NOT the epoch-0 point for this arm and is deliberately left off here. Measuring it for
# overfit30 is a separate --loss-only pass over 300 units, ~6 min.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:-${RID:-}}"
if [ -z "$RID" ] && [ -f training/runs/pose_focused_rid ]; then RID=$(cat training/runs/pose_focused_rid); fi
if [ -z "$RID" ]; then RID=$(ls -1dt training/runs/*/ | grep -v '/latest/' | head -1 | xargs basename); fi
echo "run id: $RID"
$PY -I run_logging/plot_full_train.py \
  "$RID=pose-focused, 30 structures, lr 1e-2 (13,312 params)" \
  --budget 30 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --shipped-val-pae 2.9756 \
  --out training/runs/pose_focused_plot.png

$PY -I run_logging/results_table.py \
  "$RID" \
  --budget 30 \
  --shipped-val-pae 2.9756 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --out docs/pose_focused_results.md
