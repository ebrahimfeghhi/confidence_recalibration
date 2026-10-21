#!/bin/bash
# BOTH ARMS ON ONE PLOT: pose-focused restriction against full-head finetuning.
#
# Same 30 structures, same val10, same lr 1e-2, same crop, same seeds, same loss, same monitors,
# both fresh from the shipped head. The ONLY difference is which parameters train:
#   ggfhj5+zcuiab   26,624 (0.135% of the head) -- the two distance embeddings and the PAE readout
#   $BASE           19.8M  (all of it)
# ggfhj5+zcuiab is one experiment in two segments: zcuiab resumed from ggfhj5/resume_ep11.pt after
# multi-worker prefetch stalled epoch 12. Epoch 12 exists in neither segment.
#
# Reference lines are budget 30 and must never be mixed with the budget-100 set: baseline DockQ
# 0.2909 / frac_acc 0.5556, oracle 0.5968 / 1.0000, shipped head's val10 PAE 2.9756.
# --shipped-train-pae is deliberately NOT passed: 2.8058 was measured over the 488-structure set,
# so it is not the epoch-0 point for a 30-structure arm.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
POSE=$(cat training/runs/pose_focused_rid 2>/dev/null || echo ggfhj5+zcuiab)
BASE="${1:-${BASE:-$(cat training/runs/fullhead_rid 2>/dev/null)}}"
[ -n "$BASE" ] || { echo "no baseline run id: pass it as \$1 or write training/runs/fullhead_rid"; exit 1; }
echo "pose-focused: $POSE    full-head: $BASE"
$PY -I run_logging/plot_full_train.py \
  "$POSE=pose-focused, 26,624 params (0.135%)" \
  "$BASE=full head, 19.8M params" \
  --budget 30 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --shipped-val-pae 2.9756 \
  --out training/runs/arm_comparison_plot.png

$PY -I run_logging/results_table.py "$BASE" \
  --budget 30 \
  --shipped-val-pae 2.9756 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --out docs/fullhead_overfit30_results.md
