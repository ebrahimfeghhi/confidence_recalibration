#!/bin/bash
# ALL THREE CAPACITY ARMS ON ONE PLOT. Same 30 structures, same val10, same lr 1e-2, same crop,
# same seeds, same masked PAE-only loss, all three fresh from the shipped head. The ONLY thing
# that differs is which parameters may change and how:
#   ggfhj5+zcuiab   26,624 (0.135%)   3 tensors, hard freeze on the rest      21 epochs, finished
#   a4afho         567,112 (4.42%)    141 modules, rank-8 updates everywhere  30 epochs
#   mdv9x7      19,752,270 (100%)     248 tensors, unconstrained              30 epochs, finished
#
# ggfhj5+zcuiab is ONE experiment in two segments: zcuiab resumed from ggfhj5/resume_ep11.pt after
# multi-worker prefetch stalled epoch 12. Epoch 12 exists in neither segment.
#
# Budget 30 reference lines, never mixed with the budget-100 set: baseline DockQ 0.2909 /
# frac_acc 0.5556, oracle 0.5968 / 1.0000, shipped head's val10 PAE 2.9756. --shipped-train-pae is
# NOT passed: 2.8058 was measured over the 488-structure set, not these 30.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
POSE=$(cat training/runs/pose_focused_rid 2>/dev/null || echo ggfhj5+zcuiab)
FULL=$(cat training/runs/fullhead_rid    2>/dev/null || echo mdv9x7)
LORA=$(cat training/runs/lora_rid        2>/dev/null || echo a4afho)
echo "pose-focused: $POSE   LoRA: $LORA   full head: $FULL"
$PY -I run_logging/plot_full_train.py \
  "$POSE=pose-focused, 26,624 (0.135%)" \
  "$LORA=LoRA r=8, 567,112 (4.42%)" \
  "$FULL=full head, 19.8M (100%)" \
  --budget 30 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --shipped-val-pae 2.9756 \
  --out training/runs/three_arm_plot.png

$PY -I run_logging/results_table.py "$LORA" \
  --budget 30 \
  --shipped-val-pae 2.9756 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --out docs/lora_overfit30_results.md
