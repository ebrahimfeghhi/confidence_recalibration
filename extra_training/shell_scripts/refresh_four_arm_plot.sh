#!/bin/bash
# FOUR ARMS: the three finished capacity arms plus the live pose-share arm.
# The pose-share arm is NOT a capacity point -- it changes the ARCHITECTURE (block 0's pretrained
# injection re-applied at all four pairformer blocks) and trains only that shared injection, so
# its step 0 is not the shipped head. Read it as a different experiment sharing the same axes.
# Budget 30 reference lines only; never mixed with the budget-100 set.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
POSE=$(cat training/runs/pose_focused_rid 2>/dev/null || echo ggfhj5+zcuiab)
LORA=$(cat training/runs/lora_rid 2>/dev/null || echo eewbr5)
FULL=$(cat training/runs/fullhead_rid 2>/dev/null || echo mdv9x7)
SHARE="${1:-$(cat training/runs/poseshare_rid 2>/dev/null)}"
[ -n "$SHARE" ] || { echo "no pose-share rid: pass it or write training/runs/poseshare_rid"; exit 1; }
echo "pose-focused $POSE | LoRA $LORA | full $FULL | pose-share $SHARE"
$PY -I run_logging/plot_full_train.py \
  "$POSE=pose-focused, 26,624 (0.135%)" \
  "$LORA=LoRA r=8, 286,600 effective" \
  "$FULL=full head, 19.8M (100%)" \
  "$SHARE=pose-share x4 depths, 10,240 (0.052%)" \
  --budget 30 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --shipped-val-pae 2.9756 \
  --out training/runs/four_arm_plot.png
$PY -I run_logging/results_table.py "$SHARE" --budget 30 \
  --shipped-val-pae 2.9756 --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --out docs/poseshare_overfit30_results.md
