#!/bin/bash
# FIVE ARMS: the three finished capacity arms, the finished pose-share arm, and the live
# pose-share + L2-SP arm. The two pose-share arms differ in ONE thing -- lambda * ||W - W0||^2
# on the shared injection -- so they are the only pair on this figure that is a clean A/B.
# The three capacity arms are a different axis (which parameters move) and their step 0 IS the
# shipped head, which the pose-share arms' is not. Budget 30 reference lines only.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
POSE=$(cat training/runs/pose_focused_rid 2>/dev/null || echo ggfhj5+zcuiab)
LORA=$(cat training/runs/lora_rid 2>/dev/null || echo eewbr5)
FULL=$(cat training/runs/fullhead_rid 2>/dev/null || echo mdv9x7)
SHARE=$(cat training/runs/poseshare_rid 2>/dev/null || echo 8eiek8)
L2=$(cat training/runs/poseshare_l2_rid 2>/dev/null)
[ -n "$L2" ] || { echo "no L2 rid: write training/runs/poseshare_l2_rid"; exit 1; }
echo "pose-focused $POSE | LoRA $LORA | full $FULL | pose-share $SHARE | pose-share+L2 $L2"
$PY -I run_logging/plot_full_train.py \
  "$POSE=pose-focused, 26,624 (0.135%)" \
  "$LORA=LoRA r=8, 286,600 effective" \
  "$FULL=full head, 19.8M (100%)" \
  "$SHARE=pose-share x4, 10,240, no L2" \
  "$L2=pose-share x4 + L2-SP 1e-2" \
  --budget 30 \
  --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --shipped-val-pae 2.9756 \
  --out training/runs/five_arm_plot.png
$PY -I run_logging/results_table.py "$L2" --budget 30 \
  --shipped-val-pae 2.9756 --shipped-dockq 0.2909 --shipped-frac 0.5556 \
  --oracle-dockq 0.5968 --oracle-frac 1.0000 \
  --out docs/poseshare_l2_overfit30_results.md
