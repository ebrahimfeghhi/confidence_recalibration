#!/bin/bash
# Launch the pose-focused overfit30 arm the moment the shipped-head measurement releases card 7.
# The user asked for it launched now; the measurement had ~20 min left and holds a card.
# Waits on an EXPLICIT PID, never a pattern.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PID="${1:?usage: queue_pose_focused_after_measure.sh <pid-to-wait-for>}"
LOG=training/runs/pose_focused_queue.log
exec >> "$LOG" 2>&1
echo "=== armed $(date '+%Y-%m-%d %H:%M:%S %Z'), waiting on pid $PID ==="
while [ -d "/proc/$PID" ]; do sleep 20; done
echo "$(date '+%H:%M:%S') pid $PID gone; launching"
./shell_scripts/launch_pose_focused_overfit30.sh
echo "$(date '+%H:%M:%S') launcher returned $?"
