#!/bin/bash
# Extend the row-weighted arm 71 -> 80 once t35ub6 has written resume_ep70.pt and freed the cards.
# PURPOSE IS TO KEEP THE CARDS BUSY while the centred-loss arm is being built and gated, at the
# user's direction ("don't leave the GPUs idle, if you need more time extend the current run to 80
# epochs"). These epochs are NOT expected to change any conclusion -- the arm's verdict was taken
# at epoch 55 and epoch 66 -- they are filler with a checkpoint at the end.
# WAITS ONLY; never kills anything; never touches the GPU holders. flock'd against double-launch.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
exec 9>training/runs/.ep80_chain.lock
flock -n 9 || { echo "another copy holds the lock; exiting"; exit 0; }
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
CKPT=training/runs/t35ub6/resume_ep70.pt
DEADLINE=$(( $(date +%s) + 86400 ))
echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  waiting for $CKPT"
while [ ! -f "$CKPT" ]; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: $CKPT never appeared"; exit 1; }
  sleep 60
done
echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  epoch 70 on disk; waiting for the cards"
while :; do
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: cards busy ($BUSY)"; exit 1; }
  sleep 30
done
echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  extending 71 -> 80"
exec ./shell_scripts/resume_rowweight_ep71_80.sh
