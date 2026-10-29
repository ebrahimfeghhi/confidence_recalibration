#!/bin/bash
# Run the in-sample comparison once nskyne has written resume_ep60.pt and released the cards.
# Waits only; never kills anything. One copy at a time, via flock.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
exec 9>training/runs/.compare_chain.lock
flock -n 9 || { echo "another copy holds the lock; exiting"; exit 0; }
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
CKPT=training/runs/nskyne/resume_ep60.pt
DEADLINE=$(( $(date +%s) + 86400 ))
echo "$(TZ=America/New_York date '+%I:%M %p %Z')  waiting for $CKPT"
while [ ! -f "$CKPT" ]; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: $CKPT never appeared"; exit 1; }
  sleep 120
done
echo "$(TZ=America/New_York date '+%I:%M %p %Z')  epoch 60 on disk; waiting for the cards"
while :; do
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: cards busy ($BUSY)"; exit 1; }
  sleep 30
done
echo "$(TZ=America/New_York date '+%I:%M %p %Z')  comparing in-sample selection at epoch 60"
exec $PY_BIN -I run_logging/compare_rowweight_insample.py
