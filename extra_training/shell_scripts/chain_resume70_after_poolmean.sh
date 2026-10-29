#!/bin/bash
# Resume the row-weighted arm 67 -> 70 once the mean-pooling evaluation has finished and released
# cards 4,5,7. WAITS ONLY -- never kills anything, never touches the GPU holders. flock so a
# second copy cannot double-launch the trainer.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
exec 9>training/runs/.resume70_chain.lock
flock -n 9 || { echo "another copy holds the lock; exiting"; exit 0; }
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
DONE=training/runs/rowweight_compare/rw_ep66_ep66_b10.json
DEADLINE=$(( $(date +%s) + 43200 ))

echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  waiting for the mean-pooling eval ($DONE)"
while [ ! -f "$DONE" ]; do
  grep -qE "nothing evaluated|Traceback|STOP:" training/runs/pool_mean_ep66.log 2>/dev/null \
    && { echo "the eval FAILED; not resuming training -- look at the log first"; exit 1; }
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP waiting for $DONE"; exit 1; }
  sleep 30
done

echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  eval done; waiting for cards 4,5,7"
while :; do
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: cards busy ($BUSY)"; exit 1; }
  sleep 20
done

echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  resuming d32kqp 67 -> 70"
exec ./shell_scripts/resume_rowweight_ep67_70.sh
