#!/bin/bash
# Run the HELD-OUT valfull comparison (budget 100, both arms, epoch 55) once the in-sample pass
# has finished and the cards are free. WAITS ONLY -- never kills anything, never touches the GPU
# holders. One copy at a time via flock.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
exec 9>training/runs/.val_chain.lock
flock -n 9 || { echo "another copy holds the lock; exiting"; exit 0; }
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
DEADLINE=$(( $(date +%s) + 43200 ))

echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  waiting for the in-sample pass to finish"
while [ ! -f training/runs/rowweight_compare/lineage_ep55_b10.json ]; do
  grep -qE "nothing evaluated|Traceback" training/runs/compare_ep55.log 2>/dev/null \
    && { echo "in-sample pass FAILED; not starting the val pass"; exit 1; }
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP waiting for the in-sample json"; exit 1; }
  sleep 30
done

echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  in-sample done; waiting for the cards"
while :; do
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: cards busy ($BUSY)"; exit 1; }
  sleep 30
done

echo "$(TZ=America/New_York date '+%-I:%M %p %Z')  HELD-OUT valfull, budget 100, epoch 55, both arms"
exec $PY_BIN -I run_logging/compare_rowweight_valfull.py
