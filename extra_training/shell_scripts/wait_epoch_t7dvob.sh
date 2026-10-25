#!/bin/bash
# Block until t7dvob's held-out metrics gain a row beyond $1, then exit. Re-invokes the session
# so the plot can be SENT, which the committing watcher does not do.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
HAVE="${1:-0}"
rows() {
  local f
  f=$(ls -1 training/runs/t7dvob/rank_rho_b30_v*.csv 2>/dev/null | sort -V | tail -1)
  [ -n "$f" ] && [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0
}
for i in $(seq 1 1400); do   # ~7h ceiling at 18s
  n=$(rows)
  if [ "$n" -gt "$HAVE" ] 2>/dev/null; then echo "EPOCH_READY rows=$n"; exit 0; fi
  # stop waiting if the run died
  alive=0; for P in 2972016 2972017 2972018; do [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && { echo "RUN_GONE rows=$n"; exit 0; }
  sleep 18
done
echo "TIMEOUT rows=$(rows)"
