#!/bin/bash
# Block until ij24wr's evaluated-epoch count exceeds $1, then exit so the SESSION is re-invoked
# and can SEND the plot. The committing watcher (watch_arm.sh) pushes every epoch but cannot send
# anything into the conversation, which is why epochs 10 and 11 were pushed and never reported.
#
# RANK PIDS ARE EXPLICIT, never derived by pattern: pgrep/pkill matches the shell running it.
# These are only READ (/proc/<pid> existence), nothing is signalled.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
HAVE="${1:-0}"
RANKS="3269735 3269736 3269737 "
count() { ls -1 training/runs/ij24wr/metric_train_ep*.json 2>/dev/null | wc -l; }
for i in $(seq 1 2400); do          # ~12h ceiling at 18s
  n=$(count)
  if [ "$n" -gt "$HAVE" ] 2>/dev/null; then echo "EPOCH_READY epochs=$n"; exit 0; fi
  alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && { echo "RUN_GONE epochs=$n"; exit 0; }
  sleep 18
done
echo "TIMEOUT epochs=$(count)"
