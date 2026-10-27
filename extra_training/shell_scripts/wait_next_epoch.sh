#!/bin/bash
# Block until the named arm evaluates ONE MORE epoch than it has right now, then exit so the
# SESSION is re-invoked and can send the plot.
#
# Usage:  ./shell_scripts/wait_next_epoch.sh <rid>        (as a TRACKED background task)
#
# WHY THIS REPLACES wait_epoch_<rid>.sh. That script took the count to wait past as an argument,
# and on 2026-10-26 I passed it the EPOCH NUMBER (4) when the count was 3 -- tv6hue starts at
# epoch 2, so count == epoch - 1 -- and it slept straight through epoch 5. The epoch was only
# noticed because an older waiter happened to fire. An argument that must be computed correctly
# by the caller every single time is a bug waiting its turn, so this one SNAPSHOTS the count
# itself and there is nothing to get wrong.
#
# NO PIDS, NO PATTERNS. Liveness is "the run.log was written recently". pgrep/pkill matches the
# shell running it and self-killed this session twice on 2026-10-18; a /proc cmdline scan matches
# itself. Nothing here signals anything -- it only reads.
#
# IT ALSO EXITS IF THE ARM STOPS, so a finished or crashed run wakes the session instead of
# leaving it waiting for an epoch that is never coming.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
RID="${1:?usage: wait_next_epoch.sh <rid>}"
LOG="training/runs/$RID/run.log"
count() { ls -1 "training/runs/$RID"/metric_train_ep*.json 2>/dev/null | wc -l; }
HAVE=$(count)
echo "$(TZ=America/New_York date '+%F %-I:%M:%S %p %Z') waiting for $RID to pass $HAVE evaluated epoch(s)"
STALE=0
for i in $(seq 1 2400); do          # ~12h ceiling at 18s
  n=$(count)
  if [ "$n" -gt "$HAVE" ] 2>/dev/null; then
    LAST=$(ls -1 "training/runs/$RID"/metric_train_ep*.json 2>/dev/null \
           | sed 's/.*metric_train_ep\([0-9]*\)\.json/\1/' | sort -n | tail -1)
    echo "EPOCH_READY rid=$RID count=$n latest_epoch=$LAST"
    exit 0
  fi
  if [ -f "$LOG" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$AGE" -gt 2700 ]; then STALE=$((STALE+1)); else STALE=0; fi
    # 45 min with no write is not a slow epoch; an epoch is ~30 min and writes throughout.
    if [ "$STALE" -ge 3 ]; then echo "RUN_STALLED rid=$RID count=$n log_age=${AGE}s"; exit 0; fi
  else
    echo "NO_LOG rid=$RID"; exit 0
  fi
  sleep 18
done
echo "TIMEOUT rid=$RID count=$(count)"
