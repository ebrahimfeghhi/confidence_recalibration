#!/usr/bin/env bash
# One line a minute on everything in flight, so nobody has to ask "is it moving".
#
# WHY A HEARTBEAT AND NOT A COMPLETION WAITER. A waiter that fires only at the end cannot
# distinguish "still working" from "wedged", and several jobs tonight finished minutes before
# anyone looked. This prints a stamped line every interval whether or not anything changed, and
# calls out a counter that has not moved since the previous tick.
#
#   ./run_logging/progress_watch.sh [interval_seconds] [max_ticks]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
INT="${1:-60}"
MAX="${2:-240}"
PY=/home/ebrahim/Protenix/.venv/bin/python

prev_lab=-1 prev_ep=-1 stall=0
for i in $(seq 1 "$MAX"); do
  NOW=$(TZ=America/New_York date '+%-I:%M:%S %p')
  LAB=$(ls training/fold/itm_labels_crop/*.npz 2>/dev/null | wc -l)
  MAN=$([ -f training/fold/itm_labels_crop/manifest.json ] && echo yes || echo no)
  RID=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
  EP=$(ls "training/runs/$RID"/head_ep*.pt 2>/dev/null \
       | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
  EP=${EP:-0}
  AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log" 2>/dev/null || date +%s) ))
  LOAD=$(uptime | sed 's/.*load average: //' | cut -d, -f1)

  # A COUNTER THAT HAS NOT MOVED IS NOT THE SAME AS A JOB THAT IS DONE. Labels are only "stalled"
  # while they are still short of the pool, so a finished build stops being reported as stuck.
  POOL=$(ls training/crop_final/*.npz 2>/dev/null | wc -l)
  if [ "$LAB" = "$prev_lab" ] && [ "$LAB" -lt "$POOL" ]; then stall=$((stall+1)); else stall=0; fi
  WARN=""
  [ "$stall" -ge 3 ] && WARN="  <-- labels unchanged for $stall tick(s)"
  [ "$AGE" -gt 600 ] && WARN="$WARN  <-- run.log silent ${AGE}s"

  echo "$NOW  labels $LAB/$POOL manifest=$MAN | $RID epoch $EP (log ${AGE}s ago) | load $LOAD$WARN"
  prev_lab=$LAB prev_ep=$EP
  sleep "$INT"
done
