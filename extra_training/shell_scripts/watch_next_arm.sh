#!/bin/bash
# Hand the per-epoch plot-and-push watcher to whichever arm the chain launches next.
#
# Usage:  nohup ./shell_scripts/watch_next_arm.sh ip8ga4 > training/runs/next_arm_watch.log 2>&1 &
#
# watch_arm.sh takes a rid explicitly and refuses to guess one, which is right -- but the copy
# arm's rid does not exist until the chain launches it, hours from now. So this waits for a run
# directory that is NEW (not the current arm, not one that already existed) and holds a run.log,
# then execs watch_arm.sh on it.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
CUR="${1:?usage: $0 <current rid>}"
BEFORE=$(ls training/runs | grep -E '^[a-z0-9]{6}$' | sort)
echo "$(TZ=America/New_York date '+%F %I:%M %p') waiting for the arm after $CUR ($(echo "$BEFORE" | wc -l) run dirs now)"
for i in $(seq 1 6000); do      # 50h ceiling at 30s
  for R in $(ls training/runs | grep -E '^[a-z0-9]{6}$'); do
    echo "$BEFORE" | grep -qx "$R" && continue
    [ -f "training/runs/$R/run.log" ] || continue
    echo "$(TZ=America/New_York date '+%F %I:%M %p') new arm $R -- handing over to watch_arm.sh"
    echo "$R" > training/runs/copy_rid
    exec ./shell_scripts/watch_arm.sh "$R"
  done
  sleep 30
done
echo "TIMEOUT: no new arm appeared"
