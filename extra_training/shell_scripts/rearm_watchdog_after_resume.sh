#!/bin/bash
# RE-ARM A WATCHDOG ON THE RESUMED ARM, once the predicted-crop evals have given the cards back.
#
#   nohup ./shell_scripts/rearm_watchdog_after_resume.sh > training/runs/rearm_watchdog.log 2>&1 &
#
# WHY THIS EXISTS. watchdog_uncropped.sh cannot tell a PLANNED pause from a crash: it would have
# seen the eval's stop as a death and either relaunched the trainer on top of the evals or
# refused and exited for good. So it was stopped by explicit pid before the pause, and this
# re-arms one on whatever run id the resume produced.
#
# IT WAITS FOR A RUN, NOT FOR A CLOCK. The resumed arm gets a NEW run id from launch.py, so this
# reads it from the resume log rather than guessing, and refuses rather than watching the wrong
# directory.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SENTINEL=training/runs/.predcrop_resumed
say() { echo "$(TZ=America/New_York date '+%F %I:%M:%S %p %Z') $*"; }
say "waiting for the resume to produce a run id"
DEADLINE=$(( $(date +%s) + 43200 ))
RID=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 30
  [ -f "$SENTINEL" ] || continue
  for LG in training/runs/predcrop_resume.log training/runs/predcrop_safetynet_resume.log; do
    [ -f "$LG" ] || continue
    R=$(grep -oE "^run [a-z0-9]{6}" "$LG" 2>/dev/null | head -1 | awk '{print $2}')
    [ -n "$R" ] && { RID="$R"; break; }
  done
  [ -n "$RID" ] && break
done
[ -n "$RID" ] || { say "no run id appeared; NOT arming a watchdog"; exit 1; }
[ -d "training/runs/$RID" ] || { say "run dir training/runs/$RID missing; refusing"; exit 1; }
say "resumed arm is $RID"
# A FRESH RESTART BUDGET for a fresh run id; the old state file belongs to the old chain.
echo "restarts=0" > "training/runs/${RID}_watchdog_state"
nohup ./shell_scripts/watchdog_uncropped.sh "$RID" 120 \
  > "training/runs/${RID}_watchdog.log" 2>&1 &
sleep 5
say "watchdog armed on $RID; first lines:"; head -3 "training/runs/${RID}_watchdog.log" 2>/dev/null
