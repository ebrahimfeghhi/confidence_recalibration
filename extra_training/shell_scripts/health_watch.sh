#!/bin/bash
# BLOCK UNTIL SOMETHING IS WRONG, then print the diagnosis and exit. Exiting is the point: the
# harness notifies on completion, so a clean exit IS the alert. It must never exit for a benign
# reason, or the alert means nothing.
#
# NO PATTERN MATCHING ON PROCESSES. pgrep -f / pkill -f match the shell running them and
# self-killed this session twice on 2026-10-18, so liveness is read from the CARDS (via
# nvidia-smi compute-apps, holders excluded by explicit pid) and from the trajectory advancing.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
RID="${1:?usage: health_watch.sh <run_id>}"
HOLDERS="2860931 2861781 2862274"
UUIDS="3919dad8 79b76a53 f53b08f3"
POLL="${POLL:-120}"
STALL_LIMIT="${STALL_LIMIT:-5}"   # consecutive polls with no new pose = 10 min stalled
QUOTA_FLAG_GB="${QUOTA_FLAG_GB:-2765}"
# FROM THE RUN'S OWN args.json so the target cannot disagree with what is running.
WANT_EPOCHS="${WANT_EPOCHS:-$(/home/ebrahim/Protenix/.venv/bin/python -I -c "
import json,sys
try: print(int(float(json.load(open('training/runs/$RID/args.json'))['epochs'])))
except Exception: print(0)
")}"
[ "${WANT_EPOCHS:-0}" -gt 0 ] || { echo "REFUSING: cannot read an epoch count for $RID"; exit 1; }
last=-1; stalls=0

note () { echo "[$(TZ=America/New_York date '+%-I:%M %p %Z')] $*"; }

note "watching $RID every ${POLL}s: poses advancing, ranks on cards, holders, RAM, disk"
while :; do
  sleep $POLL

  # 1. the trajectory must advance. The most direct evidence of progress there is.
  now=$(awk -F, 'NR>1{n++} END{print n+0}' "training/runs/$RID/trajectory.csv" 2>/dev/null)

  # AN EVAL IS NOT A STALL. Every rank parks at a barrier while the evaluator shards over the
  # cards, so poses stop advancing for the whole pass -- and the in-sample pass takes ~33 min
  # against a 10 min stall limit. This alarmed at 1:11 AM on egw5wq's epoch-10 in-sample eval and
  # was a FALSE POSITIVE. While an eval is in flight the stall counter is held, not incremented.
  evaling=0
  if tail -40 "training/runs/$RID/run.log" 2>/dev/null | grep -qE "evaluating epoch [0-9]+ |extra eval "; then
    _ee=$(tail -40 "training/runs/$RID/run.log" | grep -oE "evaluating epoch [0-9]+" | tail -1 | grep -oE "[0-9]+")
    if [ -n "${_ee:-}" ] && [ ! -f "training/runs/$RID/metric_insample_ep${_ee}.json" ]; then
      evaling=1
    fi
  fi
  if [ "$evaling" = 1 ]; then
    last=$now
    continue
  fi

  if [ "${now:-0}" -le "$last" ]; then
    stalls=$((stalls+1))
    if [ "$stalls" -ge "$STALL_LIMIT" ]; then
      note "ALERT $RID STALLED: $now pose(s), unchanged for $((stalls*POLL/60)) min"
      tail -6 "training/runs/$RID/run.log" 2>/dev/null | cut -c1-170
      exit 0
    fi
  else
    stalls=0
  fi
  last=$now

  # 2. a fatal line in the log. Checked even while poses advance, since rank 0 can die last.
  if grep -qE "Traceback|OutOfMemoryError|^STOP:|CUDA error" "training/runs/$RID/run.log" 2>/dev/null
  then
    note "ALERT $RID has a fatal line in its log at $now pose(s)"
    grep -nE "Traceback|OutOfMemoryError|^STOP:|CUDA error" "training/runs/$RID/run.log" \
      | head -3 | cut -c1-170
    exit 0
  fi

  # 3. the three ranks must still hold the three cards. Holders excluded by explicit pid.
  live=0
  for u in $UUIDS; do
    for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
               | grep "$u" | cut -d, -f2 | tr -d ' '); do
      keep=1; for h in $HOLDERS; do [ "$p" = "$h" ] && keep=0; done
      [ "$keep" = 1 ] && live=$((live+1))
    done
  done
  if [ "$live" -lt 3 ]; then
    note "ALERT $RID has only $live trainer process(es) on cards 4/5/7 at $now pose(s)"
    tail -6 "training/runs/$RID/run.log" 2>/dev/null | cut -c1-170
    exit 0
  fi

  # 4. the standing box conditions the user asked to be flagged.
  for h in $HOLDERS; do
    [ -d "/proc/$h" ] || { note "ALERT GPU holder $h is GONE -- the card can be taken"; exit 0; }
  done
  mem=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  [ "$mem" -lt 250 ] && { note "ALERT MemAvailable ${mem} GB, under the 250 GB floor"; exit 0; }
  # DISK IS A QUOTA, NOT df. /home is shared and its percentage says nothing about this account's
  # 3,072 GiB hard limit, which has NO grace period -- a write past it fails outright. df on /home
  # read 60-something percent while the quota was at 2,511 GiB, so this check was blind to the
  # thing it existed to catch. The flag level is the owner's: 2,765 GiB used.
  used=$(quota -u ebrahim 2>/dev/null | awk '$1 ~ /^\/dev\// {printf "%d", $2/1048576}')
  if [ -n "${used:-}" ] && [ "$used" -gt "$QUOTA_FLAG_GB" ]; then
    note "ALERT quota at ${used} GiB, over the ${QUOTA_FLAG_GB} GiB flag (3072 GiB hard limit, no grace)"
    exit 0
  fi

  # 5. and stop watching when the run has done what it was asked to.
  # THE EPOCH TARGET IS THE RUN'S, NOT A CONSTANT. This said 60 because the first arm watched ran
  # 60 epochs; against a 10-epoch arm it would never report completion and the watcher would sit
  # on a finished run looking healthy. Read from the run's own args.json, so it cannot drift.
  eps=$(grep -cE "^\s*\[rank 0\] epoch [0-9]+ took" "training/runs/$RID/run.log" 2>/dev/null)
  if [ "${eps:-0}" -ge "$WANT_EPOCHS" ]; then
    note "$RID finished all $WANT_EPOCHS epoch(s)"; exit 0
  fi
done
