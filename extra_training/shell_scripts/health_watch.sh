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
POLL=120
STALL_LIMIT=5          # consecutive polls with no new pose = 10 min stalled
last=-1; stalls=0

note () { echo "[$(TZ=America/New_York date '+%-I:%M %p %Z')] $*"; }

note "watching $RID every ${POLL}s: poses advancing, ranks on cards, holders, RAM, disk"
while :; do
  sleep $POLL

  # 1. the trajectory must advance. The most direct evidence of progress there is.
  now=$(awk -F, 'NR>1{n++} END{print n+0}' "training/runs/$RID/trajectory.csv" 2>/dev/null)
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
  disk=$(df --output=pcent /home | tail -1 | tr -dc '0-9')
  [ "$disk" -gt 90 ] && { note "ALERT /home at ${disk}%, over 90%"; exit 0; }

  # 5. and stop watching when the run has done what it was asked to.
  eps=$(grep -cE "^\s*\[rank 0\] epoch [0-9]+ took" "training/runs/$RID/run.log" 2>/dev/null)
  if [ "${eps:-0}" -ge 60 ]; then
    note "$RID finished all 60 epochs"; exit 0
  fi
done
