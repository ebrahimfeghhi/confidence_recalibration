#!/bin/bash
# ONE guarded restart of the uncropped arm, then stop and say so. Never more.
#
# WHY NOT RETRY UNTIL IT WORKS. If a particular structure reproducibly OOMs, resuming from the
# same checkpoint replays the same epoch and hits the same structure -- an unbounded retry would
# burn the cards overnight re-deriving one failure and bury the evidence under identical logs.
# One restart distinguishes a transient (allocator state, a one-off collision) from a determinism,
# which is the only thing an automatic retry can usefully learn.
#
# EPOCH 1 CANNOT BE RESUMED. The trainer writes resume_ep<N>.pt at an epoch BOUNDARY, so a run
# that dies inside its first epoch has no checkpoint and the only restart is a fresh one -- a
# decision with different consequences (it re-rolls the data order and discards the partial
# epoch). This refuses it and reports instead.
#
# PROCESSES ARE CONFIRMED GONE BEFORE RELAUNCHING, by the same contract stop_run.sh uses: pids
# come from nvidia-smi --query-compute-apps and are kept only if /proc/<pid>/cmdline carries
# --out <this run's directory>. No pattern matching, ever -- `pkill -f` has self-killed this
# project's shell twice. The GPU holders are never touched.
#
# STATE IS A FILE, not a variable, so the restart budget survives this script being restarted.
#
# Usage: watchdog_uncropped.sh <rid> [poll_s]
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: watchdog_uncropped.sh <rid> [poll_s]}"
POLL="${2:-120}"
SETTLE=180                 # seconds to let a dying run's processes exit before relaunching
TARGET_EPOCHS=60
RUNDIR="$(pwd)/training/runs/$RID"
STATE="training/runs/${RID}_watchdog_state"
LOG="training/runs/${RID}_watchdog.log"
[ -f "$STATE" ] || echo "restarts=0" > "$STATE"

say() { echo "$(TZ=America/New_York date '+%-I:%M %p') $*" | tee -a "$LOG"; }

alive_count() {
  local n=0 P CMD
  for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u); do
    case "$P" in ""|*[!0-9]*) continue ;; esac
    case " 306042 306045 306050 " in *" $P "*) continue ;; esac     # GPU holders, never counted
    [ -r "/proc/$P/cmdline" ] || continue
    CMD="$(tr '\0' ' ' < "/proc/$P/cmdline")"
    case "$CMD" in *"--out $RUNDIR "*|*"--out $RUNDIR/"*) n=$((n+1)) ;; esac
  done
  echo "$n"
}

done_epochs() {
  $PY -I -c "
import csv, collections, sys
c = collections.Counter()
try:
    for r in csv.DictReader(open('training/runs/$RID/trajectory.csv')):
        e = r.get('epoch')
        if e not in (None, ''): c[int(float(e))] += 1
except FileNotFoundError:
    pass
print(max([k for k, v in c.items() if v >= 5500] or [0]))" 2>/dev/null || echo 0
}

say "watchdog armed on $RID (poll ${POLL}s, one restart permitted)"
while true; do
  sleep "$POLL"
  N=$(alive_count)
  [ "$N" -gt 0 ] && continue
  EP=$(done_epochs)
  if [ "${EP:-0}" -ge "$TARGET_EPOCHS" ]; then
    say "WATCHDOG: $RID finished $EP epoch(s); nothing to do"; exit 0
  fi
  say "WATCHDOG: no process carries --out $RUNDIR and only $EP epoch(s) are complete -- it died"
  USED=$(sed -n 's/^restarts=//p' "$STATE")
  if [ "${USED:-0}" -ge 1 ]; then
    say "WATCHDOG: the one permitted restart is already spent. STOPPING. Cards left free."
    say "WATCHDOG: tail of the run log:"; tail -25 "$RUNDIR/run.log" >> "$LOG" 2>/dev/null
    exit 2
  fi
  CK=$(ls -t "$RUNDIR"/resume_ep*.pt 2>/dev/null | head -1)
  if [ -z "$CK" ]; then
    say "WATCHDOG: NO resume checkpoint exists -- it died inside epoch 1. A restart would be a"
    say "WATCHDOG: FRESH run, which re-rolls the data order and is a decision, not a retry."
    say "WATCHDOG: STOPPING. Cards left free."
    tail -25 "$RUNDIR/run.log" >> "$LOG" 2>/dev/null
    exit 3
  fi
  say "WATCHDOG: newest checkpoint $(basename "$CK"); settling ${SETTLE}s before relaunch"
  sleep "$SETTLE"
  N=$(alive_count)
  if [ "$N" -gt 0 ]; then
    say "WATCHDOG: $N process(es) came back on $RUNDIR -- not relaunching"; continue
  fi
  BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  if [ "$BUSY" != "0" ]; then
    say "WATCHDOG: cards report busy ($BUSY) -- refusing to relaunch onto them. STOPPING."
    exit 4
  fi
  RS="shell_scripts/_watchdog_resume_${RID}.sh"
  sed "s#^  -- --rank-loss#  -- --resume ${CK#$(pwd)/} \\\\\n     --rank-loss#" \
      shell_scripts/launch_rel_itm_beta75_uncropped.sh > "$RS"
  chmod +x "$RS"
  if ! bash -n "$RS"; then say "WATCHDOG: generated resume script is malformed. STOPPING."; exit 5; fi
  echo "restarts=1" > "$STATE"
  say "WATCHDOG: relaunching from $(basename "$CK")"
  nohup "./$RS" > "training/runs/${RID}_watchdog_relaunch.log" 2>&1 &
  sleep 60
  NEW=$(ls -td training/runs/*/ | head -1 | xargs basename)
  say "WATCHDOG: relaunched as $NEW -- this watchdog now stops; arm a new one on $NEW if wanted"
  exit 0
done
