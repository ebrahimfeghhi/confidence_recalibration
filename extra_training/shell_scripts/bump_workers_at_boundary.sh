#!/bin/bash
# RAISE --prefetch-workers AT AN EPOCH BOUNDARY, losing no completed work.
#
# At the user's direction: "go up to 16, relaunch the arm at the next epoch boundary so it
# continues smoothly onto the next epoch with this change".
#
# WHY A BOUNDARY AND NOT NOW. --prefetch-workers is a launch flag, so changing it means a new
# process. Stopping mid-epoch throws away the units walked since the last checkpoint; stopping at
# the boundary, right after resume_ep<N>.pt is written, costs nothing at all.
#
# WHY THIS IS A LEGITIMATE RESUME. --prefetch-workers is NOT among the fields the resume guard
# compares (verified), and correctly so: it changes how fast crops are prepared, not the
# objective, the population, the batch composition or the schedule. The guard will still refuse
# any of those if they moved.
#
# IT STOPS BY EXPLICIT PID, derived from the cards. NEVER pkill -f / pgrep -f: they match the
# shell running them and have self-killed this session twice.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${RID:-privsp}"
WORKERS="${WORKERS:-16}"
LOG=training/runs/bump_workers
mkdir -p "$LOG" || exit 1
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

exec 9>"$LOG/.lock"; flock -n 9 || { say "another bump already running"; exit 200; }

E0=$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | sed -E 's/.*resume_ep([0-9]+)\.pt/\1/' | sort -n | tail -1)
say "waiting for the next epoch boundary past ${E0:-0} (ceiling 90 min)"
for _ in $(seq 1 540); do
  E=$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | sed -E 's/.*resume_ep([0-9]+)\.pt/\1/' | sort -n | tail -1)
  if [ -n "$E" ] && [ "${E:-0}" -gt "${E0:-0}" ]; then break; fi
  sleep 10
done
E=$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | sed -E 's/.*resume_ep([0-9]+)\.pt/\1/' | sort -n | tail -1)
[ -n "$E" ] && [ "$E" -gt "${E0:-0}" ] || { say "REFUSING: no new boundary appeared. Nothing stopped."; exit 2; }
BUNDLE="training/runs/$RID/resume_ep${E}.pt"
say "epoch $E complete, bundle $BUNDLE ($(stat -c %s "$BUNDLE") bytes)"

# LET THE BOUNDARY FINISH. The checkpoint is written before the eval and before the next epoch's
# first unit; a few seconds avoids catching the writer mid-flush.
sleep 20

say "stopping $RID by explicit pid"
PARENT=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk -F, '{print $1}' | sort -u); do
  [ -d "/proc/$p" ] || continue
  c=$({ tr '\0' ' ' < /proc/$p/cmdline; } 2>/dev/null)
  case "$c" in *phase_k_train*"runs/$RID"*) PARENT=$(awk '/^PPid:/{print $2}' /proc/$p/status);; esac
done
[ -n "$PARENT" ] || { say "REFUSING: could not derive the parent pid. Nothing stopped."; exit 3; }
say "  torchrun parent $PARENT -- SIGTERM"
kill -TERM "$PARENT" 2>/dev/null
for _ in $(seq 1 30); do
  B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$B" = "0" ] && { say "  cards released"; break; }
  sleep 10
done
B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$B" = "0" ] || { say "REFUSING: cards still busy ('$B') after 5 min. NOT relaunching."; exit 4; }
sleep 30

PREV=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
say "LAUNCH: resume from epoch $E with --prefetch-workers $WORKERS"
RESUME="$BUNDLE" PREFETCH_WORKERS="$WORKERS" DRY=0 CROP_THREADS=3 \
  ./shell_scripts/launch_pae_crop_arm.sh > "$LOG/launch.log" 2>&1

NEW=""
for _ in $(seq 1 60); do
  C=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
  [ -n "$C" ] && [ "$C" != "${PREV:-}" ] && { NEW="$C"; break; }
  sleep 10
done
[ -n "$NEW" ] || { say "DID NOT START:"; grep -aE "REFUS|STOP|Traceback" "$LOG/launch.log" | tail -8; exit 5; }
say "new run $NEW"; echo "$NEW" > "$LOG/rid"
for _ in $(seq 1 90); do
  grep -qaE "steps/epoch|STOP:|Traceback" "training/runs/$NEW/run.log" 2>/dev/null && break
  sleep 20
done
grep -aE "RESUMED from|continuing at epoch|prefetch|steps/epoch" "training/runs/$NEW/run.log" | head -4
say "done -- $NEW continues from epoch $E with $WORKERS workers"
