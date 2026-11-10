#!/usr/bin/env bash
# Watch an arm and push every epoch's results until the run finishes.
#
# ONE SCRIPT INSTEAD OF A RE-TYPED LOOP. The waiter was being rebuilt by hand at every epoch
# boundary, which is how the push step would eventually get forgotten.
#
# WHY IT LOOPS. It used to handle ONE epoch and exit, which meant something had to re-arm it at
# every boundary. On 2026-11-10 nothing did: the waiter exited after epoch 5 and epochs 6-11 ran
# with no watcher at all, so epoch 10's metric json never reached the repo and was only found by
# hand. A one-shot waiter that must be re-armed is a waiter that will eventually be forgotten, so
# this now runs until the arm reaches its configured epoch count.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
RID="${1:?usage: epoch_watch.sh <rid>}"
PY=/home/ebrahim/Protenix/.venv/bin/python

latest_epoch() {
  ls "training/runs/$RID"/head_ep*.pt 2>/dev/null \
    | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1
}

# The epoch the run is heading for. Without it the loop would spin forever after the last epoch.
TOTAL=$("$PY" -I -c "
import json,sys
try: print(int(json.load(open('training/runs/$RID/args.json'))['epochs']))
except Exception: print(0)
" 2>/dev/null)
[ -z "$TOTAL" ] && TOTAL=0
echo "watching $RID through epoch ${TOTAL:-?}"

LAST=$(latest_epoch); LAST=${LAST:-0}
while :; do
  while [ "$(latest_epoch || echo 0)" -le "$LAST" ] 2>/dev/null; do
    # NO PROCESS PATTERN MATCHING HERE. pgrep -f / pkill -f match the shell running them and have
    # self-killed this session before, so liveness is read off the log's mtime instead: a live
    # trainer writes MEM lines continuously, so 30 minutes of silence means it is gone.
    if [ -f "training/runs/$RID/run.log" ]; then
      AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log") ))
      [ "$AGE" -gt 1800 ] && { echo "run.log silent for ${AGE}s; watcher exiting"; exit 0; }
    fi
    sleep 60
  done
  E=$(latest_epoch)
  echo "$RID EPOCH $E $(TZ=America/New_York date '+%-I:%M %p %Z')"

  # On a multiple of 5 the held-out eval runs after the checkpoint; wait for its json so the
  # pushed figure and table carry the eval rather than showing the epoch without it.
  if [ $(( E % 5 )) -eq 0 ]; then
    for _ in $(seq 1 180); do
      [ -s "training/runs/$RID/metric_valfull83_ep${E}.json" ] && {
        echo "eval json present $(TZ=America/New_York date '+%-I:%M %p %Z')"; break; }
      sleep 60
    done
    [ -s "training/runs/$RID/metric_valfull83_ep${E}.json" ] \
      || echo "  WARNING: epoch $E eval json never appeared; pushing without it"
  fi

  ./run_logging/refresh_plots.sh "$RID" figures 2>&1 | tail -3
  # the comparison against the 15-epoch restricted arm, kept current
  "$PY" -I run_logging/plot_rel_arm.py \
    peae97,"$RID" figures/fullhead_vs_restricted.png >/dev/null 2>&1 || true
  grep -aoE "epoch ${E} took [0-9.]+ min" "training/runs/$RID/run.log" | head -1
  ./run_logging/push_epoch.sh "$RID" "$E"

  LAST=$E
  [ "$TOTAL" -gt 0 ] && [ "$E" -ge "$TOTAL" ] && { echo "epoch $E is the last; watcher done"; exit 0; }
done
