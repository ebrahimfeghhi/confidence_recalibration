#!/bin/bash
# REGENERATE AN ARM'S PLOT EVERY TIME AN EPOCH LANDS, until the run reaches its own epoch count.
#
#     ./shell_scripts/plot_on_epoch.sh <rid> [<reference rid>]
#
# WHY A LOOP AND NOT A ONE-SHOT WAITER. A waiter that fires once has to be re-armed by hand after
# every epoch, and the one time it is forgotten the plot silently stops updating while the run
# carries on -- which is indistinguishable, from the outside, from a run that stopped. This loops
# to the epoch count in the run's OWN args.json, so it ends when the arm does and not a moment
# before.
#
# WHY NOT A CRON. A cron fires on the clock and an epoch lands when it lands; at 9.7 min/epoch
# against a 5-minute cron that is either a wasted redraw or a missed one. The checkpoint file
# appearing IS the epoch boundary, so that is what this waits on.
#
# LIVENESS COMES FROM run.log's MTIME, not from the process table. Deriving it from a pid means
# this script has to know which pid, and a pattern match for one is the thing that has twice
# killed this session's own shell. If the log has not been written to in STALL_S the run is gone
# and the loop exits non-zero rather than spinning until someone notices.
#
# IT NEVER TOUCHES THE RUN. Reads checkpoints and logs, writes one png. Safe to start, stop and
# restart at any point; on restart it redraws immediately and then waits for the next epoch.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python

RID="${1:?usage: plot_on_epoch.sh <rid> [<reference rid>]}"
REF="${2:-}"
STALL_S="${STALL_S:-1800}"
OUT="${OUT:-figures/rel_arm_${RID}.png}"

D="training/runs/$RID"
[ -d "$D" ] || { echo "no such run: $D"; exit 2; }

# THE ARM'S OWN EPOCH COUNT, so the loop's end is the run's end. A hardcoded 50 here would keep
# waiting through a 30-epoch arm's silence, or quit on an arm that was resumed for more.
TOTAL=$($PY -I -c "
import json
a = json.load(open('$D/args.json'))
a = a.get('args', a)
print(int(float(a.get('epochs') or 0)))" 2>/dev/null || echo 0)
[ "${TOTAL:-0}" -gt 0 ] || { echo "could not read an epoch count from $D/args.json"; exit 2; }

ARMS="$RID"
[ -n "$REF" ] && ARMS="$RID,$REF"
echo "plot_on_epoch: $RID (+ref ${REF:-none}) -> $OUT, to epoch $TOTAL, stall after ${STALL_S}s"

draw() {
    # THE FAILURE IS REPORTED, NOT SWALLOWED. A plotter that starts raising -- a renamed column, a
    # cache version bump -- would otherwise leave a stale png sitting there looking current.
    if $PY -I run_logging/plot_rel_arm.py "$ARMS" "$OUT" > "$D/plot_on_epoch.last" 2>&1; then
        echo "  epoch $1 -> $OUT  $(TZ=America/New_York date '+%-I:%M %p %Z')"
    else
        echo "  epoch $1 PLOT FAILED, tail:"
        tail -4 "$D/plot_on_epoch.last" | sed 's/^/    /'
    fi
}

_count() { ls "$D"/head_ep*.pt 2>/dev/null | wc -l; }
_latest() {
    ls "$D"/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1
}

N=$(_count)
[ "$N" -gt 0 ] && draw "$(_latest) (initial)"

while :; do
    E=$(_latest)
    if [ -n "${E:-}" ] && [ "$E" -ge "$TOTAL" ]; then
        echo "reached epoch $E of $TOTAL; done at $(TZ=America/New_York date '+%-I:%M %p %Z')"
        exit 0
    fi
    C=$(_count)
    if [ "$C" -gt "$N" ]; then
        N="$C"
        draw "$(_latest)"
        continue
    fi
    MT=$(stat -c %Y "$D/run.log" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    if [ $((NOW - MT)) -gt "$STALL_S" ]; then
        echo "STALL: $D/run.log untouched for $((  (NOW - MT) / 60 )) min at epoch ${E:-0}/$TOTAL"
        exit 3
    fi
    sleep 20
done
