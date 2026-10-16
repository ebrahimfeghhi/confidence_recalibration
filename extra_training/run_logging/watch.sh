#!/bin/bash
# Re-render a run's report every --every seconds, for as long as the run is going.
#
# WHAT THIS DOES AND DOES NOT DO. It refreshes the HTML on disk. It does NOT publish: publishing an
# artifact goes through Claude's own tool, so the PUBLISHED page updates when Claude republishes it
# (on a status check, or when asked), while the file here stays at most --every seconds stale. Said
# plainly so nobody stares at a published page wondering why it is a few minutes behind.
#
# It is cheap by design -- reading a csv and writing 18 KB of html -- so it does not compete with the
# training it is watching, and it keeps going if a render fails rather than taking the watcher down
# with it.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/home/ebrahim/Protenix/.venv/bin/python
RUN=${1:?usage: watch.sh <run out dir> [report html] [seconds]}
OUT=${2:-$R/run_logging/reports/$(basename "$RUN").html}
EVERY=${3:-300}
LOG=$R/run_logging/watch.log
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "watching $RUN -> $OUT every ${EVERY}s"
while true; do
    if ! $PY -I "$R/run_logging/render_report.py" --run "$RUN" \
            --baseline "$R/training/ft3_pilot_baseline.json" --out "$OUT" >> "$LOG" 2>&1; then
        say "render failed; keeping the previous report and trying again next tick"
    fi
    # Stop on its own once the run is gone AND one final render has been taken, so the last report
    # reflects the finished run rather than its second-to-last tick.
    # BRACKETED so the pattern cannot match this watcher's own command line. A plain
    # "phase_k_train.py" here would appear in the pgrep invocation itself and the watcher would
    # never see the run end -- the same self-match that made a status check report 3 phantom
    # training processes earlier today.
    if ! pgrep -f "[p]hase_k_train\.py" > /dev/null 2>&1; then
        say "no training process left; final report written, watcher exiting"
        exit 0
    fi
    sleep "$EVERY"
done
