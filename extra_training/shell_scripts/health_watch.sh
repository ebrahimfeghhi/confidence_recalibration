#!/bin/bash
# HEALTH WATCHER, every 10 minutes. Reports; it never launches, kills or resumes anything.
#
# WHY IT ONLY REPORTS. The fallback the user asked for -- "if you can't get the run working just
# resume this one" -- is a decision about which arm the night is spent on, and an automatic
# resumer would race whoever is mid-relaunch and could put two arms on the same three cards. The
# supervising session reads this log and acts; the one thing this must never do is act on its own.
#
# WHAT "HEALTHY" MEANS HERE, in the order a failure actually shows up:
#   1. a chain pid alive, or a scoring pass alive between segments
#   2. the three cards occupied by OUR ranks, not idle
#   3. run.log moving -- a wedged trainer holds the cards and produces nothing, and the chain's
#      own wait has no liveness check, so this is the only place that notices
#   4. the reports being non-empty, which is how the TAG-collision defect would have hidden
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

while :; do
    # THE ARM UNDER SUPERVISION, re-read every tick rather than captured once, so a relaunch
    # under a new chain pid is picked up without restarting this watcher.
    CH=$(ls -t training/runs/fullset_chain_zsoft*.log 2>/dev/null | head -1)
    RID=$(grep -oE 'segment running as [a-z0-9]{6}' "$CH" 2>/dev/null | tail -1 | awk '{print $4}')
    ALIVE=no
    for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        grep -qa 'fullset_chain_zsoft' "/proc/$p/cmdline" 2>/dev/null && { ALIVE="yes(pid $p)"; break; }
    done
    BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
    # run.log MOVEMENT, in seconds since last write. The trainer writes a MEM line per pose, so
    # anything over a few minutes is a wedge rather than a slow step.
    AGE=na
    [ -n "${RID:-}" ] && [ -f "training/runs/$RID/run.log" ] && \
        AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log") ))
    EVAL=$(ls /proc 2>/dev/null | grep -E '^[0-9]+$' | while read p; do
        grep -qa 'phase_k_eval' "/proc/$p/cmdline" 2>/dev/null && echo x; done | wc -l)
    # EMPTY REPORTS ARE A SILENT FAILURE, so they are checked explicitly rather than inferred.
    EMPTY=$(find "$SD" -maxdepth 1 -name 'report_fs_ep*.txt' -size 0 2>/dev/null | wc -l)
    N_REP=$(ls "$SD"/report_fs_ep*.txt 2>/dev/null | wc -l)
    say "chain=$ALIVE rid=${RID:-none} cards_busy=$BUSY run.log_age=${AGE}s eval_procs=$EVAL reports=$N_REP empty=$EMPTY"
    [ "$ALIVE" = "no" ] && [ "$EVAL" = "0" ] && say "  ALERT: no chain and no scoring pass -- the arm is DOWN"
    [ "$BUSY" = "0" ] && [ "$EVAL" = "0" ] && say "  ALERT: all three cards IDLE"
    [ "$AGE" != "na" ] && [ "$AGE" -gt 1200 ] && [ "$EVAL" = "0" ] && \
        say "  ALERT: run.log has not moved in ${AGE}s -- trainer may be wedged"
    [ "$EMPTY" != "0" ] && say "  ALERT: $EMPTY report(s) are EMPTY -- scoring produced no evidence"
    sleep 600
done
