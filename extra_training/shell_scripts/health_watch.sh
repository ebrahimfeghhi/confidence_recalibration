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
    # THE RID COMES FROM THE ARGUMENT, OR FROM THE LIVE POINTER -- NOT from a chain log.
    # This used to re-derive the rid from the newest fullset_chain_zsoft*.log, a chain that has
    # not run since 2026-11-16. Every watcher started since then reported on that dead run: the
    # health file for an arm read `rid=2tr3sj run.log_age=220834s` while the arm it was launched
    # for was training normally. All three ALERT conditions were therefore evaluated against a
    # run that ended days ago, so SILENCE FROM THIS LOG MEANT NOTHING. A watcher that cannot
    # fire is worse than no watcher, because it is trusted.
    RID="${1:-}"
    [ -n "$RID" ] || RID=$(cat training/runs/pose6_norms_rid 2>/dev/null)
    ALIVE=no
    # LIVENESS FROM THE RUN'S OWN LOG, not from a cmdline scan: the arm's pid belongs to torchrun
    # ranks whose argv this watcher has no reliable handle on, and a /proc pattern scan is exactly
    # what is banned in this project for matching the shell that runs it.
    if [ -n "$RID" ] && [ -f "training/runs/$RID/run.log" ]; then
        _age=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log") ))
        [ "$_age" -lt 1800 ] && ALIVE="yes(log ${_age}s old)"
    fi
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
