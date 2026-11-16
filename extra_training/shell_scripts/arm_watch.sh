#!/bin/bash
# OVERNIGHT WATCHER for the zsoftmax chain. Reports every 10 minutes; never launches or kills.
#
# IT TAKES THE CHAIN PID, IT DOES NOT GUESS IT. The first version of this scanned /proc for a
# cmdline containing "fullset_chain_zsoft" -- which also matches the nohup shell that STARTED the
# chain, so it reported "chain alive" with the chain dead. A watcher whose failure mode is a false
# healthy is worse than no watcher, so the pid is an argument and liveness is `[ -d /proc/$pid ]`.
#
# THE CHAIN LOG IS RE-DISCOVERED EVERY TICK, for the same class of reason: the previous version
# hardcoded fullset_chain_zsoft3.log and kept reporting a dead run's last line after the arm was
# relaunched as ...4.log.
#
# WHAT IT WATCHES, in the order a failure actually shows up:
#   1. the chain pid, and a scoring pass as the legitimate reason the trainer is absent
#   2. the three cards occupied rather than idle
#   3. run.log moving -- a wedged trainer holds the cards and produces nothing, and the chain's
#      own wait has no liveness check, so this is the only place that would notice
#   4. reports non-empty, which is how the scratchpad TAG collision hid itself
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
CHAIN_PID="${CHAIN_PID:?pass CHAIN_PID=<pid of the chain script>}"
# WHICH ARM. The log glob and the report tag both belong to one arm; hardcoding them
# meant the watcher kept reading the previous arm's log after a new one started.
CHAIN_GLOB="${CHAIN_GLOB:-training/runs/fullset_chain_lr1e4*.log}"
TAGP="${TAGP:-lr4}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
DOWN=0

say "watching chain pid $CHAIN_PID"
while :; do
    LOG=$(ls -t $CHAIN_GLOB 2>/dev/null | head -1)
    RID=$(grep -oE 'segment running as [a-z0-9]{6}' "$LOG" 2>/dev/null | tail -1 | awk '{print $4}')
    ALIVE=$([ -d "/proc/$CHAIN_PID" ] && echo yes || echo NO)
    BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
    EVAL=0
    for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        grep -qa 'phase_k_eval' "/proc/$p/cmdline" 2>/dev/null && EVAL=$((EVAL+1))
    done
    AGE=na
    [ -n "${RID:-}" ] && [ -f "training/runs/$RID/run.log" ] && \
        AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log") ))
    EMPTY=$(find "$SD" -maxdepth 1 -name "report_${TAGP}_ep*.txt" -size 0 2>/dev/null | wc -l)
    NREP=$(ls "$SD"/report_${TAGP}_ep*.txt 2>/dev/null | wc -l)
    EP=$(grep -oE "=== ${TAGP}_ep[0-9]+ ===" "$LOG" 2>/dev/null | tail -1)
    say "chain=$ALIVE rid=${RID:-none} cards=$BUSY run.log_age=${AGE}s eval=$EVAL reports=$NREP empty=$EMPTY last_scored=${EP:-none}"
    if [ "$ALIVE" = "NO" ] && [ "$EVAL" = "0" ]; then
        DOWN=$((DOWN+10)); say "  ALERT: chain gone and no scoring pass -- DOWN for ~${DOWN} min"
        tail -3 "$LOG" 2>/dev/null | sed 's/^/    /'
    else
        DOWN=0
    fi
    [ "$BUSY" = "0" ] && [ "$EVAL" = "0" ] && say "  ALERT: all three cards IDLE"
    [ "$AGE" != "na" ] && [ "$AGE" -gt 1200 ] && [ "$EVAL" = "0" ] && \
        say "  ALERT: run.log has not moved in ${AGE}s -- trainer may be wedged"
    [ "$EMPTY" != "0" ] && say "  ALERT: $EMPTY report(s) EMPTY -- scoring produced no evidence"
    grep -q "=== ${TAGP}_ep60 ===" "$LOG" 2>/dev/null && { say "epoch 60 scored -- COMPLETE"; exit 0; }
    sleep 600
done
