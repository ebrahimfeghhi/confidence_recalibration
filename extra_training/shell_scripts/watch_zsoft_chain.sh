#!/bin/bash
# OVERNIGHT LIVENESS WATCHER for the zsoftmax chain. Reports; never launches or kills.
#
# WHY IT LOOPS. A one-shot waiter gets forgotten the moment it fires, and the failure worth
# catching here is not the first epoch -- it is a segment boundary at 2 AM leaving the cards idle
# for five hours. So it polls until either 60 epochs are scored or the chain is gone.
#
# IT DOES NOT RESTART ANYTHING. Deciding to relaunch needs a reason, and an automatic restart
# would paper over exactly the segment-boundary bug the review is looking for.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
CHAIN_PID="${CHAIN_PID:?set CHAIN_PID to the fullset_chain_zsoft.sh pid}"
LOG=training/runs/fullset_chain_zsoft3.log
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
while :; do
    if ! [ -d "/proc/$CHAIN_PID" ]; then
        say "CHAIN GONE (pid $CHAIN_PID). Last chain lines:"
        tail -4 "$LOG"
        say "cards: $($PY -I run_logging/cards_busy.py 2>&1 | tail -1)"
        exit 1
    fi
    BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
    LAST=$(grep -E '^###' "$LOG" | tail -1)
    say "chain alive; cards_busy=$BUSY; $LAST"
    if [ "$BUSY" = "0" ]; then
        say "NOTE: cards report free while the chain is alive -- a segment boundary or a scoring"
        say "  pass between GPU phases. Flagged, not acted on."
    fi
    grep -q '=== fs_ep60 ===' "$LOG" 2>/dev/null && { say "epoch 60 scored; chain complete"; exit 0; }
    sleep 600
done
