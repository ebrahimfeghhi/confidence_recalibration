#!/bin/bash
# WAIT FOR THE NEXT SCORED EPOCH, redraw this arm's figure, print the row. Reports and never acts.
#
# WHY A SCRIPT AND NOT A ONE-LINER. The figure has to be sent by the session, not by a shell, so
# every epoch costs one re-arm; keeping the wait in a file means the re-arm is a single short
# command and cannot drift from the last one. It exits as soon as one epoch lands, on purpose:
# a loop would redraw the png repeatedly with nobody sending it.
#
# LIVENESS, SO IT CANNOT WAIT FOREVER on a dead run: it gives up if run.log has not been written
# to for STALL seconds, and says so rather than sitting silent.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
TAGP="${TAGP:-lr4}"
RID="${RID:?pass RID}"
E="${E:?pass E=<the epoch to wait for>}"
STALL="${STALL:-2400}"
OUT="${OUT:-figures/pae_lr1e4.png}"
TITLE="${TITLE:-PAE ONLY, lr 1e-4, dropout 0.25 -- val EVERY epoch   }"
while [ ! -s "$SD/report_${TAGP}_ep$E.txt" ]; do
    AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$RID/run.log" 2>/dev/null || echo 0) ))
    if [ "$AGE" -gt "$STALL" ]; then
        echo "### STALLED: run.log for $RID has not moved in ${AGE}s while waiting for epoch $E"
        exit 1
    fi
    sleep 60
done
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') epoch $E scored"
grep -E "^VAL \(141|^TRAIN SLICE \(280" "$SD/report_${TAGP}_ep$E.txt" | sed 's/^/    /'
TITLE="$TITLE" REF_RID="" $PY -I run_logging/plot_fullset.py "$OUT" 2>&1 | tail -2
