#!/bin/bash
# QUEUE THE zsoftmax 1e-4 ARM BEHIND THE RE-SCORE. The four val passes hold all three cards, so
# the arm cannot start until they finish; this waits for the re-score to say so, re-checks that
# the cards are actually free, and only then launches.
#
# cards_busy IS THE GATE, NOT THE LOG LINE. A log saying COMPLETE and a card still held is the
# case that matters -- a shard can outlive its parent -- so the launch waits on the card check and
# gives up rather than starting on top of someone else.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
WAIT_FOR=${WAIT_FOR:-$SD/rescore_zsoft.log}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

say "waiting for the re-score to finish ($WAIT_FOR)"
# THE TAIL, NOT THE WHOLE FILE. A success line from an earlier re-score into the same path
# would satisfy a whole-file grep even when this pass printed ABORT.
until tail -5 "$WAIT_FOR" 2>/dev/null | grep -qE 'RESCORE COMPLETE|ABORT|STOP:'; do sleep 60; done
if ! tail -5 "$WAIT_FOR" | grep -q 'RESCORE COMPLETE'; then
    say "NOT LAUNCHING: the re-score did not complete cleanly"
    grep -E 'ABORT|STOP:' "$WAIT_FOR" | tail -3
    exit 3
fi
say "re-score complete; waiting for the cards to clear"
for i in $(seq 1 60); do
    # EXIT 0 MEANS "I COUNTED", NOT "FREE". cards_busy.py always returns 0 when it can
    # establish occupancy and exits 2 only when it CANNOT; gating on the status therefore passed
    # immediately whatever the cards held, and waited only in the unknown case -- backwards.
    # The count is on stdout, which is what every other caller reads.
    N=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    [ "$N" = "0" ] && { say "cards clear"; break; }
    say "  $N card(s) still busy"
    [ "$i" = 60 ] && { say "NOT LAUNCHING: cards still busy after 30 min"; exit 4; }
    sleep 30
done
say "launching PAE + zsoftmax at lr 1e-4, 60 epochs, val every 2, slice every 10"
# NOHUP AND A LOG. The launcher prints the recorded argv, the card gate and any REFUSING reason;
# launched in the foreground and unredirected, all of that lived only on this script stdout.
LOG=training/runs/zsoft_lr1e4_launch.log
EPOCHS=60 EVAL_EVERY=2 LR=1e-4 nohup ./shell_scripts/launch_zsoft_lr1e4.sh > "$LOG" 2>&1 &
LPID=$!
say "launcher pid $LPID, log $LOG"
for i in $(seq 1 90); do
    grep -qE "^run [a-z0-9]{6}|REFUS|STOP:" "$LOG" 2>/dev/null && break
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        tail -5 "$LOG"; exit 5; }
    sleep 20
done
grep -E "^run [a-z0-9]{6}|REFUS|STOP:" "$LOG" | head -3
RID=$(grep -oE "^run [a-z0-9]{6}" "$LOG" | head -1 | awk "{print \$2}")
[ -n "$RID" ] && { echo "$RID" > training/runs/zsoft_lr1e4_rid; say "run $RID"; } \
    || { say "ABORT: no run id"; exit 6; }
