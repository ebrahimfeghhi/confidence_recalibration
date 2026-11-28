#!/usr/bin/env bash
# QUEUE THE SHIPPED HEAD'S PASSES FOR WHEN THE CARDS CLEAR. Fire and leave it; it launches nothing
# while a run owns the cards.
#
#   nohup ./shell_scripts/queue_shipped_head.sh > queue_shipped_head.log 2>&1 &
#
# WAITS, NEVER PRE-EMPTS. Arm E (training/runs/8yijzw, 20 epochs, ~2.5 h on cards 4,5,7) is the
# result being waited for; killing it to save twelve minutes would be the wrong trade. cards_busy.py
# reads GPU OCCUPANCY rather than command-line text -- a text check matches the checking command
# itself, which is how a queued launch once refused against an idle box -- and it fails CLOSED, so
# an unreadable nvidia-smi reads as busy.
#
# AND IT LAUNCHES ONLY WHAT IS MISSING. score_shipped_head.sh prints an existing metric json instead
# of re-scoring it, because the shipped head is fixed and so is its number for a fixed pool, budget
# and manifest. The val pass of 2026-11-11 is already on disk under both poolings
# (training/valfull83/metric_shipped_b30_141.json: max 0.5813 / 0.7234, mean 0.5735 / 0.7163), so in
# the normal case the only GPU work left here is the in-sample side.
set -uo pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
CARDS=${CARDS:-4,5,7}
SIDES=${SIDES:-both}
WAIT_MIN=${WAIT_MIN:-300}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

# THE SIDES ALREADY ON DISK ARE REPORTED BEFORE ANY WAITING. A queue script that prints nothing for
# five hours and then prints a number that was available at minute zero is a script nobody reads.
say "what the shipped head already has (no GPU, no waiting):"
REPORT_ONLY=1 SIDES=$SIDES ./shell_scripts/score_shipped_head.sh 2>&1 | sed 's/^/    /'

say "waiting for cards $CARDS (ceiling ${WAIT_MIN} min); Arm E 8yijzw is the run in front"
N=99
for _ in $(seq 1 $(( WAIT_MIN * 2 )) ); do
  N="$($PY -I run_logging/cards_busy.py "$CARDS" 2>/dev/null | tail -1)"
  case "$N" in ''|*[!0-9]*) N=99;; esac
  [ "$N" -eq 0 ] && break
  sleep 30
done
[ "$N" -eq 0 ] || { say "NOT LAUNCHED: cards still busy ($N) after ${WAIT_MIN} min. Nothing was run."; exit 3; }

say "cards clear -- scoring the SHIPPED head (no --ckpt, NO pathway installed), both poolings"
SIDES=$SIDES ./shell_scripts/score_shipped_head.sh
RC=$?
say "score_shipped_head.sh exit $RC"
exit $RC
