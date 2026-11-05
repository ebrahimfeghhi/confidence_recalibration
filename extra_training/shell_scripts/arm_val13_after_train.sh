#!/bin/bash
# Arm the 13-structure validation fold behind the training fold ACTUALLY being finished.
#
# WHY THE GATE IS FOUR CHECKS AND NOT A LOG LINE. Two earlier versions of this were wrong.
#   v1 waited on `tmux ls | grep -c w1fold` reaching 0. Stopping the workers by hand looked
#      identical to the queue finishing, so it fired, launched --fold-only onto cards the training
#      fold still held, and cost 42 units plus a doubled per-unit time.
#   v2 waited for "queue complete" in all three logs. Raised in review, and correct: folder.py
#      prints that line with its failed tally included, and then runs a SECOND PASS over whatever
#      is still unscored. The message therefore appears while the worker is still folding.
#
# So the gate is the state, not the narration: the worker processes are gone, the cards are free,
# and the queue is drained according to the trunks and the csv. All four must hold together.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

PIDS="${TRAIN_PIDS:-}"
[ -z "$PIDS" ] && { say "REFUSING to arm: set TRAIN_PIDS to the training worker pids."; exit 1; }
say "armed. training worker pids: $PIDS"
say "gate = (1) those pids gone  (2) no w1fold tmux session  (3) cards_busy 0  (4) queue drained by data"

while :; do
  ALIVE=0
  for p in $PIDS; do [ -d /proc/$p ] && ALIVE=$((ALIVE+1)); done
  SESS=$(tmux ls 2>/dev/null | grep -c w1fold || true)
  if [ "$ALIVE" -gt 0 ] || [ "$SESS" -gt 0 ]; then sleep 120; continue; fi

  B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
  if [ "${B:-1}" != "0" ]; then
    say "HOLDING: workers gone but cards_busy reports '$B'. Not launching onto occupied cards."
    sleep 300; continue
  fi

  if ! $PY -I run_logging/training_queue_drained.py; then
    say "HOLDING: the workers are gone but the training queue is NOT drained (above)."
    say "Folds are missing or unscored, so this is not the moment to launch. Staying armed."
    say "Nothing will start until the training queue is complete -- rescore or refold first."
    sleep 600; continue
  fi
  break
done

say "gate passed: workers exited, cards free, training queue drained"
say "launching the 13-structure validation fold (1,300 units)"
exec ./shell_scripts/launch_wave1_val_fold.sh
