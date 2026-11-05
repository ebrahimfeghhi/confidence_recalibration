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
say "gate = (1) those pids gone  (2) no w1fold tmux session  (3) cards_busy 0  (4) no queued unit still needs a GPU"

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

  # --gpu-only ASKS THE RIGHT QUESTION: is training finished WITH THE CARDS? A unit that is
  # folded and merely unscored, with its poses still on disk, needs ost on a CPU -- holding three
  # idle GPUs for that is waste, and the rescore runs alongside the validation fold (different
  # results.csv, so they do not contend). A unit whose poses are GONE still needs a refold, and
  # that counts as GPU work and holds. Without --gpu-only the 867-unit rescore backlog would have
  # blocked the validation launch outright.
  if ! $PY -I run_logging/training_queue_drained.py --gpu-only; then
    say "HOLDING: the workers are gone but units still need a GPU (above) -- folds missing, or"
    say "unscored with no poses to rescore from. Not the moment to launch. Staying armed."
    sleep 600; continue
  fi
  break
done

say "gate passed: workers exited, cards free, no queued training unit needs a GPU"
say "launching the 13-structure validation fold (1,300 units)"
exec ./shell_scripts/launch_wave1_val_fold.sh
