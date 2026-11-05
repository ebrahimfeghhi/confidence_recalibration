#!/bin/bash
# Arm the 13-structure validation fold behind the TRAINING QUEUE ACTUALLY FINISHING.
#
# THE ARMING CONDITION IS THE LESSON FROM TONIGHT. The previous armed script waited on
# `tmux ls | grep -c w1fold` reaching 0, so stopping the workers by hand looked identical to the
# queue finishing -- it fired, launched --fold-only onto cards the training fold still held, and
# cost 42 units plus a doubled per-unit time. folder.py prints "queue complete" only when it has
# drained the queue, so that string is required from all three workers. A manual stop cannot
# produce it.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

say "armed: waiting for 'queue complete' from all three training workers"
while :; do
  N=0
  for c in 4 5 7; do
    grep -q "queue complete" training/fold/w2_c$c.log 2>/dev/null && N=$((N+1))
  done
  [ "$N" -eq 3 ] && break
  # a worker dying without finishing is NOT a drain -- say so and stay armed rather than launch
  A=0
  for c in 4 5 7; do tmux has-session -t w1fold_$c 2>/dev/null && A=$((A+1)); done
  if [ "$A" -eq 0 ] && [ "$N" -lt 3 ]; then
    say "HOLDING: no training workers alive but only $N/3 reported 'queue complete'."
    say "The training queue did not drain, so this is NOT the moment to launch. Staying armed."
    sleep 600
  fi
  sleep 120
done
say "all three training workers reported 'queue complete'"
for c in 4 5 7; do echo "  c$c: $(grep -c 'interfaces in' training/fold/w2_c$c.log) unit(s) scored"; done
say "launching the 13-structure validation fold (1,300 units)"
exec ./shell_scripts/launch_wave1_val_fold.sh
