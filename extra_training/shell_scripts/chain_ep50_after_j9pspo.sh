#!/bin/bash
# WAIT FOR j9pspo TO FINISH EPOCH 40, THEN RESUME TO 50.
#
# NO KILL ANYWHERE. j9pspo has --epochs 40 and exits on its own; this waits for its ranks to be
# gone and the cards to be free rather than stopping anything. The worst it can do is fail to
# launch.
#
# RANK PIDS ARE EXPLICIT, never derived by pattern: pgrep/pkill match the shell running them and
# self-killed this session twice on 2026-10-18. These are only READ (/proc/<pid> existence).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RANKS="3409904 3409905 3409906"
echo $$ > training/runs/ep50_chain_pid
echo "$(TZ=America/New_York date '+%F %-I:%M %p %Z') chain armed; waiting for j9pspo (ranks $RANKS)"
for i in $(seq 1 480); do           # 4h ceiling at 30s
  alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && break
  sleep 30
done
echo "$(TZ=America/New_York date '+%-I:%M %p') j9pspo ranks gone"
# THE BUNDLE MUST EXIST, or the resume would be launched against a path that is not there and the
# failure would read as a config problem rather than a missing epoch.
B=training/runs/j9pspo/resume_ep40.pt
for i in $(seq 1 20); do [ -f "$B" ] && break; sleep 15; done
[ -f "$B" ] || { echo "STOP: $B never appeared; epoch 40 did not complete. Nothing launched."; exit 1; }
for i in $(seq 1 40); do
  [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
  sleep 30
done
exec ./shell_scripts/resume_ep41_to_50.sh
