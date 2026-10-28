#!/bin/bash
# Block until fawgtu finishes another EPOCH, then exit so the session is re-invoked and can SEND
# the train-PAE plot. This arm computes no metrics, so the epoch boundary is the only event there
# is; resume_ep<N>.pt is what marks it.
#
# RANK PIDS ARE EXPLICIT, never derived by pattern: pgrep/pkill match the shell running them and
# self-killed this session twice on 2026-10-18. These are only READ (/proc/<pid> existence) and
# nothing here signals anything.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
HAVE="${1:-0}"
RANKS="2122805 2122806 2122807"
count() { ls -1 training/runs/fawgtu/resume_ep*.pt 2>/dev/null | wc -l; }
for i in $(seq 1 3600); do          # 20h ceiling at 20s
  n=$(count)
  if [ "$n" -gt "$HAVE" ] 2>/dev/null; then echo "EPOCH_READY epochs=$n"; exit 0; fi
  alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && { echo "RUN_GONE epochs=$n"; exit 0; }
  sleep 20
done
echo "TIMEOUT epochs=$(count)"
