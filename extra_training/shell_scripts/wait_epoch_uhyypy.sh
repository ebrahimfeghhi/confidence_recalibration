#!/bin/bash
# Block until uhyypy finishes epoch $1's DockQ eval (the LAST thing in the cycle -- val_loss.csv
# lands several minutes earlier, which plots a half-filled epoch), then exit so the session can SEND.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
HAVE="${1:-0}"
rows() { grep -cE "^  epoch [0-9]+ TRAIN:" training/runs/uhyypy/run.log 2>/dev/null || echo 0; }
for i in $(seq 1 1400); do
  n=$(rows)
  if [ "$n" -gt "$HAVE" ] 2>/dev/null; then echo "EPOCH_READY epochs_evaluated=$n"; exit 0; fi
  alive=0; for P in 699045 699046 699047; do [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && { echo "RUN_GONE epochs_evaluated=$n"; exit 0; }
  sleep 18
done
echo "TIMEOUT epochs_evaluated=$(rows)"
