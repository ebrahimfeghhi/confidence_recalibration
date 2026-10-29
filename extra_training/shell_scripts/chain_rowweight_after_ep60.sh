#!/bin/bash
# QUEUE launch_rowweight_cosine.sh to start once 45q96x has written resume_ep60.pt and freed the
# cards. At the user's direction: "Just queue the run when you've addressed the sub agent bugs."
#
# WHY IT WAITS ON THE CHECKPOINT AND NOT ON THE PROCESS. A rank that is still writing
# resume_ep60.pt holds its card, and cards_busy.py is what decides whether the launch is allowed;
# waiting on the file and THEN on cards_busy covers both the "epoch finished" and the "card
# actually released" conditions, which are not the same moment.
#
# IT NEVER KILLS ANYTHING. If the cards do not free within the timeout it gives up and says so,
# leaving 45q96x alone.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training

# ONE LAUNCHER, ENFORCED BY flock. Two copies of this script both waiting on the same checkpoint
# would both see the cards free in the same instant and both launch, and the cards_busy guard
# inside the launch script cannot prevent that -- it runs before either has taken a card. flock
# makes the second copy exit immediately instead. It also means a copy whose shell died without
# the script dying cannot be double-queued: the lock is held by the process, not written to disk.
exec 9>training/runs/.rowweight_chain.lock
if ! flock -n 9; then
  echo "another copy of this chain already holds the lock; exiting"
  exit 0
fi

PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
CKPT=training/runs/45q96x/resume_ep60.pt
DEADLINE=$(( $(date +%s) + 7200 ))

echo "$(TZ=America/New_York date '+%I:%M %p %Z')  waiting for $CKPT"
while [ ! -f "$CKPT" ]; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: $CKPT never appeared"; exit 1; }
  sleep 20
done
echo "$(TZ=America/New_York date '+%I:%M %p %Z')  epoch 60 checkpoint is on disk; waiting for the cards"

while :; do
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: cards still busy ($BUSY)"; exit 1; }
  sleep 20
done
echo "$(TZ=America/New_York date '+%I:%M %p %Z')  cards free; launching"
exec ./shell_scripts/launch_rowweight_cosine.sh
