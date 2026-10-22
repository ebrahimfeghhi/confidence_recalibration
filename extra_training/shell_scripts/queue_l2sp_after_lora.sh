#!/bin/bash
# Launch the L2-SP arm when the LoRA arm releases cards 4,5,7. WAITS ON AN EXPLICIT PID.
set -u
PID="${1:?usage: queue_l2sp_after_lora.sh <torchrun-pid>}"
cd /home/ebrahim/confidence_recalibration/extra_training
LOG=training/runs/l2sp_queue.log
exec >> "$LOG" 2>&1
echo "=== armed $(date '+%Y-%m-%d %H:%M:%S %Z'), waiting on pid $PID ==="
while [ -d "/proc/$PID" ]; do sleep 20; done
echo "$(date '+%H:%M:%S') pid $PID gone"
# THE CARDS MUST ACTUALLY BE FREE, not merely the trainer gone: the final DockQ eval shards over
# 4,5,7 and outlives the ranks. A queued launch already fired into one of those transients today.
for i in $(seq 1 90); do
    BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py 2>/dev/null || echo 9)
    [ "$BUSY" = "0" ] && break
    echo "$(date '+%H:%M:%S') cards busy ($BUSY), waiting"
    sleep 20
done
echo "$(date '+%H:%M:%S') launching the L2-SP arm"
./shell_scripts/launch_l2sp_overfit30.sh
echo "$(date '+%H:%M:%S') launcher returned $?"
