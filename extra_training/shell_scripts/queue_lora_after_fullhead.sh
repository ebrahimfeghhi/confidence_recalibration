#!/bin/bash
# Launch the LoRA r=8 overfit30 arm when mdv9x7 releases cards 4,5,7.
# WAITS ON AN EXPLICIT PID, never a pattern. Pass the torchrun pid as $1.
set -u
PID="${1:?usage: queue_lora_after_fullhead.sh <torchrun-pid>}"
cd /home/ebrahim/confidence_recalibration/extra_training
LOG=training/runs/lora_queue.log
exec >> "$LOG" 2>&1
echo "=== armed $(date '+%Y-%m-%d %H:%M:%S %Z'), waiting on pid $PID ==="
while [ -d "/proc/$PID" ]; do sleep 20; done
echo "$(date '+%H:%M:%S') pid $PID gone"
# THE CARDS MUST ACTUALLY BE FREE, not merely the trainer gone: mdv9x7's last DockQ eval shards
# over 4,5,7 and outlives the ranks. Earlier a queued launch fired into a transient and refused.
for i in $(seq 1 60); do
    BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py 2>/dev/null || echo 9)
    [ "$BUSY" = "0" ] && break
    echo "$(date '+%H:%M:%S') cards busy ($BUSY), waiting"
    sleep 20
done
echo "$(date '+%H:%M:%S') launching the LoRA arm"
./shell_scripts/launch_lora_overfit30.sh
echo "$(date '+%H:%M:%S') launcher returned $?"
