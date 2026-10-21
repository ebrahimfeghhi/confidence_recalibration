#!/bin/bash
# When x5a4uj (lr 1e-2) finishes, resume pykssn (lr 5e-3, masked) from epoch 6 to 30.
#
# LAUNCHES ONLY ON A CLEAN FINISH. If x5a4uj dies instead, the cards may be wanted to diagnose or
# resume IT, and that is the user's call -- so this reports and exits rather than claiming them.
#
# CANCEL: touch the cancel file, or kill this script BY PID.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
WATCH=training/runs/owdyyp
CANCEL=training/runs/CANCEL_X5A4UJ_RESUME
LOG=training/runs/x5a4uj_resume_queue.log
exec >> "$LOG" 2>&1
echo "=== queue armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

while :; do
    [ -f "$CANCEL" ] && { echo "$(date '+%H:%M:%S') CANCELLED by $CANCEL"; exit 0; }
    if [ -f "$WATCH/status.json" ]; then
        EC=$(/home/ebrahim/Protenix/.venv/bin/python -I -c \
             "import json;print(json.load(open('$WATCH/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') x3888d finished, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT LAUNCHING: x3888d exited $EC; the cards may be wanted for it."; exit 1; }
        break
    fi
    sleep 120
done

# The newest resume bundle, not a hardcoded one: if x5a4uj's run let pykssn write more, take them.
CKPT=$(ls -1 training/runs/x5a4uj/resume_ep*.pt 2>/dev/null \
       | sed 's/.*resume_ep\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
[ -n "$CKPT" ] || { echo "NOT LAUNCHING: no resume bundle under training/runs/x5a4uj"; exit 1; }
echo "$(date '+%H:%M:%S') resuming x5a4uj (lr 1e-2) from $CKPT"

LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "RESUME of x5a4uj (PAE masked, lr 1e-2) from $CKPT, to epoch 30" \
  -- --resume "$CKPT" \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 30 --lr 0.01 --warmup-steps 0 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 17 \
  --pae-scored-pairs training/overfit30/pool.csv \
  --train-eval-pool training/overfit30/pool.csv --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "$(date '+%H:%M:%S') launcher returned $?"
