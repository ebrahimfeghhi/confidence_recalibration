#!/bin/bash
# When owdyyp ends, start a FRESH PAE-only arm for 100 epochs.
#
# NOT a resume. owdyyp is PAE-only but inherited 26 epochs of four-term training, so it is a
# hybrid; this starts from the shipped head, so the arm is PAE-only end to end. Same lr, same
# mask, same 30 structures, so it stays comparable to every other arm point for point.
#
# LAUNCHES ONLY ON exit_code 0. A crash means the cards may be wanted to diagnose owdyyp, and
# that is the user's call, so this reports and exits rather than claiming them.
#
# CANCEL: touch training/runs/CANCEL_PAEONLY_FRESH, or kill this script BY PID (ps -eo pid,cmd
# and read the pid off the row -- NEVER pkill a pattern, which matches the shell doing the
# killing and has self-killed this session twice).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
WATCH=training/runs/owdyyp
CANCEL=training/runs/CANCEL_PAEONLY_FRESH
LOG=training/runs/paeonly_fresh_queue.log
exec >> "$LOG" 2>&1
echo "=== queue armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

while :; do
    [ -f "$CANCEL" ] && { echo "$(date '+%H:%M:%S') CANCELLED by $CANCEL"; exit 0; }
    if [ -f "$WATCH/status.json" ]; then
        EC=$(/home/ebrahim/Protenix/.venv/bin/python -I -c \
             "import json;print(json.load(open('$WATCH/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') owdyyp ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT LAUNCHING: owdyyp exited $EC; the cards may be wanted for it."; exit 1; }
        break
    fi
    sleep 120
done

echo "$(date '+%H:%M:%S') launching fresh PAE-only arm, 100 epochs"
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "FRESH PAE-only from the shipped head: masked, lr 5e-3, 100 epochs, early stop patience 8. Unlike owdyyp this never saw the four-term loss." \
  -- --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 100 --lr 0.005 --warmup-steps 0 --seed 0 --accum 85 --nsample 5 \
  --early-stop-patience 8 --early-stop-min-delta 0.01 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 17 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/overfit30/pool.csv --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "$(date '+%H:%M:%S') launcher returned $?"
