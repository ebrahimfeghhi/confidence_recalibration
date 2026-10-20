#!/bin/bash
# When 335psh reaches ep50, continue the same arm to ep100. A CONTINUATION: same lr 1e-2, same
# warmup, same mask, same loss, same seed/accum/nsample, so the resume guard passes without
# --resume-allow-drift and the curve joins point for point.
#
# LAUNCHES ONLY ON exit_code 0 and only if resume_ep50.pt exists. A crash means the cards may be
# wanted to diagnose it, which is the user's call.
#
# CANCEL: touch training/runs/CANCEL_335PSH_100, or kill this script BY PID read off ps.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
W=training/runs/335psh
LOG=training/runs/335psh_to100_queue.log
exec >> "$LOG" 2>&1
echo "=== armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
while :; do
    [ -f training/runs/CANCEL_335PSH_100 ] && { echo "CANCELLED"; exit 0; }
    if [ -f "$W/status.json" ]; then
        EC=$($PY -I -c "import json;print(json.load(open('$W/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') 335psh ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT LAUNCHING: exited $EC"; exit 1; }
        break
    fi
    sleep 120
done
B=$W/resume_ep50.pt
[ -f "$B" ] || { echo "NOT LAUNCHING: $B missing; highest bundle:"; ls -1 "$W"/resume_ep*.pt 2>/dev/null | sed 's/.*resume_ep//;s/\.pt//' | sort -n | tail -1; exit 1; }
echo "$(date '+%H:%M:%S') continuing to ep100 from $B"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "CONTINUATION of the lr 1e-2 anchor to ep100: identical settings. The user asked for 100 clean epochs at this rate before adding pathway C." \
  -- --resume "$B" \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 100 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 18 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/val10/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 100 \
  --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "$(date '+%H:%M:%S') launcher returned $?"
