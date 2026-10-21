#!/bin/bash
# When hhzddi reaches ep100, continue the same arm to ep150.
#
# A CONTINUATION, not a fork: same lr (1e-2), same mask, same PAE-only loss, same 30 structures,
# same seed/accum/nsample -- so the resume guard should pass without --resume-allow-drift and the
# curve joins hhzddi's point for point. Early stop stays OFF, which is the whole point: the last
# five epochs of hhzddi were flat at ~1.756 and we do not yet know whether that is the plateau or
# another re-adaptation shelf like ep58-71 was.
#
# LAUNCHES ONLY ON exit_code 0. A crash means the cards may be wanted to diagnose hhzddi, and that
# is the user's call, so this reports and exits rather than claiming them.
#
# CANCEL: touch training/runs/CANCEL_HHZDDI_150, or kill this script BY PID (ps -eo pid,cmd and
# read the pid off the row -- NEVER pkill a pattern, which matches the shell doing the killing and
# has self-killed this session twice).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
WATCH=training/runs/hhzddi
CANCEL=training/runs/CANCEL_HHZDDI_150
LOG=training/runs/hhzddi_to150_queue.log
exec >> "$LOG" 2>&1
echo "=== queue armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

while :; do
    [ -f "$CANCEL" ] && { echo "$(date '+%H:%M:%S') CANCELLED by $CANCEL"; exit 0; }
    if [ -f "$WATCH/status.json" ]; then
        EC=$(/home/ebrahim/Protenix/.venv/bin/python -I -c \
             "import json;print(json.load(open('$WATCH/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') hhzddi ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT LAUNCHING: hhzddi exited $EC; the cards may be wanted for it."; exit 1; }
        break
    fi
    sleep 120
done

BUNDLE=$WATCH/resume_ep100.pt
if [ ! -f "$BUNDLE" ]; then
    echo "NOT LAUNCHING: $BUNDLE does not exist. hhzddi exited 0 but did not reach ep100;"
    echo "the highest bundle it wrote is:"
    ls -1 "$WATCH"/resume_ep*.pt 2>/dev/null | sed 's/.*resume_ep//; s/\.pt//' | sort -n | tail -1
    exit 1
fi

echo "$(date '+%H:%M:%S') launching the continuation to ep150 from $BUNDLE"
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "CONTINUATION of hhzddi to ep150: same lr 1e-2, masked PAE-only, early stop off. hhzddi's ep80-84 sat flat at ~1.756 after breaking through the fork-point plateau at ep72; this is to find out whether that is the floor or another shelf." \
  -- --resume "$BUNDLE" \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 150 --lr 0.01 --warmup-steps 0 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 17 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/overfit30/pool.csv --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "$(date '+%H:%M:%S') launcher returned $?"
