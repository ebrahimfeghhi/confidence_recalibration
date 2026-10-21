#!/bin/bash
# QUEUE the cropped arm: it launches itself once run 9mfb4y finishes, and not before.
#
#   cropped, all chain pairs, 5 A contact, 10 A margin; lr 5e-3, NO warmup, 15 epochs.
#
# WHY IT WAITS RATHER THAN RUNS. Cards 4/5/7 are running 9mfb4y's epochs 4-5. The holders
# (306042/306045/306050) are separate processes that outlive the run, so the cards stay ours in the
# gap between the two runs and nobody can take them while this is waiting.
#
# IT LAUNCHES ONLY ON A CLEAN FINISH. If 9mfb4y dies instead, the cards may be wanted for a resume
# and that is the user's call, so this reports and exits rather than claiming them.
#
# CANCEL: touch the file named in CANCEL below, or kill this script by PID.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
R=training/runs/9mfb4y
CANCEL=training/runs/CANCEL_CROPPED_QUEUE
LOG=training/runs/cropped_queue.log
exec >> "$LOG" 2>&1
echo "=== queue armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

while true; do
    if [ -f "$CANCEL" ]; then
        echo "$(date '+%H:%M:%S') CANCELLED by $CANCEL; not launching"; exit 0
    fi
    if [ -f "$R/status.json" ]; then
        EC=$(/home/ebrahim/Protenix/.venv/bin/python -I -c \
             "import json;print(json.load(open('$R/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') 9mfb4y finished, exit_code=$EC"
        if [ "$EC" != "0" ]; then
            echo "NOT LAUNCHING: 9mfb4y exited $EC. The cards may be wanted to resume it;"
            echo "that is the user's decision. Launch by hand when they say so."
            exit 1
        fi
        break
    fi
    # the ranks are gone but no status.json means it was killed, not finished
    if ! kill -0 333732 2>/dev/null && ! kill -0 333733 2>/dev/null && ! kill -0 333734 2>/dev/null; then
        echo "$(date '+%H:%M:%S') NOT LAUNCHING: 9mfb4y's ranks are gone with no status.json,"
        echo "so it was killed rather than finished (a hangup did exactly this at 09:56 PT)."
        echo "The cards may be wanted to resume it; waiting for the user instead."
        exit 1
    fi
    sleep 120
done

echo "$(date '+%H:%M:%S') launching the cropped arm"
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "cropped arm: 10 A margin all-pairs, lr 5e-3, no warmup, 15 epochs" \
  -- --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/runs/9mfb4y/alignment_preflight.json \
  --epochs 15 --lr 0.005 --warmup-steps 0 --seed 0 --accum 256 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 24 --eval-every-epoch --eval-budget 100 \
  --shuffle structure --seeds 500000-500009 \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "$(date '+%H:%M:%S') launcher returned $?"
