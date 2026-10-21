#!/bin/bash
# Wait for 9mfb4y to finish cleanly, then: stage-timing measurement on ONE freed card, then the
# cropped arm on all three.
#
# WHY IN THIS ORDER. The measurement needs an uncontended card, which only exists in the gap
# between the two runs, and the cropped arm would occupy all three for ~38 h. The gap is ~10 min.
#
# THE CARDS ARE NEVER FREE. The holders (306042/306045/306050) are separate processes that outlive
# both runs and release only on training/runs/HOLD_STOP_457, so nvidia-smi shows our processes on
# 4/5/7 throughout -- including while the measurement runs on one of them.
#
# CANCEL: touch training/runs/CANCEL_CROPPED_QUEUE, or kill this script by PID.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
R=training/runs/9mfb4y
CANCEL=training/runs/CANCEL_CROPPED_QUEUE
exec >> training/runs/cropped_queue.log 2>&1
echo "=== measure-then-launch armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

while true; do
    if [ -f "$CANCEL" ]; then echo "$(date '+%H:%M:%S') CANCELLED"; exit 0; fi
    if [ -f "$R/status.json" ]; then
        EC=$(/home/ebrahim/Protenix/.venv/bin/python -I -c \
             "import json;print(json.load(open('$R/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') 9mfb4y finished, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT PROCEEDING: exit $EC. Cards may be wanted to resume it."; exit 1; }
        break
    fi
    if ! kill -0 333732 2>/dev/null && ! kill -0 333733 2>/dev/null && ! kill -0 333734 2>/dev/null; then
        echo "$(date '+%H:%M:%S') NOT PROCEEDING: ranks gone with no status.json, so it was killed"
        echo "rather than finished. The cards may be wanted to resume it; waiting for the user."
        exit 1
    fi
    sleep 120
done

echo "$(date '+%H:%M:%S') --- stage-timing measurement on card 4 (uncontended now) ---"
CUDA_VISIBLE_DEVICES=4 LAYERNORM_TYPE=openfold timeout 2400 \
  /home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_eval_stage_timing.py --n 25 \
  > training/runs/eval_stage_timing.log 2>&1
echo "$(date '+%H:%M:%S') measurement exited $?; results in training/runs/eval_stage_timing.log"

if [ -f "$CANCEL" ]; then echo "$(date '+%H:%M:%S') CANCELLED before launch"; exit 0; fi
echo "$(date '+%H:%M:%S') --- launching the cropped arm on 4,5,7 ---"
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
