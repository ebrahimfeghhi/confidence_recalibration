#!/bin/bash
# Wait for e4tvxe's epoch-6 eval to land, then stop it and resume from resume_ep6.pt so the
# remaining evals get the optimisations the trainer only started passing in c1338812.
#
# WHY A RESUME RATHER THAN A FRESH RUN. Python holds the module it loaded at launch, so the
# --join-cache / --pair-iptm-only / --prefetch change cannot reach a process already running. A
# fresh run would get them but throw away six epochs; resuming keeps them. The resume is a clean
# continuation -- same lr, accum, seed, nsample, mask, loss and manifest -- so the guard passes
# without --resume-allow-drift and the curve joins point for point.
#
# WAITS FOR THE EVAL FIRST so its data point is not lost: the bundle for epoch 6 is already on
# disk, but metric_train_ep6.json is not, and killing mid-eval would discard ~9 minutes of
# finished GPU work for nothing.
#
# WHAT IT SAVES. Each eval was 8.9 min against ~1.7 min of I/O for its 156 GB of trunks; the cost
# was recomputing the pose->sample join, 1,670 OST chain mappings per shard, which is a constant
# for a fixed pool. 22 evals remain, so the saving is hours rather than minutes.
#
# CANCEL: touch training/runs/CANCEL_HANDOFF, or kill this script BY PID read off ps -- NEVER
# pkill a pattern, which matches the shell doing the killing and has self-killed this session.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
TORCHRUN=3504634
LOG=training/runs/handoff.log
exec >> "$LOG" 2>&1
echo "=== handoff armed $(date '+%H:%M:%S') for torchrun $TORCHRUN ==="

while :; do
    [ -f training/runs/CANCEL_HANDOFF ] && { echo "CANCELLED"; exit 0; }
    [ -f training/runs/e4tvxe/metric_train_ep6.json ] && { echo "$(date '+%H:%M:%S') ep6 eval landed"; break; }
    [ -d /proc/$TORCHRUN ] || { echo "$(date '+%H:%M:%S') torchrun $TORCHRUN already gone; not relaunching"; exit 1; }
    sleep 20
done

# THE LITERAL PID, never a pattern. Also give the ranks a moment: the bundle is already written,
# so there is nothing to flush, but a half-dead rank still holding GPU memory would make the next
# run's preload check fail.
echo "$(date '+%H:%M:%S') stopping torchrun $TORCHRUN"
kill -TERM $TORCHRUN
for i in $(seq 1 40); do [ -d /proc/$TORCHRUN ] || break; sleep 2; done
if [ -d /proc/$TORCHRUN ]; then echo "STILL ALIVE after 80 s; not relaunching"; exit 1; fi
echo "$(date '+%H:%M:%S') stopped"

B=training/runs/e4tvxe/resume_ep6.pt
[ -f "$B" ] || { echo "NOT LAUNCHING: $B missing"; exit 1; }
echo "$(date '+%H:%M:%S') resuming from $B"
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "CONTINUATION of e4tvxe from ep6 to ep50: identical settings, relaunched only so the evals pick up --join-cache / --pair-iptm-only / --prefetch, which the trainer began passing in c1338812 and which a running process cannot acquire. lr 1e-1, warmup 20, masked PAE-only, val10 eval every 2 epochs at budget 100." \
  -- --resume "$B" \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.1 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
