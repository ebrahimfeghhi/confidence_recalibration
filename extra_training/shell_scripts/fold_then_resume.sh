#!/bin/bash
# PAUSE THE ARM AT EPOCH 80, FOLD THE 22 DEGENERATE STRUCTURES, RESUME THE ARM TO 150.
#
# IT WAITS FOR EPOCH 80's EVAL before stopping anything: the eval is already on the cards and the
# 5-epoch cadence would otherwise lose that point for nothing. Stopping after it costs ~10 minutes.
#
# PIDS ARE EXPLICIT, NEVER A PATTERN. pkill -f matches the shell running it; this has self-killed a
# session twice. The torchrun launcher and its three ranks are passed in by number.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID=${RID:?set RID}
LAUNCH_PID=${LAUNCH_PID:?set LAUNCH_PID}
TORCHRUN_PID=${TORCHRUN_PID:?set TORCHRUN_PID}
RANK_PIDS=${RANK_PIDS:?set RANK_PIDS}     # comma separated
EP=${EP:-80}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

say "waiting for epoch $EP's two evals on $RID"
for i in $(seq 1 120); do
    [ -s "training/runs/$RID/metric_hard15_ep$EP.json" ] && \
    [ -s "training/runs/$RID/metric_hard1_ep$EP.json" ] && break
    sleep 30
done
[ -s "training/runs/$RID/metric_hard15_ep$EP.json" ] || say "WARNING: epoch $EP eval never landed; stopping anyway"
say "stopping the arm"
kill "$LAUNCH_PID" "$TORCHRUN_PID" 2>/dev/null
IFS=, read -ra RP <<< "$RANK_PIDS"
for p in "${RP[@]}"; do kill "$p" 2>/dev/null; done
for i in $(seq 1 60); do
    ALIVE=0
    for p in "$TORCHRUN_PID" "${RP[@]}"; do [ -d "/proc/$p" ] && ALIVE=$((ALIVE+1)); done
    [ "$ALIVE" -eq 0 ] && break
    sleep 5
done
for p in "$TORCHRUN_PID" "${RP[@]}"; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
sleep 10
say "cards_busy: $($PY -I run_logging/cards_busy.py 2>&1 | tail -1)"

say "=== fold ==="
./shell_scripts/regen_degenerate_seeds.sh
RC=$?
say "fold exit $RC"
[ "$RC" -eq 0 ] || { say "STOPPING: the fold did not finish cleanly; the arm is NOT resumed"; exit "$RC"; }

# RESUME FROM THE BUNDLE THE PAUSE LEFT, with the argv copied from the run's own args.json rather
# than retyped -- the same rule the resume scripts follow.
BUNDLE=$(ls -t training/runs/$RID/resume_ep*.pt | head -1)
say "=== resuming from $(basename "$BUNDLE") to epoch 150 ==="
PRE=$(ls -td training/runs/*/ | head -1 | xargs basename)
SUBSET_OK=1 \
PREFLIGHT=training/overfit_hard/alignment_preflight.json \
MANIFEST=training/overfit_hard/structures.csv \
SCORED_PAIRS=training/overfit_hard/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=hard15,pool=training/overfit_hard/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=training/overfit_hard/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5" \
CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$PWD/$BUNDLE" \
RESUME_NOTE="paused at epoch 80 to re-fold the 22 degenerate structures; nothing about the arm changed." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/hard150_resume2.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
    N=$(ls -td training/runs/*/ | head -1 | xargs basename)
    [ "$N" != "$PRE" ] && { echo "$N" > training/runs/hard150_rid; say "resumed as $N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"; exit 4; }
    sleep 20
done
say "DONE"
