#!/bin/bash
# INDEPENDENT SAFETY NET: resume erb7nd if the orchestrator died without resuming it.
#
#   nohup ./shell_scripts/predcrop_resume_safetynet.sh > training/runs/predcrop_safetynet.log 2>&1 &
#
# WHY A SECOND PROCESS. predcrop_eval_at_boundary.sh resumes from an EXIT trap, which covers a
# normal exit, an error and a SIGTERM -- verified. It does NOT cover SIGKILL, an OOM-killer, or
# the machine dropping the process. The user's instruction was unconditional ("PLEASE MAKE SURE
# TO RESUME THE UNCROPPED RUN"), so the trap gets a backstop that does not share its fate.
#
# IT CANNOT DOUBLE-LAUNCH. The orchestrator writes the sentinel BEFORE it launches, so
# "sentinel absent AND orchestrator gone" means the orchestrator died without reaching the
# resume. Both conditions must hold, plus the cards being free and no trainer alive.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID=erb7nd
EP=${1:-9}
SENTINEL=training/runs/.predcrop_resumed
PIDF=training/runs/predcrop_eval_pid
DEADLINE=$(( $(date +%s) + 43200 ))
say() { echo "$(TZ=America/New_York date '+%F %I:%M:%S %p %Z') $*"; }
say "safety net armed for $RID epoch $EP"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 120
  # 1. did the orchestrator already resume (or is it about to)?
  [ -f "$SENTINEL" ] && { say "sentinel present; orchestrator owns the resume. standing down."; exit 0; }
  # 2. is the orchestrator still alive? A DIRECT READ OF A KNOWN PID, never a pattern scan.
  OPID=$(cat "$PIDF" 2>/dev/null || echo "")
  case "$OPID" in ""|*[!0-9]*) OPID="" ;; esac
  if [ -n "$OPID" ] && [ -d "/proc/$OPID" ]; then continue; fi
  # 3. orchestrator gone, no sentinel. Is a trainer somehow running anyway?
  BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  if [ "$BUSY" != "0" ]; then
    say "orchestrator gone but cards busy ($BUSY) -- something is running; not intervening."
    continue
  fi
  CP="training/runs/$RID/resume_ep${EP}.pt"
  if [ ! -f "$CP" ]; then say "ORCHESTRATOR GONE and $CP missing. CANNOT RESUME."; exit 1; fi
  say "ORCHESTRATOR GONE WITHOUT RESUMING. Resuming $RID from epoch $EP."
  touch "$SENTINEL"
  LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
    --cards 4,5,7 --numa-map 0:0,1:0,2:1 \
    --note "SAFETY-NET RESUME of the uncropped beta-75 arm $RID from epoch $EP. The predicted-crop eval orchestrator exited without resuming; no training setting changed." \
    -- --resume "$CP" \
    --rank-loss --rank-form rel --rank-cross-seed --alpha-rank 75 \
    --rank-pool mean --rel-labels training/fold/itm_labels \
    --rank-structs-per-step 3 \
    --pose-perblock --pose-perblock-copy --frame-direction \
    --fold training/fold --crop-margin 0 \
    --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
    --manifest-selection training/full_train/structures_winnable.csv \
    --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
    --epochs 60 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
    --lr-final 1e-4 --lr-decay-shape cosine \
    --lr-decay-start-epoch 20 --lr-decay-end-epoch 50 \
    --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
    --save-every 250 \
    --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only \
    --shuffle structure --seeds 500000-500009 \
    --eval-pool training/valfull/pool.csv \
    --eval-pool-fold validation/fold --eval-pool-seed-base 400000 \
    --eval-pool-budget 30 --eval-pool-every 1 \
    --prefetch-units 2 --prefetch-workers 4 \
    --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
    > training/runs/predcrop_safetynet_resume.log 2>&1
  say "safety-net resume launcher exited $?"; tail -4 training/runs/predcrop_safetynet_resume.log
  exit 0
done
say "deadline reached without acting"
