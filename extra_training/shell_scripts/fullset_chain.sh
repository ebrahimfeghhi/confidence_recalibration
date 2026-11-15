#!/bin/bash
# THE FULL-SET ARM AS SEGMENTS, so scoring lands on chosen epochs rather than on one modulus.
#
# WHY SEGMENTS. The trainer has a single held-out trigger, (ep+1) %% EVAL_EVERY == 0, so "val after
# 1, 2, 4, 7, 10, then every 10" cannot be configured. Training therefore runs with its own eval
# OFF and stops at each listed epoch; the scoring is the same pass the trainer would have run,
# executed by shell_scripts/score_checkpoint.sh on the saved head. A resume costs ~2 min of
# startup, which is the whole overhead of the arrangement.
#
# DENSE EARLY, SPARSE LATER, AND EVERY EPOCH ON DISK. Heads and resume bundles are written every
# epoch already, so any epoch can be scored later: once val peaks, backfill the epochs either side
# of it with score_checkpoint.sh rather than guessing the cadence up front.
#
# VAL ONLY EARLY. Val is ~12 min, the slice ~13; the slice is an in-sample diagnostic and tells you
# nothing at epoch 1 that epoch 10 will not. It runs at the ten-epoch points.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/fullset
SLICE=training/slice280
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

STOPS="${STOPS:-1 2 4 7 10 20 30 40 50 60}"
SLICE_AT="${SLICE_AT:-10 20 30 40 50 60}"
RID=""
PREV_EP=0

for E in $STOPS; do
    for i in $(seq 1 120); do
        [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
        sleep 30
    done
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] || { say "ABORT: cards busy"; exit 2; }

    RES=()
    if [ -n "$RID" ]; then
        B=training/runs/$RID/resume_ep${PREV_EP}.pt
        [ -s "$B" ] || { say "ABORT: no bundle at epoch $PREV_EP under $RID"; exit 3; }
        # NO --resume-override: `epochs` is saved in the bundle but not compared, and the
        # guard refuses an override name it does not have. Only the stopping point moves.
        RES=(RESUME="$R/$B"
             RESUME_NOTE="segment to epoch $E; the arm is unchanged, only the stopping point moves so val can be scored at $STOPS.")
    fi
    PRE=$(newest_rid)
    say "=== training to epoch $E ==="
    env "${RES[@]}" \
        PREFLIGHT=$OH/alignment_preflight.json \
        MANIFEST=$OH/structures.csv \
        SCORED_PAIRS=$OH/pool.csv \
        PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
        CARDS=4,5,7 EPOCHS=$E EVAL_EVERY=0 LR=1e-3 PAIR_DROPOUT=0.25 \
        PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
        nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/fullset_seg${E}.log 2>&1 &
    LPID=$!
    NEW=""
    for i in $(seq 1 90); do
        N=$(newest_rid); [ "$N" != "$PRE" ] && { NEW="$N"; break; }
        [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
            grep -E "REFUS|STOP:" training/runs/fullset_seg${E}.log | head -5; exit 4; }
        sleep 20
    done
    [ -n "$NEW" ] || { say "ABORT: no run directory for segment $E"; exit 5; }
    RID=$NEW; echo "$RID" > training/runs/fullset_rid
    say "segment running as $RID"

    while [ ! -s "training/runs/$RID/resume_ep${E}.pt" ]; do
        if ! ps -eo args | grep -F "$RID" | grep -q phase_k_train; then
            [ -s "training/runs/$RID/resume_ep${E}.pt" ] && break
            say "ABORT: $RID exited before epoch $E"; tail -5 "training/runs/$RID/run.log"; exit 6
        fi
        sleep 60
    done
    say "epoch $E reached; stopping the segment"
    PIDS=$(ps -eo pid,args | grep -F "$RID" | grep -v ugrep | awk '/torchrun|phase_k_train|launch\.py/ {print $1}')
    for p in $PIDS; do kill "$p" 2>/dev/null; done
    for i in $(seq 1 60); do
        A=0; for p in $PIDS; do [ -d "/proc/$p" ] && A=$((A+1)); done
        [ "$A" -eq 0 ] && break; sleep 5
    done
    for p in $PIDS; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
    sleep 15

    H=$(ls training/runs/$RID/head_ep${E}_step*.pt 2>/dev/null | head -1)
    [ -s "$H" ] || { say "ABORT: no head at epoch $E"; exit 7; }
    SIDES=val
    for s in $SLICE_AT; do [ "$s" = "$E" ] && SIDES=both; done
    say "scoring epoch $E ($SIDES)"
    CKPT="$R/$H" TAG="fs_ep$E" SIDES=$SIDES ./shell_scripts/score_checkpoint.sh \
        || say "WARNING: scoring epoch $E exited $?"
    PREV_EP=$E
done
say "CHAIN COMPLETE over epochs $STOPS"
