#!/bin/bash
# PROBE 1, retried with its own alignment preflight (digest b138e23fae22d070, 1 structure, 15 units).
#
# It pauses the arm after its NEXT eval metric lands -- a natural point, not mid-epoch -- runs the
# single-item overfit, then puts the arm back. 75 candidates an epoch at ~45 s means 200 epochs is
# about 2.5 h of cards; that is the price of the answer the flat group loss cannot give.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/overfit_hard
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

RID=$(cat training/runs/hard150_rid)
WAIT_METRIC=${WAIT_METRIC:-training/runs/$RID/metric_hardall15_ep100.json}
say "waiting for $WAIT_METRIC"
for i in $(seq 1 360); do [ -s "$WAIT_METRIC" ] && break; sleep 30; done
[ -s "$WAIT_METRIC" ] || { say "ABORT: $WAIT_METRIC never appeared"; exit 2; }

PIDS=$(ps -eo pid,args | grep -F "$RID" | grep -v ugrep | awk '/torchrun|phase_k_train|launch\.py/ {print $1}')
[ -n "$PIDS" ] || { say "ABORT: no pids for $RID"; exit 3; }
say "pausing $RID; stopping pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null; done
for i in $(seq 1 60); do
    A=0; for p in $PIDS; do [ -d "/proc/$p" ] && A=$((A+1)); done
    [ "$A" -eq 0 ] && break; sleep 5
done
for p in $PIDS; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
sleep 15

PRE=$(newest_rid)
say "probe1: single-item overfit on 8v91_BF, constant lr 1e-3, 200 epochs"
SUBSET_OK=1 \
PREFLIGHT=training/single_item/alignment_preflight.json \
MANIFEST=training/single_item/structures.csv \
SCORED_PAIRS=training/single_item/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
CARDS=4,5,7 EPOCHS=200 EVAL_EVERY=0 LR=1e-3 \
PREFETCH_WORKERS=8 TRUNK_MEMMAP=1 DRY=0 \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/probe1_launch.log 2>&1 &
LPID=$!
SRID=""
for i in $(seq 1 60); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { SRID="$N"; break; }
    [ -d "/proc/$LPID" ] || { say "probe1: launcher exited early"
        grep -E "REFUS|STOP:" training/runs/probe1_launch.log | head -3; break; }
    sleep 20
done
if [ -n "$SRID" ]; then
    echo "$SRID" > training/runs/single_item_rid
    say "probe1 running as $SRID"
    while ps -eo args | grep -F "$SRID" | grep -q phase_k_train; do sleep 60; done
    say "probe1 finished"
    grep -oE "epoch [0-9]+ PAE BY POSE GROUP.*" training/runs/$SRID/run.log | tail -2 | cut -c1-170
fi

for i in $(seq 1 40); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 15
done
BUNDLE=$(ls -t training/runs/$RID/resume_ep*.pt | head -1)
PRE=$(newest_rid)
say "=== resuming the arm from $(basename "$BUNDLE") ==="
SUBSET_OK=1 \
PREFLIGHT=$OH/alignment_preflight.json \
MANIFEST=$OH/structures.csv \
SCORED_PAIRS=$OH/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=hardall15,pool=$OH/pool_all15.csv,fold=training/fold,seed_base=500000,budget=15,every=5" \
CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$R/${BUNDLE#./}" \
RESUME_NOTE="paused for probe 1 (single-item overfit); nothing about the arm changed." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/resume_after_probe1.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { echo "$N" > training/runs/hard150_rid; say "arm resumed as $N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"; exit 4; }
    sleep 20
done
say "DONE"
