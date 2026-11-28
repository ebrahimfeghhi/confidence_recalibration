#!/bin/bash
# PAUSE AT AN EPOCH BOUNDARY, RUN TWO PROBES, RESUME.
#
# PROBE 1 -- CAN THE HEAD SEPARATE ONE ITEM'S CANDIDATES AT ALL? One item (8v91_BF: 75 candidates,
# 7 good, quality range 0.96), same loss, same mask, 200 epochs, nothing else in the batch. The
# group loss has been flat at ~3.08/2.26 for twenty epochs, which rules out "not done training".
# What is left is that the inputs do not distinguish these candidates, or the head cannot use them.
# If the loss will not go well below 3 on ONE item with unlimited passes, it is the former.
#
# PROBE 2 -- IS 29% TOP-1 A LEVEL OR A SPIKE? The epoch-90 jump came with a FLAT loss, and flat
# predictions reordering candidates is exactly what near-ties flipping looks like. The heads for
# 88, 89 and 91 are already on disk, so this is three eval passes and no training.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/overfit_hard
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

RID=$(cat training/runs/hard150_rid)
B0=$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)
WAIT_METRIC="${WAIT_METRIC:-}"
if [ -n "$WAIT_METRIC" ]; then
    say "waiting for $WAIT_METRIC (then stopping before the redundant second pass)"
    for i in $(seq 1 180); do
        [ -s "$WAIT_METRIC" ] && break
        sleep 10
    done
    [ -s "$WAIT_METRIC" ] || say "WARNING: $WAIT_METRIC never appeared; pausing anyway"
else
    say "waiting for $RID's next epoch boundary"
    for i in $(seq 1 120); do
        [ "$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)" -gt "$B0" ] && break
        sleep 20
    done
fi
EP=$(ls training/runs/$RID/resume_ep*.pt | sed -E 's/.*resume_ep([0-9]+)\.pt/\1/' | sort -n | tail -1)
PIDS=$(ps -eo pid,args | grep -F "$RID" | grep -v ugrep | awk '/torchrun|phase_k_train|launch\.py/ {print $1}')
[ -n "$PIDS" ] || { say "ABORT: no pids for $RID"; exit 2; }
say "pausing $RID at epoch $EP; stopping pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null; done
for i in $(seq 1 60); do
    A=0; for p in $PIDS; do [ -d "/proc/$p" ] && A=$((A+1)); done
    [ "$A" -eq 0 ] && break; sleep 5
done
for p in $PIDS; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
sleep 15

# ---------------------------------------------------------------- probe 2 first: it is 3 evals
for E in 88 89 91; do
    H=$(ls training/runs/xi5gbj/head_ep${E}_step*.pt 2>/dev/null | head -1)
    [ -s "$H" ] || { say "probe2: no head for epoch $E, skipping"; continue; }
    say "probe2: evaluating epoch $E on the original 130"
    mkdir -p training/runs/probe_nearties
    EXTRA="--fold $R/training/fold --results $R/training/fold/results.csv"
    EXTRA="$EXTRA --pool $R/$OH/pool_b15.csv --roles $R/training/fold/chain_roles_all.csv"
    EXTRA="$EXTRA --seed-base 500000 --pose-perblock --pose-perblock-copy --frame-direction"
    EXTRA="$EXTRA --join-cache $R/$OH/join_cache.json --drop-unmapped-poses"
    LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
        --cards 4,5,7 --ckpt "$R/$H" --budget 15 \
        --work "training/runs/probe_nearties/shards_ep${E}" \
        --out "training/runs/probe_nearties/metric_ep${E}.json" \
        --extra "$EXTRA" >> training/runs/probe_nearties.log 2>&1 \
        || say "probe2: epoch $E exited $?"
done

# ---------------------------------------------------------------- probe 1: one item, 200 epochs
say "probe1: single-item overfit on 8v91_BF"
PRE=$(newest_rid)
SUBSET_OK=1 \
MANIFEST=training/single_item/structures.csv \
SCORED_PAIRS=training/single_item/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
CARDS=4,5,7 EPOCHS=200 EVAL_EVERY=0 LR=1e-3 \
PREFETCH_WORKERS=8 TRUNK_MEMMAP=1 DRY=0 \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/single_item_launch.log 2>&1 &
LPID=$!
SRID=""
for i in $(seq 1 60); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { SRID="$N"; break; }
    [ -d "/proc/$LPID" ] || { say "probe1: launcher exited before a run appeared"
        grep -E "REFUS|STOP:" training/runs/single_item_launch.log | head -5; break; }
    sleep 20
done
if [ -n "$SRID" ]; then
    echo "$SRID" > training/runs/single_item_rid
    say "probe1 running as $SRID; waiting for it to finish"
    while ps -eo args | grep -F "$SRID" | grep -q phase_k_train; do sleep 60; done
    say "probe1 done"
    grep -oE "epoch [0-9]+ PAE BY POSE GROUP.*" training/runs/$SRID/run.log 2>/dev/null | tail -3 | cut -c1-160
fi

# ---------------------------------------------------------------- back to the arm
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
RESUME_NOTE="paused at epoch $EP for two probes: a single-item overfit (can the head separate one item's candidates at all) and three re-evals of saved heads (is 29% top-1 a level or a spike). Nothing about the arm changed." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/resume_after_probes.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { echo "$N" > training/runs/hard150_rid; say "arm resumed as $N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"; exit 4; }
    sleep 20
done
say "DONE"
