#!/bin/bash
# FOLD THE NEW DATA INTO EVAL. Everything in the training pool is now evaluated at budget 15.
#
# WHY hard1 GOES. Its 21 lists were budget 1 only because the DockQ-SD<0.05 prune had left their
# structures with one trunk. All 21 now have 15 seeds, so a budget-1 pass over them measures a
# restriction that no longer exists -- 19 of those 21 were unwinnable by construction.
#
# TRAIN AND HELD-OUT, ONE PASS EACH. hard15's 130 lists are a subset of hardall15's 155, so the
# old series can still be recomputed from the same shards offline -- it does not need its own GPU
# pass. The second pass is valfull83 instead: this arm has run 90 epochs with no held-out number
# at all, which is the one thing that says whether any of it generalises.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/overfit_hard
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

RID=$(cat training/runs/hard150_rid)
N15=$(( $(wc -l < "$OH/pool_all15.csv") - 1 ))
[ "$N15" -eq 155 ] || { say "ABORT: pool_all15.csv has $N15 rows, expected 155"; exit 2; }

B0=$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)
say "waiting for $RID's next epoch boundary"
for i in $(seq 1 120); do
    [ "$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)" -gt "$B0" ] && break
    sleep 30
done
EP=$(ls training/runs/$RID/resume_ep*.pt | sed -E 's/.*resume_ep([0-9]+)\.pt/\1/' | sort -n | tail -1)
PIDS=$(ps -eo pid,args | grep -F "$RID" | grep -v ugrep | awk '/torchrun|phase_k_train|launch\.py/ {print $1}')
[ -n "$PIDS" ] || { say "ABORT: no pids for $RID"; exit 3; }
say "pausing at epoch $EP; stopping pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null; done
for i in $(seq 1 60); do
    A=0; for p in $PIDS; do [ -d "/proc/$p" ] && A=$((A+1)); done
    [ "$A" -eq 0 ] && break; sleep 5
done
for p in $PIDS; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
sleep 15

echo "$EP: eval = 155 train lists + held-out valfull83" >> training/runs/arm_events.txt
BUNDLE=$(ls -t training/runs/$RID/resume_ep*.pt | head -1)
PRE=$(newest_rid)
say "=== resuming from $(basename "$BUNDLE"), eval over all 155 lists ==="
SUBSET_OK=1 \
PREFLIGHT=$OH/alignment_preflight.json \
MANIFEST=$OH/structures.csv \
SCORED_PAIRS=$OH/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=hardall15,pool=$OH/pool_all15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard15,pool=$OH/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5" \
CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$R/${BUNDLE#./}" \
RESUME_NOTE="epoch $EP: eval folded in the re-folded data -- all 155 training lists at budget 15 (hardall15). hard15 keeps the original 130 so the epoch 1-$EP curve stays comparable. hard1 dropped: its 21 lists now have 15 seeds, so a budget-1 pass measures a restriction that no longer exists." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/resume_evalall.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
    N=$(newest_rid)
    [ "$N" != "$PRE" ] && { echo "$N" > training/runs/hard150_rid; say "resumed as $N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:" training/runs/resume_evalall.log | head -5; exit 4; }
    sleep 20
done
say "DONE"
