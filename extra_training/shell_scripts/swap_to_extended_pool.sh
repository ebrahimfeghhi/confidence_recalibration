#!/bin/bash
# PAUSE THE ARM AT AN EPOCH BOUNDARY AND BRING IT BACK ON THE EXTENDED POOL.
#
# WHAT CHANGES: the 18 re-folded structures declare seeds 500000-500014 instead of 500000, so
# their hard poses are loadable for the first time, and 4 interfaces that now qualify are added.
# 1,173 units -> 1,425. Nothing else about the arm moves: same lr, same losses, same eval pools
# plus hardnew15 over the 4 new interfaces.
#
# THE EPOCH IS RECORDED so the plots can draw it. A step at that epoch is a different training
# set, not learning, and a resume marker alone does not say that.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/overfit_hard
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

RID=$(cat training/runs/hard150_rid)
[ -s "$SD/pf_ext.json" ] || { say "ABORT: no passing extended preflight at $SD/pf_ext.json"; exit 2; }
$PY -I -c "
import json,sys
d=json.load(open('$SD/pf_ext.json'))
sys.exit(0 if d.get('ok') and d.get('complete') and not d.get('mismatches') else 1)" \
    || { say "ABORT: the extended preflight did not pass"; exit 3; }

B0=$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)
say "waiting for $RID's next epoch boundary"
for i in $(seq 1 120); do
    [ "$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)" -gt "$B0" ] && break
    sleep 30
done
EP=$(ls training/runs/$RID/resume_ep*.pt | sed -E 's/.*resume_ep([0-9]+)\.pt/\1/' | sort -n | tail -1)
say "pausing at epoch $EP"
PIDS=$(ps -eo pid,args | grep -F "$RID" | grep -v ugrep | awk '/torchrun|phase_k_train|launch\.py/ {print $1}')
[ -n "$PIDS" ] || { say "ABORT: no pids found for $RID"; exit 4; }
say "stopping pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null; done
for i in $(seq 1 60); do
    A=0; for p in $PIDS; do [ -d "/proc/$p" ] && A=$((A+1)); done
    [ "$A" -eq 0 ] && break; sleep 5
done
for p in $PIDS; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
sleep 15

# the extended files, staged while the arm kept training
cp -p "$SD/extended_pool/pool.csv"       "$OH/pool.csv"
cp -p "$SD/extended_pool/structures.csv" "$OH/structures.csv"
cp -p "$SD/extended_pool/pool_new15.csv" "$OH/pool_new15.csv"
cp -p "$SD/pf_ext.json"                  "$OH/alignment_preflight.json"
echo "$EP" > "$OH/extended_at_epoch"
say "pool: $(( $(wc -l < $OH/pool.csv) - 1 )) interface(s); manifest digest $($PY -I -c "
import json; print(json.load(open('$OH/alignment_preflight.json'))['manifest_digest'])")"

NEW_ROWS=$(( $(wc -l < "$OH/pool_new15.csv") - 1 ))
EVALS="name=hard15,pool=$OH/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5"
EVALS="$EVALS;name=hard1,pool=$OH/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5"
[ "$NEW_ROWS" -gt 0 ] && EVALS="$EVALS;name=hardnew15,pool=$OH/pool_new15.csv,fold=training/fold,seed_base=500000,budget=15,every=5"

BUNDLE=$(ls -t training/runs/$RID/resume_ep*.pt | head -1)
PRE=$(newest_rid)
say "=== resuming from $(basename "$BUNDLE") on the EXTENDED pool, to epoch 150 ==="
SUBSET_OK=1 \
PREFLIGHT=$OH/alignment_preflight.json \
MANIFEST=$OH/structures.csv \
SCORED_PAIRS=$OH/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="$EVALS" \
CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$R/${BUNDLE#./}" \
RESUME_OVERRIDE=manifest_digest \
RESUME_NOTE="epoch $EP: the 22 re-folded structures now declare seeds 500000-500014, so their hard poses are loadable for the first time (1,173 -> 1,425 units), and 4 newly qualifying interfaces joined the training pool. Everything else is unchanged." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/resume_extended.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
    N=$(newest_rid)
    [ "$N" != "$PRE" ] && { echo "$N" > training/runs/hard150_rid; say "resumed as $N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:" training/runs/resume_extended.log | head -5; exit 5; }
    sleep 20
done
say "DONE"
