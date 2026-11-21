#!/bin/bash
# THE CARDS DO NOT WAIT FOR THE CPU. The moment the fold ends the arm resumes on the pool it
# already has, and the label rebuild -- which is CPU only -- finishes alongside it. When the labels
# and the pool extension are ready, the arm is paused at an epoch boundary and resumed on the
# extended pool. One extra pause, against ~5 h of idle GPU.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/overfit_hard
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

FOLD_PID=${FOLD_PID:?set FOLD_PID}

resume_arm(){   # $1 = bundle, $2 = eval string, $3 = note, $4 = override (may be empty)
    local PRE; PRE=$(newest_rid)
    local OV=()
    [ -n "$4" ] && OV=(RESUME_OVERRIDE="$4")
    env SUBSET_OK=1 \
        PREFLIGHT=$OH/alignment_preflight.json \
        MANIFEST=$OH/structures.csv \
        SCORED_PAIRS=$OH/pool.csv \
        PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
        EXTRA_EVAL="$2" \
        CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
        PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
        RESUME="$R/${1#./}" RESUME_NOTE="$3" "${OV[@]}" \
        nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/resume_$(date +%s).log 2>&1 &
    local LPID=$!
    for i in $(seq 1 90); do
        local N; N=$(newest_rid)
        [ "$N" != "$PRE" ] && { echo "$N" > training/runs/hard150_rid; say "resumed as $N"; return 0; }
        [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"; return 1; }
        sleep 20
    done
    say "ABORT: no run directory after 30 min"; return 1
}

EV2="name=hard15,pool=$OH/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5"
EV2="$EV2;name=hard1,pool=$OH/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5"

say "waiting for the fold (pid $FOLD_PID)"
while [ -d "/proc/$FOLD_PID" ]; do sleep 30; done
say "fold done; putting the cards back to work before anything else"
for i in $(seq 1 40); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 15
done
BUNDLE=$(ls -t training/runs/ctgfwj/resume_ep*.pt | head -1)
resume_arm "$BUNDLE" "$EV2" "paused at epoch 80 for the re-fold; resumed on the UNEXTENDED pool so the cards are not idle while the labels rebuild. The extended pool follows at the next pause." "" \
    || { say "STOPPING: the interim resume failed"; exit 2; }
RID=$(cat training/runs/hard150_rid)

# ---- CPU side, while that trains: the last seeds' labels, then the pool extension.
say "building the remaining labels (seeds 500013-500014)"
LIST=docs/degenerate_regen_structures.txt LO=500013 HI=500014 JOBS=12 \
    ./shell_scripts/build_extended_train_labels.sh >> training/runs/labels_degen_tail.log 2>&1 || true
say "waiting for the 500001-500012 build to finish"
while pgrep -u "$USER" -f phase_j_build_cache >/dev/null 2>&1; do sleep 60; done
NLAB=$($PY -I -c "
import glob
S=[l.strip() for l in open('docs/degenerate_regen_structures.txt') if l.strip() and l[0]!='#']
print(sum(1 for p in S for s in range(500001,500015) if glob.glob(f'training/fold/labels/{p}/free_{s}.npz')))")
say "labels on disk for the new seeds: $NLAB of 308"
[ "$NLAB" -ge 300 ] || { say "STOPPING: only $NLAB labels; not extending the pool"; exit 3; }

$PY -I run_logging/extend_hard_pool.py --list docs/degenerate_regen_structures.txt --out "$OH" --apply \
    || { say "STOPPING: pool extension refused"; exit 4; }
$PY -I phases/phase_k_alignment_preflight.py --manifest-selection "$OH/structures.csv" \
    --manifest-skip "4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m" \
    --out "$OH/alignment_preflight.json" --workers 12 \
    || { say "STOPPING: preflight failed"; exit 5; }

NEW_ROWS=0
[ -f "$OH/pool_new15.csv" ] && NEW_ROWS=$(( $(wc -l < "$OH/pool_new15.csv") - 1 ))
EVALS="$EV2"
[ "$NEW_ROWS" -gt 0 ] && EVALS="$EVALS;name=hardnew15,pool=$OH/pool_new15.csv,fold=training/fold,seed_base=500000,budget=15,every=5"
say "extended pool ready; hardnew15 over $NEW_ROWS interface(s)"

# ---- pause the arm at an epoch boundary and bring it back on the extended pool
say "pausing $RID at the next epoch boundary"
B0=$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)
for i in $(seq 1 120); do
    [ "$(ls training/runs/$RID/resume_ep*.pt 2>/dev/null | wc -l)" -gt "$B0" ] && break
    sleep 60
done
PIDS=$(ps -eo pid,args | grep -F "$RID" | grep -v ugrep | awk '$2 ~ /torchrun|phase_k_train|launch.py/ {print $1}')
say "stopping pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null; done
for i in $(seq 1 60); do
    A=0; for p in $PIDS; do [ -d "/proc/$p" ] && A=$((A+1)); done
    [ "$A" -eq 0 ] && break; sleep 5
done
for p in $PIDS; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
sleep 15

BUNDLE=$(ls -t training/runs/$RID/resume_ep*.pt | head -1)
resume_arm "$BUNDLE" "$EVALS" "resumed on the EXTENDED pool: the 22 re-folded structures now declare seeds 500000-500014 and the interfaces that qualify as hard are in the training pool." "manifest_digest" \
    || { say "STOPPING: the extended resume failed"; exit 6; }
say "DONE"
