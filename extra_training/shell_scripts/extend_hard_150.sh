#!/bin/bash
# EXTEND THE HARD-OVERFIT ARM FROM EPOCH 70 TO EPOCH 150.
#
# ONE CHANGE, THE USER'S: the epoch count 70 -> 150. Everything else is 55reqf's own, copied from
# resume_ambgom_5e4.sh rather than retyped: the 151 hard lists, PAE-only with the per-list class
# balance at DockQ 0.23, lr 5e-4, no dropout, no held-out eval, the two uncropped extra-evals
# every 5 epochs.
#
# lr IS UNCHANGED at 5e-4, so no --resume-override is needed; the guard will match.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
say() { echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid() { ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

# WAIT FOR EPOCH 70's BUNDLE, not for the process to exit: the final eval runs after the bundle is
# written, and the cards are not free until the whole thing is down.
for i in $(seq 1 240); do
    [ -s training/runs/55reqf/resume_ep70.pt ] && break
    sleep 30
done
[ -s training/runs/55reqf/resume_ep70.pt ] || { say "ABORT: resume_ep70.pt never appeared"; exit 2; }
say "resume_ep70.pt present; waiting for the cards"
for i in $(seq 1 80); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 15
done
[ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] || { say "ABORT: cards still busy"; exit 3; }

PRE=$(newest_rid)
say "=== resuming the hard-overfit arm at epoch 71, lr 5e-4, to epoch 150 ==="
SUBSET_OK=1 \
PREFLIGHT=training/overfit_hard/alignment_preflight.json \
MANIFEST=training/overfit_hard/structures.csv \
SCORED_PAIRS=training/overfit_hard/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=hard15,pool=training/overfit_hard/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=training/overfit_hard/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5" \
CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$R/training/runs/55reqf/resume_ep70.pt" \
RESUME_NOTE="epoch count extended 70 -> 150 at the user's request; nothing else changed from 55reqf." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/hard150_launch.log 2>&1 &
LPID=$!
RID=""
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { RID="$N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:|not found" training/runs/hard150_launch.log | head -5; exit 4; }
    sleep 20
done
[ -n "$RID" ] || { say "ABORT: no run directory after 30 min"; exit 5; }
echo "$RID" > training/runs/hard150_rid
echo 70 > "training/runs/${RID}_last_sent"
say "resumed as $RID, to epoch 150"
