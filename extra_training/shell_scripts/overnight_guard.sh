#!/bin/bash
# OVERNIGHT GUARD. The probe chain resumes the arm itself; this exists for the case where it does
# not -- a probe that exits non-zero, a launcher that refuses, a card someone else took. It checks
# every ten minutes that SOMETHING of ours is on the cards, and if nothing is and no chain is
# alive, it resumes the arm from its newest bundle rather than leaving three cards idle till morning.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/overfit_hard
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

IDLE=0
for i in $(seq 1 120); do          # 20 h at 10 min
    sleep 600
    B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
    CHAIN=$(ps -eo args | grep -cE "pause_probe_resume\.sh|queue_poseinject_twosite\.sh" || true)
    if [ "$B" != "0" ]; then IDLE=0; continue; fi
    # cards free. If a chain is mid-flight it is between steps; give it two cycles.
    IDLE=$((IDLE + 1))
    say "cards free (cycle $IDLE), chains alive: $CHAIN"
    [ "$IDLE" -lt 2 ] && continue
    if [ "$CHAIN" -gt 1 ]; then say "a chain is alive; not intervening"; IDLE=0; continue; fi
    RID=$(cat training/runs/hard150_rid)
    BUNDLE=$(ls -t training/runs/$RID/resume_ep*.pt 2>/dev/null | head -1)
    [ -s "$BUNDLE" ] || { say "no bundle for $RID; nothing to resume"; continue; }
    EP=$(echo "$BUNDLE" | sed -E 's/.*resume_ep([0-9]+)\.pt/\1/')
    [ "$EP" -ge 150 ] && { say "$RID already at epoch $EP; nothing to resume"; continue; }
    say "RESUMING $RID from epoch $EP -- cards were idle and no chain was alive"
    PRE=$(newest_rid)
    SUBSET_OK=1 \
    PREFLIGHT=$OH/alignment_preflight.json \
    MANIFEST=$OH/structures.csv \
    SCORED_PAIRS=$OH/pool.csv \
    PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
    EXTRA_EVAL="name=hardall15,pool=$OH/pool_all15.csv,fold=training/fold,seed_base=500000,budget=15,every=5" \
    CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
    RESUME="$R/${BUNDLE#./}" \
    RESUME_NOTE="overnight guard: the cards were idle with no chain alive, so the arm was resumed from epoch $EP." \
        nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/overnight_resume.log 2>&1 &
    sleep 120
    N=$(newest_rid)
    [ "$N" != "$PRE" ] && { echo "$N" > training/runs/hard150_rid; say "resumed as $N"; IDLE=0; }
done
say "guard finished its window"
