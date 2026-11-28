#!/bin/bash
# PAE ONLY AT lr 1e-4, VALIDATED EVERY EPOCH. Same arm as the 1e-3 PAE-only reference in
# training/runs/pae_ref_scores (no ranking term, same pool, same manifest); the learning rate and
# the scoring cadence are the only differences, so the two are read against each other directly.
#
# EVERY EPOCH IS A SEGMENT. STOPS is 1..60, so each epoch is scored by the same
# shell_scripts/score_checkpoint.sh pass that produced every number we already have. The
# in-trainer trigger (EVAL_EVERY=1) would also run each epoch but writes its own artefacts, and
# comparability with the reference arm is worth the ~2 min resume per epoch.
#
# NO EPOCH-2 GATE. The reference arm's easy-bin check is removed: at a tenth the learning rate
# epoch 2 has barely moved, so the gate would abort an arm that is behaving exactly as expected.
#
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

STOPS="${STOPS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60}"
SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
export SD
mkdir -p "$SD"
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
        CARDS=4,5,7 EPOCHS=$E EVAL_EVERY=0 LR=${LR:-1e-4} PAIR_DROPOUT=0.25 \
        PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
        nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/lr1e4_seg${E}.log 2>&1 &
    LPID=$!
    NEW=""
    for i in $(seq 1 90); do
        N=$(newest_rid); [ "$N" != "$PRE" ] && { NEW="$N"; break; }
        [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
            grep -E "REFUS|STOP:" training/runs/lr1e4_seg${E}.log | head -5; exit 4; }
        sleep 20
    done
    [ -n "$NEW" ] || { say "ABORT: no run directory for segment $E"; exit 5; }
    RID=$NEW; echo "$RID" > training/runs/fullset_rid
    say "segment running as $RID"

    # LET IT STOP ITSELF. The segment was launched with EPOCHS=$E, so the trainer exits at that
    # boundary on its own. Killing it on sight of resume_ep<E>.pt is both unnecessary and a race:
    # rank 0 writes the resume bundle FIRST and the head after, so a watchdog that fires on the
    # bundle can kill the process while the head is still being written -- and then refuse the
    # segment for a head that was moments away. Wait for the exit, then require a clean status
    # and BOTH artefacts.
    say "waiting for the segment to finish at epoch $E"
    while ps -eo args | grep -F "$RID" | grep -q phase_k_train; do sleep 60; done
    for i in $(seq 1 30); do [ -s "training/runs/$RID/status.json" ] && break; sleep 10; done
    RC=$($PY -I -c "
import json,sys
try: print(json.load(open('training/runs/$RID/status.json')).get('exit_code'))
except Exception: print('missing')" 2>/dev/null)
    [ "$RC" = "0" ] || { say "ABORT: segment $E ended with exit_code=$RC"
        tail -5 "training/runs/$RID/run.log"; exit 6; }
    [ -s "training/runs/$RID/resume_ep${E}.pt" ] || { say "ABORT: no resume bundle at epoch $E"; exit 7; }

    H=$(ls training/runs/$RID/head_ep${E}_step*.pt 2>/dev/null | head -1)
    [ -s "$H" ] || { say "ABORT: no head at epoch $E"; exit 8; }
    SIDES=val
    for s in $SLICE_AT; do [ "$s" = "$E" ] && SIDES=both; done
    say "scoring epoch $E ($SIDES)"
    # FAIL CLOSED. A chain that scores nothing and trains to 60 anyway has spent the GPU and
    # produced no evidence, which is the one outcome worth refusing.
    # A TAG OF ITS OWN. score_checkpoint.sh writes its rows into $SD/val30_$TAG/, and a tag
    # reused from an earlier arm makes the row files ACCUMULATE: score_report.py then trips its
    # 141x150 assertion, and because the last line of score_checkpoint.sh is a pipeline without
    # pipefail the failure is swallowed and tee truncates the report to nothing. That emptied
    # all ten reports of the 2026-11-16 arm. TAGP keeps each arm in its own namespace.
    CKPT="$R/$H" TAG="${TAGP:-lr4}_ep$E" SIDES=$SIDES ./shell_scripts/score_checkpoint.sh \
        || { say "ABORT: scoring epoch $E failed"; exit 9; }

    # NO EPOCH-2 RULE HERE. See the header: the reference arm's easy-bin gate assumes lr 1e-3
    # progress by epoch 2 and would abort this arm for being slower on purpose.
    PREV_EP=$E
done
say "CHAIN COMPLETE over epochs $STOPS"
