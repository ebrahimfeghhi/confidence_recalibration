#!/bin/bash
# THE FULL-SET ARM PLUS THE Z-SCORED SOFTMAX RANKING LOSS, AS SEGMENTS.
#
# A COPY OF fullset_chain.sh, not an edit of it. That chain is the PAE-only reference this arm is
# read against, and editing it would make the comparison unreproducible; it also ran live while
# this was written, and editing a running bash script desyncs it by byte offset.
#
# WHAT IS THE SAME AS THE REFERENCE ARM: the manifest (853 rows, 846 trained), the shipped init,
# alpha_pae 1.0, the per-item class balance at DockQ 0.23, constant lr 1e-3, pair dropout 0.25,
# 60 epochs, and the segment boundaries.
#
# WHAT IS NOT, and both are forced by the loss rather than chosen:
#   - --rank-cross-seed, which the trainer refuses to run this loss without. It changes the EPOCH
#     BUILDER: every seed of a structure must sit in one optimiser step, because a candidate's
#     coefficient is a softmax over all of its item's candidates. So a step is 3 structures x all
#     their seeds instead of ~48 distinct structures. Same number of poses, different mix -- each
#     step's PAE gradient sees fewer structures than the reference arm's did.
#   - SCORED_PAIRS is pool_labelled.csv, 1,694 rows against pool.csv's 1,711. The 17 dropped
#     interfaces have no DockQ at any seed -- OST's chain mapping refuses them, three are already
#     written up in docs/unrecoverable_interfaces.csv -- and a ranking loss cannot label their
#     candidates. The pool sets the PAE mask too, so those interfaces leave both terms. Every
#     earlier ranking arm used a pool with the same interfaces already absent
#     (training/combined/pool.csv), so this is the established set and not a new deviation.
#
# ALPHA 1, WHICH IS PARITY AND NOT A GUESS. Measured on this exact pool and manifest at the
# shipped weights with lr 0, every step of a partial epoch: median alpha_for_parity 1.061 over 94
# steps (run ycmfnz, p10 0.48, p90 7.96). The spread is 16x, so the median is the statistic and a
# single-step probe would have been worthless -- the five historical probes that set alpha 75 for
# the rel form span ratio 0.0041 to 0.4728 for a DIFFERENT term.
#
# READ THE COSINE ALONGSIDE IT: median +0.0295, and NEGATIVE on 43 of 94 steps. The two terms are
# close to orthogonal on average and genuinely conflict on half the steps, so parity is not a
# neutral choice -- it is the largest defensible one.
#
# VAL AND THE SLICE ON THE SAME SCHEDULE, which is the one change to the arrangement rather than
# to the arm. The reference chain scored val at 1,2,4,7,10,... and the slice only from 10, so an
# in-sample and a held-out number could not be compared before epoch 10 -- and the generalisation
# gap is the whole quantity of interest here. Both now run at every stop, at ~25 min a stop
# instead of ~12 early on.
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
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
OH=training/fullset
SLICE=training/slice280
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

STOPS="${STOPS:-1 2 4 7 10 20 30 40 50 60}"
SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
export SD
mkdir -p "$SD"
# THE REFERENCE ARM'S SCHEDULE, UNCHANGED: val at every stop, the slice at the ten-epoch points.
# I briefly made the slice run at every stop so an in-sample and a held-out number would exist at
# the same epoch; reverted at the user's direction, because matching the reference arm exactly is
# worth more than an early gap reading and the slice is ~29 min a pass.
SLICE_AT="${SLICE_AT:-10 20 30 40 50 60}"
ALPHA_RANK="${ALPHA_RANK:-1}"
RANK_ACCEPT="${RANK_ACCEPT:-0.23}"
RANK_S="${RANK_S:-3}"
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
        SCORED_PAIRS=$OH/pool_labelled.csv \
        PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
        RANK_FORM=zsoftmax ALPHA_RANK="$ALPHA_RANK" RANK_ACCEPT="$RANK_ACCEPT" \
        RANK_S="$RANK_S" \
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
    CKPT="$R/$H" TAG="fs_ep$E" SIDES=$SIDES ./shell_scripts/score_checkpoint.sh \
        || { say "ABORT: scoring epoch $E failed"; exit 9; }

    # NO EARLY STOPPING. The reference arm carried an epoch-2 gate -- stop if the easy bin
    # sits more than EASY_DROP points below shipped -- and it is deliberately absent here at
    # the user's direction: this arm runs all 60 epochs whatever epoch 2 looks like. The
    # reason to want that is specific rather than general. zsoftmax is a SELECTION term and
    # the easy bin is where selection is already right, so early damage there is the cost the
    # arm is paying for the hard bins, not evidence it has gone wrong -- and the gate could
    # not tell those apart. Scoring still happens at every stop, so a genuine collapse is
    # visible at epoch 1 and the run can be stopped by hand.
    PREV_EP=$E
done
say "CHAIN COMPLETE over epochs $STOPS"
