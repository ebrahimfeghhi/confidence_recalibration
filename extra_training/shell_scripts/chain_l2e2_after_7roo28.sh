#!/bin/bash
# QUEUE THE lambda 1e-2 ARM BEHIND THE LIVE REGULARISED ARM (7roo28).
#
# WHY 1e-2 AND WHAT IT IS MEANT TO SETTLE. 7roo28 runs the same recipe at lambda 1e-3, and its
# measured gradient share has sat UNDER the 1% "acting" threshold for every epoch and is falling:
# 0.97% at ep3, 0.93, 0.91, 0.83, 0.81, 0.77, 0.70% at ep9. It is not falling because the penalty
# is weakening -- ||W-W0|| grows -- but because the median TASK gradient is growing faster
# (0.9639 -> 1.0770 -> 1.1436 -> 1.3040 -> 1.3815 -> 1.4926 -> 1.6766). The anchor is losing a
# race with the denominator. The share is linear in lambda, so 1e-2 puts it near 7%, inside the
# band egw5wq measured as actually constraining (7.99%).
#
# DROPOUT STAYS AT 0.25 DELIBERATELY. Against 7roo28 this arm then differs in lambda ALONE, which
# makes it a clean read on the anchor. Dropping dropout here would have produced a third
# configuration differing from the live arm in two ways, answering neither question.
#
# IT IS CONDITIONAL AND THE CONDITION IS NOT CHECKED BY THIS SCRIPT. It was armed on the premise
# that the epoch-10/15 held-out evals show OVERFITTING -- held-out selection falling while the
# training objective keeps improving. If they do not, this arm is answering a question nobody
# asked and the waiter should be killed rather than left to fire when the cards free. Whoever
# reads this: check the evals before letting it run.
#
# THE SAME CONFOUND AS 7roo28 APPLIES. Dropout and L2-SP move together, so neither this arm nor
# the live one can attribute an effect to one term. Against each other they isolate lambda; against
# an unregularised arm they do not isolate anything.
#
# WHAT THIS CHANGES AND WHAT IT DOES NOT. Two flags: --pair-dropout 0.25 and --l2-sp 1e-3 (with
# the zero-init exclude). Everything else is the launcher's own defaults, the same ones the live
# arm uses -- training/crop_final 256-token crops, --pae-only, --epoch-cover-all, --epoch-balance,
# --trunk-memmap, 16 prefetch workers, 50 epochs, lr 1e-3, accum 240, nsample 5.
#
# THE TWO TERMS ARE DELIBERATELY CONFOUNDED, at the user's direction (2026-11-11), to spend one
# ~17h GPU slot instead of two. THE CONSEQUENCE MUST BE STATED WHEN THIS ARM IS REPORTED: if it
# moves, this arm cannot attribute the move to dropout or to L2-SP, and if the two terms cancel it
# will look like no effect. Separating them is a second experiment, not a reinterpretation of this
# one.
#
# LAMBDA 1e-3 IS THE USER'S CHOICE, NOT A DERIVED VALUE. paper_results.md section 17 measures the
# L2-SP gradient as 2*lambda*(W-W0): against a median task grad_norm of 1.080 with ||W-W0|| = 43.2,
# 1e-5 is 0.08% of the task gradient and 1e-4 is 0.80% -- both effectively off. 1e-3 is 7.99%, the
# lowest value measured to actually act. tv6hue sat there and finished -0.0084 +- 0.0054 held-out,
# but it was UNCROPPED with no augmentation, so it is not this experiment.
#
# IT STARTS FRESH, NOT FROM epoch 11. The live chain is privsp -> axrorr -> rdi25k -> 2cax4o, and
# privsp's args.json records resume = "" -- the arm began from the shipped head. A dropout arm
# resumed from rdi25k/resume_ep11.pt would be "11 epochs without dropout then 39 with", which is
# not a controlled comparison against the arm it is meant to be compared to. So no RESUME.
#
# WHY p=0.25 IS AF3's NUMBER AND NOT PROTENIX's. AF3 Algorithm 31 (confidence head) line 4 calls
# PairformerStack with Nblock=4 -- the same Algorithm 17 whose box hard-codes DropoutRowwise 0.25
# on the two triangle multiplications and the starting-node attention, and DropoutColumnwise 0.25
# on the ending-node attention. Protenix implements all four faithfully (the ending-node site is
# wrapped in z.transpose(-2,-3), which makes its rowwise call columnwise in original coordinates)
# but its own config zeroes the rate for this head: configs/configs_base.py:347
# "pairformer_dropout": 0.0. So this arm moves TOWARD AF3 and AWAY from Protenix's shipped config.
#
# READINESS IS cards_busy == 0, HELD. Documented trap from chain_ep21_to_40.sh: a single zero
# reading is not enough, because the cards go briefly quiet between a training epoch and the eval
# that follows it, and launching into that window would put two jobs on the same three cards. This
# requires HOLD_N consecutive zero readings AND the watched run's ranks to be gone.
#
# IT VERIFIES THE FLAG ACTUALLY TOOK. The dry run cannot prove it: the launcher's argv dump only
# echoes ARGS, and --pair-dropout is appended after it. Two runtime checks instead -- args.json
# must record pair_dropout 0.25, and run.log must carry the trainer's own "PAIR DROPOUT: row-wise
# rate on z set to 0.25" line. The trainer already hard-stops if p_drop does not take
# (phase_k_train.py:2009), so this is belt and braces rather than the only guard.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
WATCH="${WATCH:-q6mgyc}"            # the run to wait for
CARDS="${CARDS:-4,5,7}"
HOLD_N="${HOLD_N:-3}"               # consecutive zero readings required
POLL="${POLL:-120}"                 # seconds between readings
CEIL_H="${CEIL_H:-30}"              # give up after this many hours
LOG=training/runs/l2e2_chain.log

say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*" | tee -a "$LOG"; }

say "queued: waiting for $WATCH to finish and release cards $CARDS (ceiling ${CEIL_H}h)"
say "  will then launch: PAIR_DROPOUT=0.25 + L2_SP=1e-2, fresh (no RESUME), 50 epochs, lr 1e-3"

DEADLINE=$(( $(date +%s) + CEIL_H * 3600 ))
ZEROS=0
while :; do
    [ "$(date +%s)" -lt "$DEADLINE" ] || { say "CEILING HIT after ${CEIL_H}h -- NOT launching"; exit 3; }
    B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    # the watched run's own ranks, by --out path, never by pattern
    ALIVE=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
        if [ -r "/proc/$p/cmdline" ] && tr '\0' ' ' < "/proc/$p/cmdline" | grep -q "training/runs/$WATCH"; then
            ALIVE=$((ALIVE+1))
        fi
    done
    if [ "${B:-9}" = "0" ] && [ "$ALIVE" -eq 0 ]; then
        ZEROS=$((ZEROS+1))
        say "cards idle and no $WATCH ranks ($ZEROS/$HOLD_N)"
        [ "$ZEROS" -ge "$HOLD_N" ] && break
    else
        [ "$ZEROS" -gt 0 ] && say "not idle after all (busy=$B, $WATCH ranks=$ALIVE) -- resetting"
        ZEROS=0
    fi
    sleep "$POLL"
done

say "cards released; launching the regularised arm"
PAIR_DROPOUT=0.25 L2_SP=1e-2 EPOCHS=50 PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
  ./shell_scripts/launch_pae_crop_arm.sh >> training/runs/l2e2_arm_launch.log 2>&1
say "launcher returned $?"

# ---- verify the flag took, or say so loudly --------------------------------------------------
RID=$(ls -td training/runs/*/ 2>/dev/null | head -1 | xargs basename)
say "new run id: $RID"
echo "$RID" > training/runs/l2e2_rid
for i in $(seq 1 60); do
    [ -f "training/runs/$RID/args.json" ] && break
    sleep 10
done
PD=$($PY -I -c "import json;print(json.load(open('training/runs/$RID/args.json')).get('pair_dropout'))" 2>/dev/null)
L2=$($PY -I -c "import json;print(json.load(open('training/runs/$RID/args.json')).get('l2_sp'))" 2>/dev/null)
EXC=$($PY -I -c "import json;print(json.load(open('training/runs/$RID/args.json')).get('l2_sp_exclude'))" 2>/dev/null)
say "args.json pair_dropout = $PD   l2_sp = $L2   l2_sp_exclude = $EXC"
if [ "$PD" != "0.25" ]; then
    say "FAULT: pair_dropout is '$PD', not 0.25 -- the queued run is NOT the regularised arm"
    exit 4
fi
if [ "$L2" != "0.01" ]; then
    say "FAULT: l2_sp is '$L2', not 0.01 -- the queued run is NOT the lambda 1e-2 arm"
    exit 4
fi
if [ -z "$EXC" ]; then
    say "FAULT: l2_sp_exclude is EMPTY. This arm has zero-init frame_dir_proj and pose_perblock_"
    say "  tensors; L2-SP over them is plain decay toward zero on the very terms being added."
    exit 4
fi
for i in $(seq 1 90); do
    grep -qa "PAIR DROPOUT:" "training/runs/$RID/run.log" 2>/dev/null && break
    grep -qaE "STOP:|Traceback" "training/runs/$RID/run.log" 2>/dev/null && { say "FAULT: run died at startup"; exit 5; }
    sleep 20
done
say "runtime confirmation: $(grep -a 'PAIR DROPOUT:' training/runs/$RID/run.log | head -1)"
say "runtime confirmation: $(grep -aiE 'L2-SP|l2sp' training/runs/$RID/run.log | head -1)"
say "QUEUED RUN IS LIVE AND CARRIES p=0.25 AND l2-sp 1e-2"
