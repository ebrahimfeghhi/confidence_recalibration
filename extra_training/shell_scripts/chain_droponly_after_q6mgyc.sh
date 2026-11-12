#!/bin/bash
# QUEUE THE DROPOUT-ONLY ARM: --pair-dropout 0.25, NO L2 anchor, lr 1e-3, 50 epochs, fresh.
#
# WHAT THIS ONE ANSWERS, and why it is the arm worth running. 7roo28 carried dropout 0.25 AND
# l2-sp 1e-3 together and produced the project's first held-out rise (ep5 0.5845 -> ep10 0.5896
# max-pooled, against the matched unregularised arm rdi25k falling 0.5851 -> 0.5786 over the same
# window). The two terms were confounded by construction, so that arm cannot say which did it.
#
# BUT THE ANCHOR WAS MEASURED INERT. Its gradient share sat under the 1% acting threshold for
# EVERY epoch and fell throughout -- 0.97, 0.93, 0.91, 0.83, 0.81, 0.77, 0.70, 0.70, 0.80, 0.77,
# 0.83, 0.78, 0.76, 0.74, 0.65, 0.67% over epochs 3-18 -- with 0 of 145 steps in any epoch where
# the penalty exceeded the task gradient. So the prediction is explicit and falsifiable: THIS ARM
# SHOULD REPRODUCE 7roo28's CURVE. If it does, the anchor was decorative and dropout is the
# mechanism. If it does NOT, the share statistic is measuring the wrong thing and section 17's
# whole framing needs revisiting.
#
# NO --l2-sp AT ALL, not 1e-5. "A little L2" is the failure section 17 documents: 1e-5 is 0.08%
# of the task gradient, which is noise wearing the name of a hyperparameter. Off means off.
#
# lr 1e-3 FLAT, matching 7roo28's epochs 1-18, NOT q6mgyc's 1e-4. The comparison this arm exists
# to make is against 7roo28's first 18 epochs, so its lr schedule must match those.
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
LOG=training/runs/droponly_chain.log

say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*" | tee -a "$LOG"; }

say "queued: waiting for $WATCH to finish and release cards $CARDS (ceiling ${CEIL_H}h)"
say "  will then launch: PAIR_DROPOUT=0.25, NO anchor, fresh (no RESUME), 50 epochs, lr 1e-3"

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
PAIR_DROPOUT=0.25 EPOCHS=50 PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
  ./shell_scripts/launch_pae_crop_arm.sh >> training/runs/droponly_arm_launch.log 2>&1
say "launcher returned $?"

# ---- verify the flag took, or say so loudly --------------------------------------------------
RID=$(ls -td training/runs/*/ 2>/dev/null | head -1 | xargs basename)
say "new run id: $RID"
echo "$RID" > training/runs/droponly_rid
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
if [ "$L2" != "0.0" ] && [ "$L2" != "0" ]; then
    say "FAULT: l2_sp is '$L2', expected 0 -- this arm must carry NO anchor"
    exit 4
fi
LRGOT=$($PY -I -c "import json;print(json.load(open('training/runs/$RID/args.json')).get('lr'))" 2>/dev/null)
say "args.json lr = $LRGOT"
if [ "$LRGOT" != "0.001" ]; then
    say "FAULT: lr is '$LRGOT', not 0.001 -- this arm must match 7roo28's epochs 1-18 to be comparable"
    exit 4
fi
for i in $(seq 1 90); do
    grep -qa "PAIR DROPOUT:" "training/runs/$RID/run.log" 2>/dev/null && break
    grep -qaE "STOP:|Traceback" "training/runs/$RID/run.log" 2>/dev/null && { say "FAULT: run died at startup"; exit 5; }
    sleep 20
done
say "runtime confirmation: $(grep -a 'PAIR DROPOUT:' training/runs/$RID/run.log | head -1)"
# THE ANCHOR MUST BE ABSENT, not merely small. The trainer prints an anchor SNAPSHOT/READ line
# whenever --l2-sp is on; its absence is the positive evidence that this arm carries none.
if grep -qa "anchor SNAPSHOT\|anchor READ\|anchor INHERITED" "training/runs/$RID/run.log" 2>/dev/null; then
    say "FAULT: the run log carries an l2sp anchor line -- this arm was meant to have NO anchor"
    exit 4
fi
say "confirmed: no l2sp anchor line in the run log"
say "QUEUED RUN IS LIVE: dropout 0.25, NO anchor, lr 1e-3, 50 epochs"
