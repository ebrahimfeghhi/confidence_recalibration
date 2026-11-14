#!/bin/bash
# THREE lr ARMS, ONE AFTER ANOTHER: 5e-4, then 1e-4, then 8e-4. 20 epochs each, held-out eval
# EVERY epoch, everything else identical to 7mo52r (PAE + rel at alpha 59, cropped, dropout 0.25,
# cross-seed 3 structures/step, crop allowlist on).
#
# ORDER IS THE USER'S: 5e-4 first, 1e-4 second, 8e-4 third. Not sorted, because the point is to
# see the middle of the range before its ends.
#
# EACH ARM STARTS FROM SHIPPED, not from the previous arm. A sweep where arm two inherits arm
# one's weights measures a schedule, not a learning rate.
#
# WALL CLOCK: an epoch is ~11.4 min and an eval ~11.2 min on the same three cards, so eval-every-
# epoch roughly doubles an epoch's cost. 20 epochs is about 7.5 h, and the three run about 23 h
# in total. That is the price of the cadence and it was asked for deliberately.
#
# IT WAITS FOR THE CARDS RATHER THAN TAKING THEM. launch_pae_crop_arm.sh runs cards_busy itself
# and refuses occupied cards, so each arm here waits for the previous one's processes to exit
# before trying. A refusal stops the chain instead of silently skipping an arm.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

EPOCHS="${EPOCHS:-20}"
STALL_S="${STALL_S:-2400}"

for LR in 5e-4 1e-4 8e-4; do
    say "=== arm lr=$LR, $EPOCHS epochs, eval every epoch ==="

    # THE CARDS MUST BE FREE FIRST. Polling cards_busy rather than a pid list: the previous arm's
    # pids are not this script's to know, and matching a pattern for them is what has twice
    # killed this session's own shell.
    for i in $(seq 1 120); do
        B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py 2>&1 | tail -1)
        [ "$B" = "0" ] && break
        sleep 30
    done
    B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py 2>&1 | tail -1)
    [ "$B" = "0" ] || { say "ABORT: cards still busy ('$B') after 60 min"; exit 3; }

    LOG="training/runs/lrsweep_${LR}.log"
    CARDS=4,5,7 EPOCHS="$EPOCHS" LR="$LR" PAIR_DROPOUT=0.25 \
    ALPHA_RANK=59 RANK_S=3 EVAL_EVERY=1 \
    REL_LABELS=training/fold/itm_labels_crop \
    PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
        ./shell_scripts/launch_pae_crop_arm.sh > "$LOG" 2>&1
    say "launcher for $LR returned $?"

    RID=$(grep -oE "^run [a-z0-9]{6}" "$LOG" | head -1 | awk '{print $2}')
    if [ -z "${RID:-}" ]; then
        say "ABORT: no run id in $LOG"
        grep -E "REFUS|STOP" "$LOG" | head -3
        exit 4
    fi
    say "lr=$LR -> run $RID"
    echo "$RID" > "training/runs/lrsweep_${LR}_rid"

    # THE WATCHERS BELONG TO THE ARM, started here so each arm has them from its first epoch.
    for i in $(seq 1 60); do [ -f "training/runs/$RID/args.json" ] && break; sleep 10; done
    nohup ./shell_scripts/health_watch.sh "$RID" > "training/runs/$RID/health.log" 2>&1 &
    nohup ./shell_scripts/plot_on_epoch.sh "$RID" > "training/runs/$RID/plot_on_epoch.log" 2>&1 &

    # WAIT FOR THIS ARM TO REACH ITS EPOCH COUNT, or die. Liveness from run.log's mtime for the
    # same reason the watchers use it.
    while :; do
        E=$(ls "training/runs/$RID"/head_ep*.pt 2>/dev/null \
            | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
        [ "${E:-0}" -ge "$EPOCHS" ] && { say "lr=$LR finished at epoch $E"; break; }
        if grep -qE "^STOP:|Traceback \(most recent call last\)" \
                "training/runs/$RID/run.log" 2>/dev/null; then
            say "ABORT: $RID ($LR) refused or raised at epoch ${E:-0}"
            grep -E "^STOP:" "training/runs/$RID/run.log" | head -2
            exit 5
        fi
        MT=$(stat -c %Y "training/runs/$RID/run.log" 2>/dev/null || echo 0)
        NOW=$(date +%s)
        [ $((NOW - MT)) -gt "$STALL_S" ] && {
            say "ABORT: $RID ($LR) silent for $(( (NOW - MT) / 60 )) min at epoch ${E:-0}"; exit 6; }
        sleep 60
    done

    # ITS OWN PROCESSES MUST GO BEFORE THE NEXT ARM ASKS FOR THE CARDS. The trainer exits on its
    # own at the epoch count; this is the wait for that, not a kill.
    for i in $(seq 1 60); do
        B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py 2>&1 | tail -1)
        [ "$B" = "0" ] && break
        sleep 30
    done
done
say "ALL THREE ARMS DONE"
