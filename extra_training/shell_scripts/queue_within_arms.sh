#!/bin/bash
# RUN FOUR WITHIN-SEED ARMS BACK TO BACK, after zg5fnw releases cards 4,5,7.
#
#   L8    --rank-within-weight 8                  within-seed at 37.1% of pairwise gradient mass
#   L1    --rank-within-weight 1                  within-seed at 7.0%, its natural share -- CONTROL
#   L8K2  --rank-within-weight 8 --rank-topk-decay 2
#   L1K2  --rank-within-weight 1 --rank-topk-decay 2
#
# All four at --rank-pairs all, --rank-tau 0.0270, 25 epochs, overfit30 train / val10 held out.
# ~3h each at ~7 min/epoch, so ~12h for the set.
#
# A FULL 2x2, which is why both K=2 arms are queued rather than only the winning L. Running K=2
# on one L alone would confound the decay with the weight if the two interacted -- and they
# plausibly do, since the decay concentrates weight on pairs touching rank 1 while L concentrates
# it on within-seed pairs, and a pose's own seed-mates are its nearest competitors for rank 1.
# The K=0 pair runs first because the K=2 arms are read against them.
#
# WHY L=8 RUNS FIRST. It is the arm carrying the hypothesis, and at ~7 min/epoch each arm takes
# ~3h, so ordering decides which answer exists 3 hours sooner. 24u342 already serves as a partial
# control (cross-seed only, tau 0.0305), so if L=8 shows nothing against it the question is
# largely settled; if it shows something, L=1 is needed to say whether the WEIGHTING or merely
# the PRESENCE of within-seed pairs did it. Flip the two lines below to reverse.
#
# WHY IT WAITS RATHER THAN DISPLACING. zg5fnw is at epoch 8 of 12 on the full fold and is the
# only arm beating shipped there; stopping it short of ten epochs would forfeit the trend claim
# under this project's ten-epoch minimum. It releases the cards on its own.
#
# NEVER KILLS ANYTHING. It polls for zg5fnw's status.json and for cards_busy.py reporting 0; the
# GPU holders (pids 306042, 306045, 306050) are never touched.
#
# CANCEL WITH:  touch training/runs/CANCEL_WITHIN_QUEUE
# It is checked before each wait tick, so a cancel lands before the next arm starts but does not
# stop an arm already running.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=training/runs/within_queue.log
exec >>"$LOG" 2>&1
echo "=== within-seed queue started $(date '+%F %T %Z') ==="

# Only directories, 6 lowercase chars, and NEVER `latest` -- which is itself six lowercase
# characters and has resolved to the wrong run twice in this project.
newest_rid() {
    ls -t training/runs 2>/dev/null \
      | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
      | grep -E '^[a-z0-9]{6}$' | grep -vx latest | head -1
}

wait_for_cards() {
    for _ in $(seq 100000); do
        [ -f training/runs/CANCEL_WITHIN_QUEUE ] && { echo "CANCELLED"; exit 0; }
        if [ "$($PY -I run_logging/cards_busy.py)" = "0" ]; then return 0; fi
        sleep 60
    done
    return 1
}

run_arm() {
    local L="$1" K="$2" label="$3"
    echo "--- $(date '+%T') waiting for cards before $label ---"
    wait_for_cards || { echo "never got the cards for $label"; return 1; }
    local before; before=$(newest_rid)
    echo "$(date '+%T') launching $label (newest rid before: ${before:-none})"
    ./shell_scripts/launch_within_arm.sh "$L" "$K" &
    local launcher=$!
    local RID=""
    for _ in $(seq 600); do
        RID=$(newest_rid)
        [ -n "$RID" ] && [ "$RID" != "$before" ] && break
        sleep 2
        RID=""
    done
    if [ -z "$RID" ]; then
        echo "$label: no new run directory appeared; not watching"
    else
        echo "$RID" > "training/runs/within_rid_$label"
        echo "$(date '+%T') $label is $RID"
        nohup ./shell_scripts/watch_arm.sh "$RID" >>"training/runs/watch_$RID.log" 2>&1 &
    fi
    wait "$launcher"
    echo "$(date '+%T') $label launcher returned $? (rid ${RID:-unknown})"
}

run_arm 8 0 L8
run_arm 1 0 L1
run_arm 8 2 L8K2
run_arm 1 2 L1K2
echo "=== within-seed queue done $(date '+%F %T %Z') ==="
