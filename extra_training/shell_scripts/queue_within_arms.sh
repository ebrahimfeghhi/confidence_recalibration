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

# THE SET OF RUN DIRECTORIES, not the newest one. `ls -t` orders by MTIME, and any write inside
# an existing run directory bumps it to the top -- rank_rho.load does os.replace into
# <rid>/rank_rho_b30_v3.csv on every plot refresh, and a sibling watch_arm.sh does that during
# this very detection window. "newest differs from before" would then latch onto an unrelated
# finished run, record the wrong rid, and leave the real arm unwatched and unpushed. A set
# difference cannot be fooled by mtime. `latest` is excluded by name because it is itself six
# lowercase characters and has resolved to the wrong run twice in this project.
rid_set() {
    ls training/runs 2>/dev/null \
      | while read -r n; do [ -d "training/runs/$n" ] && echo "$n"; done \
      | grep -E '^[a-z0-9]{6}$' | grep -vx latest | sort
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
    local before; before=$(rid_set)
    echo "$(date '+%T') launching $label ($(echo "$before" | grep -c . ) run dir(s) before)"
    ./shell_scripts/launch_within_arm.sh "$L" "$K" &
    local launcher=$!
    local RID=""
    for _ in $(seq 600); do
        RID=$(comm -13 <(echo "$before") <(rid_set) | head -1)
        [ -n "$RID" ] && break
        # Stop waiting if the launcher has already died -- otherwise a refusal (cards busy)
        # burns the full 20 minutes before the queue notices.
        kill -0 "$launcher" 2>/dev/null || break
        sleep 2
    done
    if [ -z "$RID" ]; then
        echo "$label: no new run directory appeared; not watching"
    else
        echo "$RID" > "training/runs/within_rid_$label"
        echo "$(date '+%T') $label is $RID"
        nohup ./shell_scripts/watch_arm.sh "$RID" >>"training/runs/watch_$RID.log" 2>&1 &
    fi
    # CAPTURE $? BEFORE ANY OTHER COMMAND RUNS. `echo "$(date) ... $?"` prints 0 always: the
    # command substitution executes during argument expansion and resets $? before it is read.
    wait "$launcher"
    local rc=$?
    echo "$(date '+%T') $label launcher returned $rc (rid ${RID:-unknown})"
    return "$rc"
}

# A FAILED ARM STOPS THE QUEUE. run_arm used to print the exit code and discard it, so a refusal
# (cards busy) or a crash in launch.py would be logged as a successful launch and the next arm
# would start on top of whatever caused it. Four arms sharing one set of cards means the first
# failure is usually a reason the rest will fail too, and 12 hours of them failing unattended is
# worse than stopping.
# ARMS COME FROM THE COMMAND LINE, defaulting to all four. L8 ran first at 25 epochs and was
# stopped at 15 once the target changed, which exited this queue by design; the remainder are
# then started by naming them, rather than by editing this list and risking a re-run of an arm
# whose results are already on disk.
#   ./shell_scripts/queue_within_arms.sh "1 0 L1" "8 2 L8K2" "1 2 L1K2"
ARMS=("$@")
[ "${#ARMS[@]}" -gt 0 ] || ARMS=("8 0 L8" "1 0 L1" "8 2 L8K2" "1 2 L1K2")
for spec in "${ARMS[@]}"; do
    [ -f training/runs/CANCEL_WITHIN_QUEUE ] && { echo "CANCELLED before $spec"; exit 0; }
    # shellcheck disable=SC2086
    if ! run_arm $spec; then
        echo "STOPPING THE QUEUE: $spec failed; the remaining arms were NOT started."
        exit 1
    fi
done
echo "=== within-seed queue done $(date '+%F %T %Z') ==="
