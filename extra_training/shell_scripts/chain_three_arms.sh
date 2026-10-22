#!/bin/bash
# RUN THREE 100-EPOCH ARMS BACK TO BACK, in the order the user asked for:
#   1. pose-share            (shared injection at 4 depths, no L2)
#   2. pose-share + L2-SP    (same, plus lambda 1e-2 anchoring the shared injection to shipped)
#   3. full-head L2-SP       (all 19.8M trainable, lambda 1e-5) -- this one launched early as
#                             jjvdhk and was stopped so it could run last as intended.
#
# ~100 epochs x ~7 min each is roughly 12 h per arm, so about 35 h for the chain. Each arm waits
# for cards_busy.py to reach 0 rather than merely for the previous trainer to exit, because the
# final DockQ eval shards over 4,5,7 and outlives the ranks -- a queued launch already fired into
# one of those transients today.
#
# STOPS THE CHAIN if an arm's launcher returns non-zero: a crash means the cards may be wanted to
# diagnose it, and running the next arm would bury the evidence.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LOG=training/runs/chain_three_arms.log
exec >> "$LOG" 2>&1
echo "=== chain armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
wait_for_cards () {
    for i in $(seq 1 180); do
        B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py 2>/dev/null || echo 9)
        [ "$B" = "0" ] && return 0
        echo "$(date '+%H:%M:%S') cards busy ($B), waiting"
        sleep 20
    done
    echo "$(date '+%H:%M:%S') GIVING UP: cards never freed"
    return 1
}
for arm in launch_poseshare_overfit30.sh launch_poseshare_l2_overfit30.sh launch_l2sp_overfit30.sh; do
    echo "=== $(date '+%H:%M:%S') next arm: $arm ==="
    wait_for_cards || exit 1
    ./shell_scripts/$arm
    rc=$?
    echo "$(date '+%H:%M:%S') $arm launcher returned $rc"
    [ "$rc" = "0" ] || { echo "STOPPING THE CHAIN: $arm exited $rc"; exit 1; }
done
echo "=== $(date '+%H:%M:%S') chain complete ==="
