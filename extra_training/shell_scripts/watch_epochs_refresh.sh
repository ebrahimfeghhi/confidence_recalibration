#!/bin/bash
# Regenerate the plot and the results doc the moment an epoch's FOURTH metric lands.
#
# WHY AN EVENT AND NOT A TIMER. An epoch is 55-80 min and the figure that completes it -- the DockQ
# eval -- lands minutes after the others. A 2-hour cron either reports a stale epoch or the same
# one twice.
#
# WHY "TRAIN: mean_dockq" IS THE TRIGGER. The epoch boundary runs in a fixed order: timing
# breakdown, memory peaks, resume bundle, fingerprint cache, held-out PAE, then the DockQ eval. That
# last line therefore means all four metrics for the epoch exist. Triggering on the timing line
# would publish an epoch with two of its four values blank.
#
# RESTART-SAFE, which the first version was not. `tail -n0` discards history, so a watcher that
# crashed or was restarted after an epoch finished would never replay that epoch and the artifacts
# could stay stale indefinitely. This records the last published epoch in a state file, reconciles
# against the run's own outputs on startup, and republishes anything missed. Events are then
# deduplicated by epoch number, so two watchers racing cannot double-publish.
#
# A FAILED REFRESH IS LOUD. set -o pipefail and an explicit exit check: silently proceeding to the
# next event is how a stale doc looks identical to a current one.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
LOG=${1:-training/runs/2w5to5/run.log}
STATE=${2:-training/runs/epoch_refresh.state}

publish() {   # $1 = epoch that triggered it
    local ep=$1
    local last=0
    [ -f "$STATE" ] && last=$(cat "$STATE" 2>/dev/null || echo 0)
    if [ "$ep" -le "$last" ]; then
        echo "  epoch $ep already published (last=$last); skipping"
        return 0
    fi
    if ./shell_scripts/refresh_full_train_plot.sh 2>&1 | sed 's/^/  /'; then
        echo "$ep" > "$STATE"
        echo "  published through epoch $ep"
    else
        # NOT RECORDED, so the next event retries rather than skipping past the gap.
        echo "  REFRESH FAILED for epoch $ep; state left at $last so the next event retries"
        return 1
    fi
}

# RECONCILE BEFORE WATCHING. The run's own log is the authority on what has completed; the state
# file only records what has been published.
LATEST=$(grep -hoE "epoch [0-9]+ TRAIN: mean_dockq" "$LOG" 2>/dev/null \
         | grep -oE "[0-9]+" | sort -n | tail -1)
echo "watching $LOG, $(date '+%Y-%m-%d %H:%M:%S %Z')"
if [ -n "${LATEST:-}" ]; then
    echo "log's latest completed epoch: $LATEST"
    publish "$LATEST" || true
else
    echo "no completed epoch in the log yet"
fi

# -n0 from here on is correct: history is already reconciled above.
tail -F -n0 "$LOG" 2>/dev/null | grep --line-buffered -oE "epoch [0-9]+ TRAIN: mean_dockq" | \
while IFS= read -r line; do
    EP=$(echo "$line" | grep -oE "[0-9]+")
    echo "$(date '+%H:%M:%S') epoch $EP completed"
    publish "$EP" || true
done
