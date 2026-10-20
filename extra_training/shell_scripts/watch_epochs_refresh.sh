#!/bin/bash
# Regenerate the plot and the results doc the moment an epoch's FOURTH metric lands.
#
# WHY A WATCHER AND NOT A TIMER. An epoch is ~55-80 min and the figure that completes it -- the
# DockQ eval -- lands several minutes after the others. A 2-hour cron either reports a stale epoch
# or the same one twice. This fires on the event itself.
#
# WHY "TRAIN: mean_dockq" IS THE TRIGGER. The epoch boundary runs in a fixed order: timing
# breakdown, memory peaks, resume bundle, fingerprint cache, held-out PAE, then the DockQ eval.
# That last line therefore means all four metrics for the epoch exist. Triggering on the epoch
# timing line instead would regenerate before the val numbers were written and show an epoch with
# two of its four values missing.
#
# IT DOES NOT MESSAGE ANYONE. It keeps the artifacts current so the plot and docs are never stale
# regardless of who is watching; the separate Monitor on the same log is what prompts a report.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
LOG=${1:-training/runs/2w5to5/run.log}
echo "watching $LOG for epoch completions, $(date '+%Y-%m-%d %H:%M:%S %Z')"
# -n0 so a restart does not re-fire on history; -F so log rotation or a recreated file is followed.
tail -F -n0 "$LOG" 2>/dev/null | grep --line-buffered -oE "epoch [0-9]+ TRAIN: mean_dockq .*" | \
while IFS= read -r line; do
    echo "$(date '+%H:%M:%S') $line"
    ./shell_scripts/refresh_full_train_plot.sh 2>&1 | sed 's/^/  /'
done
