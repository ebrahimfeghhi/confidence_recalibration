#!/bin/bash
# Plot and push one arm every epoch, until it writes status.json. Takes the rid explicitly --
# NEVER guesses it, because `ls -t training/runs | head -1` has twice picked a log file or the
# `latest` symlink (itself six lowercase characters, so it passes a naive rid filter).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:-}"
[ -n "$RID" ] || { echo "usage: $0 <rid>"; exit 1; }
[ -f "training/runs/${RID}_last_plotted" ] || echo 0 > "training/runs/${RID}_last_plotted"
echo "$(date '+%H:%M:%S') watching $RID"
while :; do
    [ -f training/runs/CANCEL_WITHIN_WATCH ] && { echo "CANCELLED"; exit 0; }
    LASTP=$(cat "training/runs/${RID}_last_plotted" 2>/dev/null || echo 0)
    EP=$($PY -I -c "
import sys; sys.path.insert(0,'.')
from run_logging import plot_three as P3
d=P3.load_chain('$RID',30)
print(max(d) if d else 0)
" 2>/dev/null || echo 0)
    if [ "${EP:-0}" -gt "${LASTP:-0}" ] 2>/dev/null; then
        # Shards must be non-empty before anything is recorded -- never a hardcoded row count,
        # since val10 and valfull have different interface counts.
        N=$(cat training/runs/$RID/train_eval_shards_ep$EP/rows_b30_*.csv 2>/dev/null | grep -vc '^iface' || echo 0)
        if [ "${N:-0}" -gt 0 ] 2>/dev/null; then
            # last_plotted ADVANCES ONLY ON SUCCESS. It used to be written in both branches, so
            # one transient refresh_three.sh or cp failure permanently skipped that epoch's panel
            # AND its push -- the loop only ever looks for an epoch GREATER than last_plotted.
            # Leaving it alone means the next tick retries the same epoch.
            if ./shell_scripts/refresh_three.sh "$RID" >/dev/null 2>&1 \
               && cp training/runs/three_panels.png "training/runs/${RID}_panels.png"; then
                echo "$(date '+%H:%M:%S') plotted ep$EP ($N shard rows)"
                echo "$EP" > "training/runs/${RID}_last_plotted"
                ./shell_scripts/push_findings.sh "$RID" "$EP" 2>&1 | tail -1
            else
                echo "$(date '+%H:%M:%S') PLOT FAILED ep$EP, will retry next tick"
            fi
        else
            echo "$(date '+%H:%M:%S') ep$EP shards empty, leaving last_plotted alone"
        fi
    fi
    if [ -f "training/runs/$RID/status.json" ]; then
        echo "$(date '+%H:%M:%S') $RID ended: $(tr -d '\n ' < training/runs/$RID/status.json)"
        exit 0
    fi
    sleep 60
done
