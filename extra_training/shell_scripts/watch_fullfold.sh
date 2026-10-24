#!/bin/bash
# OVERNIGHT WATCHER FOR THE FULL-FOLD ARM. Plots and pushes every epoch without needing anyone
# awake. Runs until the arm finishes or CANCEL_FULLFOLD_WATCH appears.
#
# WHY THIS EXISTS AND A MONITOR DOES NOT SUFFICE. The queued arm gets a NEW run id that nobody
# knows until launch.py picks it, so a watcher pointed at 24u342's log would quietly tail a dead
# file after the handoff and report nothing while twelve epochs went by unplotted.
#
# HOW THE RUN IS IDENTIFIED, and why not "newest directory". The queue runs a --grad-probe FIRST,
# which also creates a run directory, so "newest" would latch onto the probe and never advance.
# `latest` is a six-character symlink and would match a bare id pattern too. So the arm is
# identified by what it IS, from its own args.json: 12 epochs AND a full_train manifest. That
# cannot match the probe (1 epoch) or any earlier arm (overfit30).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=training/runs/fullfold_watch.log
exec >> "$LOG" 2>&1
echo "=== full-fold watcher started $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

find_rid() {
    $PY -I - <<'INNER'
import glob, json, os
best = None
for f in glob.glob('training/runs/*/args.json'):
    rid = os.path.basename(os.path.dirname(f))
    if rid == 'latest':
        continue
    try:
        a = json.load(open(f))
    except Exception:
        continue
    sel = a.get('manifest_selection') or ''
    if 'full_train' in sel and float(a.get('epochs') or 0) == 12.0 and not a.get('grad_probe'):
        m = os.path.getmtime(f)
        if best is None or m > best[0]:
            best = (m, rid)
print(best[1] if best else '')
INNER
}

RID=""
for _ in $(seq 100000); do
    [ -f training/runs/CANCEL_FULLFOLD_WATCH ] && { echo "CANCELLED"; exit 0; }
    RID=$(find_rid)
    [ -n "$RID" ] && break
    sleep 60
done
[ -n "$RID" ] || { echo "never found the full-fold run"; exit 1; }
echo "$RID" > training/runs/fullfold_rid
echo "$(date '+%H:%M:%S') watching $RID"
[ -f "training/runs/${RID}_last_plotted" ] || echo 0 > "training/runs/${RID}_last_plotted"

while :; do
    [ -f training/runs/CANCEL_FULLFOLD_WATCH ] && { echo "CANCELLED"; exit 0; }
    LASTP=$(cat "training/runs/${RID}_last_plotted" 2>/dev/null || echo 0)
    EP=$($PY -I -c "
import sys; sys.path.insert(0,'.')
from run_logging import plot_three as P3
d=P3.load_chain('$RID',30)
print(max(d) if d else 0)
" 2>/dev/null || echo 0)
    if [ "${EP:-0}" -gt "${LASTP:-0}" ] 2>/dev/null; then
        # shards must be non-empty before anything is recorded, never a hardcoded count
        N=$(cat training/runs/$RID/train_eval_shards_ep$EP/rows_b30_*.csv 2>/dev/null | grep -vc '^iface' || echo 0)
        if [ "${N:-0}" -gt 0 ] 2>/dev/null; then
            ./shell_scripts/refresh_three.sh "$RID" >/dev/null 2>&1 \
                && echo "$(date '+%H:%M:%S') plotted ep$EP ($N shard rows)" \
                || echo "$(date '+%H:%M:%S') PLOT FAILED ep$EP"
            echo "$EP" > "training/runs/${RID}_last_plotted"
            # Only 12 epochs in this arm, so every one is worth a commit: the GitHub trail is the
            # thing that can be read in the morning without this session being awake.
            ./shell_scripts/push_findings.sh "$RID" "$EP" 2>&1 | tail -1
        else
            echo "$(date '+%H:%M:%S') ep$EP shards empty, leaving last_plotted alone"
        fi
    fi
    if [ -f "training/runs/$RID/status.json" ]; then
        echo "$(date '+%H:%M:%S') $RID ended: $(cat training/runs/$RID/status.json | tr -d '\n ')"
        exit 0
    fi
    sleep 120
done
