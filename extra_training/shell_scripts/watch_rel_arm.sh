#!/bin/bash
# PER-EPOCH: plot both training terms and both diagnostics, then COMMIT AND PUSH, for every
# completed epoch of the rel-iTM arm. At the user's request.
#
# WHAT IS PUSHED: the figure and a one-line-per-epoch results table. NOT trajectory.csv -- it is
# appended per pose and would be a 5,520-row diff an epoch. The table is the reviewable artifact.
#
# WAITS ONLY. It never kills, launches or resumes anything, and it touches no GPU. If the run
# stops, this loop simply stops seeing new epochs; it is not a supervisor.
#
# flock'd: two copies pushing to the same branch would race on the index.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
exec 9>training/runs/.relwatch.lock
flock -n 9 || { echo "another copy holds the lock; exiting"; exit 0; }
RID="${1:-gu3kh7}"
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
LAST=0
[ -f "training/runs/${RID}_last_pushed" ] && LAST=$(cat "training/runs/${RID}_last_pushed")
DEADLINE=$(( $(date +%s) + 172800 ))

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  DONE=$($PY_BIN -I -c "
import csv, collections
c = collections.Counter()
try:
    for r in csv.DictReader(open('training/runs/$RID/trajectory.csv')):
        e = r.get('epoch')
        if e not in (None, ''): c[int(float(e))] += 1
except FileNotFoundError:
    pass
print(max([k for k, v in c.items() if v >= 5500] or [0]))
" 2>/dev/null || echo 0)
  if [ "${DONE:-0}" -gt "$LAST" ]; then
    $PY_BIN -I run_logging/plot_rel_arm.py "$RID" > "training/runs/${RID}_table.txt" 2>&1
    echo "$DONE" > "training/runs/${RID}_last_pushed"
    git add -f "figures/rel_arm_${RID}.png" "training/runs/${RID}_table.txt" \
              "training/runs/${RID}_last_pushed" >/dev/null 2>&1
    # THE EPOCH'S OWN NUMBERS IN THE COMMIT MESSAGE, so the history is readable without
    # checking out the figure.
    ROW=$(tail -1 "training/runs/${RID}_table.txt")
    git commit -q -m "$RID epoch $DONE: $ROW

Per-epoch artefact push for the rel-iTM arm (beta 75, mean pooling). Columns are
PAE loss, the relative-iTM term, within-interface Spearman against DockQ, and
std(predicted)/std(true iTM). The two diagnostics are the arm's prediction: the
head measured +0.3170 at d32kqp epoch 66 and the TRUE iTM reaches +0.6900, so
Spearman should move between them, and a spread ratio falling below 1 is the
margin-collapse shape every ranking arm here showed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014t55QPNWnMkjqFi95YVBtG" >/dev/null 2>&1 \
      && git push -q origin main >/dev/null 2>&1 \
      && echo "$(TZ=America/New_York date '+%-I:%M %p') pushed epoch $DONE: $ROW" \
      || echo "$(TZ=America/New_York date '+%-I:%M %p') epoch $DONE: commit/push FAILED"
    LAST=$DONE
  fi
  sleep 60
done
echo "watcher deadline reached"
