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
exec 9>"training/runs/.relwatch.${1:-default}.lock"
flock -n 9 || { echo "another copy holds the lock; exiting"; exit 0; }
RID="${1:-gu3kh7}"
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
# THE FIGURES ARE NAMED AFTER THE CHAIN ROOT, not after RID. A resume gets a fresh run id, so
# naming by RID would move the arm's figures to a new path halfway through and leave the repo
# with two partial figures of one run. The root is resolved from args.json's recorded --resume
# path by the same function the plotters use.
STEM=$($PY_BIN -I -c "
import sys; sys.path.insert(0, 'run_logging'); import plot_three
print(list(plot_three.run_chain('$RID'))[0])" 2>/dev/null || echo "$RID")
[ -n "$STEM" ] || STEM="$RID"
echo "watching $RID; figures and tables keyed to the chain root $STEM"
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
# THE RUN SAYS WHEN AN EPOCH IS DONE. A pose-count threshold of 5500 was inherited from an arm
# with a different epoch size; on a 6100-pose epoch it fired at 5600 and this watcher pushed
# epoch 1 with rel, rho and sd_ratio all '-' and a PAE averaged over 92% of the poses. The
# five-minute wait below could not save it -- the missing 500 poses were twelve minutes away.
import re, os
_lg = 'training/runs/$RID/run.log'
_t = open(_lg, errors='replace').read() if os.path.exists(_lg) else ''
_e = [int(m) for m in re.findall(r'\[rank 0\] epoch (\d+) rank-loss', _t)]
print(max(_e) if _e else 0)
" 2>/dev/null || echo 0)
  if [ "${DONE:-0}" -gt "$LAST" ]; then
    # WAIT FOR THE EPOCH'S OWN LOG LINES BEFORE PLOTTING. The trigger is the pose count in
    # trajectory.csv, but the epoch-summary lines -- rank-loss, WITHIN-INTERFACE SPEARMAN, SCORE
    # SPREAD -- are printed AFTER the last pose, so the first push plotted them as "-" and lost
    # that epoch's diagnostics. Waits up to 5 minutes, then plots anyway rather than stalling the
    # whole watcher on a line that may never come.
    for _ in $(seq 1 60); do
      grep -q "\[rank 0\] epoch ${DONE} rank-loss" "training/runs/$RID/run.log" 2>/dev/null \
        && break
      sleep 5
    done
    $PY_BIN -I run_logging/plot_rel_arm.py "$RID" "figures/rel_arm_${STEM}.png" \
      > "training/runs/${STEM}_table.txt" 2>&1
    # THE HELD-OUT BAR FIGURE TOO, whenever an eval has produced rows. Only the per-epoch 2x2
    # was being pushed, so the figure that carries the actual held-out result -- shipped against
    # both poolings, with a dot per interface -- existed only on disk.
    # BOTH NAMING CONVENTIONS -- the eval dirs are <pool>_eval_shards_ep<N> now, and a resumed
    # chain holds the legacy train_eval_shards_ep<N> under the parent run id.
    if ls -d "training/runs/$RID"/*_eval_shards_ep* >/dev/null 2>&1 \
       || ls -d "training/runs/$STEM"/*_eval_shards_ep* >/dev/null 2>&1; then
      $PY_BIN -I run_logging/plot_rel_bars.py "$RID" --best --out "figures/rel_bars_${STEM}.png" \
        > "training/runs/${STEM}_heldout.txt" 2>&1 || true
    fi
    echo "$DONE" > "training/runs/${RID}_last_pushed"
    git add -f "figures/rel_arm_${STEM}.png" "training/runs/${STEM}_table.txt" \
              "training/runs/${RID}_last_pushed" >/dev/null 2>&1
    [ -f "figures/rel_bars_${STEM}.png" ] && git add -f "figures/rel_bars_${STEM}.png" \
              "training/runs/${STEM}_heldout.txt" >/dev/null 2>&1
    # THE EPOCH'S OWN NUMBERS IN THE COMMIT MESSAGE, so the history is readable without
    # checking out the figure.
    # THE LAST COMPLETED EPOCH, not the last line. plot_rel_arm prints the in-flight epoch as a
    # PARTIAL row, and tail -1 picked it up the moment a resume had already started the next
    # epoch before the watcher's first pass -- putting "29 ... PARTIAL" in the epoch-28 commit.
    ROW=$(grep -v PARTIAL "training/runs/${STEM}_table.txt" | tail -1)
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
