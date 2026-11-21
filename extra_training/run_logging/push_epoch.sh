#!/usr/bin/env bash
# Commit and push ONE epoch's results. Plots and numbers only -- no checkpoints, no run logs,
# no trajectories. Called by the epoch waiter so the repo tracks the run as it happens.
#
# WHAT IS DEPLIBERATELY NOT PUSHED: *.pt is gitignored anyway, and run.log (49 MB) plus the three
# trajectory csvs (11 MB each) would add ~80 MB per arm. The figures are ~130 KB and the metric
# jsons under 1 KB, which is the whole point -- the record should be cheap enough to push every
# epoch without thinking about it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
RID="${1:?usage: push_epoch.sh <rid> [epoch]}"
EP="${2:-?}"

# THE INTERPRETER IS LOAD-BEARING. This was `python3`, the system python, which has no numpy --
# results_table.py imports eval_curve, which imports numpy, so every call died on ModuleNotFound.
# The `> RESULTS.md` truncated the file FIRST and `2>/dev/null || true` swallowed the traceback,
# so RESULTS.md sat at zero bytes for every epoch and got committed that way. Build to a temp file
# and move it into place only on success, so a future failure leaves the last good table standing
# instead of silently publishing an empty one.
PY=/home/ebrahim/Protenix/.venv/bin/python
if "$PY" -I run_logging/results_table.py "$RID" > RESULTS.md.tmp 2>RESULTS.md.err && \
   [ -s RESULTS.md.tmp ]; then
  mv RESULTS.md.tmp RESULTS.md
  rm -f RESULTS.md.err
else
  echo "  WARNING: results_table.py failed; keeping the previous RESULTS.md"
  head -3 RESULTS.md.err 2>/dev/null
  rm -f RESULTS.md.tmp
fi

cd ..
git add -- "extra_training/figures/${RID}_train.png" \
           "extra_training/figures/${RID}_validation.png" \
           "extra_training/figures/fullhead_vs_restricted.png" \
           "extra_training/RESULTS.md" 2>/dev/null
git add -- extra_training/training/runs/"$RID"/metric_valfull83_ep*.json 2>/dev/null
git diff --cached --quiet && { echo "  nothing new to push"; exit 0; }
git commit -q -m "$RID epoch $EP: plots and held-out numbers

Pushed automatically at the epoch boundary. Figures and metric json only -- no checkpoints,
no run log, no trajectories.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014t55QPNWnMkjqFi95YVBtG"
git push -q origin "$(git branch --show-current)" && echo "  pushed $(git rev-parse --short HEAD)"
