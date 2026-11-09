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

python3 run_logging/results_table.py "$RID" > RESULTS.md 2>/dev/null || true

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
