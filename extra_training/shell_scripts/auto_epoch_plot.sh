#!/usr/bin/env bash
# THE PER-EPOCH PLOT, FOR WHATEVER IS RUNNING, WITHOUT BEING RE-ARMED.
#
# Every previous version of this was a cron pinned to one run id, so each new arm needed a new
# cron and every handover lost the epochs between the old job being deleted and the new one
# firing -- which happened at Arm A->B, at B->decay, and twice for Arm E. The user should not
# have to ask for the plot.
#
# THE LIVE RUN IS DISCOVERED, NOT CONFIGURED: the run directory whose run.log was written most
# recently, within STALE seconds. The reference arm (drawn grey) comes from a file that can be
# repointed once per experiment rather than per run.
#
# Prints one of:
#   SEND <rid> <epoch>   -- a new epoch landed and the figure has been redrawn
#   QUIET                -- nothing new, or nothing training
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/ebrahim/Protenix/.venv/bin/python
STALE=${STALE:-600}
FIG=${FIG:-figures/paeonly_group_loss.png}
REF_FILE=training/runs/plot_reference_rid
REF=$(cat "$REF_FILE" 2>/dev/null || echo "")

now=$(date +%s); live=""; newest=0
for d in training/runs/*/; do
    # `latest` IS A SYMLINK TO THE LIVE RUN, not a run of its own. Including it made the newest
    # run.log appear twice, and whichever name won the tie became the figure's label -- the
    # watcher announced "latest" instead of the arm, and wrote its epoch marker under that name,
    # which would then let the real run's next epoch send twice or not at all.
    case "$(basename "$d")" in latest|archive) continue;; esac
    l="$d/run.log"; [ -f "$l" ] || continue
    m=$(stat -c %Y "$l"); age=$(( now - m ))
    [ "$age" -le "$STALE" ] || continue
    [ "$m" -gt "$newest" ] && { newest=$m; live=$(basename "$d"); }
done
[ -n "$live" ] || { echo "QUIET"; exit 0; }
# A DIAGNOSTIC IS NOT AN ARM. The live run with the newest log can be a one-epoch probe
# (gradient split, memory walk, smoke gate) whose loss curve means nothing next to a 20-epoch
# arm -- drawing it would relabel the figure mid-experiment and send a one-point line. Any run
# with a `_noplot` marker is skipped here and the watcher stays QUIET until a real arm starts.
[ -f "training/runs/${live}_noplot" ] && { echo "QUIET"; exit 0; }
[ "$live" = "$REF" ] && { echo "QUIET"; exit 0; }

# `|| true` IS LOAD-BEARING. This file runs under `set -e` with pipefail, and a brand-new arm has
# no head_ep*.pt yet, so `ls` exits 2, the pipeline inherits it, and the assignment killed the
# whole script -- the watcher exited 2 instead of printing QUIET for the first ~8 minutes of
# every new arm, which is exactly the handover it exists to cover.
ep=$(ls "training/runs/$live"/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' \
     | sort -n | tail -1 || true)
ep=${ep:-0}
# EPOCH 0 IS NOT AN EPOCH. A run that has started but not finished its first epoch has nothing
# new to draw, and `prev` defaults to -1, so without this it would send an empty figure once.
[ "$ep" -gt 0 ] || { echo "QUIET"; exit 0; }
mark="training/runs/${live}_last_autoplot"
prev=$(cat "$mark" 2>/dev/null || echo -1)
[ "$ep" -gt "$prev" ] || { echo "QUIET"; exit 0; }

# THE REFERENCE IS DRAWN ONLY IF IT IS A DIFFERENT, READABLE RUN. A missing or equal reference
# would otherwise make plot_group_loss_arms refuse, and a missing plot is how this went unnoticed.
name(){ cat "training/runs/$1_name" 2>/dev/null || echo "ref"; }
args=()
[ -n "$REF" ] && [ -d "training/runs/$REF" ] && args+=("$(name "$REF") ($REF)=$REF")
args+=("$(name "$live") ($live)=$live")
TITLE="PAE training loss on good vs bad poses" "$PY" -I run_logging/plot_group_loss_arms.py \
    "$FIG" "${args[@]}" >/dev/null 2>&1 || { echo "QUIET"; exit 0; }
echo "$ep" > "$mark"
echo "SEND $live $ep"
