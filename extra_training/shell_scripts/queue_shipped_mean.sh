#!/usr/bin/env bash
# SUPERSEDED by queue_shipped_head.sh -- DO NOT RUN.
# Why: it drives the superseded score_shipped.sh.
# Kept for the record only; it refuses below.
echo "REFUSING: this script is superseded by queue_shipped_head.sh. See the header." >&2
exit 64
# Scores the SHIPPED head on val with mean pooling, once the in-sample pass frees the cards.
# Waits on the cards rather than pre-empting: the Arm B epoch-10 in-sample pass is the result
# being waited for, and killing it to save twelve minutes would be the wrong trade.
set -euo pipefail
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python

say "waiting for the cards (Arm B ep10 in-sample pass is finishing)"
for i in $(seq 1 240); do
    n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    case "$n" in ''|*[!0-9]*) n=99;; esac
    [ "$n" -eq 0 ] && break
    sleep 30
done
[ "${n:-99}" -eq 0 ] || { say "NOT LAUNCHED: cards still busy ($n) after 2 h"; exit 3; }
say "cards clear; scoring the SHIPPED head on val, budget 30, both poolings"

# CKPT deliberately EMPTY -- that is what selects the shipped head.
CKPT="" TAG=shipped_mean SIDES=val \
    ./shell_scripts/score_shipped.sh
say "done -> $SD/val30_shipped_mean.json"
