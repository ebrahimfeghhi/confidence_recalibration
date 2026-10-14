#!/bin/bash
# GPU guard: every 60 s, records who holds each GPU and whether OUR work is still on the cards.
#
# WHY: on 2026-10-10 the fold workers died when the quota filled, the chain script read both workers
# exiting as "ALL PHASES COMPLETE", and ~1,000 of 1,230 folds were silently skipped. quota_guard.sh
# now watches the disk; this watches the other half -- a session alive but its GPU process gone, or
# someone else taking a card we were using.
#
# IT DOES NOT KILL ANYTHING. Other users share GPUs 2-7, our fold workers are relaunched by
# watch_free_extra10.sh, and poses do not reproduce, so the right action here is to record and
# escalate in the log rather than act.
#
# USAGE: tmux new-session -d -s gpu_guard "bash gpu_guard.sh"
set -u

# the repo root, derived from this file so a rename of the checkout needs no edit here
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOG="${LOG:-$R/extra_training/gpu_guard.log}"
INTERVAL="${INTERVAL:-60}"
OURS="${OURS:-0,1}"          # the GPUs this project is allowed to use

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "gpu guard started (ours: $OURS, every ${INTERVAL}s)"
prev_sessions=""

while true; do
    # our tmux sessions that should own a GPU
    sessions=$(tmux ls 2>/dev/null | grep -oE '^fx_[a-z0-9]+_[0-9]+' | sort | tr '\n' ' ')
    # our processes actually resident on a GPU
    mine=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read p; do
               [ -n "$p" ] && [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "ebrahim" ] && echo "$p"
           done | wc -l)

    if [ "$sessions" != "$prev_sessions" ]; then
        log "sessions changed: '${prev_sessions}' -> '${sessions}'"
        prev_sessions="$sessions"
    fi

    # a worker session alive but nothing of ours on a GPU means it is wedged, not working
    if [ -n "$sessions" ] && [ "$mine" -eq 0 ]; then
        log "ESCALATE worker sessions exist ($sessions) but NO ebrahim process is on any GPU"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 \
            | sed 's/^/    /' >> "$LOG"
    fi

    # someone else on a GPU we claim
    for g in ${OURS//,/ }; do
        uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
               | awk -F', ' -v g="$g" '$1==g {print $2}')
        [ -z "$uuid" ] && continue
        others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
                 | awk -F', ' -v u="$uuid" '$1==u {print $2}' | while read p; do
                     u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
                     [ -n "$u" ] && [ "$u" != "ebrahim" ] && echo "$u:$p"
                 done | tr '\n' ' ')
        [ -n "$others" ] && log "WARN GPU $g also held by: $others"
    done

    if [ $(( 10#$(date +%M) % 15 )) -eq 0 ]; then
        log "ok $mine of our processes on GPUs; sessions: ${sessions:-none}"
    fi
    sleep "$INTERVAL"
done
