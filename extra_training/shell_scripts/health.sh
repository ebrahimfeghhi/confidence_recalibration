#!/bin/bash
# Consolidated health report every 30 minutes: GPUs, quota, sessions, and progress of each running job.
#
# This is the REPORT. The two acting watchers stay separate and run every 60 s:
#   quota_guard.sh  stops the MSA build at 90% of quota
#   gpu_guard.sh    escalates when a worker session is alive but nothing of ours is on a GPU
# Keeping them apart means a slow report can never delay a stop.
#
# USAGE: tmux new-session -d -s health "bash health.sh"
set -u
# the repo root, derived from this file so a rename of the checkout needs no edit here
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOG=$R/extra_training/health.log
PXM=$R/experimental_data/protenix/pxm2025/free_extra10
W=$R/extra_training/building_msa/work
INTERVAL="${INTERVAL:-1800}"

say() { echo "$*" >> "$LOG"; }

while true; do
    say ""
    say "========== $(date '+%F %H:%M') =========="

    say "-- quota"
    quota -s 2>/dev/null | awk '/nvme/{printf "   %s of %s (%d%%)\n", $2, $4, (substr($2,1,length($2)-1)/substr($4,1,length($4)-1))*100}' >> "$LOG" 2>/dev/null

    say "-- gpus (index, util, mem used)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 \
        | sed 's/^/   /' >> "$LOG"
    say "-- our processes on gpus"
    n=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
        [ "$u" = "ebrahim" ] && n=$((n+1))
    done
    say "   $n"

    say "-- sessions"
    say "   $(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"

    say "-- pxm sweep"
    done_n=$(find $PXM/poses -name run.json 2>/dev/null | wc -l)
    rows=$(($(wc -l < $PXM/results.csv 2>/dev/null || echo 1)-1))
    fail=$(grep -c FAILED $PXM/w0.log $PXM/w1.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
    say "   folds $done_n/$(wc -l < $PXM/queue.tsv 2>/dev/null) scored rows $rows   failed $fail"

    say "-- msa pipeline"
    say "   $(tail -1 $R/extra_training/overnight.log 2>/dev/null | cut -c1-100)"
    say "   a3m: $(ls $W/msa_initial_idx/*.a3m 2>/dev/null | wc -l)   pairing.a3m: $(find $W/mmcif_msa -name pairing.a3m 2>/dev/null | wc -l)"

    say "-- escalations in the last 30 min"
    e=$(grep -h ESCALATE $R/extra_training/quota_guard.log $R/extra_training/gpu_guard.log 2>/dev/null \
        | awk -v cut="$(date -d '30 minutes ago' '+%F %T')" '$0 > "["cut' | tail -5)
    say "   ${e:-none}"

    sleep "$INTERVAL"
done
