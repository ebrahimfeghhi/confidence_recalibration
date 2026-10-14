#!/bin/bash
# Quota guard: checks every 60 s and STOPS the MSA build before it fills the disk.
#
# WHY THIS EXISTS: on 2026-10-10 `mmseqs createindex` wrote a 284 GB partial index on top of a
# 232 GB database, exhausted the 1 TB quota, and killed the running Protenix fold workers -- their
# logs show dumper.py failing on mkdir. The chain script then saw both workers exit and declared
# "ALL PHASES COMPLETE", so roughly 1,000 of 1,230 PXM folds were silently skipped. Nothing was
# watching the quota, and `df` does not reflect it, so there was no warning at all.
#
# WHAT IT PROTECTS: only the MSA build is ever stopped. Fold workers are never touched -- they are
# the victims here, not the cause, and poses do not reproduce.
#
# USAGE: tmux new-session -d -s quota_guard "bash quota_guard.sh"
set -u

LOG="${LOG:-/home/ebrahim/tts_abag/extra_training/quota_guard.log}"
WARN_PCT="${WARN_PCT:-80}"
STOP_PCT="${STOP_PCT:-90}"
INTERVAL="${INTERVAL:-60}"
# Processes this guard is allowed to stop, matched on the command line. Deliberately narrow.
GUARDED_RE="${GUARDED_RE:-phase1_uniref|setup_databases.sh|tsv2exprofiledb|createindex|indexdb|colabfold_search}"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

used_pct() {
    # quota charges by file OWNER and df does not reflect it, so parse quota(1) directly.
    quota -s 2>/dev/null | awk '
        /nvme/ {
            u=$2; l=$4
            gsub(/[^0-9.]/,"",u); gsub(/[^0-9.]/,"",l)
            if (l+0 > 0) printf "%d", (u/l)*100
        }'
}

log "quota guard started (warn ${WARN_PCT}%, stop ${STOP_PCT}%, every ${INTERVAL}s)"
log "guarded pattern: ${GUARDED_RE}"

stopped=0
while true; do
    pct=$(used_pct)
    if [ -z "$pct" ]; then
        log "WARN could not read quota"
        sleep "$INTERVAL"; continue
    fi

    if [ "$pct" -ge "$STOP_PCT" ] && [ "$stopped" -eq 0 ]; then
        log "ESCALATE quota at ${pct}% >= ${STOP_PCT}% -- stopping the MSA build"
        # Kill by explicit PID, never by pattern: a pattern kill matches this script too.
        pids=$(ps -eo pid=,args= | grep -E "$GUARDED_RE" | grep -v grep | grep -v quota_guard | awk '{print $1}')
        if [ -z "$pids" ]; then
            log "ESCALATE nothing matching the guarded pattern is running -- cannot self-remediate"
        else
            for p in $pids; do
                log "  killing PID $p: $(ps -p "$p" -o args= 2>/dev/null | cut -c1-110)"
                kill "$p" 2>/dev/null
            done
            sleep 5
            for p in $pids; do
                if kill -0 "$p" 2>/dev/null; then log "  PID $p still alive after TERM"; fi
            done
            stopped=1
            log "ESCALATE stopped. Free space, then re-run phase1_uniref.sh -- it resumes."
        fi
        # name the biggest offenders so the fix is obvious
        du -sh /home/ebrahim/tts_abag/tools/colabfold/db 2>/dev/null >> "$LOG"
        ls -S /home/ebrahim/tts_abag/tools/colabfold/db 2>/dev/null | head -5 | sed 's/^/    /' >> "$LOG"
    elif [ "$pct" -ge "$WARN_PCT" ]; then
        log "WARN quota at ${pct}%"
    else
        # quiet unless something changed, so the log stays readable over a long run
        if [ $(( 10#$(date +%M) % 15 )) -eq 0 ]; then log "ok quota at ${pct}%"; fi
        stopped=0
    fi
    sleep "$INTERVAL"
done
