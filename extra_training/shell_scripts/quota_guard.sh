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

# the repo root, derived from this file so a rename of the checkout needs no edit here
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOG="${LOG:-$R/extra_training/quota_guard.log}"
# ABSOLUTE GB, not a percentage. The threshold is a decision about how much headroom a fold needs
# to finish its current write, which is a number of gigabytes, not a fraction of whatever the quota
# happens to be -- and "90%" could not express the 3,000 GB asked for on 2026-10-14. A percentage is
# still accepted and still honoured, whichever triggers first.
STOP_GB="${STOP_GB:-3000}"
WARN_GB="${WARN_GB:-2900}"
# KILLING A DELIBERATE DATABASE BUILD IS NOT REMEDIATION. The MSA rebuild of 2026-10-14 fills the
# disk ON PURPOSE -- UniRef30 is 450 GB extracted plus a 301 GB index -- so a guard that TERMs it
# partway loses hours of transfer and leaves a half-built index that the next run cannot resume from.
# With ALLOW_KILL=0 the guard reports loudly and takes no action; set it to 1 to restore the original
# behaviour once the build is done.
ALLOW_KILL="${ALLOW_KILL:-0}"
WARN_PCT="${WARN_PCT:-80}"
STOP_PCT="${STOP_PCT:-99}"
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

used_gb() {
    # the used column, normalised to GB. quota(1) prints 1886G, or 1886M / 1.8T on other volumes.
    quota -s 2>/dev/null | awk '
        /nvme/ {
            u=$2
            unit=substr(u, length(u), 1)
            gsub(/[^0-9.]/,"",u)
            if (unit == "T") u = u * 1024
            else if (unit == "M") u = u / 1024
            else if (unit == "K") u = u / 1048576
            printf "%d", u
        }'
}

log "quota guard started (stop at ${STOP_GB} GB or ${STOP_PCT}%, warn at ${WARN_GB} GB or ${WARN_PCT}%, every ${INTERVAL}s)"
log "guarded pattern: ${GUARDED_RE}"

stopped=0
while true; do
    pct=$(used_pct)
    gb=$(used_gb)
    if [ -z "$pct" ] || [ -z "$gb" ]; then
        log "WARN could not read quota"
        sleep "$INTERVAL"; continue
    fi

    if { [ "$gb" -ge "$STOP_GB" ] || [ "$pct" -ge "$STOP_PCT" ]; } && [ "$stopped" -eq 0 ]; then
        log "ESCALATE quota at ${gb} GB (${pct}%) >= ${STOP_GB} GB / ${STOP_PCT}% -- stopping the MSA build"
        # Kill by explicit PID, never by pattern: a pattern kill matches this script too.
        if [ "$ALLOW_KILL" != "1" ]; then
            log "ESCALATE ALLOW_KILL=0 -- NOT killing anything. Free space by hand; the MSA build is"
            log "ESCALATE filling the disk deliberately and a TERM would lose the transfer."
            du -sh $R/tools/colabfold/db 2>/dev/null >> "$LOG"
            stopped=1
            sleep "$INTERVAL"; continue
        fi
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
        du -sh $R/tools/colabfold/db 2>/dev/null >> "$LOG"
        ls -S $R/tools/colabfold/db 2>/dev/null | head -5 | sed 's/^/    /' >> "$LOG"
    elif [ "$gb" -ge "$WARN_GB" ] || [ "$pct" -ge "$WARN_PCT" ]; then
        log "WARN quota at ${gb} GB (${pct}%)"
    else
        # quiet unless something changed, so the log stays readable over a long run
        if [ $(( 10#$(date +%M) % 15 )) -eq 0 ]; then log "ok quota at ${gb} GB (${pct}%)"; fi
        stopped=0
    fi
    sleep "$INTERVAL"
done
