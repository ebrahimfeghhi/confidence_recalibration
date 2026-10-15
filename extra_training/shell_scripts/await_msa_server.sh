#!/bin/bash
# Wait for opendata.mmseqs.org to come back, then start phase D on the corrected queries.
#
# WHY: the server returned HTTP 503 on every path at 2026-10-14 16:53 PT, so the 155 corrected
# queries have nothing to search against. It is a public dataset host and usually recovers within
# hours; this starts the search the moment it does rather than whenever someone next looks.
#
# GATED ON A REAL FILE, NOT A STATUS CODE ALONE. A 503 page still returns 200 from some caches, and
# its body is 17 bytes. So the gate is: HTTP 200 AND a content-length large enough to be the actual
# tarball. Anything else counts as still down.
#
# FREE_DB=0 ON PURPOSE: uniref30 stays on disk after the search. 1,042 GB of databases fits inside
# the 1,411 GB free, and keeping it means a failure in the envdb half does not force a 450 GB
# re-download -- the resume path that exists but has never been exercised twice.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W=$R/extra_training/building_msa/work
Q=$R/extra_training/building_msa/queries_corrected/query.fasta
URL=https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz
LOG=$W/await_msa_server.log
INTERVAL="${INTERVAL:-300}"      # 5 min: the canonical host is the simpler path, so catch it early
MIN_BYTES="${MIN_BYTES:-1000000000}" # a real tarball is hundreds of GB; 1 GB is a generous floor
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "waiting for $URL (every $((INTERVAL/60)) min); will start phase D on $(grep -c '^>' "$Q") queries"
tries=0
while :; do
    tries=$((tries+1))
    hdr=$(curl -sI --max-time 60 "$URL" 2>/dev/null)
    code=$(printf '%s' "$hdr" | head -1 | awk '{print $2}')
    len=$(printf '%s' "$hdr" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
    if [ "${code:-0}" = "200" ] && [ "${len:-0}" -ge "$MIN_BYTES" ] 2>/dev/null; then
        say "SERVER BACK after $tries check(s): HTTP $code, content-length $len"
        break
    fi
    [ $((tries % 12)) -eq 1 ] && say "still down: HTTP ${code:-?}, content-length ${len:-?}"
    sleep "$INTERVAL"
done

# ONE DATABASE, ONE BUILDER. The mirror build and this poller write THE SAME PLACES -- the database
# in tools/colabfold/db that both routes search, and msa_corrected_idx. With no lock, a canonical
# host returning mid-build would start phase D against a half-extracted database: the searches would
# read whatever happened to be on disk at that instant and report success, because phase D gates on
# data files existing and they would. So this does NOT start phase D while the mirror build holds the
# lock -- it says so and leaves the decision to a person, since switching routes mid-build means
# choosing which half-finished thing to abandon.
LOCK=$R/tools/colabfold/db/.msa_build.lock
if [ -e "$LOCK" ]; then
    owner=$(cat "$LOCK" 2>/dev/null)
    if kill -0 "${owner%% *}" 2>/dev/null; then
        say "SERVER IS BACK but the mirror build holds $LOCK (owner $owner)."
        say "  NOT starting phase D: it would search a half-built database and report success."
        say "  Decide which route to keep, stop the other, remove the lock, then re-run this."
        exit 0
    fi
    say "stale lock from a dead owner ($owner); removing it"
    rm -f "$LOCK"
fi
echo "$$ await_msa_server.sh $(date '+%F %T')" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# DO NOT START IF THE CARDS ARE ALREADY BUSY WITH SOMETHING THAT NEEDS THE CPUs. The search is
# taskset-pinned to 64 cores and the fold workers need CPU for featurisation; 288 cores means both
# fit, but the check is recorded so a future reader knows it was considered.
say "GPU workers running: $(tmux ls 2>/dev/null | grep -cE '^(fbtop|tcache)_[013]:')"
say "starting phase D"
mkdir -p "$W/msa_corrected_idx"
tmux new-session -d -s msa_d "bash $R/extra_training/phases/phase_d_search_uniref.sh \
    $Q $W/msa_corrected_idx corrected >> $W/phase_d_corrected.outer.log 2>&1"
sleep 30
if tmux has-session -t msa_d 2>/dev/null; then
    say "phase D launched (tmux msa_d); log $W/phase_d_corrected.log"
else
    say "ESCALATE phase D exited within 30 s -- see $W/phase_d_corrected.outer.log"
    tail -5 "$W/phase_d_corrected.outer.log" >> "$LOG" 2>/dev/null
fi
say "done waiting"
