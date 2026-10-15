#!/bin/bash
# Fetch one large file over N parallel byte-range connections, then verify it exactly.
#
# WHY: the GWDG mirror throttles PER CONNECTION. Measured 2026-10-14 -- a single stream sustains
# 0.8 MB/s while this machine pulls 5.3 MB/s from RCSB, and three extra ranged connections added
# 1.9 MB/s on top of the main stream rather than dividing it. So 96 GB is 32 hours on one connection
# and a few on eight. aria2c is not installed; curl ranges do the same thing.
#
# EVERY CHUNK IS SIZE-CHECKED, THEN THE WHOLE FILE. A silently short chunk is the obvious failure
# mode of this approach and it would produce a corrupt archive that still looks complete, so: each
# part must be exactly the bytes requested, the concatenation must equal the server's
# content-length, and the caller still verifies the md5sums the archive ships.
set -u
URL="${1:?usage: parallel_fetch.sh <url> <out> [connections]}"
OUT="${2:?}"
N="${3:-8}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

hdr=$(curl -sIL --max-time 60 "$URL" 2>/dev/null)
code=$(printf '%s' "$hdr" | grep -E '^HTTP' | tail -1 | awk '{print $2}')
TOTAL=$(printf '%s' "$hdr" | awk 'tolower($1)=="content-length:"{print $2}' | tail -1 | tr -d '\r')
[ "${code:-0}" = "200" ] || { say "STOP: HTTP $code"; exit 1; }
[ "${TOTAL:-0}" -ge 1000000000 ] 2>/dev/null || { say "STOP: content-length ${TOTAL:-?} is not a real file"; exit 1; }
say "$TOTAL bytes over $N connections -> $OUT"

# Fetch one part's remaining range over `per` sub-connections, and append ONLY when every sub-range
# is exactly the size it was asked for.
#
# WHY THAT CONDITION IS THE WHOLE POINT. The first version did `wait || true` and then appended any
# sub-file that was merely NON-EMPTY. With sub0 short and sub1 complete -- which is exactly what
# happened here on p2 -- it would concatenate sub0 then sub1, leaving a GAP where sub0's missing
# bytes belong and sub1's data at the wrong offset. The resume loop would then top the part up at the
# end, so the part reaches its expected SIZE holding the wrong bytes. The md5 verification would
# eventually catch it, at the cost of the entire 96 GB download.
fetch_part_parallel() {
    local i="$1" per="$2" s e want have rs left sub a b k f sw sh pids=""
    s=$(( i * CH )); e=$(( s + CH - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
    want=$(( e - s + 1 )); have=0; [ -s "$D/p$i" ] && have=$(stat -c %s "$D/p$i")
    [ "$have" -ge "$want" ] && return 0
    rs=$(( s + have )); left=$(( e - rs + 1 ))
    sub=$(( (left + per - 1) / per ))
    for k in $(seq 0 $((per-1))); do
        a=$(( rs + k * sub )); b=$(( a + sub - 1 )); [ "$b" -gt "$e" ] && b=$e
        [ "$a" -gt "$e" ] && continue
        # NAMED BY ITS ABSOLUTE RANGE, not its index. `p$i.sub$k` was reused across runs, but which
        # absolute range sub$k covers depends on how much the part already held -- so after an
        # interruption the ranges shift and a leftover sub0 holding the OLD range's bytes was reused
        # for the NEW one: right byte count, wrong bytes. Binding the name to "$a-$b" means a file
        # from a superseded split can never match a current range again.
        f="$D/p$i.sub.$a-$b"; sw=$(( b - a + 1 )); sh=0; [ -s "$f" ] && sh=$(stat -c %s "$f")
        if [ "$sh" -gt "$sw" ]; then say "  $(basename "$f") is $sh over $sw; discarding"; rm -f "$f"; sh=0; fi
        [ "$sh" -eq "$sw" ] && continue
        ( curl -sL --fail --retry 8 --retry-delay 15 -r "$(( a + sh ))-$b" "$URL" >> "$f" ) &
        pids="$pids $!"
    done
    for k in $pids; do wait "$k" || true; done
    # EVERY sub-range exact, or nothing is appended and the caller tries again.
    local ok=1
    for k in $(seq 0 $((per-1))); do
        a=$(( rs + k * sub )); b=$(( a + sub - 1 )); [ "$b" -gt "$e" ] && b=$e
        [ "$a" -gt "$e" ] && continue
        f="$D/p$i.sub.$a-$b"; sw=$(( b - a + 1 )); sh=0; [ -s "$f" ] && sh=$(stat -c %s "$f")
        if [ "$sh" -ne "$sw" ]; then
            say "  $(basename "$f") short: $sh of $sw -- NOT appending any sub-range of p$i"
            ok=0
        fi
    done
    [ "$ok" -eq 1 ] || return 1
    for k in $(seq 0 $((per-1))); do
        a=$(( rs + k * sub )); b=$(( a + sub - 1 )); [ "$b" -gt "$e" ] && b=$e
        [ "$a" -gt "$e" ] && continue
        f="$D/p$i.sub.$a-$b"
        [ -e "$f" ] || continue
        # THE PART MUST ALREADY END EXACTLY WHERE THIS RANGE BEGINS, or the append lands at the
        # wrong offset. Checked per append, so a stale file cannot slip in.
        local at=$(( s + $(stat -c %s "$D/p$i") ))
        if [ "$at" -ne "$a" ]; then
            say "  refusing $(basename "$f"): the part ends at absolute $at, not $a"
            return 1
        fi
        cat "$f" >> "$D/p$i" || return 1
        rm -f "$f"
    done
    return 0
}

if [ -s "$OUT" ] && [ "$(stat -c %s "$OUT")" = "$TOTAL" ]; then
    say "already complete"; exit 0
fi
D="$OUT.parts"; mkdir -p "$D"
CH=$(( (TOTAL + N - 1) / N ))

# VERIFY WHAT IS ALREADY THERE BEFORE BUILDING ON IT. A chunk from an earlier run may have been
# fetched with the broken `-C - -r` logic, which produces a plausible byte count of the wrong data,
# so resuming on top of it would extend a corrupt chunk and every size check would still pass.
if ls "$D"/p* >/dev/null 2>&1; then
    say "existing parts found; verifying them against the server before resuming"
    bash "$(dirname "${BASH_SOURCE[0]}")/verify_parts.sh" "$URL" "$D" "$TOTAL" "$N"         || { say "STOP: a part holds the wrong data. Delete the parts it named and re-run."; exit 1; }
fi
# COUNT WHAT IS ACTUALLY LEFT BEFORE LAUNCHING ANYTHING. The first version launched one connection
# per INCOMPLETE PART and only redistributed after those finished -- so on a resume with 2 parts left
# it ran 2 connections and the redistribution loop below was dead code, reached when there was
# nothing left to redistribute. 6 of 8 connections idle while 10 GiB remained, on a server that
# throttles per connection. The work has to be sized from the remaining BYTES, not the part count.
nshort_pre=0
for i in $(seq 0 $((N-1))); do
    s=$(( i * CH )); e=$(( s + CH - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
    w=$(( e - s + 1 )); h=0; [ -s "$D/p$i" ] && h=$(stat -c %s "$D/p$i")
    [ "$h" -lt "$w" ] && nshort_pre=$(( nshort_pre + 1 ))
done
if [ "$nshort_pre" -gt 0 ] && [ "$nshort_pre" -lt "$N" ]; then
    say "$nshort_pre of $N part(s) incomplete -- splitting their remaining ranges across all $N"
    per=$(( N / nshort_pre )); [ "$per" -lt 1 ] && per=1
    for i in $(seq 0 $((N-1))); do
        s=$(( i * CH )); e=$(( s + CH - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
        w=$(( e - s + 1 )); h=0; [ -s "$D/p$i" ] && h=$(stat -c %s "$D/p$i")
        [ "$h" -ge "$w" ] && continue
        fetch_part_parallel "$i" "$per" || say "  p$i incomplete this round; will retry"
    done
fi

pids=""
for i in $(seq 0 $((N-1))); do
    s=$(( i * CH )); e=$(( s + CH - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
    want=$(( e - s + 1 ))
    part="$D/p$i"
    # RESUME BY ABSOLUTE RANGE, NEVER WITH -C -. `curl -C - -r "$s-$e"` is broken for this: on a
    # partial part curl REWRITES the Range header from the LOCAL file offset, so a part that had
    # 100 bytes requested "bytes=100-" when it needed "bytes=1124-2047". It would silently fetch the
    # wrong region of the file, and every size check downstream would still pass because the byte
    # COUNT comes out right. A fresh part is unaffected, which is why this survived the first run.
    #
    # So the offset is computed here -- chunk start plus what the part already holds -- and curl is
    # given that absolute range with no -C at all, appending to the part.
    have=0
    [ -s "$part" ] && have=$(stat -c %s "$part")
    if [ "$have" -ge "$want" ]; then
        [ "$have" -eq "$want" ] || { say "STOP: $part is $have bytes, over the $want requested"; exit 1; }
        continue
    fi
    rs=$(( s + have ))
    ( curl -sL --fail --retry 8 --retry-delay 15 -r "$rs-$e" "$URL" >> "$part" ) &
    pids="$pids $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done

# REDISTRIBUTE, rather than letting finished connections idle. The first version launched one
# connection per chunk and never reassigned them, so as chunks completed their throughput was simply
# lost: 8 connections at 41 MB/s decayed to 2 at 2 MB/s with 10 GiB still to fetch, turning four
# minutes into an hour. The server throttles PER CONNECTION, so the fix is to keep N of them busy.
#
# Each incomplete part's REMAINING range is split across the connections the finished parts freed,
# fetched into sub-files, then appended in order. Every sub-range is absolute, so appending in order
# reconstructs the part exactly -- the same property that makes the resume correct.
round=0
while :; do
    short=""
    for i in $(seq 0 $((N-1))); do
        s=$(( i * CH )); e=$(( s + CH - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
        want=$(( e - s + 1 )); have=0; [ -s "$D/p$i" ] && have=$(stat -c %s "$D/p$i")
        [ "$have" -lt "$want" ] && short="$short $i"
    done
    [ -n "$short" ] || break
    round=$(( round + 1 ))
    [ "$round" -gt 12 ] && { say "STOP: still short after $round redistribution rounds"; exit 1; }
    nshort=$(echo $short | wc -w)
    per=$(( N / nshort )); [ "$per" -lt 1 ] && per=1
    say "round $round: $nshort part(s) short; $per connection(s) each"
    for i in $short; do
        fetch_part_parallel "$i" "$per" || say "  p$i still incomplete; next round"
    done
done
[ "$fail" -eq 0 ] || { say "STOP: a connection failed; re-run to resume"; exit 1; }

for i in $(seq 0 $((N-1))); do
    s=$(( i * CH )); e=$(( s + CH - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
    want=$(( e - s + 1 )); got=$(stat -c %s "$D/p$i" 2>/dev/null || echo 0)
    [ "$got" = "$want" ] || { say "STOP: part $i is $got bytes, wanted $want -- a short chunk is the"; \
        say "  failure mode this check exists for; re-run to resume that part"; exit 1; }
done
say "all $N parts the exact size requested; concatenating"
cat $(for i in $(seq 0 $((N-1))); do echo "$D/p$i"; done) > "$OUT.tmp" || { say "STOP: concat failed"; exit 1; }
got=$(stat -c %s "$OUT.tmp")
[ "$got" = "$TOTAL" ] || { say "STOP: assembled $got bytes, server says $TOTAL"; exit 1; }
mv -f "$OUT.tmp" "$OUT"
# THE PARTS ARE KEPT. Removing them here meant a failure in the caller's md5 verification left
# nothing to resume from and cost the whole 96 GB. The caller deletes $D only once the archive has
# verified end to end.
say "complete: $(du -h "$OUT" | cut -f1); parts kept in $D until the archive verifies"
