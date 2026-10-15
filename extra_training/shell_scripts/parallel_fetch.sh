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

if [ -s "$OUT" ] && [ "$(stat -c %s "$OUT")" = "$TOTAL" ]; then
    say "already complete"; exit 0
fi
D="$OUT.parts"; mkdir -p "$D"
CH=$(( (TOTAL + N - 1) / N ))
pids=""
for i in $(seq 0 $((N-1))); do
    s=$(( i * CH )); e=$(( s + CH - 1 )); [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
    want=$(( e - s + 1 ))
    part="$D/p$i"
    # RESUME PER PART: -C - continues this part from wherever it stopped, so an interrupted run does
    # not restart 96 GB. The range is absolute, so a resumed part still lands at the right offset.
    if [ -s "$part" ] && [ "$(stat -c %s "$part")" = "$want" ]; then continue; fi
    ( curl -sL --fail --retry 8 --retry-delay 15 -C - -r "$s-$e" -o "$part" "$URL" ) &
    pids="$pids $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
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
rm -rf "$D"
say "complete: $(du -h "$OUT" | cut -f1)"
