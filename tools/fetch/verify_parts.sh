#!/bin/bash
# Verify each downloaded chunk really holds the bytes of the file it claims to.
#
# WHY: the first parallel_fetch used `curl -C - -r "$s-$e"`, where curl rewrites the Range header
# from the LOCAL file offset. A FRESH chunk is unaffected, but `--retry` could have re-requested
# mid-chunk with the broken offset, and the result still has a plausible byte COUNT -- every size
# check downstream passes. Proven on this server: a 100-byte partial resuming 1124-2047 returned
# 45,997,834 bytes of the wrong region.
#
# HOW: for each part, re-fetch a small window from the server at the part's ABSOLUTE start and at its
# tail, and compare. Head and tail together catch a chunk that began correctly and drifted.
set -u
URL="${1:?usage: verify_parts.sh <url> <parts-dir> <total-bytes> <n-chunks>}"
D="${2:?}"; TOTAL="${3:?}"; N="${4:?}"
W=4096
CH=$(( (TOTAL + N - 1) / N ))
bad=0
for i in $(seq 0 $((N-1))); do
    part="$D/p$i"
    [ -s "$part" ] || { echo "  p$i: absent, will be fetched fresh"; continue; }
    have=$(stat -c %s "$part")
    s=$(( i * CH ))
    for where in head tail; do
        if [ "$where" = head ]; then
            off=$s; local_skip=0
        else
            [ "$have" -le "$W" ] && continue
            off=$(( s + have - W )); local_skip=$(( have - W ))
        fi
        curl -sL --max-time 120 -r "$off-$(( off + W - 1 ))" -o /tmp/vp_srv.$$ "$URL" 2>/dev/null
        dd if="$part" of=/tmp/vp_loc.$$ bs=1 skip="$local_skip" count="$W" 2>/dev/null
        if cmp -s /tmp/vp_srv.$$ /tmp/vp_loc.$$; then
            echo "  p$i $where ok (absolute byte $off)"
        else
            echo "  p$i $where MISMATCH at absolute byte $off -- this chunk holds the wrong data"
            bad=1
        fi
    done
done
rm -f /tmp/vp_srv.$$ /tmp/vp_loc.$$
[ "$bad" -eq 0 ] && echo "all parts verified at head and tail" || echo "SOME PARTS ARE WRONG; delete those and refetch"
exit $bad
