# Source this. Resumable download whose completeness is decided by the SERVER and by the archive
# itself, never by a constant in this repo.
#
# WHY: a hardcoded expected size cost us 2.4 h of idle machine. uniref30_2302.db.tar.gz is
# 106,729,528,496 bytes; the constant written here said 106,688,276,925. The download finished, the
# equality gate rejected it, curl had nothing left to resume, and the chain died holding a complete
# and correct file. The server publishes the length on every request, so ask it.
#
# TWO SOURCES OF TRUTH, in order:
#   1. Content-Length from a HEAD on the final URL after redirects. Cheap and exact.
#   2. `tar -tzf` reading the whole stream. Slow (minutes) but definitive: a truncated gzip member
#      cannot be listed. This is the fallback when the server reports no length, and the tie-breaker
#      when it reports one we cannot reach.
#
# FUNCTIONS
#   remote_size <url>            bytes the server says, or 0 if it will not say
#   archive_whole <path>         0 if the gzip stream reads end to end
#   fetch_tarball <url> <path>   resume until complete; 0 on success

remote_size() {
    # -L follows redirects, so several Content-Length headers can come back; the last one belongs to
    # the final 200 response. IGNORECASE because header case is not guaranteed, and \r must go
    # because HTTP line endings are CRLF.
    curl --no-progress-meter -sIL --retry 3 --retry-delay 5 --max-time 120 "$1" 2>/dev/null \
      | awk 'BEGIN{IGNORECASE=1} /^content-length:/{n=$2} END{gsub(/[^0-9]/,"",n); print n+0}'
}

archive_whole() { tar -tzf "$1" >/dev/null 2>&1; }

fetch_tarball() {
    local url=$1 path=$2 want got try
    want=$(remote_size "$url")
    if [ "${want:-0}" -gt 0 ]; then
        say "  server reports $(numfmt --to=iec "$want") for $(basename "$path")"
    else
        say "  server reports no length; completeness will be judged by reading the archive"
    fi
    mkdir -p "$(dirname "$path")"
    for try in 1 2 3 4 5; do
        got=$(stat -c%s "$path" 2>/dev/null || echo 0)
        if [ "$want" -gt 0 ] && [ "$got" -eq "$want" ]; then
            say "  $(basename "$path") complete at $(numfmt --to=iec "$got") (matches the server)"
            return 0
        fi
        if [ "$want" -eq 0 ] && [ "$got" -gt 0 ] && archive_whole "$path"; then
            say "  $(basename "$path") reads end to end at $(numfmt --to=iec "$got")"
            return 0
        fi
        [ "$try" -gt 1 ] && say "  attempt $try: have $(numfmt --to=iec "$got"), resuming"
        curl --no-progress-meter -L -C - --retry 5 --retry-delay 10 -o "$path" "$url" >> "${LOG:-/dev/null}" 2>&1
    done
    # Out of attempts. Before failing, let the archive speak: a size that disagrees with the server by
    # a little is not evidence of corruption, and a stream that lists cleanly is complete by
    # definition.
    got=$(stat -c%s "$path" 2>/dev/null || echo 0)
    if [ "$want" -gt 0 ] && [ "$got" -eq "$want" ]; then return 0; fi
    if [ "$got" -gt 0 ] && archive_whole "$path"; then
        say "  size is $got against the server's $want, but the archive reads end to end -- accepting"
        return 0
    fi
    say "  $(basename "$path") is $got bytes and does not read as a whole archive"
    return 1
}
