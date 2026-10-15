# fetch

Generic, resumable, parallel download of one large file. Written for the 96 GB UniRef30 and 118 GB
colabfold_envdb archives, but nothing in it is specific to those.

| file | what it does |
|---|---|
| `parallel_fetch.sh <url> <out> [connections]` | fetches over N byte-range connections, redistributing them as chunks finish, and assembles the file |
| `verify_parts.sh <url> <parts-dir> <total> <n>` | re-fetches a 4 KB window at each chunk's absolute start and tail and compares, to prove a chunk holds the bytes it claims |

## Why it exists

The GWDG mirror throttles PER CONNECTION: one stream sustained 0.8 MB/s while the same machine pulled
5.3 MB/s from RCSB, and extra ranged connections ADD throughput rather than dividing it. 96 GB is 32
hours on one connection and under an hour on eight.

## The four bugs it was built around

Each of these produced a plausible byte COUNT of the wrong data, which is why every one of them got
past a size check:

1. **`curl -C - -r "$s-$e"` does not resume a range.** On a partial file curl rewrites the Range
   header from the LOCAL offset, so a chunk holding 100 bytes asked for `bytes=100-` when it needed
   `bytes=1124-2047`. Measured: 45,997,834 bytes of the wrong region where 924 were wanted. The offset
   is computed here instead and curl is given an absolute range with no `-C`.
2. **Finished connections sat idle.** One connection per chunk, never reassigned, so 8 connections at
   41 MB/s decayed to 2 at 2 MB/s with 10 GiB left. Work is now sized from the remaining BYTES before
   anything launches.
3. **A short sub-range was appended anyway.** `wait || true` plus an `-s` test meant a short slice
   followed by a complete one was concatenated, leaving a gap with the later slice's data at the wrong
   offset. Every sub-range must now be exactly the size requested before ANY is appended.
4. **Sub-files were reused across runs.** `sub0` means "the first slice of what's left", and what is
   left changes after an interruption -- so a leftover `sub0` holding an older range was reused for a
   new one. Sub-files are now named by their absolute range, `sub.<start>-<end>`, so a file from a
   superseded split can never match current work. The part's own length is also re-checked against
   the range about to be appended.

## Contract

* every chunk must be exactly the bytes requested, and the assembly must equal the server's
  `content-length`
* the parts directory SURVIVES assembly. The caller deletes it only after verifying the archive
  itself -- md5 or equivalent -- so a bad member is refetchable without the whole transfer
* `verify_parts.sh` runs automatically before any resume, because building on an unverified chunk
  extends a corrupt one invisibly
