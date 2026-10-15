#!/bin/bash
# Steps 1-4 of docs/msa_rebuild_plan.md: fetch uniref30_2302 from the GWDG mirror, verify it against
# the md5sums it ships, convert it to a searchable database, and apply the new taxonomy.
#
# WHY THE MIRROR. opendata.mmseqs.org has been 503 on every path since 2026-10-14 16:53 PT. The GWDG
# host (the Soding lab's own) serves the same releases, TSV packaging only -- both .db.tar.gz are 404
# there.
#
# WHY THIS IS WORTH STARTING NOW, when an hour ago it was not. The blocker was envdb: its TSV form
# "produced 265 GB that converted to nothing and still exited 0", which made the uniref half useless
# in both branches. That was never a conversion failure -- setup_databases.sh:20 defaults
# FAST_PREBUILT_DATABASES=1, and the flag controls BOTH the download URL and whether to convert, so
# phase E's own download bypassed the first and not the second. Measured on the real script: 0 calls
# to tsv2exprofiledb at the default, 1 call with FAST_PREBUILT_DATABASES=0. Nothing failed to
# convert; nothing was asked to.
#
# EVERY GATE IS A DATA FILE, NEVER AN EXIT CODE. That is not a style preference here: the conversion
# exits 0 when it does nothing, which is the whole reason this took two days to find.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DB=$R/tools/colabfold/db
W=$R/extra_training/building_msa/work
M=$R/tools/mmseqs/bin/mmseqs
LOG=$W/build_uniref_mirror.log
URL=https://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2302.tar.gz
TAR=$DB/uniref30_2302.tar.gz
CORES="${CORES:-0-63}"
export PATH="$R/tools/mmseqs/bin:$PATH"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die(){ say "STOP: $*"; exit 1; }
free_gb(){ quota -s 2>/dev/null | awk 'NR==3{u=$2; gsub(/[^0-9.]/,"",u); printf "%d", 3072-u}'; }

# THE LOCK, taken before anything is written. await_msa_server.sh launches phase D against the same
# database directory, so without this a canonical-host recovery mid-build would start a search over a
# half-extracted database and report success -- phase D gates on data files existing, and they would.
LOCK=$DB/.msa_build.lock
if [ -e "$LOCK" ]; then
    owner=$(cat "$LOCK" 2>/dev/null)
    kill -0 "${owner%% *}" 2>/dev/null && die "another MSA builder holds $LOCK (owner $owner)"
    say "stale lock from a dead owner ($owner); removing it"
    rm -f "$LOCK"
fi
echo "$$ build_uniref_from_mirror.sh $(date '+%F %T')" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

say "=== uniref30_2302 from the GWDG mirror. free: $(free_gb) GB ==="
[ "$(free_gb)" -ge 1000 ] || die "only $(free_gb) GB free; need ~936 for archive + TSVs + db"

# ---------------------------------------------------------------- 1. fetch
if [ ! -s "$DB/uniref30_2302_db_seq" ]; then
  if [ ! -s "$TAR" ] || [ "$(stat -c %s "$TAR")" -lt 100000000000 ]; then
    say "step 1: fetching (~96 GB). Resumable: -C - continues a partial file."
    hdr=$(curl -sIL --max-time 60 "$URL" 2>/dev/null)
    code=$(printf '%s' "$hdr" | grep -E '^HTTP' | tail -1 | awk '{print $2}')
    len=$(printf '%s' "$hdr" | awk 'tolower($1)=="content-length:"{print $2}' | tail -1 | tr -d '\r')
    say "  HTTP $code, content-length ${len:-?}"
    [ "${code:-0}" = "200" ] && [ "${len:-0}" -ge 1000000000 ] 2>/dev/null \
        || die "the mirror is not serving a real file (HTTP $code, length ${len:-?})"
    # PARALLEL RANGES. The mirror throttles per connection: 0.8 MB/s on one stream against 5.3 MB/s
    # from RCSB on this machine, and extra ranged connections ADD throughput rather than dividing it.
    # One stream is 32 hours for 96 GB. Each part is size-checked and the whole file against the
    # server's content-length, because a silently short chunk is this method's failure mode.
    bash "$R/extra_training/shell_scripts/parallel_fetch.sh" "$URL" "$TAR" "${CONNS:-8}" >> "$LOG" 2>&1 \
        || die "download failed; re-run to resume (parts are kept)"
    say "  fetched $(du -h "$TAR" | cut -f1)"
  else
    say "step 1: archive already present, $(du -h "$TAR" | cut -f1)"
  fi
  # COMPLETENESS FROM THE SERVER, not a constant: a hardcoded size cost this project 2.4 h once.
  want=$(curl -sIL --max-time 60 "$URL" 2>/dev/null | awk 'tolower($1)=="content-length:"{print $2}' | tail -1 | tr -d '\r')
  got=$(stat -c %s "$TAR")
  [ "$got" = "$want" ] || die "archive is $got bytes, the server says $want -- incomplete"
  tar -tzf "$TAR" >/dev/null 2>&1 || die "archive does not list cleanly; it is corrupt"
  say "  archive complete and listable"

  # ------------------------------------------------------------ 2. extract and verify
  say "step 2: extracting"
  ( cd "$DB" && tar -xzf uniref30_2302.tar.gz ) || die "extraction failed"
  [ -s "$DB/uniref30_2302.md5sum" ] || die "no md5sum file in the archive"
  say "  verifying every member against the bundled md5sum"
  bad=0
  while read -r sum name; do
    [ -s "$DB/$name" ] || { say "  MISSING $name"; bad=1; continue; }
    got=$(md5sum "$DB/$name" | cut -d' ' -f1)
    if [ "$got" = "$sum" ]; then say "  ok   $name"
    else say "  FAIL $name: $got != $sum"; bad=1; fi
  done < "$DB/uniref30_2302.md5sum"
  [ "$bad" -eq 0 ] || die "md5 verification failed; not converting a database we cannot trust"
  say "  every member verified. free: $(free_gb) GB"

  # ------------------------------------------------------------ 3. convert
  # FAST_PREBUILT_DATABASES=0 is the fix. MMSEQS_NO_INDEX=1 keeps setup_databases.sh from building
  # the index; phase D's own BUILD_INDEX=0 keeps IT from building one, and the disk budget needs both.
  say "step 3: tsv2exprofiledb (FAST_PREBUILT_DATABASES=0, MMSEQS_NO_INDEX=1)"
  rm -f "$DB/UNIREF30_READY"
  ( cd "$R/tools/colabfold" && FAST_PREBUILT_DATABASES=0 MMSEQS_NO_INDEX=1 UNIREF30DB=uniref30_2302 \
      taskset -c "$CORES" bash setup_databases.sh "$DB" ) >> "$W/build_uniref_setup.log" 2>&1
  say "  setup_databases.sh exit $? (NOT trusted; it exits 0 when it converts nothing)"
  for f in _db.dbtype _db_seq _db_aln _db_h; do
    [ -s "$DB/uniref30_2302$f" ] || die "missing or empty uniref30_2302$f -- the conversion produced nothing"
  done
  say "  conversion produced: $(du -ch $DB/uniref30_2302_db* 2>/dev/null | tail -1 | cut -f1)"
else
  say "steps 1-3: uniref30_2302_db_seq already present, skipping"
fi

# ---------------------------------------------------------------- 4. new taxonomy
NT=$DB/uniref30_2302_newtaxonomy.tar.gz
if [ -s "$NT" ]; then
  say "step 4: applying the new taxonomy from $(basename "$NT")"
  ( cd "$DB" && tar -xzf "$(basename "$NT")" ) || die "newtaxonomy extraction failed"
  got=$(md5sum "$DB/uniref30_2302_db_taxonomy" | cut -d' ' -f1)
  [ "$got" = "83105388ca315687dee85e6a26eaedf3" ] \
      || die "taxonomy md5 is $got, expected 83105388... (the copy this project already held)"
  say "  taxonomy matches the copy we already had"
else
  say "step 4: SKIPPED -- no newtaxonomy tarball at $NT"
fi

say "=== done. free: $(free_gb) GB ==="
say "NEXT: step 5, the conversion control -- one UNCHANGED query through phase D with BUILD_INDEX=0,"
say "comparing phase D's own uniref.a3m against the stored one BEFORE phase F merges anything."
say "The batch is NOT authorised until that passes."
