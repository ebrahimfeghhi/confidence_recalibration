#!/bin/bash
# Phase 1 of the staged MSA database build: uniref30_2302 only.
#
# WHY NOT JUST RUN setup_databases.sh: its DOWNLOADS_READY block fetches uniref30 AND the 128 GB
# colabfold_envdb in one go, so both tarballs plus both unpacked databases would coexist -- about
# 825 GB, over our 606 GB quota headroom. So the downloads happen here, one database at a time, and
# the script is then invoked with the other stages' marker files pre-created so it only extracts and
# indexes uniref30.
#
# RESUMABLE BY DESIGN. The server restarts ~17:00 PST 2026-10-10 and the upstream script's own
# downloader uses `curl -L -o` with no resume flag, so a killed download would restart from zero.
# Every download here uses `curl -C -` and is retried until the file reaches its expected size, so
# re-running this script after the restart continues where it stopped.
#
# USAGE:  bash phase1_uniref.sh            (safe to re-run; skips what is already complete)
set -u
cd "$(dirname "$0")"

# THE REPO ROOT IS DERIVED, NOT HARDCODED. These three paths were absolute under the repo's
# old name (tts_abag) and silently broke when it was renamed; two levels up from
# tools/colabfold is the root whatever the directory is called.
REPO="$(cd ../.. && pwd)"

DB="${DB:-$REPO/tools/colabfold/db}"
CORES="${CORES:-0-63}"   # 64 is the user-stated maximum
mkdir -p "$DB"
export PATH="$REPO/tools/mmseqs/bin:$PATH"
MMSEQS_BIN="$REPO/tools/mmseqs/bin/mmseqs"

log() { echo "[$(date '+%F %T')] $*"; }

# name -> expected byte size, so a truncated file is detected rather than trusted
declare -A WANT=(
  ["uniref30_2302.tar.gz"]=106688276925
  ["uniref30_2302_newtaxonomy.tar.gz"]=0
  ["pdb100_230517.fasta.gz"]=0
)
declare -A URL=(
  ["uniref30_2302.tar.gz"]="https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz"
  ["uniref30_2302_newtaxonomy.tar.gz"]="https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz"
  ["pdb100_230517.fasta.gz"]="https://opendata.mmseqs.org/colabfold/pdb100_230517.fasta.gz"
)

for f in uniref30_2302.tar.gz uniref30_2302_newtaxonomy.tar.gz pdb100_230517.fasta.gz; do
  u="${URL[$f]}"
  want="${WANT[$f]}"
  if [ "$want" = "0" ]; then
    want=$(curl -sIL "$u" | awk 'BEGIN{IGNORECASE=1}/^content-length/{c=$2} END{gsub(/\r/,"",c); print c+0}')
  fi
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    have=$(stat -c %s "$DB/$f" 2>/dev/null || echo 0)
    if [ "$have" -ge "$want" ] && [ "$want" -gt 0 ]; then
      log "$f complete ($have bytes)"
      break
    fi
    log "$f: $have / $want bytes, attempt $attempt"
    curl -fsSL -C - -o "$DB/$f" "$u" || log "$f: curl returned $?, will retry"
  done
  have=$(stat -c %s "$DB/$f" 2>/dev/null || echo 0)
  if [ "$have" -lt "$want" ]; then
    log "FAILED to fully download $f ($have / $want) -- re-run this script"
    exit 1
  fi
done

log "all phase-1 downloads complete"

# Pre-create the markers for every stage that is NOT phase 1, so setup_databases.sh only extracts and
# indexes uniref30. COLABDB_READY keeps it away from colabfold_envdb (phase 2); PDB_READY,
# PDB100_READY and PDB_MMCIF_READY keep it away from templates and the 92.3 GB PDB rsync (phase 3).
touch "$DB/DOWNLOADS_READY" "$DB/COLABDB_READY" "$DB/PDB_READY" "$DB/PDB100_READY" "$DB/PDB_MMCIF_READY"
log "markers set; building uniref30 only"

# MMSEQS_NO_INDEX=1 is REQUIRED, not an optimisation. createindex writes a .idx of roughly 300 GB
# on top of the 232 GB extracted database; it exhausted a 1 TB quota, killed the running fold
# workers, and left a corrupt partial index. setup_databases.sh itself calls the index "not useful
# for colabfold_search in most cases", and Protenix's scripts/colabfold_msa.py defaults to
# db_load_mode 0, which does not use one. Search with --db-load-mode 0.
MMSEQS_NO_INDEX=1 UNIREF30DB=uniref30_2302 taskset -c "$CORES" bash setup_databases.sh "$DB"
rc=$?
log "setup_databases.sh exit=$rc"
# UNIREF30_READY alone is NOT evidence of success: setup_databases.sh touches it before the
# taxonomy and index steps, so a failed run still reports ready. Check the databases are readable.
if [ "$rc" -eq 0 ] && [ -f "$DB/UNIREF30_READY" ] \
   && "$MMSEQS_BIN" dbtype "$DB/uniref30_2302_db" >/dev/null 2>&1 \
   && "$MMSEQS_BIN" dbtype "$DB/uniref30_2302_db_seq" >/dev/null 2>&1 \
   && [ -s "$DB/uniref30_2302_db_taxonomy" ]; then
  log "uniref30_2302 built and readable"
  # the tarball is extracted; holding 107 GB of it is what filled the quota
  rm -f "$DB/uniref30_2302.tar.gz"
  log "removed the extracted tarball"
  du -sh "$DB" 2>/dev/null
else
  log "uniref30 build incomplete -- re-run this script, it resumes"
fi
exit $rc
