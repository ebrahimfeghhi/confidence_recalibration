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

DB="${DB:-/home/ebrahim/tts_abag/tools/colabfold/db}"
CORES="${CORES:-0-63}"   # 64 is the user-stated maximum
mkdir -p "$DB"
export PATH="/home/ebrahim/tts_abag/tools/mmseqs/bin:$PATH"

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

UNIREF30DB=uniref30_2302 taskset -c "$CORES" bash setup_databases.sh "$DB"
rc=$?
log "setup_databases.sh exit=$rc"
if [ "$rc" -eq 0 ] && [ -f "$DB/UNIREF30_READY" ]; then
  log "UNIREF30_READY present -- uniref30_2302 built"
  du -sh "$DB" 2>/dev/null
else
  log "uniref30 build incomplete -- re-run this script, it resumes"
fi
exit $rc
