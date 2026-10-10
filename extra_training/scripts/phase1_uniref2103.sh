#!/bin/bash
# Phase 1 of the staged MSA database build: uniref30_2103 only.
#
# WHY 2103 AND NOT THE NEWER 2302: every msa.sh in both of our benchmark prep trees uses
# uniref30_2103_db -- 131 of 131 for PXM-2025 and 113 of 113 for FoldBench -- and Protenix's own
# documented pipeline (scripts/msa/step2-get_msa.ipynb) uses it too. Building training MSAs on 2302
# would make the training and test sets differ in how their MSAs were constructed, which is a
# confound we do not need. 2103 also carries the taxonomy information that pairing requires; the
# notebook flags this explicitly.
#
# WHY THE DOWNLOAD IS NOT setup_databases.sh's JOB: opendata.mmseqs.org 404s for 2103, and the
# script's DOWNLOADS_READY block would fetch uniref30 AND the 118 GB envdb together, so both
# tarballs plus both unpacked databases would coexist. Downloads happen here, from gwdg, one database
# at a time; the script is then invoked with the other stages' markers pre-created so it only builds
# uniref30.
#
# 2103 IS THE OLD FORMAT, so FAST_PREBUILT_DATABASES=0 and mmseqs tsv2exprofiledb has to run. That is
# CPU-heavy but we have the cores; it is why peak disk here exceeds the download size.
#
# RESUMABLE BY DESIGN: the server restarts ~17:00 PST 2026-10-10 and the upstream downloader uses
# `curl -L -o` with no resume flag. Every download here uses `curl -C -` and retries until the file
# reaches its expected size, so re-running this script continues where it stopped.
#
# USAGE:  bash phase1_uniref2103.sh        (safe to re-run; skips what is already complete)
set -u
cd "$(dirname "$0")"

DB="${DB:-/home/ebrahim/tts_abag/tools/colabfold/db}"
CORES="${CORES:-0-31}"          # 64 is the hard cap on this box; 32 leaves room for the fold workers
mkdir -p "$DB"
export PATH="/home/ebrahim/tts_abag/tools/mmseqs/bin:$PATH"

log() { echo "[$(date '+%F %T')] $*"; }

BASE="https://wwwuser.gwdg.de/~compbiol/colabfold"
FILES=("uniref30_2103.tar.gz" "uniref30_2103_taxonomy.tar.gz")

for f in "${FILES[@]}"; do
  u="$BASE/$f"
  want=$(curl -sIL --max-time 60 "$u" | awk 'BEGIN{IGNORECASE=1}/^content-length/{c=$2} END{gsub(/\r/,"",c); print c+0}')
  if [ "$want" -le 0 ]; then log "could not determine size of $f -- aborting"; exit 1; fi
  for attempt in $(seq 1 12); do
    have=$(stat -c %s "$DB/$f" 2>/dev/null || echo 0)
    if [ "$have" -ge "$want" ]; then log "$f complete ($have bytes)"; break; fi
    log "$f: $have / $want bytes, attempt $attempt"
    curl -fsSL -C - -o "$DB/$f" "$u" || log "$f: curl exit $?, retrying"
  done
  have=$(stat -c %s "$DB/$f" 2>/dev/null || echo 0)
  if [ "$have" -lt "$want" ]; then log "FAILED $f ($have / $want) -- re-run to resume"; exit 1; fi
done
log "downloads complete"

# The taxonomy tarball is a separate artefact the upstream script knows nothing about for 2103, so
# unpack it here. It supplies uniref30_2103_db_taxonomy / _mapping, which is what makes pairing work.
if [ ! -f "$DB/TAXONOMY_EXTRACTED" ]; then
  log "extracting taxonomy"
  tar xzf "$DB/uniref30_2103_taxonomy.tar.gz" -C "$DB" && touch "$DB/TAXONOMY_EXTRACTED"
  log "taxonomy extracted"
fi

# Pre-create markers for every stage that is NOT phase 1: COLABDB_READY keeps the script away from
# colabfold_envdb (phase 2), and PDB_READY / PDB100_READY / PDB_MMCIF_READY away from templates and
# the PDB rsync. Templates come from Protenix's own runner/template_search.py (hmmsearch against
# pdb_seqres), not from ColabFold, so the 92 GB PDB mirror is never needed.
touch "$DB/DOWNLOADS_READY" "$DB/COLABDB_READY" "$DB/PDB_READY" "$DB/PDB100_READY" "$DB/PDB_MMCIF_READY"
log "markers set; building uniref30_2103 (tsv2exprofiledb + createindex)"

FAST_PREBUILT_DATABASES=0 UNIREF30DB=uniref30_2103 \
  taskset -c "$CORES" bash setup_databases.sh "$DB"
rc=$?
log "setup_databases.sh exit=$rc"
if [ "$rc" -eq 0 ] && [ -f "$DB/UNIREF30_READY" ]; then
  log "UNIREF30_READY present -- uniref30_2103 built"
  ls "$DB" | grep -c uniref30_2103 | xargs -I{} log "{} uniref30_2103 files"
  du -sh "$DB"
else
  log "incomplete -- re-run this script, it resumes"
fi
exit $rc
