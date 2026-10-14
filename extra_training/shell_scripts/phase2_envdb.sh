#!/bin/bash
# Download and build colabfold_envdb_202108, the second (environmental) MSA database.
#
# WHY A SEPARATE SCRIPT, mirroring phase1_uniref.sh: setup_databases.sh gates its download block on
# DOWNLOADS_READY, which phase1 pre-touched so that only uniref30 was built. Removing that marker
# would re-download the 180 GB uniref30 tarball we no longer need. So the tarball is fetched here
# (resumable) and only COLABDB_READY is removed, which makes setup_databases.sh build the envdb and
# nothing else.
#
# TWO BUGS THIS FIXES, both from the 2026-10-10 overnight chain:
#   1. setup_databases.sh:35 requires `mmseqs` ON PATH. Ours lives in tools/mmseqs/bin, so the run
#      died instantly with "mmseqs command not found" -- before downloading anything.
#   2. The old gate was `mmseqs dbtype <db>`, which RETURNS 0 ON A MISSING DATABASE (verified).
#      The gate here checks the data file exists and is non-empty instead.
#
# NO INDEX (MMSEQS_NO_INDEX=1). The uniref30 index was built because taxonomy lookup needs it:
# convertalis resolves taxids through uniref30_2302_db.idx_taxonomy. Pairing uses UniRef hits only --
# Protenix's step4 splits on the ">UniRef100" header prefix -- so no env hit is ever paired and no env
# taxonomy is read. An envdb index would cost ~300 GB and hours for nothing, and it would put the
# peak near the quota ceiling again.
#
# DISK: 118 GB tarball + ~320 GB tsv + ~160 GB db, peak under 600 GB, against 971 GB free.
#
# MODE (argument 1): "download" fetches the tarball only, "build" extracts it, "all" (the default)
# does both. They are separable because uniref30 must come back for the short-peptide pass and the two
# databases do not fit on disk together, so the order is: fetch this tarball, run the uniref30 pass,
# free uniref30, only then extract this one.
set -u
MODE="${1:-all}"
R=/home/ebrahim/tts_abag
DB=$R/tools/colabfold/db
# THE PREBUILT DATABASE, not the .tsv one. setup_databases.sh defaults FAST_PREBUILT_DATABASES=1
# (line 20), and in that mode it extracts ${CFDB}.tar.gz straight to MMseqs db format and never calls
# tsv2exprofiledb. phase1_uniref.sh:34 did the same for uniref30: it fetched uniref30_2302.db.tar.gz
# and saved it under the plain name the script looks for. Verified by listing the archive head: its
# first member is "colabfold_envdb_202108_db".
TAR=$DB/colabfold_envdb_202108.tar.gz
URL=https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz
LOG=$R/extra_training/building_msa/work/phase2_envdb.log
export PATH="$R/tools/mmseqs/bin:$PATH"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

# Completeness of a 100+ GB download is judged by the server's Content-Length and by whether the gzip
# stream reads end to end -- never by a constant in this file. See lib_fetch.sh.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib_fetch.sh"

command -v mmseqs >/dev/null || die "mmseqs still not on PATH"

# ---------------------------------------------------------------- download (resumable)
if [ "$MODE" = "build" ]; then
    say "mode=build: skipping the download"
else
    say "fetching the envdb tarball"
    fetch_tarball "$URL" "$TAR" || die "envdb tarball incomplete"
fi
if [ "$MODE" = "download" ]; then
    say "mode=download: stopping before the build, as asked"
    exit 0
fi

# ---------------------------------------------------------------- build
rm -f "$DB/COLABDB_READY"
say "building envdb (tsv2exprofiledb, index-free)"
cd "$R/tools/colabfold" || die "no tools/colabfold"
CFDB=colabfold_envdb_202108 MMSEQS_NO_INDEX=1 taskset -c 0-63 \
    bash setup_databases.sh "$DB" >> "$R/extra_training/building_msa/work/phase2_setup.log" 2>&1
rc=$?
say "setup_databases.sh exit $rc (not trusted; checking the files)"

# THE GATE: the data files themselves, never an exit code -- setup_databases.sh touches its READY
# marker and returns 0 even when it converted nothing.
# These are exactly the three paths colabfold_search opens for the env pass: search.py checks
# {metagenomic_db}.dbtype, then reads {metagenomic_db}_seq (dbSuffix1) and _aln (dbSuffix2) when no
# index is present.
for f in _db.dbtype _db_seq _db_aln _db_h; do
    p="$DB/colabfold_envdb_202108$f"
    [ -s "$p" ] || die "missing or empty $p"
    say "  ok $(basename "$p") $(du -h "$p" | cut -f1)"
done
# LEAVE THE MARKER BEHIND. setup_databases.sh decides what to build from these markers, so without
# COLABDB_READY the NEXT call -- phase1b_short.sh's, for uniref30 -- extracts this 291 GB database all
# over again alongside it. That is exactly how both databases ended up on disk at once on 2026-10-11.
touch "$DB/COLABDB_READY"
rm -f "$TAR" "$DB"/colabfold_envdb_202108_*.tsv "$DB"/colabfold_envdb_202108.tsv
say "envdb READY; tarball and tsv removed; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2}')"
