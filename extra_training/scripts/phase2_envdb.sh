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
set -u
R=/home/ebrahim/tts_abag
DB=$R/tools/colabfold/db
TAR=$DB/colabfold_envdb_202108.tar.gz
URL=https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.tar.gz
WANT=117965643010
LOG=$R/extra_training/msa_work/phase2_envdb.log
export PATH="$R/tools/mmseqs/bin:$PATH"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

command -v mmseqs >/dev/null || die "mmseqs still not on PATH"

# ---------------------------------------------------------------- download (resumable)
if [ ! -s "$TAR" ] || [ "$(stat -c%s "$TAR")" -ne "$WANT" ]; then
    say "downloading envdb tarball ($(numfmt --to=iec $WANT))"
    for try in 1 2 3 4 5; do
        curl -L -C - --retry 5 --retry-delay 10 -o "$TAR" "$URL" >> "$LOG" 2>&1
        got=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
        [ "$got" -eq "$WANT" ] && break
        say "  attempt $try stopped at $(numfmt --to=iec "$got"); resuming"
    done
fi
got=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
[ "$got" -eq "$WANT" ] || die "tarball is $got bytes, expected $WANT"
say "tarball complete and the right size"

# ---------------------------------------------------------------- build
rm -f "$DB/COLABDB_READY"
say "building envdb (tsv2exprofiledb, index-free)"
cd "$R/tools/colabfold" || die "no tools/colabfold"
CFDB=colabfold_envdb_202108 MMSEQS_NO_INDEX=1 taskset -c 0-63 \
    bash setup_databases.sh "$DB" >> "$R/extra_training/msa_work/phase2_setup.log" 2>&1
rc=$?
say "setup_databases.sh exit $rc (not trusted; checking the files)"

# the gate: the data file itself, not an exit code
for f in _db _db_h _db_seq _db_aln; do
    p="$DB/colabfold_envdb_202108$f"
    [ -s "$p" ] || die "missing or empty $p"
    say "  ok $(basename "$p") $(du -h "$p" | cut -f1)"
done
rm -f "$TAR" "$DB"/colabfold_envdb_202108_*.tsv "$DB"/colabfold_envdb_202108.tsv
say "envdb READY; tarball and tsv removed; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2}')"
