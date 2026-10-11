#!/bin/bash
# PHASE D -- search every sequence phase C listed against uniref30_2302.
#
# WHAT THIS PRODUCES, per query sequence, in <out-dir>:
#     <name>.a3m        the alignment: that sequence's evolutionary relatives, one per line
#     uniref.a3m        the raw search output, KEPT -- see "the disk trick" below
#     uniref_tax.m8     the NCBI taxonomy id of every hit, which phase F needs to pair chains
#
# These are not yet MSAs Protenix can read. Phase E deepens them against the environmental database
# and phase F writes the taxids into the headers and splits each alignment into pairing.a3m and
# non_pairing.a3m. Phase D's job is the uniref30 half.
#
# THE DISK TRICK. uniref30 (450 GB extracted plus a 301 GB index) and colabfold_envdb (291 GB) cannot
# both sit inside the 1,536 GB quota. So: search uniref30, KEEP its intermediates
# (COLABFOLD_KEEP_INTERMEDIATES=1), delete the database, and let phase E resume from uniref.a3m
# without uniref30 on disk. That resume is only possible because of
# patches/colabfold_search_staging.patch; without it colabfold_search refuses to start when uniref30
# is missing. Verify the patch is applied before trusting any of this:
#     envs/colabfold/bin/python -c "import colabfold.mmseqs.search as m; print(m._emit_taxid())"
#
# WHY THE 301 GB INDEX IS BUILT, since setup_databases.sh calls it "not useful for colabfold_search
# in most cases" and it buys nothing on search speed at this batch size. It is not for speed. It is
# the only route to taxonomy: convertalis resolves taxids through uniref30_2302_db.idx_mapping and
# .idx_taxonomy, and an index-free search instead targets _db_seq, whose 350,950,053 member keys do
# not live in the same key space as the mapping table's 36,293,491 representatives. No index means no
# taxids, which means no paired MSA, which means phase F has nothing to split on. Earlier comments in
# this repo called the index unnecessary; that was about speed and it read as though it were about
# correctness.
#
# USAGE
#     phase_d_search_uniref.sh <query.fasta> <out-dir> [tag]
#
# ENVIRONMENT
#     FREE_DB=1      delete uniref30 and its index after a successful search. Default 0, so several
#                    batches can run back to back without re-downloading 100 GB each time.
#     ALLOW_THIN=1   accept a batch that retrieves little or nothing. Correct for short peptides --
#                    4 of FoldBench's 8 chains under 20 aa come back query-only -- and wrong for a
#                    full batch, where an empty result means the search is broken. Default 0.
#     MIN_Q=N        refuse to run if the query fasta holds fewer than N records. Default 1.
#     CORES=0-63     taskset range. 64 cores is the stated maximum for this project.
#
# BATCHES RUN SO FAR, all with these parameters:
#     query.fasta                  -> msa_initial_idx        5,305 chains of 20+ residues
#     query_short.fasta            -> msa_short_idx            235 antigen chains of 5-19 residues
#     query_short_TODO_bench.fasta -> msa_bench_short_idx        11 benchmark chains of 5-19 residues
# The first two were run by scripts/phase1_uniref.sh + phase1_index.sh + phase1_search_idx.sh and
# scripts/phase1b_short.sh, which this script replaces; those are preserved in git at c1cbe087^.
#
# SEPARATE BATCHES, NOT ONE. ColabFold keys its query database by position, so appending queries
# shifts every key and the kept uniref.a3m could no longer be resumed from. Each batch therefore gets
# its own output directory, its own uniref.a3m and its own uniref_tax.m8.
set -u
R=/home/ebrahim/tts_abag
DB=$R/tools/colabfold/db
W=$R/extra_training/msa_work
M=$R/tools/mmseqs/bin/mmseqs
CORES="${CORES:-0-63}"
export PATH="$R/tools/mmseqs/bin:$PATH"

QFA="${1:-}"
OUT="${2:-}"
TAG="${3:-$(basename "${OUT:-batch}")}"
LOG=$W/phase_d_$TAG.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../scripts/lib_fetch.sh"      # fetch_tarball: completeness from the server, not a constant

[ -n "$QFA" ] && [ -n "$OUT" ] || die "usage: phase_d_search_uniref.sh <query.fasta> <out-dir> [tag]"
[ -s "$QFA" ] || die "no query fasta at $QFA"
NQ=$(grep -c '^>' "$QFA")
[ "$NQ" -ge "${MIN_Q:-1}" ] || die "only $NQ queries in $QFA (expected >= ${MIN_Q:-1})"
mkdir -p "$OUT"
say "=== phase D: $NQ queries from $(basename "$QFA") -> $OUT ==="

# ---------------------------------------------------------------- 1. the database
# GATED ON DATA FILES, NEVER ON AN EXIT CODE OR A MARKER. setup_databases.sh touches UNIREF30_READY
# before the steps that can fail, and `mmseqs dbtype <missing-db>` returns 0 -- verified -- so the
# check that used to live here passed whether or not the database existed.
if [ ! -s "$DB/uniref30_2302_db_seq" ]; then
    say "uniref30 absent; fetching"
    fetch_tarball https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz \
        "$DB/uniref30_2302.tar.gz" || die "uniref30 tarball incomplete"
    # The taxonomy tarball is small and setup_databases.sh needs it for _db_taxonomy.
    fetch_tarball https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz \
        "$DB/uniref30_2302_newtaxonomy.tar.gz" || die "taxonomy tarball incomplete"

    # POSITIVE CONTROL OVER WHAT GETS BUILT. setup_databases.sh decides from the READY markers:
    # removing UNIREF30_READY makes it rebuild uniref30, and the ABSENCE of COLABDB_READY makes it
    # extract the 291 GB env database in the same call. That happened on 2026-10-11 -- both databases
    # on disk at once, 760 GB, quota at 91%, and the quota guard killed the index build. Create the
    # markers for everything we want left alone instead of assuming they are there.
    ENVDB_BEFORE=0
    [ -s "$DB/colabfold_envdb_202108_db_seq" ] && ENVDB_BEFORE=1
    for m in COLABDB_READY PDB_READY PDB100_READY PDB_MMCIF_READY; do
        [ -e "$DB/$m" ] || { touch "$DB/$m"; say "  created $m so setup_databases.sh skips it"; }
    done
    rm -f "$DB/UNIREF30_READY"
    cd "$R/tools/colabfold" || die "no tools/colabfold"
    # MMSEQS_NO_INDEX=1: we build the index ourselves below, so setup_databases.sh must not.
    MMSEQS_NO_INDEX=1 UNIREF30DB=uniref30_2302 taskset -c "$CORES" \
        bash setup_databases.sh "$DB" >> "$W/phase_d_setup_$TAG.log" 2>&1
    say "setup_databases.sh exit $? (not trusted; checking files)"
    for f in _db.dbtype _db_seq _db_aln _db_mapping _db_taxonomy; do
        [ -s "$DB/uniref30_2302$f" ] || die "missing or empty $DB/uniref30_2302$f"
    done
    if [ "$ENVDB_BEFORE" -eq 0 ] && [ -s "$DB/colabfold_envdb_202108_db_seq" ]; then
        die "setup_databases.sh also extracted the env database; the two do not both fit"
    fi
    rm -f "$DB/uniref30_2302.tar.gz"
    say "uniref30 ready: $(du -sh --apparent-size "$DB/uniref30_2302_db_seq" | cut -f1) of members"
fi

# ---------------------------------------------------------------- 2. the index, for taxonomy only
if [ ! -s "$DB/uniref30_2302_db.idx" ]; then
    say "building the index (taxonomy, not speed -- see the header)"
    cd "$DB" || die "no db dir"
    rm -rf tmp1
    taskset -c "$CORES" $M createindex uniref30_2302_db tmp1 --remove-tmp-files 1 --threads 64 \
        >> "$W/phase_d_index_$TAG.log" 2>&1
    say "createindex exit $? (not trusted; checking files)"
    [ -s "$DB/uniref30_2302_db.idx" ] || die "no uniref30_2302_db.idx after createindex"
    say "index ready: $(du -sh --apparent-size "$DB/uniref30_2302_db.idx" | cut -f1)"
fi
# .idx_mapping and .idx_taxonomy are SYMLINKS, and that is correct: setup_databases.sh:168,171 makes
# them with `ln -sf` and createindex does not make them at all. An earlier gate here mistook them for
# hand-made leftovers and demanded they postdate the index build, which failed a finished 301 GB
# index twice. We create them (we skip the part of setup_databases.sh that would have) and gate on
# what they RESOLVE to -- `-s` follows a symlink, so a dangling one fails here.
for l in mapping taxonomy; do
    [ -s "$DB/uniref30_2302_db_$l" ] || die "missing $DB/uniref30_2302_db_$l -- taxids cannot resolve"
    ln -sf "uniref30_2302_db_$l" "$DB/uniref30_2302_db.idx_$l"
    [ -s "$DB/uniref30_2302_db.idx_$l" ] || die "$DB/uniref30_2302_db.idx_$l does not resolve"
done
say "taxonomy tables resolve"

# ---------------------------------------------------------------- 3. the search
# --use-env 0: the environmental database is phase E, and it is not on disk now.
# --db-load-mode 0: no index-into-memory; the index exists for taxonomy, not for the search.
# RESUMABLE: colabfold_search skips a query whose <name>.a3m already exists, so re-running continues.
say "searching (resumable: existing .a3m are skipped)"
COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c "$CORES" \
    $R/envs/colabfold/bin/colabfold_search --mmseqs $M --threads 64 \
    --use-env 0 --use-templates 0 --db-load-mode 0 \
    "$QFA" "$DB" "$OUT" >> "$W/phase_d_search_$TAG.log" 2>&1
say "colabfold_search exit $? (not trusted; checking output)"

# ---------------------------------------------------------------- 4. the gates
NA=$(ls "$OUT"/*.a3m 2>/dev/null | grep -vc '/uniref.a3m$')
say "alignments written: $NA for $NQ queries"
[ "$NA" -ge "$NQ" ] || die "$NA alignments for $NQ queries -- the search did not cover the batch"

TAXF=$OUT/uniref_tax.m8
NTAX=0
[ -s "$TAXF" ] && NTAX=$(awk -F'\t' 'NR<=2000 && $3+0>0 {n++} END{print n+0}' "$TAXF")
if [ "$NTAX" -le "${MIN_TAX:-10}" ]; then
    if [ "${ALLOW_THIN:-0}" = "1" ]; then
        say "NOTE only $NTAX usable taxids. For short peptides that is a real outcome rather than a"
        say "     fault -- they retrieved little or nothing -- so they will pair on what depth exists."
    else
        die "uniref_tax.m8 has $NTAX usable taxids in its first 2000 rows -- pairing would be impossible"
    fi
fi
if [ ! -s "$OUT/uniref.a3m" ]; then
    [ "${ALLOW_THIN:-0}" = "1" ] && say "NOTE no uniref.a3m kept; phase E cannot resume for this batch" \
        || die "uniref.a3m not kept -- phase E could not resume without uniref30 on disk"
fi
say "depth of the first few:"
for f in $(ls "$OUT"/*.a3m 2>/dev/null | grep -v '/uniref.a3m$' | head -5); do
    say "   $(basename "$f") $(grep -c '^>' "$f") sequences"
done
say "=== phase D OK for $TAG: $NA alignments, taxids $([ "$NTAX" -gt 0 ] && echo present || echo thin) ==="

# ---------------------------------------------------------------- 5. free the database, on request
if [ "${FREE_DB:-0}" = "1" ]; then
    say "freeing uniref30 and its index so the env database fits"
    rm -rf "$DB"/uniref30_2302_db* "$DB"/UNIREF30_READY
    say "freed; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
else
    say "uniref30 left on disk (FREE_DB=1 to remove it); quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
fi
