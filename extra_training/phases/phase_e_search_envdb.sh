#!/bin/bash
# PHASE E -- deepen phase D's alignments against colabfold_envdb_202108.
#
# WHAT THIS DOES. Phase D searched uniref30, which holds sequences from named organisms. The
# environmental database holds sequences assembled from soil, seawater and gut samples, where nobody
# knows which organism they came from. It is larger and far more diverse, so it adds depth uniref30
# cannot -- and depth is what the model reads structure out of. Each query's a3m is appended to.
#
# WHAT MAKES THIS POSSIBLE WITHOUT uniref30 ON DISK. The env search is driven from `prof_res`, the
# uniref profile phase D kept, not from the alignments. Given prof_res and the env database the
# result is identical to an unstaged single run -- verified by code (search.py:157 defines one
# filter_param, :172 and :210 apply it per pass, :231 mergedbs concatenates by key without filtering
# the union) and by experiment (scripts/verify_minidb.sh: 6 of 6 alignments byte-identical, staged
# against unstaged). The resume needs patches/colabfold_search_staging.patch, gated below.
#
# THE FASTA MUST BE THE ONE PHASE D SEARCHED. search.py:517 rebuilds qdb from the query file on every
# run and ColabFold keys everything by position, so a fasta that has gained or lost a record since
# the uniref pass silently mis-assigns alignments to chains. Not hypothetical: phase C's
# antigen-type filter removed one RNA chain from query.fasta, at key 479 of 5,305, which would have
# shifted 4,825 keys. So phase E reads each batch's frozen <out-dir>/query.searched.fasta, verifies
# its checksum, and never reads msa_work/query.fasta.
#
# USAGE
#     phase_e_search_envdb.sh [out-dir ...]      default: all three phase D output directories
#
# ENVIRONMENT
#     FREE_DB=1      delete the env database once every batch succeeds. Default 1 -- phase F does not
#                    need it and 291 GB is most of the disk headroom. On failure it is always kept,
#                    so a retry costs no re-download.
#     KEEP_TARBALL=1 keep the 120 GB tarball after extracting. Default 0.
#     CORES=0-63     taskset range; --threads is derived from it, never hardcoded.
set -u
R=/home/ebrahim/tts_abag
DB=$R/tools/colabfold/db
W=$R/extra_training/msa_work
M=$R/tools/mmseqs/bin/mmseqs
CFDB=colabfold_envdb_202108
CORES="${CORES:-0-63}"
NTHREADS=$(taskset -c "$CORES" nproc 2>/dev/null || echo 64)
TAR=$DB/$CFDB.tar.gz
URL=https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz
LOG=$W/phase_e.log
export PATH="$R/tools/mmseqs/bin:$PATH"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../scripts/lib_fetch.sh"

# Every removal below writes ${VAR:?}: if the variable were ever unset or empty the shell stops with
# an error instead of running rm against / -- these paths are 291 GB of database and 100 GB tarballs.
BATCHES=("$@")
[ ${#BATCHES[@]} -gt 0 ] || BATCHES=("$W/msa_initial_idx" "$W/msa_short_idx" "$W/msa_bench_short_idx")

say "=== phase E: env pass for ${#BATCHES[@]} batch(es), $NTHREADS threads on cores $CORES ==="

# ---------------------------------------------------------------- 0. the patch, and every input
SEARCHPY=$R/envs/colabfold/lib/python3.12/site-packages/colabfold/mmseqs/search.py
$R/envs/colabfold/bin/python -c 'import sys, colabfold.mmseqs.search as m; sys.exit(0 if all(hasattr(m,f) for f in ("_emit_taxid","_keep_intermediates")) else 1)' \
    || die "the staging patch is not applied to $SEARCHPY -- phase E cannot resume without uniref30"
NMARK=$(grep -c 'PATCHED (tts_abag)' "$SEARCHPY" 2>/dev/null || true)
[ "${NMARK:-0}" -ge 4 ] || die "only ${NMARK:-0} PATCHED markers in search.py, expected at least 4"
say "staging patch applied"

# EVERY BATCH CHECKED BEFORE ANY WORK, so a missing input is not discovered after 90 minutes of
# downloading and extracting.
for OUT in "${BATCHES[@]}"; do
    [ -d "$OUT" ] || die "no such batch directory: $OUT"
    QFA=$OUT/query.searched.fasta
    [ -s "$QFA" ] || die "$QFA missing -- phase D freezes the fasta it searched and phase E needs it"
    ( cd "$OUT" && sha256sum -c query.searched.fasta.sha256 >/dev/null 2>&1 ) \
        || die "$QFA does not match its recorded checksum. ColabFold keys by position, so a changed
     fasta mis-assigns alignments to chains. Do not proceed."
    NQ=$(grep -c '^>' "$QFA")
    for f in uniref.a3m prof_res; do
        [ -s "$OUT/$f" ] || die "$OUT/$f missing -- the env search is driven from prof_res and resumes
     from uniref.a3m, so phase D's intermediates must still be present"
        NE=$(wc -l < "$OUT/$f.index")
        [ "$NE" -eq "$NQ" ] || die "$OUT/$f holds $NE entries for $NQ queries -- incomplete"
    done
    say "  $(basename "$OUT"): $NQ queries, checksum ok, uniref.a3m and prof_res complete"
done

# ---------------------------------------------------------------- 1. the env database
if [ ! -s "$DB/${CFDB}_db_seq" ]; then
    [ -s "$DB/uniref30_2302_db_seq" ] && die "uniref30 is still on disk; the two databases do not both
     fit in the quota. Free uniref30 first (phase D's FREE_DB=1 does it)."
    say "env database absent; fetching"
    fetch_tarball "$URL" "$TAR" || die "env tarball incomplete"
    # THE PREBUILT DATABASE. setup_databases.sh defaults FAST_PREBUILT_DATABASES=1, which extracts
    # ${CFDB}.tar.gz straight to MMseqs format and never calls tsv2exprofiledb. The URL above is the
    # .db.tar.gz for that reason: the plain .tar.gz is the tsv form, and extracting it once produced
    # 265 GB that converted to nothing and still exited 0.
    #
    # POSITIVE MARKER CONTROL, and the markers we create are removed again. A missing READY marker
    # makes setup_databases.sh build that database too; a marker left behind makes a later bare call
    # skip its own downloads and report success.
    CREATED=""
    for m in UNIREF30_READY PDB_READY PDB100_READY PDB_MMCIF_READY; do
        if [ ! -e "$DB/$m" ]; then touch "$DB/$m"; CREATED="$CREATED $m"; fi
    done
    rm -f "${DB:?}/COLABDB_READY"
    cd "$R/tools/colabfold" || die "no tools/colabfold"
    CFDB=$CFDB MMSEQS_NO_INDEX=1 taskset -c "$CORES" \
        bash setup_databases.sh "$DB" >> "$W/phase_e_setup.log" 2>&1
    say "setup_databases.sh exit $? (not trusted; checking files)"
    # THE GATE IS THE DATA FILES, never the exit code or the marker. These are exactly the paths
    # colabfold_search opens for the env pass: it checks {metagenomic_db}.dbtype, then reads _seq
    # (dbSuffix1) and _aln (dbSuffix2) when no index is present.
    for f in _db.dbtype _db_seq _db_aln _db_h; do
        [ -s "$DB/${CFDB}$f" ] || die "missing or empty $DB/${CFDB}$f"
    done
    [ -s "$DB/uniref30_2302_db_seq" ] && die "setup_databases.sh also extracted uniref30 -- the two
     databases do not both fit"
    touch "$DB/COLABDB_READY"
    for m in $CREATED; do rm -f "${DB:?}/${m:?}"; done
    [ -n "$CREATED" ] && say "  removed the markers this run created:$CREATED"
    [ "${KEEP_TARBALL:-0}" = "1" ] || rm -f "${TAR:?}"
    rm -f "${DB:?}"/${CFDB}_*.tsv "${DB:?}"/${CFDB}.tsv
fi
say "env database ready: $(du -sh --apparent-size "$DB/${CFDB}_db_seq" | cut -f1) of members"

# ---------------------------------------------------------------- 2. the env pass, per batch
# GATED ON GROWTH, NOT ON A LOG LINE. What matters is that env sequences were appended. Total bytes
# before and after proves exactly that, and catches the failure an earlier string match was there
# for: had the pass tried to re-search the deleted uniref30 it would have died and added nothing.
# Matching colabfold's own wording would break the moment that message is reworded, on a run that
# worked perfectly, so the log is only noted.
a3m_bytes() { find "$1" -maxdepth 1 -name '*.a3m' ! -name 'uniref.a3m' -printf '%s\n' 2>/dev/null \
                | awk '{s+=$1} END{print s+0}'; }

FAILED=0
for OUT in "${BATCHES[@]}"; do
    LBL=$(basename "$OUT")
    QFA=$OUT/query.searched.fasta
    NQ=$(grep -c '^>' "$QFA")
    BEFORE=$(a3m_bytes "$OUT")
    say "--- $LBL: $NQ queries, alignments currently $(numfmt --to=iec "$BEFORE")"
    COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c "$CORES" \
        $R/envs/colabfold/bin/colabfold_search --mmseqs $M --threads "$NTHREADS" \
        --use-env 1 --use-templates 0 --db-load-mode 0 \
        "$QFA" "$DB" "$OUT" >> "$W/phase_e_search_$LBL.log" 2>&1
    say "  colabfold_search exit $? (not trusted; the data decides)"
    grep -q "Skipping" "$W/phase_e_search_$LBL.log" \
        && say "  note: the log mentions skipping a search, consistent with resuming from uniref.a3m"

    AFTER=$(a3m_bytes "$OUT")
    # BY NAME, as in phase D: a count is satisfied by any directory holding enough files.
    MISSING=$(awk '/^>/{sub(/^>/,""); print}' "$QFA" | while read -r n; do
                  [ -s "$OUT/$n.a3m" ] || echo "$n"
              done)
    NMISS=$(printf '%s' "$MISSING" | grep -c . || true)
    say "  alignments $(numfmt --to=iec "$BEFORE") -> $(numfmt --to=iec "$AFTER"); $NMISS of $NQ without one"
    if [ "${NMISS:-0}" -gt 0 ]; then
        say "  first few missing: $(printf '%s' "$MISSING" | head -5 | tr '\n' ' ')"
        say "  ESCALATE $LBL: the env pass did not cover the batch"; FAILED=$((FAILED+1)); continue
    fi
    # The intermediate colabfold keeps for the env half, named for the databases it merges. Its
    # presence distinguishes "the env search ran" from "the uniref alignments were rewritten".
    ENVINT=$(ls "$OUT"/bfd.mgnify30.metaeuk30.smag30.a3m 2>/dev/null | wc -l)
    if [ "$AFTER" -le "$BEFORE" ]; then
        say "  ESCALATE $LBL: no sequences added ($BEFORE -> $AFTER bytes); env intermediate: $ENVINT"
        FAILED=$((FAILED+1)); continue
    fi
    say "  $LBL OK: +$(numfmt --to=iec $((AFTER - BEFORE))), env intermediate present: $ENVINT"
done

[ "$FAILED" -eq 0 ] || die "$FAILED of ${#BATCHES[@]} batches failed. The env database is LEFT ON DISK
     so they can be retried without a 120 GB re-download."

# ---------------------------------------------------------------- 3. free the database
if [ "${FREE_DB:-1}" = "1" ]; then
    say "freeing the env database; phase F does not need it"
    rm -rf "${DB:?}"/${CFDB}_db* "${DB:?}"/COLABDB_READY
fi
say "=== phase E OK, all ${#BATCHES[@]} batches; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}') ==="
