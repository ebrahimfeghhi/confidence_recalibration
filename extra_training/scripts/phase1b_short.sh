#!/bin/bash
# The second uniref30 pass: MSAs for the 235 antigen chains of 5-19 residues that the first pass's
# 20-residue minimum left out.
#
# WHY IT EXISTS. FoldBench, one of the two benchmarks we evaluate on, contains protein chains under
# 20 aa and searches them normally: of its 8 such chains, 4 come back query-only but 4 carry real
# depth (7, 4, 431 and 3,320 sequences). Dropping short antigens would have cost 103 of 557 clusters
# and 13 of the 50 validation clusters, and faking a query-only MSA would misrepresent half of them.
#
# WHY A SEPARATE BATCH. ColabFold keys its query database by position, so appending queries to the
# first fasta would shift every key and the kept uniref.a3m could no longer be resumed from. This
# batch gets its own output directory, its own uniref.a3m and its own uniref_tax.m8; step3 then runs
# once per batch into a shared output directory, which also avoids concatenating the 82 GB m8.
#
# WHY uniref30 COMES BACK. Their pairing hits are UniRef100 entries, and a taxid-tagged header is what
# makes them pairable -- convertalis only finds taxonomy through uniref30_2302_db.idx_taxonomy, so the
# index is required, exactly as in the first pass. uniref30 and the env database cannot both be on
# disk at this quota, so this runs BEFORE the env database is built and frees itself when done.
#
# DISK: tarball 106 GB -> database 232 GB -> index 302 GB, peak about 1,170 GB of the 1,536 GB quota
# with the env tarball already present. All of it is removed at the end.
set -u
R=/home/ebrahim/tts_abag
P=/home/ebrahim/Protenix
PY=$P/.venv/bin/python
C=$R/diffusion_commit_clean_v2
S=$R/extra_training/scripts
W=$R/extra_training/msa_work
DB=$R/tools/colabfold/db
M=$R/tools/mmseqs/bin/mmseqs
LOG=$W/phase1b_short.log
TAR=$DB/uniref30_2302.tar.gz
URL=https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz
WANT=106688276925
export PATH="$R/tools/mmseqs/bin:$PATH"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

# ---------------------------------------------------------------- 1. the query set
if [ ! -s "$W/query_short.fasta" ]; then
    say "building the short-antigen query set"
    taskset -c 0-15 $PY $C/extract_msa_queries.py --min-len 5 --which train \
        --out-dir $W/msa_queries_min5 >> "$LOG" 2>&1 || die "extract_msa_queries failed"
    $PY $S/extract_short_queries.py --all $W/msa_queries_min5/train.fasta \
        --already $W/query.fasta --out $W/query_short.fasta >> "$LOG" 2>&1 \
        || die "extract_short_queries failed"
fi
NQ=$(grep -c '^>' "$W/query_short.fasta")
[ "$NQ" -ge 100 ] || die "only $NQ short queries (expected ~235)"
say "short queries: $NQ"

# ---------------------------------------------------------------- 2. uniref30 database
if ! [ -s "$DB/uniref30_2302_db_seq" ]; then
    if [ ! -s "$TAR" ] || [ "$(stat -c%s "$TAR")" -ne "$WANT" ]; then
        say "downloading uniref30_2302.db.tar.gz ($(numfmt --to=iec $WANT))"
        for try in 1 2 3 4 5; do
            curl -L -C - --retry 5 --retry-delay 10 -o "$TAR" "$URL" >> "$LOG" 2>&1
            got=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
            [ "$got" -eq "$WANT" ] && break
            say "  attempt $try stopped at $(numfmt --to=iec "$got"); resuming"
        done
    fi
    [ "$(stat -c%s "$TAR" 2>/dev/null || echo 0)" -eq "$WANT" ] || die "uniref30 tarball incomplete"
    say "tarball complete; building (index built separately below)"
    # UNIREF30_READY must go so the uniref30 branch runs; COLABDB_READY stays so the env database is
    # not touched here. The gate below is the data files, never the exit code.
    rm -f "$DB/UNIREF30_READY"
    cd "$R/tools/colabfold" || die "no tools/colabfold"
    MMSEQS_NO_INDEX=1 taskset -c 0-63 bash setup_databases.sh "$DB" >> $W/phase1b_setup.log 2>&1
    say "setup_databases.sh exit $? (not trusted; checking files)"
    for f in _db.dbtype _db_seq _db_aln; do
        [ -s "$DB/uniref30_2302$f" ] || die "missing or empty $DB/uniref30_2302$f"
    done
    rm -f "$TAR"
    say "uniref30 database ready: $(du -sh $DB/uniref30_2302_db_seq | cut -f1) of members"
fi

# ---------------------------------------------------------------- 3. index, for taxonomy
if [ ! -s "$DB/uniref30_2302_db.idx" ]; then
    say "building the uniref30 index (needed for convertalis to resolve taxids)"
    cd "$DB" || die "no db dir"
    taskset -c 0-63 $M createindex uniref30_2302_db tmp1 --remove-tmp-files 1 --threads 64 \
        >> $W/phase1b_index.log 2>&1
    say "createindex exit $? (not trusted; checking files)"
    [ -s "$DB/uniref30_2302_db.idx" ] || die "no uniref30_2302_db.idx after createindex"
    say "index ready: $(du -sh $DB/uniref30_2302_db.idx | cut -f1)"
fi
for l in idx_mapping idx_taxonomy; do
    [ -e "$DB/uniref30_2302_db.$l" ] || die "missing $DB/uniref30_2302_db.$l -- taxids cannot resolve"
done

# ---------------------------------------------------------------- 4. the search
say "searching $NQ short queries against uniref30 (--use-env 0, as the first pass did)"
COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c 0-63 \
    $R/envs/colabfold/bin/colabfold_search --mmseqs $M --threads 64 \
    --use-env 0 --use-templates 0 --db-load-mode 0 \
    "$W/query_short.fasta" "$DB" "$W/msa_short_idx" >> $W/phase1b_search.log 2>&1
say "colabfold_search exit $? (not trusted; checking output)"
NA=$(ls $W/msa_short_idx/*.a3m 2>/dev/null | grep -vc '/uniref.a3m$')
TAXF=$W/msa_short_idx/uniref_tax.m8
[ "$NA" -ge "$NQ" ] || die "$NA a3m for $NQ queries"
[ -s "$TAXF" ] || die "no uniref_tax.m8 -- these chains could not be paired"
NTAX=$(awk -F'\t' 'NR<=2000 && $3+0>0 {n++} END{print n+0}' "$TAXF")
[ "$NTAX" -gt 10 ] || die "uniref_tax.m8 has $NTAX usable taxids in its first 2000 rows"
[ -s "$W/msa_short_idx/uniref.a3m" ] || die "uniref.a3m not kept -- the env pass could not resume"
say "stage OK: $NA a3m, uniref_tax.m8 with real taxids, uniref.a3m kept for the env pass"
say "  MSA depth of the first few:"
for f in $(ls $W/msa_short_idx/*.a3m | grep -v '/uniref.a3m$' | head -5); do
    say "    $(basename $f) $(grep -c '^>' $f) sequences"
done

# ---------------------------------------------------------------- 5. free uniref30
say "removing uniref30 and its index so the env database fits"
rm -rf "$DB"/uniref30_2302_db* "$DB"/UNIREF30_READY
say "done; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
