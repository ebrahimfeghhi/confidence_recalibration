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
PH=$R/extra_training/phases
W=$R/extra_training/msa_work
DB=$R/tools/colabfold/db
M=$R/tools/mmseqs/bin/mmseqs
LOG=$W/phase1b_short.log
TAR=$DB/uniref30_2302.tar.gz
URL=https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz
export PATH="$R/tools/mmseqs/bin:$PATH"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

# Completeness of a 100+ GB download is judged by the server's Content-Length and by whether the gzip
# stream reads end to end -- never by a constant in this file. See lib_fetch.sh.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib_fetch.sh"

# ---------------------------------------------------------------- 1. the query set
if [ ! -s "$W/query_short.fasta" ]; then
    say "building the short-antigen query set"
    # ONE call: phase C does the set difference itself under --short-vs, so the fasta it writes and
    # the fasta it subtracts can never come from different runs.
    #
    # --which BOTH, not train. Batch 1 used --min-len 20 for both sets, so the short chains it skipped
    # exist on both sides. Running batch 2 over training chains only meant a benchmark complex with a
    # 5-19 residue antigen was treated differently from its training counterparts -- the same
    # train/test asymmetry the shared uniref30_2302 rebuild exists to remove.
    taskset -c 0-15 $PY $PH/phase_c_extract_msa_queries.py --min-len 5 --which both \
        --out-dir $W/msa_queries_min5 --short-vs $W/query.fasta \
        --short-out $W/query_short.fasta >> "$LOG" 2>&1 \
        || die "phase C batch-2 failed"
fi
NQ=$(grep -c '^>' "$W/query_short.fasta")
[ "$NQ" -ge 100 ] || die "only $NQ short queries (expected ~235)"
say "short queries: $NQ"

# ---------------------------------------------------------------- 2. uniref30 database
if ! [ -s "$DB/uniref30_2302_db_seq" ]; then
    say "fetching uniref30_2302.db.tar.gz"
    fetch_tarball "$URL" "$TAR" || die "uniref30 tarball incomplete"
    say "tarball complete; building (index built separately below)"
    # POSITIVE CONTROL OVER WHAT setup_databases.sh BUILDS, not an assumption about it. It decides
    # from the READY markers in $DB: removing UNIREF30_READY makes it rebuild uniref30, and the
    # ABSENCE of COLABDB_READY makes it extract the 291 GB env database too, in the very same call.
    # That is what happened on 2026-10-11: an earlier phase2_envdb.sh run had deleted COLABDB_READY
    # before a build that then failed, this script's comment asserted the marker "stays", and one call
    # put both databases on disk at once -- 760 GB, quota at 91%, and the quota guard killed the index
    # build. Assuming a marker is present is not the same as ensuring it is.
    ENVDB_BEFORE=0
    [ -s "$DB/colabfold_envdb_202108_db_seq" ] && ENVDB_BEFORE=1
    for m in COLABDB_READY PDB100_READY PDB_READY PDB_MMCIF_READY; do
        [ -e "$DB/$m" ] || { touch "$DB/$m"; say "  created $m so setup_databases.sh leaves it alone"; }
    done
    rm -f "$DB/UNIREF30_READY"
    cd "$R/tools/colabfold" || die "no tools/colabfold"
    MMSEQS_NO_INDEX=1 taskset -c 0-63 bash setup_databases.sh "$DB" >> $W/phase1b_setup.log 2>&1
    say "setup_databases.sh exit $? (not trusted; checking files)"
    for f in _db.dbtype _db_seq _db_aln; do
        [ -s "$DB/uniref30_2302$f" ] || die "missing or empty $DB/uniref30_2302$f"
    done
    # Did it build anything we did not ask for? Checked by data file, not by marker.
    if [ "$ENVDB_BEFORE" -eq 0 ] && [ -s "$DB/colabfold_envdb_202108_db_seq" ]; then
        die "setup_databases.sh also extracted the env database; uniref30 and envdb do not both fit"
    fi
    rm -f "$TAR"
    say "uniref30 database ready: $(du -sh $DB/uniref30_2302_db_seq | cut -f1) of members"
fi

# ---------------------------------------------------------------- 3. index, for taxonomy
# convertalis resolves taxids only through .idx_mapping and .idx_taxonomy.
#
# THESE ARE SYMLINKS, AND THAT IS CORRECT. setup_databases.sh:168,171 creates them with `ln -sf`
# pointing at _db_mapping and _db_taxonomy. An earlier version of this block assumed they were
# leftovers from a hand-rolled attempt, deleted them before the build, and then required them to be
# NEWER than the build -- which createindex never satisfies, because createindex does not create
# them. That gate failed a complete 301 GB index. The links are recreated here rather than relied
# upon, because we run createindex ourselves with MMSEQS_NO_INDEX=1 and so skip the part of
# setup_databases.sh that would have made them.
if [ ! -s "$DB/uniref30_2302_db.idx" ]; then
    say "building the uniref30 index (needed for convertalis to resolve taxids)"
    cd "$DB" || die "no db dir"
    rm -rf tmp1
    taskset -c 0-63 $M createindex uniref30_2302_db tmp1 --remove-tmp-files 1 --threads 64 \
        >> $W/phase1b_index.log 2>&1
    say "createindex exit $? (not trusted; checking files)"
    [ -s "$DB/uniref30_2302_db.idx" ] || die "no uniref30_2302_db.idx after createindex"
    say "index ready: $(du -sh $DB/uniref30_2302_db.idx | cut -f1)"
fi
# Link, then gate on what the link RESOLVES to: -s follows the symlink, so an empty or dangling link
# fails here. That is the property that matters -- convertalis reads through it either way.
for l in mapping taxonomy; do
    [ -s "$DB/uniref30_2302_db_$l" ] || die "missing $DB/uniref30_2302_db_$l -- taxids cannot resolve"
    ln -sf "uniref30_2302_db_$l" "$DB/uniref30_2302_db.idx_$l"
    [ -s "$DB/uniref30_2302_db.idx_$l" ] || die "$DB/uniref30_2302_db.idx_$l does not resolve"
    say "  idx_$l -> $(readlink "$DB/uniref30_2302_db.idx_$l") ($(du -shL "$DB/uniref30_2302_db.idx_$l" | cut -f1))"
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
