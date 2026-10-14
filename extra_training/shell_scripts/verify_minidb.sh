#!/bin/bash
# Two verification experiments for phase D, on the 280 MB mini_swissprot database.
#
# DEBUG_MINI_DB=1 makes setup_databases.sh extract mini_swissprot TWICE -- once as uniref30_2302_db
# and once as colabfold_envdb_202108_db, each with its _taxonomy and _mapping tables. So both halves
# of the staged search exist in 280 MB, in a database directory of their own, and neither experiment
# can touch the real one.
#
# EXPERIMENT 1 -- IS THE 301 GB INDEX REQUIRED FOR TAXIDS?
#   Phase D builds it solely so convertalis can resolve taxids through <db>.idx_mapping and
#   .idx_taxonomy. But --db-load-mode 0 means the search itself never uses an index, and phase D
#   creates both of those sidecars by hand as symlinks to _db_mapping and _db_taxonomy. So: run with
#   the sidecars present and NO .idx, and look at uniref_tax.m8. Taxids resolving means the index
#   buys nothing and 301 GB per batch goes away.
#
# EXPERIMENT 2 -- IS THE STAGED SEARCH EQUAL TO AN UNSTAGED ONE?
#   Staged: search uniref, keep the intermediates, DELETE the uniref database, then run the env pass
#   resuming from uniref.a3m. Unstaged: one run with both databases present. Diff the final a3m.
#   Reading search.py says they must agree -- filtering is applied per pass at :172 and :210 with the
#   same filter_param from :157, and :231 mergedbs concatenates by key without filtering the union --
#   so this is confirmation of a code-level result, not the only evidence for it.
#
# USAGE: bash verify_minidb.sh [n_queries]   (default 5)
set -u
R=/home/ebrahim/tts_abag
W=$R/extra_training/building_msa/work
MDB=$W/minidb
M=$R/tools/mmseqs/bin/mmseqs
CORES="${CORES:-0-7}"
NTHREADS=$(taskset -c "$CORES" nproc)
NQ="${1:-5}"
LOG=$W/verify_minidb.log
export PATH="$R/tools/mmseqs/bin:$PATH"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

say "=== phase D verification on mini_swissprot, $NTHREADS threads on cores $CORES ==="

# ---------------------------------------------------------------- the mini databases
if [ ! -s "$MDB/uniref30_2302_db_seq" ] || [ ! -s "$MDB/colabfold_envdb_202108_db_seq" ]; then
    mkdir -p "$MDB"
    say "building both mini databases (280 MB download)"
    cd "$R/tools/colabfold" || die "no tools/colabfold"
    # FETCH THE MINI TARBALL OURSELVES, THEN TOUCH DOWNLOADS_READY. The per-database READY markers
    # only skip a database's BUILD; the downloads are gated on DOWNLOADS_READY as one block, so
    # touching PDB100_READY alone still pulled pdb100_foldseek_230517.tar.gz -- 15.5 GB we have no
    # use for. Fetching the 280 MB we need and then declaring downloads done skips the rest.
    [ -s "$MDB/mini_swissprot2503.tar.gz" ] || \
        curl -fsSL -o "$MDB/mini_swissprot2503.tar.gz" \
            https://opendata.mmseqs.org/colabfold/mini_swissprot2503.tar.gz \
        || die "could not fetch mini_swissprot2503.tar.gz"
    say "mini tarball: $(du -h --apparent-size "$MDB/mini_swissprot2503.tar.gz" | cut -f1)"
    touch "$MDB/DOWNLOADS_READY"
    # PDB markers so it skips templates and the 92 GB PDB rsync at build time too.
    for m in PDB_READY PDB100_READY PDB_MMCIF_READY; do touch "$MDB/$m"; done
    DEBUG_MINI_DB=1 MMSEQS_NO_INDEX=1 taskset -c "$CORES" bash setup_databases.sh "$MDB" \
        >> "$W/verify_minidb_setup.log" 2>&1
    say "setup_databases.sh exit $? (not trusted; checking files)"
fi
for d in uniref30_2302 colabfold_envdb_202108; do
    for f in _db.dbtype _db_seq _db_aln _db_mapping _db_taxonomy; do
        [ -s "$MDB/$d$f" ] || die "missing $MDB/$d$f"
    done
done
say "both mini databases present: $(du -sh --apparent-size "$MDB" | cut -f1)"

# The sidecars phase D creates by hand, because we run createindex ourselves (and here, never).
for d in uniref30_2302 colabfold_envdb_202108; do
    for l in mapping taxonomy; do
        ln -sf "${d}_db_$l" "$MDB/${d}_db.idx_$l"
        [ -s "$MDB/${d}_db.idx_$l" ] || die "$MDB/${d}_db.idx_$l does not resolve"
    done
done
say "sidecar symlinks created and resolving; NO .idx exists: $(ls "$MDB"/*.idx 2>/dev/null | wc -l) files"

# ---------------------------------------------------------------- the query set
QFA=$W/verify_minidb_queries.fasta
head -$((NQ * 2)) "$W/msa_initial_idx/query.searched.fasta" > "$QFA"
say "queries: $(grep -c '^>' "$QFA") taken from the frozen batch-1 searched fasta"

run_search() {   # out-dir, use-env, label
    local out=$1 env=$2 lbl=$3
    mkdir -p "$out"
    COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c "$CORES" \
        $R/envs/colabfold/bin/colabfold_search --mmseqs $M --threads "$NTHREADS" \
        --use-env "$env" --use-templates 0 --db-load-mode 0 \
        "$QFA" "$MDB" "$out" >> "$W/verify_minidb_$lbl.log" 2>&1
    say "  $lbl: colabfold_search exit $?"
}

# ---------------------------------------------------------------- experiment 1
say "--- EXPERIMENT 1: taxids with the sidecars but no .idx"
rm -rf "$W/verify_noidx"
run_search "$W/verify_noidx" 0 noidx
T=$W/verify_noidx/uniref_tax.m8
if [ -s "$T" ]; then
    ROWS=$(wc -l < "$T")
    WITH=$(awk -F'\t' '$3+0>0{n++} END{print n+0}' "$T")
    say "  uniref_tax.m8: $ROWS rows, $WITH with a nonzero taxid"
    say "  sample: $(head -1 "$T" | cut -f2,3,4)"
    [ "$WITH" -gt 0 ] && say "  RESULT: taxids RESOLVE without an index -- the 301 GB index is not needed for this" \
                      || say "  RESULT: rows but no taxids -- the index IS needed"
else
    say "  uniref_tax.m8 absent or empty"
    say "  RESULT: no taxonomy output without an index -- the index IS needed"
fi

# ---------------------------------------------------------------- experiment 2
say "--- EXPERIMENT 2: staged vs unstaged"
rm -rf "$W/verify_unstaged" "$W/verify_staged"
run_search "$W/verify_unstaged" 1 unstaged
say "  staged, pass 1: uniref only"
run_search "$W/verify_staged" 0 staged1
say "  hiding the uniref database, exactly as phase D deletes it"
mkdir -p "$MDB/hidden"
for f in "$MDB"/uniref30_2302_db*; do
    case "$f" in *.idx_*) ;; *) mv "$f" "$MDB/hidden/" ;; esac
done
say "  staged, pass 2: env, resuming from uniref.a3m with uniref30 gone"
run_search "$W/verify_staged" 1 staged2
say "  restoring the uniref database"
mv "$MDB"/hidden/* "$MDB"/ 2>/dev/null; rmdir "$MDB/hidden" 2>/dev/null

say "--- comparing the two"
SAME=0; DIFF=0
for f in "$W/verify_unstaged"/*.a3m; do
    b=$(basename "$f"); [ "$b" = "uniref.a3m" ] && continue
    g=$W/verify_staged/$b
    if [ ! -s "$g" ]; then say "  MISSING in staged: $b"; DIFF=$((DIFF+1)); continue; fi
    if cmp -s "$f" "$g"; then SAME=$((SAME+1)); else
        DIFF=$((DIFF+1))
        say "  DIFFERS: $b  unstaged $(grep -c '^>' "$f") seqs, staged $(grep -c '^>' "$g") seqs"
    fi
done
say "  byte-identical: $SAME    differing: $DIFF"
[ "$DIFF" -eq 0 ] && say "  RESULT: staged == unstaged, byte for byte" \
                  || say "  RESULT: staged differs from unstaged -- the staging argument fails"
say "=== done ==="
