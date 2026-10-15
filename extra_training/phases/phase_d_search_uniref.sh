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
# phases/phase_d_colabfold_search_staging.patch; without it colabfold_search refuses to start when uniref30
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
#     MIN_LONG_HIT_PCT=N  minimum percentage of queries of 20+ residues that must retrieve at least
#                    one UniRef hit. Default 90; measured 100 on both completed batches. There is no
#                    per-batch "allow thin" flag: whether a batch is expected to be thin is a property
#                    of its query LENGTHS, not of its size, so the gate reads the lengths instead.
#     MIN_Q=N        refuse to run if the query fasta holds fewer than N records. Default 1.
#     MIN_TAX_PCT=N  minimum percentage of the alignments' UniRef hits that must carry a taxid in
#                    uniref_tax.m8. Default 95. This checks the taxid MECHANISM, not search depth.
#     TAX_SAMPLE=N   how many alignments to sample for that check. Default 60.
#     BUILD_INDEX=0  skip the 301 GB index and give _db_seq the taxonomy sidecars instead. Measured
#                    to give identical taxids on mini_swissprot; the default is 1 because uniref30's
#                    key-space gap is ~10x against mini's 4.5x and that scale is untested.
#     CORES=0-63     taskset range. 64 cores is the project cap (MSA_RUNBOOK.md:119); --threads is
#                    derived from it with `taskset -c "$CORES" nproc`, never hardcoded.
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
# the repo root, derived from this file so a rename of the checkout needs no edit here
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DB=$R/tools/colabfold/db
W=$R/extra_training/building_msa/work
M=$R/tools/mmseqs/bin/mmseqs
CORES="${CORES:-0-63}"
# THREADS DERIVED FROM THE TASKSET, not hardcoded. `--threads 64` under a narrower --cpu-list
# oversubscribes: the threads exist but the scheduler has only the pinned cores to put them on.
# `taskset -c "$CORES" nproc` reports exactly how many CPUs that range allows, whatever its syntax,
# so the two cannot disagree. The project cap is 64 cores (MSA_RUNBOOK.md:119 -- more has crashed
# this server), which is the default here.
NTHREADS=$(taskset -c "$CORES" nproc 2>/dev/null || echo 64)
export PATH="$R/tools/mmseqs/bin:$PATH"

QFA="${1:-}"
OUT="${2:-}"
TAG="${3:-$(basename "${OUT:-batch}")}"
LOG=$W/phase_d_$TAG.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../shell_scripts/lib_fetch.sh"      # fetch_tarball: completeness from the server, not a constant

# ---------------------------------------------------------------- 0. the patch, checked not assumed
# EVERYTHING BELOW DEPENDS ON THIS, so it is a gate rather than a comment telling the operator to
# look. Without the staging patch there is no uniref_tax.m8 (so no paired MSA) and no resume from
# uniref.a3m (so phase E cannot run once uniref30 is deleted). Unpatched, that surfaced only after
# the 100 GB download and the search -- hours in, at the taxid gate. Four lines here instead.
SEARCHPY=$R/envs/colabfold/lib/python3.12/site-packages/colabfold/mmseqs/search.py
check_patch() {
    [ -s "$SEARCHPY" ] || return 1
    # The functions the patch adds, asked of the imported module rather than grepped for. Their
    # RETURN values depend on the environment, so only their existence is checked.
    $R/envs/colabfold/bin/python -c 'import sys, colabfold.mmseqs.search as m; sys.exit(0 if all(hasattr(m,f) for f in ("_emit_taxid","_keep_intermediates")) else 1)' || return 1
    # Each of the four hunks carries this marker; counting them catches a partial application.
    local n; n=$(grep -c 'PATCHED (tts_abag)' "$SEARCHPY" 2>/dev/null || true)
    [ "${n:-0}" -ge 4 ] || { say "  only ${n:-0} PATCHED markers in search.py, expected at least 4"; return 1; }
}
if ! check_patch; then
    say "the staging patch is not applied to $SEARCHPY"
    say "  without it: no uniref_tax.m8 (no paired MSA), and phase E cannot resume once uniref30 is gone"
    say "  apply extra_training/phases/phase_d_colabfold_search_staging.patch to that file, then re-run"
    die "colabfold_search is unpatched"
fi
say "staging patch applied"

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
    # MARKERS WE CREATE ARE REMOVED AGAIN, so the prior state is restored exactly. Leaving
    # PDB_READY / PDB100_READY / PDB_MMCIF_READY behind makes a later bare setup_databases.sh call --
    # which is how the template step runs -- skip its own downloads and report success. The
    # discipline already exists elsewhere in the repo: phase2_envdb.sh removes COLABDB_READY before
    # building. Track exactly what we touched; never remove a marker that was already there.
    CREATED=""
    # ON ANY EXIT, PUT THE MARKERS BACK THE WAY THEY WERE. The removal used to sit only at the end
    # of the happy path, so every `die` between here and there -- a truncated tarball, a missing
    # _db_taxonomy, an index shorter than its own index file, the env-database guard -- left
    # PDB_READY / PDB100_READY / PDB_MMCIF_READY / COLABDB_READY on disk. A later bare
    # setup_databases.sh call, which is how the template step runs, then SKIPS those databases and
    # reports success, so a retry silently builds on a database that was never built. An EXIT trap
    # cannot be forgotten on a path added later; `rm -f` over an empty CREATED is a no-op, so the
    # happy path runs it too. Never removes a marker that was already there.
    restore_markers() {
        local m
        for m in $CREATED; do rm -f "$DB/$m"; done
        [ -n "$CREATED" ] && say "  removed the markers this run created:$CREATED"
        CREATED=""
    }
    trap restore_markers EXIT
    for m in COLABDB_READY PDB_READY PDB100_READY PDB_MMCIF_READY; do
        if [ ! -e "$DB/$m" ]; then
            touch "$DB/$m"; CREATED="$CREATED $m"
            say "  created $m so setup_databases.sh skips that database"
        fi
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
    # NON-EMPTY IS NOT WHOLE. An intact .index can sit beside a truncated data file -- a tar that
    # stopped part-way leaves exactly that -- and every count downstream would still agree. An MMseqs
    # .index line is "key<TAB>offset<TAB>length", so the furthest byte any entry claims must lie
    # inside the data file. A file shorter than its own index is truncated, full stop.
    for base in uniref30_2302_db uniref30_2302_db_seq uniref30_2302_db_aln; do
        idx=$DB/$base.index
        [ -s "$idx" ] || die "missing $idx, so $base cannot be checked for completeness"
        read -r NEED < <(awk -F'\t' '{e=$2+$3; if (e>m) m=e} END{print m+0}' "$idx")
        HAVE=$(stat -c%s "$DB/$base")
        if [ "$HAVE" -lt "$NEED" ]; then
            die "$base is $HAVE bytes but its index claims data out to $NEED -- the file is
     truncated. Delete the database and re-extract; the counts downstream cannot see this."
        fi
        say "  $base complete: $HAVE bytes, index claims $NEED"
    done
    if [ "$ENVDB_BEFORE" -eq 0 ] && [ -s "$DB/colabfold_envdb_202108_db_seq" ]; then
        die "setup_databases.sh also extracted the env database; the two do not both fit"
    fi
    rm -f "$DB/uniref30_2302.tar.gz"
    restore_markers
    say "uniref30 ready: $(du -sh --apparent-size "$DB/uniref30_2302_db_seq" | cut -f1) of members"
fi

# ---------------------------------------------------------------- 2. the index, for taxonomy only
#
# MEASURED 2026-10-11: THE INDEX IS NOT ACTUALLY REQUIRED FOR TAXONOMY. search.py:134-139 sets
# dbSuffix1 to "_seq" when no index is present and ".idx" when one is, and convertalis then wants the
# taxonomy sidecars for whichever suffix it is handed. With an index that is .idx_mapping /
# .idx_taxonomy; without one it is _db_seq_mapping / _db_seq_taxonomy, which nothing creates -- so an
# index-free run failed with exit 1 and `uniref_tax.m8 not produced`, and the index looked mandatory.
# It is not: creating the _db_seq sidecars as symlinks makes an index-free run produce taxids, and on
# mini_swissprot the two routes gave IDENTICAL assignments -- 610 targets each, 610 shared, 0
# disagreeing, none unique to either (shell_scripts/verify_minidb.sh).
#
# WHY THE INDEX IS STILL BUILT BY DEFAULT. The mini database's _db and _db_seq differ 4.5x in size
# (127,203 against 573,661 entries), so the key-space concern was exercised -- but the real uniref30
# differs nearly 10x (36,293,491 representatives against 350,950,053 members), and that scale is
# untested. A wrong-but-nonzero taxid is the dangerous failure here, far worse than a missing one, so
# the verified-safe route stays the default.
#
# BUILD_INDEX=0 takes the cheap route. Worth doing: it saves 301 GB and ~5 minutes per batch, and it
# would let uniref30 (450 GB) and colabfold_envdb (291 GB) sit on disk together, which removes the
# entire staging arrangement -- the two-pass split, the kept intermediates, and the reason the
# staging patch exists. Before flipping the default, run one real-uniref30 batch both ways and
# confirm the m8 target->taxid assignments are identical, as they were on mini.
if [ "${BUILD_INDEX:-1}" = "0" ]; then
    say "BUILD_INDEX=0: skipping the index; giving _db_seq its own taxonomy sidecars instead"
    for l in mapping taxonomy; do
        [ -s "$DB/uniref30_2302_db_$l" ] || die "missing $DB/uniref30_2302_db_$l"
        ln -sf "uniref30_2302_db_$l" "$DB/uniref30_2302_db_seq_$l"
        [ -s "$DB/uniref30_2302_db_seq_$l" ] || die "$DB/uniref30_2302_db_seq_$l does not resolve"
        say "  db_seq_$l -> $(readlink "$DB/uniref30_2302_db_seq_$l")"
    done
elif [ ! -s "$DB/uniref30_2302_db.idx" ]; then
    say "building the index (taxonomy, not speed -- see the header)"
    cd "$DB" || die "no db dir"
    rm -rf tmp1
    taskset -c "$CORES" $M createindex uniref30_2302_db tmp1 --remove-tmp-files 1 --threads "$NTHREADS" \
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
if [ "${BUILD_INDEX:-1}" != "0" ]; then
    for l in mapping taxonomy; do
        [ -s "$DB/uniref30_2302_db_$l" ] || die "missing $DB/uniref30_2302_db_$l -- taxids cannot resolve"
        ln -sf "uniref30_2302_db_$l" "$DB/uniref30_2302_db.idx_$l"
        [ -s "$DB/uniref30_2302_db.idx_$l" ] || die "$DB/uniref30_2302_db.idx_$l does not resolve"
    done
fi
say "taxonomy tables resolve"

# ---------------------------------------------------------------- 3. the search
# --use-env 0: the environmental database is phase E, and it is not on disk now.
# --db-load-mode 0: no index-into-memory; the index exists for taxonomy, not for the search.
#
# HOW RE-RUNNING ACTUALLY BEHAVES. Not per query. colabfold_search's skip logic is per DATABASE
# SEARCH, keyed on the kept intermediate: search.py:191 logs "Skipping <uniref_db> search because
# uniref.a3m already exists". So a re-run skips the expensive search and regenerates every per-query
# a3m from uniref.a3m. Both this script and the phase1_search_idx.sh it replaces claimed it "skips a
# query whose <name>.a3m already exists" -- that is not what the code does. The practical difference:
# a re-run is cheap, but it rewrites all the per-query files rather than filling in gaps, so a
# partial output cannot be topped up query-by-query. It also means the count below should match
# exactly rather than merely not fall short.
# FREEZE THE FASTA THAT IS ACTUALLY SEARCHED, beside its outputs, with a checksum.
# search.py:517 runs `createdb <query_file> qdb --shuffle 0` unconditionally, and ColabFold keys
# everything by POSITION in that database. So phase E resuming from uniref.a3m and prof_res is only
# correct if it is handed the byte-identical fasta in the same order. This is not hypothetical: on
# 2026-10-11 phase C's new antigen-type filter removed one RNA chain from query.fasta, which sat at
# key 479 of 5,305 -- rebuilding qdb from the shortened file would have shifted 4,825 keys and
# mis-assigned that many alignments to the wrong chains, silently. Phase E must read
# <out-dir>/query.searched.fasta and verify this checksum, never msa_work/query.fasta, which is
# phase C's current output and may legitimately have moved on.
# COMPARE BEFORE WRITING. Overwriting unconditionally meant a rerun stamped a NEW checksum onto
# OLD intermediates: point this at an existing output directory with a reordered or changed fasta and
# uniref.a3m/prof_res still describe the previous query order, while the checksum says everything
# agrees. Phase E then trusts it. So an existing frozen fasta is authoritative -- if it differs, this
# is a different batch and needs its own directory.
FROZ=$OUT/query.searched.fasta
if [ -s "$FROZ" ]; then
    if ! cmp -s "$QFA" "$FROZ"; then
        die "$OUT already holds a DIFFERENT searched fasta ($(grep -c '^>' "$FROZ") records against
     $NQ in $QFA). Its uniref.a3m and prof_res are keyed by that older query order, and ColabFold
     keys by position, so reusing this directory would mis-assign alignments. Use a fresh output
     directory for a changed query set."
    fi
    say "frozen fasta already present and identical: $(cut -c1-16 "$FROZ.sha256" 2>/dev/null)"
else
    cp -f "$QFA" "$FROZ"
    ( cd "$OUT" && sha256sum query.searched.fasta > query.searched.fasta.sha256 )
    say "froze the searched fasta: $(cut -c1-16 "$FROZ.sha256")"
fi

say "searching"
COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c "$CORES" \
    $R/envs/colabfold/bin/colabfold_search --mmseqs $M --threads "$NTHREADS" \
    --use-env 0 --use-templates 0 --db-load-mode 0 \
    "$QFA" "$DB" "$OUT" >> "$W/phase_d_search_$TAG.log" 2>&1
say "colabfold_search exit $? (not trusted; checking output)"

# ---------------------------------------------------------------- 4. the gates
# BY NAME, NOT BY COUNT. A count with -ge is satisfied by any directory holding enough files,
# including ones left by an earlier batch, so it cannot tell "this batch is covered" from "this
# directory happens to be full". Every query header must have its own a3m.
NA=$(ls "$OUT"/*.a3m 2>/dev/null | grep -vc '/uniref.a3m$')
MISSING=$(awk '/^>/{sub(/^>/,""); print}' "$QFA" | while read -r n; do
              [ -s "$OUT/$n.a3m" ] || echo "$n"
          done)
NMISS=$(printf '%s' "$MISSING" | grep -c . || true)
say "alignments written: $NA in the directory; $NQ queries in this batch; $NMISS without an a3m"
if [ "${NMISS:-0}" -gt 0 ]; then
    say "  first few missing: $(printf '%s' "$MISSING" | head -5 | tr '\n' ' ')"
    die "$NMISS of $NQ queries have no alignment -- the search did not cover the batch"
fi

# WHAT THE m8 IS ACTUALLY FOR, and therefore what to measure. Protenix's step3 reads it as a
# dictionary from COLUMN 2, the UniRef accession, to COLUMN 3, the taxid
# (step3-uniref_add_taxid.py:253-255). The query column is never read -- and could not be: ColabFold
# writes a constant query identifier, so every row of this file says `101`. The a3m files carry the
# same constant as their first header. Any gate keyed on the query column therefore measures nothing;
# an earlier version of this one did exactly that and always reported a single query.
#
# So the mechanism check is: of the UniRef hits that appear in the alignments, how many have a taxid?
# That is what decides how much of each alignment can be paired. It should be ~100%; if convertalis
# or the patch failed it collapses to 0 while the alignments themselves look perfectly normal.
#
# DEPTH IS GATED BY QUERY LENGTH, not by a per-batch flag. The expectation is a property of the
# query: a 12-residue peptide retrieving nothing is correct, a 300-residue chain retrieving nothing is
# a fault. Measured on the completed batches, the split is clean -- of 200 sampled queries of 20+
# residues, 200 have UniRef hits (100%); of 200 sampled under 20 residues, 25 do (12%). So the gate
# requires hits for the long queries and exempts the short ones, which makes a small peptide batch
# pass for the right reason rather than by being small, and keeps a mixed batch honest.
#
# SAMPLED a3m, FULL m8 PASS. The sample bounds memory (batch 1's alignments hold far too many
# accessions to hold all at once); the m8 must be streamed in full because membership is the
# question. Tens of GB of sequential read, minutes against an hour of search.
TAXF=$OUT/uniref_tax.m8
say "measuring depth by query length, and taxid coverage of the hits ..."
COV=$($R/envs/colabfold/bin/python - "$OUT" "$TAXF" "${TAX_SAMPLE:-200}" <<'PYCOV'
import glob, os, sys
out, m8, nsample = sys.argv[1], sys.argv[2], int(sys.argv[3])
fa, n = {}, None
with open(os.path.join(out, "query.searched.fasta")) as fh:
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            n = line[1:]; fa[n] = ""
        elif n:
            fa[n] += line
files = sorted(f for f in glob.glob(os.path.join(out, "*.a3m"))
               if os.path.basename(f) != "uniref.a3m")
step = max(1, len(files) // nsample)
sample = files[::step][:nsample]
# long = 20+ residues, the floor phase C's first batch used
long_tot = long_hit = short_tot = short_hit = 0
hits = set()
for p in sample:
    q = fa.get(os.path.basename(p)[:-4])
    if q is None:
        continue
    n_hit = 0
    with open(p) as fh:
        for line in fh:
            if line.startswith(">UniRef100_"):
                hits.add(line[1:].split()[0]); n_hit += 1
    if len(q) >= 20:
        long_tot += 1; long_hit += 1 if n_hit else 0
    else:
        short_tot += 1; short_hit += 1 if n_hit else 0
found = set()
if hits and os.path.exists(m8):
    with open(m8) as fh:
        for line in fh:
            f = line.split("\t", 3)
            if len(f) > 2 and f[1] in hits and f[2].strip() not in ("", "0"):
                found.add(f[1])
lpct = 100 * long_hit // long_tot if long_tot else -1
tpct = 100 * len(found) // len(hits) if hits else -1
print(f"{len(sample)} {long_tot} {long_hit} {lpct} {short_tot} {short_hit} {len(hits)} {len(found)} {tpct}")
PYCOV
)
# A CRASHED VALIDATOR IS NOT A CLEAN BATCH. COV was consumed without checking the exit code, so a
# traceback left it empty, `set --` produced no positional parameters, every ${n:-default} supplied
# a zero, and both gates below are guarded by `[ "$LTOT" -gt 0 ]` -- which is false at zero. The
# batch then passed without a single check having run. phase_e_search_envdb.sh:271-281 already
# escalates on exactly this; phase D did not. The field count is checked too: a validator that
# prints a partial line is as uninformative as one that prints nothing.
RC=$?
if [ "$RC" -ne 0 ] || [ -z "$COV" ]; then
    die "the coverage validator exited $RC (output: ${COV:-<empty>}). Its checks did not run, so
     nothing about this batch has been verified -- depth and taxid coverage are both unknown."
fi
set -- $COV
if [ "$#" -ne 9 ]; then
    die "the coverage validator returned $# fields, expected 9: $COV"
fi
NS=${1:-0}; LTOT=${2:-0}; LHIT=${3:-0}; LPCT=${4:--1}; STOT=${5:-0}; SHIT=${6:-0}
NHITS=${7:-0}; NFOUND=${8:-0}; TPCT=${9:--1}
say "  sampled $NS alignments"
[ "$LTOT" -gt 0 ] && say "    20+ residues: $LHIT of $LTOT have UniRef hits (${LPCT}%)"
[ "$STOT" -gt 0 ] && say "    under 20:     $SHIT of $STOT have UniRef hits -- query-only is expected here"
say "    $NHITS distinct UniRef hits, $NFOUND carry a taxid (${TPCT}%)"

# DEPTH, gated only on the long queries.
if [ "$LTOT" -gt 0 ] && [ "$LPCT" -lt "${MIN_LONG_HIT_PCT:-90}" ]; then
    die "only ${LPCT}% of sampled queries of 20+ residues retrieved any UniRef hit (floor
     ${MIN_LONG_HIT_PCT:-90}%). A chain that long returning nothing is a fault rather than a property
     of the query -- on both completed batches this was 200 of 200."
fi

# TAXID MECHANISM. Only meaningful where there are hits to attach taxids to: a batch of short
# peptides that retrieved nothing legitimately has no m8, and that is not a fault. Which is why no
# per-batch flag is needed -- "is this expected to be thin" is answered by the query lengths.
if [ "$NHITS" -gt 0 ]; then
    [ -s "$TAXF" ] || die "no uniref_tax.m8 although the alignments carry $NHITS UniRef hits --
     convertalis did not run, so nothing in this batch can be paired. Check the staging patch."
    if [ "$TPCT" -lt "${MIN_TAX_PCT:-95}" ]; then
        die "only ${TPCT}% of the alignments' UniRef hits carry a taxid (floor ${MIN_TAX_PCT:-95}%).
     step3 builds its taxid dictionary from this file, so the rest cannot be paired."
    fi
elif [ "$LTOT" -gt 0 ]; then
    die "no UniRef hits at all in a sample holding $LTOT queries of 20+ residues -- the search
     returned nothing where it should have returned a great deal."
else
    say "  no UniRef hits in the sample and every sampled query is under 20 residues: expected"
fi

# THE INTERMEDIATES MUST BE COMPLETE, NOT MERELY PRESENT -- and unlike the depth check above, this
# one has no exemption for short queries. A batch without complete intermediates simply never gets
# phase E, which is a different failure from a batch that legitimately retrieved little.
#
# WHY COMPLETENESS AND NOT EXISTENCE. colabfold_search skips the uniref search when uniref.a3m
# exists (search.py:191) -- it does not check that it is COMPLETE. A search killed mid-write leaves a
# truncated intermediate, and the re-run then expands that truncation into per-query files. Tail
# truncation shows up as missing a3m and the by-name gate above catches it; a boundary query whose
# record was cut mid-record comes out non-empty but thin, and nothing above would notice. The index
# entry counts are what close that: they must equal the number of queries exactly.
for f in uniref.a3m prof_res; do
    [ -s "$OUT/$f" ] || die "$OUT/$f missing -- phase E cannot resume without uniref30 on disk"
    [ -s "$OUT/$f.index" ] || die "$OUT/$f.index missing -- cannot verify the intermediate is complete"
    N=$(wc -l < "$OUT/$f.index")
    [ "$N" -eq "$NQ" ] || die "$f holds $N entries for $NQ queries -- the intermediate is incomplete.
     colabfold_search skips the search when uniref.a3m merely EXISTS, so a re-run would expand this
     truncation into per-query alignments. Delete $OUT/$f* and re-run to search again."
    say "  $f complete: $N entries for $NQ queries"
done
say "depth of the first few:"
for f in $(ls "$OUT"/*.a3m 2>/dev/null | grep -v '/uniref.a3m$' | head -5); do
    say "   $(basename "$f") $(grep -c '^>' "$f") sequences"
done
# $NTAX no longer exists: the taxid gate was rewritten to measure coverage of the alignments' hits,
# and this line still referenced the old variable. Under set -u that produced an unbound-variable
# diagnostic inside a command substitution, where it does not fail the surrounding command -- so the
# run reported success while printing an error. It now reports what was actually measured.
say "=== phase D OK for $TAG: $NA alignments; $NFOUND of $NHITS sampled UniRef hits carry a taxid ==="

# ---------------------------------------------------------------- 5. free the database, on request
# VALIDATE THE INTERMEDIATES BEFORE DELETING THE DATABASE THEY CANNOT BE REBUILT WITHOUT.
# phase E resumes from uniref.a3m and prof_res; once uniref30 is gone, a truncated intermediate means
# re-downloading 96 GB and re-searching. `-s` is not enough: an mmseqs db is a data file plus an
# .index of "key offset length", and a 3-byte data file whose index claims 1,000 bytes passes every
# existence check there is. So the index is compared against the data it points into.
check_mmseqs_db() {
    local base="$1" data idx end sz
    data="$base"; idx="$base.index"
    [ -s "$data" ] || { say "  MISSING or empty $data"; return 1; }
    [ -s "$idx" ]  || { say "  MISSING or empty $idx";  return 1; }
    sz=$(stat -c %s "$data")
    # the furthest byte any key claims; it must lie inside the data file
    end=$(awk -F'\t' 'NF>=3 {e=$2+$3; if (e>m) m=e} END{printf "%d", m}' "$idx")
    if [ "${end:-0}" -gt "$sz" ]; then
        say "  TRUNCATED $data: the index reaches byte $end, the file is $sz"
        return 1
    fi
    say "  ok $(basename "$data"): $(awk 'END{print NR}' "$idx") keys, index reaches $end of $sz bytes"
    return 0
}

if [ "${FREE_DB:-0}" = "1" ]; then
    say "validating the retained intermediates before freeing uniref30"
    bad=0
    # BOTH BATCH DIRECTORIES, when a second is named. The controls are their own batch and phase E
    # processes both, so a truncated intermediate in EITHER costs the 96 GB download and a re-search.
    # ALSO_VALIDATE is a space-separated list of other batch directories to check before freeing.
    for b in "$OUT/uniref.a3m" "$OUT/prof_res" $(for o in ${ALSO_VALIDATE:-}; do echo "$o/uniref.a3m" "$o/prof_res"; done); do
        [ -e "$b" ] || { say "  absent: $b"; bad=1; continue; }
        check_mmseqs_db "$b" || bad=1
    done
    [ "$bad" -eq 0 ] || die "a retained intermediate is missing or truncated; NOT freeing uniref30 --
     phase E resumes from these, and rebuilding them means another 96 GB download and search"
    say "freeing uniref30 and its index so the env database fits"
    rm -rf "$DB"/uniref30_2302_db* "$DB"/UNIREF30_READY
    say "freed; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
else
    say "uniref30 left on disk (FREE_DB=1 to remove it); quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
fi
