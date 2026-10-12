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
#     FREE_DB=1      delete the env database once every batch succeeds. DEFAULT 0, deliberately:
#                    phase F is the first thing that reads the env half of these alignments, and
#                    anything it reveals would otherwise cost a 120 GB re-download. The orchestrator
#                    frees it after phase F passes. 291 GB is most of the disk headroom, so this is
#                    not free -- it is cheap insurance against repeating an hour and a half.
#     KEEP_TARBALL=1 keep the 120 GB tarball after extracting. Default 0.
#     MIN_LONG_GROW_PCT=N  minimum percentage of chains of 20+ residues that must gain at least one
#                    sequence from the env pass. Default 90.
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

# THE ENV INTERMEDIATE, VALIDATED AGAINST ITS OWN INDEX. "The file is nonempty" is not evidence that
# a batch was deepened: an interrupted search leaves a truncated data file whose index still lists
# every query, and an index whose entries point past the end of the data is exactly what a kill
# mid-write produces. So: one entry per query, every length positive, and the furthest byte an entry
# claims must actually exist in the data file.
env_ok() {
    local out=$1 nq=$2 idx=$1/bfd.mgnify30.metaeuk30.smag30.a3m.index
    local dat=$1/bfd.mgnify30.metaeuk30.smag30.a3m
    [ -s "$idx" ] && [ -s "$dat" ] || return 1
    [ "$(wc -l < "$idx")" -eq "$nq" ] || return 1
    local size end zero
    size=$(stat -c %s "$dat")
    # colabfold/mmseqs index lines are "<key>\t<offset>\t<length>"
    end=$(awk -F'\t' 'BEGIN{m=0} {if($2+$3>m) m=$2+$3} END{print m+0}' "$idx")
    zero=$(awk -F'\t' '$3+0<=0 {n++} END{print n+0}' "$idx")
    [ "$zero" -eq 0 ] || return 1
    [ "$end" -le "$size" ] || return 1
    return 0
}

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
    rm -f "${DB:?}"/"${CFDB:?}"_*.tsv "${DB:?}"/"${CFDB:?}".tsv
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

# PER-QUERY SIZES, snapshotted before the pass. A directory total is satisfied by other queries'
# growth, so it cannot see a chain that received nothing while the rest grew -- and the by-name check
# cannot either, because phase D already left every query a non-empty a3m. The only thing that
# answers "did THIS chain gain env sequences" is its own size, before against after.
snapshot() { find "$1" -maxdepth 1 -name '*.a3m' ! -name 'uniref.a3m' \
                  ! -name 'bfd.mgnify30.metaeuk30.smag30.a3m' -printf '%f\t%s\n' 2>/dev/null | sort; }

FAILED=0
for OUT in "${BATCHES[@]}"; do
    LBL=$(basename "$OUT")
    QFA=$OUT/query.searched.fasta
    NQ=$(grep -c '^>' "$QFA")
    BEFORE=$(a3m_bytes "$OUT")
    SNAP=$W/phase_e_sizes_$LBL
    snapshot "$OUT" > "$SNAP"
    # HAS THIS BATCH ALREADY BEEN DEEPENED? colabfold keeps bfd.mgnify30.metaeuk30.smag30.a3m once
    # the env half has run, and a second pass over finished alignments is a no-op by design. The
    # growth gate would then see zero growth and reject a correct result -- which matters because the
    # orchestrator reruns the whole chain when a LATER phase fails. So growth is required only on the
    # first pass; on a rerun the same correctness checks run and growth is reported, not demanded.
    # A RERUN IS ESTABLISHED BY A COMPLETE INTERMEDIATE, NOT A NONEMPTY ONE. Setting this from mere
    # presence meant an interrupted search -- which leaves a truncated data file -- disabled BOTH
    # growth gates, so a batch that had actually gained nothing passed as "already deepened".
    ALREADY=0
    if env_ok "$OUT" "$NQ"; then
        ALREADY=1
        say "    (env intermediate complete for all $NQ queries: this is a rerun, growth not required)"
    elif [ -s "$OUT/bfd.mgnify30.metaeuk30.smag30.a3m" ]; then
        say "    env intermediate present but INCOMPLETE or inconsistent with its index -- treating
     this as a first pass, so growth is required and the checks below must pass on their own"
    fi
    say "--- $LBL: $NQ queries, alignments currently $(numfmt --to=iec "$BEFORE")"
    COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c "$CORES" \
        $R/envs/colabfold/bin/colabfold_search --mmseqs $M --threads "$NTHREADS" \
        --use-env 1 --use-templates 0 --db-load-mode 0 \
        "$QFA" "$DB" "$OUT" >> "$W/phase_e_search_$LBL.log" 2>&1
    SRC=$?
    # NONZERO IS PROOF OF FAILURE, zero is not proof of success. The original note here -- "not
    # trusted; the data decides" -- was right about zero and wrong about nonzero: setup_databases.sh
    # returning 0 having done nothing is why the data checks exist, but a search that exited 99 did
    # not finish, and continuing past it let a truncated intermediate reach the gates below.
    say "  colabfold_search exit $SRC"
    if [ "$SRC" -ne 0 ]; then
        say "  ESCALATE $LBL: colabfold_search exited $SRC. See $W/phase_e_search_$LBL.log. The
     alignments it left are not a completed pass, whatever their byte totals say."
        FAILED=$((FAILED+1)); continue
    fi
    grep -q "Skipping" "$W/phase_e_search_$LBL.log" \
        && say "  note: the log mentions skipping a search, consistent with resuming from uniref.a3m"

    AFTER=$(a3m_bytes "$OUT")
    snapshot "$OUT" > "$SNAP.after"
    # The intermediate colabfold keeps for the env half, named for the databases it merges. Its
    # presence distinguishes "the env search ran" from "the uniref alignments were rewritten".
    ENVINT=$(ls "$OUT"/bfd.mgnify30.metaeuk30.smag30.a3m 2>/dev/null | wc -l)
    say "  alignments $(numfmt --to=iec "$BEFORE") -> $(numfmt --to=iec "$AFTER"); env intermediate: $ENVINT"

    # PER QUERY, AND THE QUERY LINE RE-ASSERTED. Two questions a directory total cannot answer:
    #   did each chain of 20+ residues actually gain sequences, and
    #   does each a3m still begin with ITS OWN query sequence?
    # The second is the direct test that qdb keys did not shift. The env pass is where mergedbs could
    # misalign them, which is the failure the checksum gate on the input exists to prevent -- so the
    # same check phase D's review ran by hand on the input is run here on the output.
    # ITS EXIT STATUS IS CHECKED. On failure the shell substituted an empty string, `set --` left
    # every field unset, and the ${x:-0} defaults made a crashed validator look like a batch with
    # nothing to report -- absent files and sequence mismatches included.
    REPORT=$($R/envs/colabfold/bin/python - "$OUT" "$QFA" "$SNAP" "$SNAP.after" <<'PYCHK'
import os, sys
out, qfa, before_f, after_f = sys.argv[1:5]
fa, n = {}, None
with open(qfa) as fh:
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            n = line[1:]; fa[n] = ""
        elif n:
            fa[n] += line
def sizes(p):
    d = {}
    with open(p) as fh:
        for line in fh:
            k, v = line.rstrip("\n").split("\t")
            d[k[:-4]] = int(v)
    return d
before, after = sizes(before_f), sizes(after_f)
lt = lg = st = sg = 0
mism, absent, unexplained = [], [], []
for name, seq in fa.items():
    p = os.path.join(out, name + ".a3m")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        absent.append(name); continue
    grew = after.get(name, 0) > before.get(name, 0)
    if len(seq) >= 20:
        lt += 1; lg += 1 if grew else 0
    else:
        st += 1; sg += 1 if grew else 0
    with open(p) as fh:
        fh.readline()
        line2 = fh.readline().strip()
    if line2.upper() != seq.upper():
        mism.append(name)
        # EXPLAINED OR NOT. The expected transformation is a nucleic-acid chain the search wrote as
        # X: same length, and every differing position a nucleotide that became exactly X.
        nuc = set("ACGTUN")
        if len(line2) != len(seq) or not all(
                b.upper() == "X" and a.upper() in nuc
                for a, b in zip(seq, line2) if a.upper() != b.upper()):
            unexplained.append(name)
lpct = 100 * lg // lt if lt else -1
print(f"{lt} {lg} {lpct} {st} {sg} {len(mism)} {len(absent)} {len(unexplained)}")
print(" ".join(unexplained[:5]) or " ".join(mism[:5]))
print(" ".join(absent[:5]))
PYCHK
)
    RC=$?
    if [ "$RC" -ne 0 ] || [ -z "$REPORT" ]; then
        say "  ESCALATE $LBL: the per-query validator exited $RC (output: ${REPORT:-<empty>})."
        say "    Its checks did not run, so nothing about this batch has been verified."
        FAILED=$((FAILED+1)); continue
    fi
    set -- $(printf '%s' "$REPORT" | head -1)
    if [ "$#" -ne 8 ]; then
        say "  ESCALATE $LBL: validator returned $# fields, expected 8: $REPORT"
        FAILED=$((FAILED+1)); continue
    fi
    LTOT=${1:-0}; LGREW=${2:-0}; LPCT=${3:--1}; STOT=${4:-0}; SGREW=${5:-0}; NMISM=${6:-0}
    NABS=${7:-0}; NUNEXP=${8:-0}
    [ "$LTOT" -gt 0 ] && say "    20+ residues: $LGREW of $LTOT gained sequences (${LPCT}%)"
    [ "$STOT" -gt 0 ] && say "    under 20:     $SGREW of $STOT gained sequences -- many legitimately gain none"
    say "    query-line mismatches: $NMISM ($NUNEXP unexplained)   absent or empty: $NABS"
    # THE ENV INTERMEDIATE, CHECKED DIRECTLY. Byte growth is an indirect measure: it says the files
    # changed, not that env sequences were added. colabfold keeps the env half as its own MMseqs
    # database, so its index has one entry per query when the env search actually covered the batch.
    ENVIDX=$OUT/bfd.mgnify30.metaeuk30.smag30.a3m.index
    if [ -s "$ENVIDX" ]; then
        NENV=$(wc -l < "$ENVIDX")
        say "    env intermediate: $NENV entries for $NQ queries"
        # THE DATA, NOT JUST THE ENTRY COUNT. An index listing every query says nothing about
        # whether the bytes it points at were written.
        if ! env_ok "$OUT" "$NQ"; then
            say "  ESCALATE $LBL: the env intermediate's index does not agree with its data file
     (entries pointing past the end, or a zero-length entry). The search was interrupted."
            FAILED=$((FAILED+1)); continue
        fi
        if [ "$NENV" -ne "$NQ" ]; then
            say "  ESCALATE $LBL: the env intermediate holds $NENV entries for $NQ queries, so the
     env search did not cover the batch whatever the byte totals say"
            FAILED=$((FAILED+1)); continue
        fi
    else
        say "  ESCALATE $LBL: no env intermediate index; the env half did not run"
        FAILED=$((FAILED+1)); continue
    fi

    if [ "$NABS" -gt 0 ]; then
        say "  ESCALATE $LBL: $NABS queries have no alignment: $(printf '%s' "$REPORT" | sed -n 3p)"
        FAILED=$((FAILED+1)); continue
    fi
    # EVERY MISMATCH MUST BE EXPLAINED, not merely rare. A 1% tolerance accepted up to 53 swapped or
    # corrupted alignments on batch 1, and a handful of wrong alignments is still wrong. The
    # validator classifies each one: the expected case is a nucleic-acid chain whose nucleotides the
    # search wrote as X, and anything else is reported by name and fails regardless of how few there
    # are. A key shift shows up as thousands, which the same check catches a fortiori.
    if [ "${NUNEXP:-0}" -gt 0 ]; then
        say "  ESCALATE $LBL: $NUNEXP a3m differ from their query for reasons other than
     nucleotide->X: $(printf '%s' "$REPORT" | sed -n 2p)"
        FAILED=$((FAILED+1)); continue
    fi
    MAXMISM=$(( NQ / 100 )); [ "$MAXMISM" -lt 2 ] && MAXMISM=2
    if [ "$NMISM" -gt "$MAXMISM" ]; then
        say "  ESCALATE $LBL: $NMISM a3m do not begin with their own query sequence (tolerance
     $MAXMISM). That is the signature of shifted qdb keys, not of a transcribed base.
     First few: $(printf '%s' "$REPORT" | sed -n 2p)"
        FAILED=$((FAILED+1)); continue
    fi
    if [ "$ALREADY" -eq 0 ] && [ "$LTOT" -gt 0 ] && [ "$LPCT" -lt "${MIN_LONG_GROW_PCT:-90}" ]; then
        say "  ESCALATE $LBL: only ${LPCT}% of chains of 20+ residues gained any env sequence
     (floor ${MIN_LONG_GROW_PCT:-90}%). A chain that long gaining nothing from a database this large
     is a fault, not a property of the query."
        FAILED=$((FAILED+1)); continue
    fi
    if [ "$ALREADY" -eq 0 ] && [ "$AFTER" -le "$BEFORE" ]; then
        say "  ESCALATE $LBL: the directory did not grow at all ($BEFORE -> $AFTER bytes)"
        FAILED=$((FAILED+1)); continue
    fi
    say "  $LBL OK: +$(numfmt --to=iec $((AFTER - BEFORE)))"
done

[ "$FAILED" -eq 0 ] || die "$FAILED of ${#BATCHES[@]} batches failed. The env database is LEFT ON DISK
     so they can be retried without a 120 GB re-download."

# ---------------------------------------------------------------- 3. free the database
if [ "${FREE_DB:-0}" = "1" ]; then
    say "freeing the env database as asked (FREE_DB=1)"
    rm -rf "${DB:?}"/"${CFDB:?}"_db* "${DB:?}"/COLABDB_READY
fi
[ "${FREE_DB:-0}" = "1" ] || say "env database LEFT ON DISK so phase F can be retried without a
     120 GB re-download. Free it with FREE_DB=1 once phase F has passed."
say "=== phase E OK, all ${#BATCHES[@]} batches; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}') ==="
