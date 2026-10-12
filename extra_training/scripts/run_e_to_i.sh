#!/bin/bash
# Phases E through I, unattended. Ordering and escalation only -- every phase owns its own gates.
#
# WHERE THIS PICKS UP: the three query batches are already searched against uniref30 (5,305 + 235 +
# 11 alignments, phases D and its two top-ups). The env database was deliberately freed after the
# last phase F, so stage 0 brings it back before phase E can deepen anything.
#
# PHASE A RUNS ALONGSIDE, NOT IN THE CHAIN. Review round 3 widened its leakage search (short chains,
# peptide targets, full benchmark coverage), which can only shrink the 537-cluster pool. That changes
# which clusters may be VALIDATION clusters, so stage 3a consults phase A's fresh output before phase
# G builds the jobs, and prunes the split if a drawn cluster has since been excluded. Nothing has been
# folded yet, so pruning costs nothing.
#
# WHAT IT NEVER DOES: it does not delete a pose and it does not use --drop-cifs. It stops the PXM
# sweep only at stage 5, where phase I needs both cards, and phase I restores it on every exit path.
set -u
R=/home/ebrahim/tts_abag
P=/home/ebrahim/Protenix
PY=$P/.venv/bin/python
S=$R/extra_training/scripts
PH=$R/extra_training/phases
W=$R/extra_training/msa_work
DB=$R/tools/colabfold/db
LOG=$R/extra_training/run_e_to_i.log
STATE=$R/extra_training/run_e_to_i.state
CORES=0-95

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; echo "FAILED" > "$STATE"; exit 1; }
mark() { echo "$1" > "$STATE"; say "--> stage now: $1"; }

say "=== run_e_to_i starting (pid $$) ==="

# ---------------------------------------------------------------- resume: are the MSAs already done?
# STAGES 0-2 ARE SKIPPED WHEN PHASE F HAS FINISHED, and this is not an optimisation. Stage 0 is gated
# on the env database's data file, which phase F's success DELETES (160 GB freed, by design, because
# nothing downstream reads it). So a plain relaunch after a later stage failed would re-download 120
# GB and rebuild a database no remaining stage uses -- about 90 minutes to arrive back where it
# started. The completion test is phase F's own report against its own outputs: one renamed alignment
# per row, one directory with BOTH halves of the split per alignment.
# THE COMPLETION RECORD, NOT A COUNT. phase_f_report.csv is written in step 1 -- before taxids and
# before the split -- so its existence proves only that the headers were renamed. Comparing its row
# count against aggregate totals of pairing.a3m and non_pairing.a3m let a FAILED phase F pass the
# resume test: leftover files from an earlier attempt can make the totals agree while a specific
# expected chain has neither half, and stages E and F would then be skipped along with every gate
# they carry. phase_f_complete.json is written only after all of F's gates pass; it names the batches
# it consumed with their frozen-fasta checksums and lists every chain it produced, and each of those
# chains is verified individually here.
FREC=$W/pdb_seqs/phase_f_complete.json
FDONE=0
if [ -s "$FREC" ]; then
    FCHECK=$($PY - "$FREC" "$W" <<'PYF'
import hashlib, json, os, sys
rec, W = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(rec))
except (OSError, ValueError) as e:
    print(f"unreadable record: {e}"); sys.exit(1)
if d.get("version") != 1:
    print(f"record version {d.get('version')}, expected 1"); sys.exit(1)
want = [f"{W}/msa_initial_idx", f"{W}/msa_short_idx", f"{W}/msa_bench_short_idx"]
got = [b["dir"] for b in d.get("batches", [])]
if sorted(os.path.abspath(x) for x in want) != sorted(got):
    print(f"record covers {got}, chain needs {want}"); sys.exit(1)
# THE BATCHES MUST STILL BE THE ONES F CONSUMED. A re-searched batch invalidates the split.
for b in d["batches"]:
    sha = os.path.join(b["dir"], "query.searched.fasta.sha256")
    if b["sha256"] is None or not os.path.exists(sha):
        print(f"no frozen checksum for {b['dir']}"); sys.exit(1)
    if open(sha).read().split()[0] != b["sha256"]:
        print(f"{b['dir']} was re-searched since phase F ran"); sys.exit(1)
# EVERY CHAIN, INDIVIDUALLY, AND BY SIZE. This is the check a total cannot make -- and existence
# alone is not enough either: an interrupted rerun can leave every recorded file truncated to zero
# bytes, which `os.path.exists` happily accepts. The record carries the byte counts phase F produced,
# so any change is detectable without having to decide which emptiness is legitimate (a short peptide
# that retrieved nothing has a legitimately empty pairing.a3m).
msa = d["msa_dir"]; bad = []
ch = d.get("chains", {})
if not isinstance(ch, dict):
    print("record predates byte-count verification; re-run phase F"); sys.exit(1)
for name, sizes in ch.items():
    for half, want in zip(("pairing.a3m", "non_pairing.a3m"), sizes):
        f = os.path.join(msa, name, half)
        if not os.path.exists(f):
            bad.append(f"{name}/{half} absent")
        elif os.path.getsize(f) != want:
            bad.append(f"{name}/{half} is {os.path.getsize(f)}B, record says {want}B")
if bad:
    print(f"{len(bad)} output(s) changed since phase F, e.g. {bad[:3]}"); sys.exit(1)
print(f"ok {len(ch)} chains over {len(d['batches'])} batches, byte counts match")
PYF
)
    if [ $? -eq 0 ]; then
        FDONE=1
        say "resume: phase F complete and verified ($FCHECK) -- skipping stages 0-2."
        say "  The env database was freed when F passed and no later stage reads it, so rebuilding"
        say "  it would cost 120 GB and 90 minutes for nothing."
    else
        say "resume: phase F record present but NOT usable -- $FCHECK"
        say "  running stages 0-2 from the top; a partial phase F must not be skipped."
    fi
elif [ -s "$W/pdb_seqs/phase_f_report.csv" ]; then
    say "resume: phase_f_report.csv exists but phase_f_complete.json does not. The report is written
     in step 1, before taxids and the split, so this is an UNFINISHED phase F. Running stages 0-2."
fi

if [ "$FDONE" -eq 0 ]; then
mark "stage0_envdb"

# ---------------------------------------------------------------- stage 0: the env database
# GATED ON THE DATA FILE, not on COLABDB_READY. That marker survives the deletion of the database it
# describes -- it is still on disk right now while the database is not -- so a build gated on the
# marker would skip and phase E would search against nothing.
if [ -s "$DB/colabfold_envdb_202108_db_seq" ]; then
    say "stage 0: env database already on disk"
else
    say "stage 0: fetching and building the env database (~118 GB down, ~160 GB built)"
    rm -f "${DB:?}/COLABDB_READY"
    bash $S/phase2_envdb.sh all >> $LOG 2>&1 || die "env database fetch/build failed (msa_work/phase2_envdb.log)"
    [ -s "$DB/colabfold_envdb_202108_db_seq" ] || die "env build returned 0 but wrote no data file"
fi
say "stage 0 OK: $(du -sh $DB 2>/dev/null | cut -f1) in db/"

# ---------------------------------------------------------------- stage 1: phase E
mark "stage1_phaseE"
say "stage 1: phase E -- deepen all three batches against colabfold_envdb"
bash $PH/phase_e_search_envdb.sh $W/msa_initial_idx $W/msa_short_idx $W/msa_bench_short_idx \
    >> $LOG 2>&1 || die "phase E failed (msa_work/phase_e.log)"
say "stage 1 OK"

# ---------------------------------------------------------------- stage 2: phase F
mark "stage2_phaseF"
say "stage 2: phase F -- taxids, then the pairing/non_pairing split"
$PY $PH/phase_f_pair_and_split.py $W/msa_initial_idx $W/msa_short_idx $W/msa_bench_short_idx \
    >> $W/phase_f.log 2>&1 || die "phase F failed (msa_work/phase_f.log, phase_f_step*.log)"
NPAIR=$(find $W/mmcif_msa -name pairing.a3m 2>/dev/null | wc -l)
NNON=$(find $W/mmcif_msa -name non_pairing.a3m 2>/dev/null | wc -l)
say "stage 2 OK: $NPAIR pairing.a3m, $NNON non_pairing.a3m"

# FREE THE ENV DATABASE ONLY NOW. Phase F is the first reader of the env half of those alignments, so
# freeing 160 GB before it passed would make anything F reveals cost a 118 GB re-download.
if [ -s "$DB/colabfold_envdb_202108_db_seq" ]; then
    rm -rf "${DB:?}"/colabfold_envdb_202108_db* "${DB:?}"/COLABDB_READY
    say "env database freed after phase F; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
fi

fi   # end of the stages 0-2 block skipped on resume

# ---------------------------------------------------------------- stage 3a: the validation split
mark "stage3a_split"
say "stage 3a: reconciling the validation split with phase A's widened leakage search"
# ONE EXCLUSION ARTIFACT, PASSED TO EVERY PHASE THAT READS ONE. Writing the round-3 exclusions to a
# second file left B and G defaulting to a stale 24-cluster copy, and four known benchmark leaks
# survived in the committed training list. This is now the canonical path phase A writes by default,
# and it is passed explicitly here so the chain does not depend on a module default either.
AOUT=$R/extra_training/results/excluded_homolog_clusters.csv
[ -s "$AOUT" ] || die "no exclusion artifact at $AOUT -- run phase A before the chain"
$PY $PH/phase_b_define_val_clusters.py --prune --excluded "$AOUT" >> $W/phase_b.log 2>&1 \
    && say "stage 3a OK: split reconciled against $(basename $AOUT)" \
    || die "phase B could not reconcile the split against $AOUT (msa_work/phase_b.log). If it
     reports clusters in BOTH the training list and the exclusions, phase A must be re-run first."

# ---------------------------------------------------------------- stage 3: phase G
mark "stage3_phaseG"
say "stage 3: phase G -- validation job jsons"
$PY $PH/phase_g_build_val_jobs.py --reps 2 --msa-root $W/mmcif_msa --excluded "$AOUT" \
    --seq-index $W/pdb_seqs/seq_to_pdb_index.json --out-dir $R/extra_training/val_jobs \
    >> $W/phase_g.log 2>&1 || die "phase G failed (msa_work/phase_g.log)"
NJOB=$(ls $R/extra_training/val_jobs/*-final-updated.json 2>/dev/null | wc -l)
[ "$NJOB" -ge 1 ] || die "phase G wrote no *-final-updated.json; folder.py would find no jobs"
say "stage 3 OK: $NJOB job jsons"

# ---------------------------------------------------------------- stage 4: phase H
mark "stage4_phaseH"
say "stage 4: phase H -- templates, date-filtered against self-templating"
taskset -c $CORES timeout 21600 $PY $PH/phase_h_search_templates.py \
    --jobs-dir $R/extra_training/val_jobs --msa-dir $W/mmcif_msa \
    >> $W/phase_h.log 2>&1 || die "phase H failed (msa_work/phase_h.log)"
NTMPL=$(find $W/mmcif_msa -name hmmsearch.a3m 2>/dev/null | wc -l)
say "stage 4 OK: $NTMPL hmmsearch.a3m"

# ---------------------------------------------------------------- stage 5: phase I
# SMOKE FIRST, and the chain stops if it fails. One unit proves the job json, the trunk dump, all
# twelve cached tensors and OST DockQ before 1,540 units of GPU are committed.
mark "stage5_smoke"
say "stage 5: phase I smoke (one complex, one seed; PXM untouched)"
bash $PH/phase_i_fold_validation.sh smoke >> $LOG 2>&1 || die "phase I smoke failed (phase_i.log).
     Nothing was committed to the GPUs beyond one unit, and the PXM sweep is untouched."
say "stage 5 OK: smoke passed"

mark "stage6_fold"
say "stage 6: phase I run -- pausing PXM, folding on GPUs 0 and 1, restoring PXM afterwards"
bash $PH/phase_i_fold_validation.sh run >> $LOG 2>&1
RC=$?
say "phase I returned $RC"
mark "verify"
bash $PH/phase_i_fold_validation.sh verify >> $LOG 2>&1 && mark "COMPLETE" || mark "INCOMPLETE"
say "=== run_e_to_i finished: $(cat $STATE) ==="
