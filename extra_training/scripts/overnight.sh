#!/bin/bash
# MSA + template preparation for the confidence-head finetune, then the validation job jsons.
#
# EVERY STAGE IS GATED ON CONTENT, never on an exit code. setup_databases.sh touches its READY marker
# and returns 0 even when it converted nothing, and `mmseqs dbtype <missing-db>` also returns 0, so a
# command that "succeeded" proves nothing. If a gate fails the chain stops, leaves the PXM sweep
# running, and says why.
#
# ORDER IS FORCED BY DISK. uniref30 (232 GB + a 302 GB index) and colabfold_envdb (about 250 GB) do
# not fit together in the 1,536 GB quota, so: fetch the env tarball, run the short-peptide uniref pass
# and free uniref30, only then extract the env database. The env database is kept until BOTH query
# batches have had their env pass.
#
# WHAT IT NEVER DOES: it does not touch the fold workers, it does not delete a pose, and it stops
# before any folding. see_it_through.sh takes it from there.
#
# USAGE: tmux new-session -d -s overnight "bash overnight.sh"
set -u

R=/home/ebrahim/tts_abag
P=/home/ebrahim/Protenix
PY=$P/.venv/bin/python
S=$R/extra_training/scripts
PH=$R/extra_training/phases
W=$R/extra_training/msa_work
DB=$R/tools/colabfold/db
LOG=$R/extra_training/overnight.log
PXM=$R/experimental_data/protenix/pxm2025/free_extra10
REPS=2
SEEDS=20
CORES=0-63

say()  { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die()  { say "STOP: $*"; say "PXM sweep left running. Nothing else started."; exit 1; }

say "=== overnight chain starting (reps=$REPS seeds=$SEEDS) ==="

# ---------------------------------------------------------------- stage 1: batch 1 already searched
say "stage 1: verifying the first uniref pass (5,305 chains of 20+ residues)"
while tmux has-session -t msa_search 2>/dev/null; do sleep 120; done
TAX=$W/msa_initial_idx/uniref_tax.m8
[ -s "$TAX" ] || die "uniref_tax.m8 missing or empty -- pairing impossible"
NTAX=$(awk -F'\t' 'NR<=2000 && $3+0>0 {n++} END{print n+0}' "$TAX")
[ "$NTAX" -gt 100 ] || die "uniref_tax.m8 has $NTAX usable taxids in its first 2000 rows"
NA3M=$(ls "$W"/msa_initial_idx/*.a3m 2>/dev/null | grep -vc '/uniref.a3m$')
[ "$NA3M" -ge 5000 ] || die "only $NA3M a3m from the first pass (expected 5305)"
[ -s "$W/msa_initial_idx/uniref.a3m" ] || die "uniref.a3m not kept -- the env pass cannot resume"
say "stage 1 OK: $NA3M a3m, real taxids, uniref.a3m kept"

# ---------------------------------------------------------------- stage 2: env tarball (download only)
say "stage 2: fetching the env database tarball (no extraction yet -- uniref30 needs the space)"
bash $S/phase2_envdb.sh download || die "env tarball download failed (phase2_envdb.log)"
say "stage 2 OK"

# ---------------------------------------------------------------- stage 3: batch 2, short antigens
# FoldBench contains protein chains under 20 aa and searches them normally, so the 235 antigen chains
# of 5-19 residues that the first pass skipped are searched too. This brings uniref30 back and frees
# it again; without it, 13 of the 50 validation clusters would have no antigen at all.
# GATED ON ITS OWN OUTPUTS, NOT ON THE DATABASE. phase1b_short.sh rebuilds uniref30 whenever
# uniref30 is absent -- which is its normal end state, since it deletes the database on the way out.
# So calling it unconditionally after a successful run re-downloads 100 GB and rebuilds a 450 GB
# database to redo work already on disk, about 1.5 h for nothing. The outputs that matter are the
# a3m and the kept intermediates phase E resumes from.
say "stage 3: short-antigen uniref pass"
NSHORT=$(ls $W/msa_short_idx/*.a3m 2>/dev/null | grep -vc '/uniref.a3m$')
if [ "$NSHORT" -ge 100 ] && [ -s "$W/msa_short_idx/uniref.a3m" ] && [ -s "$W/msa_short_idx/prof_res" ]; then
    say "stage 3 already done: $NSHORT a3m with uniref.a3m and prof_res kept; skipping the search"
else
    bash $S/phase1b_short.sh || die "short-antigen pass failed (phase1b_short.log)"
    NSHORT=$(ls $W/msa_short_idx/*.a3m 2>/dev/null | grep -vc '/uniref.a3m$')
    [ "$NSHORT" -ge 100 ] || die "only $NSHORT short-antigen a3m"
fi
[ -e "$DB/uniref30_2302_db_seq" ] && die "uniref30 is still on disk -- the env build will not fit"
say "stage 3 OK: $NSHORT short-antigen a3m, uniref30 not on disk"

# ---------------------------------------------------------------- stage 4: env database
say "stage 4: extracting the env database"
bash $S/phase2_envdb.sh build || die "env database build failed (phase2_envdb.log)"
[ -s "$DB/colabfold_envdb_202108_db_seq" ] || die "env database data files missing"
say "stage 4 OK: $(du -sh $DB 2>/dev/null | cut -f1) in db/"

# ================================================================ stages 5-8: one phase, one script
# Each phase script below owns its own gates -- the orchestrator's job is order and escalation, not
# re-checking what the phase already checked. See extra_training/phases/README.md.

# ---------------------------------------------------------------- stage 5: phase E, the env pass
say "stage 5: phase E -- deepen the alignments against colabfold_envdb"
bash $PH/phase_e_search_envdb.sh || die "phase E failed (msa_work/phase_e.log)"
say "stage 5 OK"

# ---------------------------------------------------------------- stage 6: phase F, taxids + split
# This is the phase that produces files Protenix can read; nothing downstream works without it.
say "stage 6: phase F -- taxids, then the pairing/non_pairing split"
$PY $PH/phase_f_pair_and_split.py >> $W/phase_f.log 2>&1 \
    || die "phase F failed (msa_work/phase_f.log and phase_f_step*.log)"
NPAIR=$(find $W/mmcif_msa -name pairing.a3m 2>/dev/null | wc -l)
say "stage 6 OK: $NPAIR chains with pairing.a3m"

# FREE THE ENV DATABASE HERE, NOT IN PHASE E. Phase F is the first thing that reads the env half of
# those alignments, so freeing 291 GB before it has passed would make anything F reveals cost a
# 120 GB re-download and another 90 minutes. Now that F has succeeded, the database has no further
# reader in this pipeline.
if [ -s "$DB/colabfold_envdb_202108_db_seq" ]; then
    rm -rf "${DB:?}"/colabfold_envdb_202108_db* "${DB:?}"/COLABDB_READY
    say "env database freed after phase F; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
fi

# ---------------------------------------------------------------- stage 7: phase G, the job jsons
say "stage 7: phase G -- validation job jsons"
$PY $PH/phase_g_build_val_jobs.py --reps $REPS --msa-root $W/mmcif_msa \
    --seq-index $W/pdb_seqs/seq_to_pdb_index.json --out-dir $R/extra_training/val_jobs \
    >> $W/phase_g.log 2>&1 || die "phase G failed (msa_work/phase_g.log)"
# ORCHESTRATION-LEVEL GATES: these compare phase G's two outputs against each other, which is the
# orchestrator's business rather than the phase's. manifest.csv is the denominator, so no literal
# count goes stale when --reps or the cluster list changes.
NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
NPICK=$(($(wc -l < $R/extra_training/val_jobs/manifest.csv 2>/dev/null || echo 1)-1))
[ "$NPICK" -ge 1 ] || die "manifest.csv is empty -- phase G picked nothing"
[ "$NJOB" -eq "$NPICK" ] || die "$NJOB jsons for $NPICK picked structures -- some chain lacks an MSA"
POOL=$R/extra_training/val_fold/pool.csv
[ -s "$POOL" ] || die "no val_fold/pool.csv -- the fold would score no interfaces"
NIF=$(($(wc -l < $POOL)-1))
NPDB_POOL=$(tail -n +2 $POOL | cut -d, -f1 | sort -u | wc -l)
[ "$NPDB_POOL" -eq "$NJOB" ] || die "pool.csv covers $NPDB_POOL structures but there are $NJOB jsons"
say "stage 7 OK: $NJOB jsons, $NIF interfaces over $NPDB_POOL structures"

# ---------------------------------------------------------------- stage 8: phase H, templates
say "stage 8: phase H -- templates (hmmsearch vs pdb_seqres)"
taskset -c $CORES timeout 21600 $PY $PH/phase_h_search_templates.py \
    --jobs-dir $R/extra_training/val_jobs --msa-dir $W/mmcif_msa \
    >> $W/phase_h.log 2>&1 || die "phase H failed (msa_work/phase_h.log)"
NTMPL=$(find $W/mmcif_msa -name hmmsearch.a3m 2>/dev/null | wc -l)
say "stage 8 OK: $NTMPL hmmsearch.a3m"

say "=== PREP COMPLETE. Folding is NOT started automatically. ==="
say "  bash $R/extra_training/phases/phase_i_fold_validation.sh smoke     then     ... run"
say "PXM sweep still running; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
