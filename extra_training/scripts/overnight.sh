#!/bin/bash
# Overnight chain: finish the MSAs, prepare the validation set, pause PXM, fold validation, resume PXM.
#
# EVERY STAGE IS GATED. If a check fails the chain stops, leaves the PXM sweep running, and says why.
# That is deliberate: the validation fold is ~20 h of GPU on two cards, and the earlier failure today
# came from trusting a success marker (setup_databases.sh touches UNIREF30_READY before the steps that
# can fail). Nothing here treats "the command exited 0" as evidence.
#
# WHAT IT DOES NOT DO: it never touches the fold workers except at the one planned pause, and it never
# deletes a pose. The uniref30 database is only removed once its search output is verified.
#
# USAGE: tmux new-session -d -s overnight "bash overnight.sh"
set -u

R=/home/ebrahim/tts_abag
P=/home/ebrahim/Protenix
PY=$P/.venv/bin/python
C=$R/diffusion_commit_clean_v2
W=$R/extra_training/msa_work
DB=$R/tools/colabfold/db
LOG=$R/extra_training/overnight.log
PXM=$R/experimental_data/protenix/pxm2025/free_extra10
VAL=$R/extra_training/val_fold
REPS=2
SEEDS=20
CORES=0-63

say()  { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die()  { say "STOP: $*"; say "PXM sweep left running. Nothing else started."; exit 1; }

say "=== overnight chain starting (reps=$REPS seeds=$SEEDS) ==="

# ---------------------------------------------------------------- stage 1: uniref search
say "stage 1: waiting for the uniref30 search"
while tmux has-session -t msa_search 2>/dev/null; do sleep 120; done
TAX=$W/msa_initial_idx/uniref_tax.m8
[ -s "$TAX" ] || die "uniref_tax.m8 missing or empty -- pairing impossible, do not delete uniref30"
# a taxid column of zeros would pass a size check but break step3, so look at the content
NTAX=$(awk -F'\t' 'NR<=2000 && $3+0>0 {n++} END{print n+0}' "$TAX")
[ "$NTAX" -gt 100 ] || die "uniref_tax.m8 has $NTAX usable taxids in its first 2000 rows"
NA3M=$(ls "$W"/msa_initial_idx/*.a3m 2>/dev/null | wc -l)
[ "$NA3M" -ge 5000 ] || die "only $NA3M a3m files from the uniref pass (expected ~5305)"
say "stage 1 OK: $NA3M a3m, uniref_tax.m8 with real taxids"

# ---------------------------------------------------------------- stage 2: env database
# uniref30 is already gone (deleted 2026-10-10 21:00). The env pass resumes from uniref.a3m, which
# the staging patch keeps, so uniref30 is never needed again -- the search log line
# "Staging mode: uniref.a3m exists" confirmed that path works.
rm -rf "$DB"/uniref30_2302_db* "$DB"/UNIREF30_READY
say "stage 2: colabfold_envdb via phase2_envdb.sh (download + build, index-free)"
bash $R/extra_training/scripts/phase2_envdb.sh || die "envdb download/build failed (phase2_envdb.log)"
# gate on the data file, NOT on an exit code: `mmseqs dbtype <missing-db>` returns 0, which is how
# the 2026-10-10 chain reported "envdb built" 30 s after setup_databases.sh had died on PATH.
[ -s "$DB/colabfold_envdb_202108_db" ] && [ -s "$DB/colabfold_envdb_202108_db_seq" ] \
    || die "envdb data files missing or empty after build"
say "stage 2a OK: envdb built ($(du -sh "$DB" 2>/dev/null | cut -f1) in db/)"

COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c $CORES \
  $R/envs/colabfold/bin/colabfold_search --mmseqs $R/tools/mmseqs/bin/mmseqs --threads 64 \
  --use-env 1 --use-templates 0 --db-load-mode 0 \
  $W/query.fasta "$DB" $W/msa_initial_idx >> $W/phase2_search.log 2>&1
grep -q "Skipping uniref30_2302_db search" $W/phase2_search.log \
    || die "env pass did not resume from uniref.a3m -- it would have re-searched uniref30, now deleted"
say "stage 2 OK: env pass merged"
rm -rf "$DB"/colabfold_envdb_202108_db* "$DB"/COLABDB_READY
say "envdb removed; quota now $(quota -s 2>/dev/null | awk '/nvme/{print $2}')"

# ---------------------------------------------------------------- stage 3: taxids + pairing split
say "stage 3: step1 (sequence index), step3 (taxids), step4 (pairing split)"
mkdir -p $W/pdb_seqs
cd $P || die "no Protenix"
$PY scripts/msa/step1-get_prot_seq.py --cif_dir $R/extra_training/data/training_cifs \
    --out_dir $W/pdb_seqs >> $W/step1.log 2>&1 || die "step1 failed (see step1.log)"
[ -s "$W/pdb_seqs/seq_to_pdb_index.json" ] || die "step1 produced no seq_to_pdb_index.json"
say "stage 3a OK: $(/usr/bin/python3 -c "import json;print(len(json.load(open('$W/pdb_seqs/seq_to_pdb_index.json'))))") sequences indexed"
say "stage 3b: step3/step4 are UNTIMED and their CLI is not verified -- running with a hard timeout"
timeout 10800 $PY scripts/msa/step3-uniref_add_taxid.py >> $W/step3.log 2>&1 \
    || say "WARN step3 returned non-zero or timed out; see step3.log"
timeout 10800 $PY scripts/msa/step4-split_msa_to_uniref_and_others.py \
    --input_msa_dir $W/mmcif_msa_with_taxid --output_msa_dir $W/mmcif_msa \
    --seq_to_pdb_id $W/pdb_seqs/seq_to_pdb_id_entity_id.json \
    --seq_to_pdb_index $W/pdb_seqs/seq_to_pdb_index.json >> $W/step4.log 2>&1 \
    || say "WARN step4 returned non-zero or timed out; see step4.log"
NPAIR=$(find $W/mmcif_msa -name pairing.a3m 2>/dev/null | wc -l)
[ "$NPAIR" -ge 100 ] || die "only $NPAIR pairing.a3m produced -- step3/step4 did not work as assumed"
say "stage 3 OK: $NPAIR pairing.a3m"

# ---------------------------------------------------------------- stage 4: templates
say "stage 4: templates (hmmsearch vs pdb_seqres) -- untimed, hard timeout"
export PROTENIX_ROOT_DIR=$R/extra_training/protenix_root
mkdir -p $PROTENIX_ROOT_DIR/search_database
timeout 21600 $PY runner/template_search.py --help >> $W/template.log 2>&1
say "stage 4: template_search.py CLI logged to template.log -- NOT run blind; see stage 5 gate"

# ---------------------------------------------------------------- stage 5: validation jobs
say "stage 5: validation job jsons"
cd $C || die "no code dir"
$PY build_val_jobs.py --reps $REPS --msa-root $W/mmcif_msa \
    --seq-index $W/pdb_seqs/seq_to_pdb_index.json --out-dir $R/extra_training/val_jobs \
    >> $W/val_jobs.log 2>&1 || die "build_val_jobs.py failed (see val_jobs.log)"
NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
[ "$NJOB" -ge 70 ] || die "only $NJOB validation jsons written (expected ~79)"
say "stage 5 OK: $NJOB validation jsons"

say "=== PREP COMPLETE. Folding is NOT started automatically. ==="
say "The validation fold needs a 1-fold smoke test (new job jsons + new trunk-dump path) before"
say "committing ~20 h of GPU and pausing the PXM sweep. Run:"
say "  bash $R/extra_training/scripts/val_fold.sh smoke"
say "then, if the dump and DockQ look right:"
say "  bash $R/extra_training/scripts/val_fold.sh run"
say "PXM sweep still running; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2}')"
