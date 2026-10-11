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
# uniref30's database is already gone and is not needed again: the env pass resumes from uniref.a3m,
# which the staging patch keeps ("Staging mode: uniref.a3m exists" in the search log).
# UNIREF30_READY MUST STAY. setup_databases.sh is re-entered here to build the env database, and
# without that marker it first tries to rebuild uniref30 from a tarball that no longer exists.
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
# step1 is NOT run. It reads _entity_poly.pdbx_seq_one_letter_code_can from the mmCIF, which the
# SAbDab2 cifs do not carry (it skipped 4,550 of 4,555), and its sequences would be the full deposited
# constructs rather than the SAbDab2 chain sequences our MSAs were searched with. prep_step4_inputs.py
# writes the two index files from the query fasta instead, and gives each a3m its own query header --
# our ColabFold writes a constant ">101" in all 5,305, which would collapse step4's output into one
# directory.
say "stage 3a: per-query a3m headers + step4 index files"
$PY $R/extra_training/scripts/prep_step4_inputs.py --a3m-dir $W/msa_initial_idx \
    --fasta $W/query.fasta --out-dir $W/msa_initial_named --index-dir $W/pdb_seqs \
    >> $W/prep_step4.log 2>&1 || die "prep_step4_inputs.py failed (see prep_step4.log)"
[ -s "$W/pdb_seqs/seq_to_pdb_index.json" ] || die "prep_step4_inputs.py wrote no seq_to_pdb_index.json"
NNAMED=$(ls $W/msa_initial_named/*.a3m 2>/dev/null | wc -l)
[ "$NNAMED" -ge 5000 ] || die "only $NNAMED a3m carry a per-query header (expected ~5304)"
say "stage 3a OK: $NNAMED a3m renamed, $(grep -c . $W/pdb_seqs/seq_to_pdb_index.json 2>/dev/null) index file written"

say "stage 3b: step3 -- write NCBI taxids into the UniRef headers"
cd $P || die "no Protenix"
taskset -c $CORES timeout 14400 $PY scripts/msa/step3-uniref_add_taxid.py \
    --input_msa_dir $W/msa_initial_named --output_msa_dir $W/mmcif_msa_with_taxid \
    --num_workers 16 --mp_read_workers 16 >> $W/step3.log 2>&1 \
    || say "WARN step3 returned non-zero or timed out; see step3.log"
# GATE ON CONTENT, not on file count: step3 writes an output file whether or not it found a taxid, and
# a taxid-less run is exactly the silent failure that makes pairing impossible. A tagged header looks
# like ">UniRef100_A0A123_9606/".
NOUT=$(ls $W/mmcif_msa_with_taxid/*.a3m 2>/dev/null | wc -l)
NTAGGED=0
for f in $(ls $W/mmcif_msa_with_taxid/*.a3m 2>/dev/null | head -20); do
    grep -qE '^>UniRef100_.*_[0-9]+/' "$f" && NTAGGED=$((NTAGGED+1))
done
say "stage 3b: $NOUT files out, $NTAGGED of the first 20 carry taxid-tagged headers"
[ "$NOUT" -ge 5000 ] || die "step3 wrote only $NOUT a3m"
[ "$NTAGGED" -ge 15 ] || die "only $NTAGGED of 20 sampled a3m have taxid-tagged headers -- pairing would be empty"
say "stage 3b OK"

say "stage 3c: step4 -- split each a3m into pairing.a3m (UniRef100) and non_pairing.a3m"
taskset -c $CORES timeout 14400 $PY scripts/msa/step4-split_msa_to_uniref_and_others.py \
    --input_msa_dir $W/mmcif_msa_with_taxid --output_msa_dir $W/mmcif_msa \
    --seq_to_pdb_id $W/pdb_seqs/seq_to_pdb_id_entity_id.json \
    --seq_to_pdb_index $W/pdb_seqs/seq_to_pdb_index.json --num_workers 32 >> $W/step4.log 2>&1 \
    || say "WARN step4 returned non-zero or timed out; see step4.log"
NPAIR=$(find $W/mmcif_msa -name pairing.a3m 2>/dev/null | wc -l)
# ONE DIRECTORY PER QUERY. The old threshold of 100 would have passed the header-collision bug, which
# produces exactly one directory, so the gate is the query count.
[ "$NPAIR" -ge 5000 ] || die "only $NPAIR pairing.a3m -- expected one per query (~5304)"
NEMPTY=0
for f in $(find $W/mmcif_msa -name pairing.a3m | head -40); do
    [ "$(wc -l < "$f")" -le 2 ] && NEMPTY=$((NEMPTY+1))
done
say "stage 3 OK: $NPAIR pairing.a3m, $NEMPTY of the first 40 hold only the query"
[ "$NEMPTY" -le 20 ] || die "$NEMPTY of 40 pairing.a3m are empty -- taxid tagging did not take"

# ---------------------------------------------------------------- stage 4: validation job jsons
# BEFORE the template search, which walks the jsons to find each chain's MSA directory.
say "stage 4: validation job jsons"
cd $C || die "no code dir"
$PY build_val_jobs.py --reps $REPS --msa-root $W/mmcif_msa \
    --seq-index $W/pdb_seqs/seq_to_pdb_index.json --out-dir $R/extra_training/val_jobs \
    >> $W/val_jobs.log 2>&1 || die "build_val_jobs.py failed (see val_jobs.log)"
NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
[ "$NJOB" -ge 70 ] || die "only $NJOB validation jsons written (expected ~79)"
# pool.csv decides which interfaces a target=* unit is scored on. folder.py's interfaces_by_pdb()
# opens it unconditionally, so a missing or empty file means the fold scores nothing at all.
POOL=$R/extra_training/val_fold/pool.csv
[ -s "$POOL" ] || die "no val_fold/pool.csv -- the fold would score no interfaces"
NIF=$(($(wc -l < $POOL)-1))
NPDB_POOL=$(tail -n +2 $POOL | cut -d, -f1 | sort -u | wc -l)
say "stage 4: $NJOB jsons, pool.csv has $NIF interfaces over $NPDB_POOL structures"
[ "$NPDB_POOL" -eq "$NJOB" ] || die "pool.csv covers $NPDB_POOL structures but there are $NJOB jsons -- every folded structure needs at least one interface"
say "stage 4 OK"

# ---------------------------------------------------------------- stage 5: templates
# pxm2025 folds use templates (free_templates defaults True and their jsons carry real hmmsearch.a3m),
# so the validation set is built the same way. update_template_info() is only called from Protenix's
# batch_inference.py, never from the inference.py path folder.py uses, so it must be run here or every
# templatesPath would dangle. Single-process and up front: MSA directories are keyed by sequence, so
# two workers sharing a chain would otherwise run hmmsearch into the same path at once.
say "stage 5: templates (hmmsearch vs pdb_seqres) -- untimed, hard timeout"
export PROTENIX_ROOT_DIR=$R/extra_training/protenix_root
mkdir -p $PROTENIX_ROOT_DIR/search_database
cd $P || die "no Protenix"
taskset -c $CORES timeout 21600 $PY $R/extra_training/scripts/run_template_search.py \
    --jobs-dir $R/extra_training/val_jobs >> $W/template.log 2>&1 \
    || die "template search incomplete (see template.log)"
NTMPL=$(find $W/mmcif_msa -name hmmsearch.a3m 2>/dev/null | wc -l)
say "stage 5 OK: $NTMPL hmmsearch.a3m written"

say "=== PREP COMPLETE. Folding is NOT started automatically. ==="
say "The validation fold needs a 1-fold smoke test (new job jsons + new trunk-dump path) before"
say "committing ~20 h of GPU and pausing the PXM sweep. Run:"
say "  bash $R/extra_training/scripts/val_fold.sh smoke"
say "then, if the dump and DockQ look right:"
say "  bash $R/extra_training/scripts/val_fold.sh run"
say "PXM sweep still running; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2}')"
