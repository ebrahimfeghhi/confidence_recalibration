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
C=$R/diffusion_commit_clean_v2
S=$R/extra_training/scripts
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
say "stage 3: short-antigen uniref pass (brings uniref30 back, then frees it)"
bash $S/phase1b_short.sh || die "short-antigen pass failed (phase1b_short.log)"
NSHORT=$(ls $W/msa_short_idx/*.a3m 2>/dev/null | grep -vc '/uniref.a3m$')
[ "$NSHORT" -ge 100 ] || die "only $NSHORT short-antigen a3m"
[ -e "$DB/uniref30_2302_db_seq" ] && die "uniref30 is still on disk -- the env build will not fit"
say "stage 3 OK: $NSHORT short-antigen a3m, uniref30 removed"

# ---------------------------------------------------------------- stage 4: env database
say "stage 4: extracting the env database"
bash $S/phase2_envdb.sh build || die "env database build failed (phase2_envdb.log)"
[ -s "$DB/colabfold_envdb_202108_db_seq" ] || die "env database data files missing"
say "stage 4 OK: $(du -sh $DB 2>/dev/null | cut -f1) in db/"

# ---------------------------------------------------------------- stage 5: env pass, both batches
# COLABFOLD_KEEP_INTERMEDIATES=1 is what lets the staging patch drop uniref30 from the precondition
# check and resume from each batch's kept uniref.a3m. The "Skipping" line is the proof it resumed
# rather than silently trying to re-search a database that is gone.
env_pass() {
    local fa=$1 out=$2 lbl=$3
    say "stage 5: env pass for $lbl"
    COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c $CORES \
      $R/envs/colabfold/bin/colabfold_search --mmseqs $R/tools/mmseqs/bin/mmseqs --threads 64 \
      --use-env 1 --use-templates 0 --db-load-mode 0 \
      "$fa" "$DB" "$out" >> $W/phase2_search_$lbl.log 2>&1
    say "  colabfold_search exit $? (not trusted; checking the log)"
    grep -q "Skipping uniref30_2302_db search" $W/phase2_search_$lbl.log \
        || die "$lbl env pass did not resume from uniref.a3m"
    say "  $lbl OK"
}
env_pass "$W/query.fasta"       "$W/msa_initial_idx" main
env_pass "$W/query_short.fasta" "$W/msa_short_idx"   short
rm -rf "$DB"/colabfold_envdb_202108_db* "$DB"/COLABDB_READY
say "stage 5 OK: both env passes merged; env database removed; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2}')"

# ---------------------------------------------------------------- stage 6: taxids + pairing split
# step1 is NOT used: it reads _entity_poly.pdbx_seq_one_letter_code_can, which the SAbDab2 cifs do not
# carry (4,550 of 4,555 skipped), and its sequences are the deposited constructs rather than the
# chains we searched. prep_step4_inputs.py writes step4's two index files from the query fastas and
# gives each a3m its own query header -- our ColabFold writes a constant ">101" in every file, which
# would make step4 resolve one key for all of them and collapse its output into a single directory.
say "stage 6a: per-query a3m headers + step4 index files (both batches)"
rm -f $W/pdb_seqs/seq_to_pdb_index.json $W/pdb_seqs/seq_to_pdb_id_entity_id.json \
      $W/pdb_seqs/prep_step4_report.csv
$PY $S/prep_step4_inputs.py --a3m-dir $W/msa_initial_idx --fasta $W/query.fasta \
    --out-dir $W/msa_named_main --index-dir $W/pdb_seqs >> $W/prep_step4.log 2>&1 \
    || die "prep_step4_inputs.py failed on the main batch (prep_step4.log)"
$PY $S/prep_step4_inputs.py --a3m-dir $W/msa_short_idx --fasta $W/query_short.fasta \
    --out-dir $W/msa_named_short --index-dir $W/pdb_seqs >> $W/prep_step4.log 2>&1 \
    || die "prep_step4_inputs.py failed on the short batch (prep_step4.log)"
NIDX=$($PY -c "import json;print(len(json.load(open('$W/pdb_seqs/seq_to_pdb_index.json'))))" 2>/dev/null || echo 0)
[ "$NIDX" -ge 5400 ] || die "the sequence index covers only $NIDX sequences (expected ~5540)"
say "stage 6a OK: $NIDX sequences indexed"

say "stage 6b: step3 -- write NCBI taxids into the UniRef headers (once per batch)"
cd $P || die "no Protenix"
for b in main short; do
    taskset -c $CORES timeout 14400 $PY scripts/msa/step3-uniref_add_taxid.py \
        --input_msa_dir $W/msa_named_$b --output_msa_dir $W/mmcif_msa_with_taxid \
        --num_workers 16 --mp_read_workers 16 >> $W/step3_$b.log 2>&1 \
        || say "WARN step3 on $b returned non-zero or timed out; see step3_$b.log"
done
NOUT=$(ls $W/mmcif_msa_with_taxid/*.a3m 2>/dev/null | wc -l)
NTAGGED=0
for f in $(ls $W/mmcif_msa_with_taxid/*.a3m 2>/dev/null | head -20); do
    grep -qE '^>UniRef100_.*_[0-9]+/' "$f" && NTAGGED=$((NTAGGED+1))
done
say "stage 6b: $NOUT files out, $NTAGGED of the first 20 carry taxid-tagged headers"
[ "$NOUT" -ge 5400 ] || die "step3 wrote only $NOUT a3m"
[ "$NTAGGED" -ge 15 ] || die "only $NTAGGED of 20 sampled a3m have taxid-tagged headers"
say "stage 6b OK"

say "stage 6c: step4 -- split each a3m into pairing.a3m and non_pairing.a3m"
taskset -c $CORES timeout 14400 $PY scripts/msa/step4-split_msa_to_uniref_and_others.py \
    --input_msa_dir $W/mmcif_msa_with_taxid --output_msa_dir $W/mmcif_msa \
    --seq_to_pdb_id $W/pdb_seqs/seq_to_pdb_id_entity_id.json \
    --seq_to_pdb_index $W/pdb_seqs/seq_to_pdb_index.json --num_workers 32 >> $W/step4.log 2>&1 \
    || say "WARN step4 returned non-zero or timed out; see step4.log"
NPAIR=$(find $W/mmcif_msa -name pairing.a3m 2>/dev/null | wc -l)
# ONE DIRECTORY PER SEQUENCE. A threshold of 100 would have passed the header-collision bug, which
# produces exactly one directory, so the gate is the sequence count.
[ "$NPAIR" -ge 5400 ] || die "only $NPAIR pairing.a3m -- expected one per sequence (~5540)"
say "stage 6 OK: $NPAIR pairing.a3m"

# ---------------------------------------------------------------- stage 7: validation job jsons
say "stage 7: validation job jsons"
cd $C || die "no code dir"
$PY build_val_jobs.py --reps $REPS --msa-root $W/mmcif_msa \
    --seq-index $W/pdb_seqs/seq_to_pdb_index.json --out-dir $R/extra_training/val_jobs \
    >> $W/val_jobs.log 2>&1 || die "build_val_jobs.py failed (see val_jobs.log)"
NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
[ "$NJOB" -ge 70 ] || die "only $NJOB validation jsons written (expected 79)"
POOL=$R/extra_training/val_fold/pool.csv
[ -s "$POOL" ] || die "no val_fold/pool.csv -- the fold would score no interfaces"
NIF=$(($(wc -l < $POOL)-1))
NPDB_POOL=$(tail -n +2 $POOL | cut -d, -f1 | sort -u | wc -l)
say "stage 7: $NJOB jsons, pool.csv has $NIF interfaces over $NPDB_POOL structures"
[ "$NPDB_POOL" -eq "$NJOB" ] || die "pool.csv covers $NPDB_POOL structures but there are $NJOB jsons"
say "stage 7 OK"

# ---------------------------------------------------------------- stage 8: templates
# update_template_info() is called only from Protenix's batch_inference.py, never from the
# inference.py path folder.py uses, so without this every templatesPath would dangle. pxm2025 folds
# use templates, so the validation set is built the same way. Once and single-process: MSA
# directories are keyed by sequence, so two fold workers sharing a chain would otherwise run
# hmmsearch into the same path at the same time.
say "stage 8: templates (hmmsearch vs pdb_seqres)"
export PROTENIX_ROOT_DIR=$R/extra_training/protenix_root
mkdir -p $PROTENIX_ROOT_DIR/search_database
cd $P || die "no Protenix"
taskset -c $CORES timeout 21600 $PY $S/run_template_search.py \
    --jobs-dir $R/extra_training/val_jobs >> $W/template.log 2>&1 \
    || die "template search incomplete (see template.log)"
NTMPL=$(find $W/mmcif_msa -name hmmsearch.a3m 2>/dev/null | wc -l)
say "stage 8 OK: $NTMPL hmmsearch.a3m written"

say "=== PREP COMPLETE. Folding is NOT started automatically. ==="
say "  bash $S/val_fold.sh smoke     then     bash $S/val_fold.sh run"
say "PXM sweep still running; quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"
