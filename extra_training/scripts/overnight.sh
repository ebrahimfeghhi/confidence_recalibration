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
# check and resume from each batch's kept uniref.a3m.
#
# THE GATE IS GROWTH, NOT A LOG LINE. What matters is that env sequences were actually appended to
# every query's alignment. Measuring the a3m set's total bytes before and after proves exactly that,
# and it also catches the failure the old string match was there for: if the pass had tried to
# re-search the deleted uniref30 it would have died and added nothing. Matching colabfold's own
# wording ("Skipping ... search") would break the moment that message is reworded, on a run that
# worked perfectly -- so it is logged as a note and never decides anything.
a3m_bytes() { find "$1" -maxdepth 1 -name '*.a3m' ! -name 'uniref.a3m' -printf '%s\n' 2>/dev/null \
                | awk '{s+=$1} END{print s+0}'; }
env_pass() {
    local fa=$1 out=$2 lbl=$3 before after nq na
    nq=$(grep -c '^>' "$fa")
    before=$(a3m_bytes "$out")
    say "stage 5: env pass for $lbl ($nq queries, a3m set $(numfmt --to=iec $before) before)"
    COLABFOLD_KEEP_INTERMEDIATES=1 COLABFOLD_EMIT_TAXID=1 taskset -c $CORES \
      $R/envs/colabfold/bin/colabfold_search --mmseqs $R/tools/mmseqs/bin/mmseqs --threads 64 \
      --use-env 1 --use-templates 0 --db-load-mode 0 \
      "$fa" "$DB" "$out" >> $W/phase2_search_$lbl.log 2>&1
    say "  colabfold_search exit $? (not trusted; the data decides)"
    grep -q "Skipping" $W/phase2_search_$lbl.log \
        && say "  note: the log mentions skipping a search, consistent with resuming from uniref.a3m"
    after=$(a3m_bytes "$out")
    na=$(find "$out" -maxdepth 1 -name '*.a3m' ! -name 'uniref.a3m' 2>/dev/null | wc -l)
    say "  $lbl: $na a3m, $(numfmt --to=iec $before) -> $(numfmt --to=iec $after)"
    [ "$na" -ge "$nq" ] || die "$lbl env pass left $na a3m for $nq queries"
    [ "$after" -gt "$before" ] || die "$lbl env pass added no sequences ($before -> $after bytes)"
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
# EXPECTATIONS COME FROM THE INPUT, not from numbers typed here: the two query fastas are what the
# whole chain is processing, so they define how many sequences every later stage must account for.
# A couple of queries legitimately drop out (the known one is an RNA antigen whose a3m does not match
# its fasta sequence), hence 98% rather than equality.
NQ=$(( $(grep -c '^>' $W/query.fasta) + $(grep -c '^>' $W/query_short.fasta) ))
NEXP=$(( NQ * 98 / 100 ))
say "stage 6: $NQ queries in the two fastas; later stages must account for at least $NEXP"
NIDX=$($PY -c "import json;print(len(json.load(open('$W/pdb_seqs/seq_to_pdb_index.json'))))" 2>/dev/null || echo 0)
[ "$NIDX" -ge "$NEXP" ] || die "the sequence index covers $NIDX of $NQ queries (want >= $NEXP)"
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
# A SAMPLE, AND A RATIO OVER THE FILES THE QUESTION APPLIES TO. A short peptide can legitimately
# retrieve no UniRef hits at all, and such a file has no header to tag -- counting it as a failure
# would fail the run on correct data. So the denominator is files that contain a UniRef100 header,
# and the gate is the fraction of those that step3 actually tagged.
# Every 97th file rather than the first few: the first files in the listing are not a sample, and a
# fault that hits one batch and not the other would hide behind them.
NSEEN=0; NTAGGED=0
for f in $(ls $W/mmcif_msa_with_taxid/*.a3m 2>/dev/null | awk 'NR % 97 == 1'); do
    grep -qm1 '^>UniRef100' "$f" || continue
    NSEEN=$((NSEEN+1))
    if grep -qm1 -E '^>UniRef100_.+_[0-9]+/' "$f"; then NTAGGED=$((NTAGGED+1)); fi
done
say "stage 6b: $NOUT files out; of $NSEEN sampled files with UniRef hits, $NTAGGED carry taxids"
[ "$NOUT" -ge "$NEXP" ] || die "step3 wrote $NOUT a3m for $NQ queries (want >= $NEXP)"
[ "$NSEEN" -ge 5 ] || die "only $NSEEN sampled a3m had any UniRef hits -- the env pass looks wrong"
[ $((NTAGGED * 100 / NSEEN)) -ge 90 ] \
    || die "only $NTAGGED of $NSEEN a3m with UniRef hits carry taxids -- step3 did not do its job"
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
[ "$NPAIR" -ge "$NEXP" ] || die "only $NPAIR pairing.a3m for $NQ queries (want >= $NEXP)"
say "stage 6 OK: $NPAIR pairing.a3m"

# ---------------------------------------------------------------- stage 7: validation job jsons
say "stage 7: validation job jsons"
cd $C || die "no code dir"
$PY build_val_jobs.py --reps $REPS --msa-root $W/mmcif_msa \
    --seq-index $W/pdb_seqs/seq_to_pdb_index.json --out-dir $R/extra_training/val_jobs \
    >> $W/val_jobs.log 2>&1 || die "build_val_jobs.py failed (see val_jobs.log)"
# build_val_jobs.py records every structure it picked in manifest.csv, including any it could not
# write a json for, so the manifest is the denominator -- no literal count to go stale when --reps or
# the cluster list changes.
NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
NPICK=$(($(wc -l < $R/extra_training/val_jobs/manifest.csv 2>/dev/null || echo 1)-1))
say "stage 7: $NJOB jsons for $NPICK picked structures"
[ "$NPICK" -ge 1 ] || die "manifest.csv is empty -- build_val_jobs.py picked nothing"
[ "$NJOB" -eq "$NPICK" ] || die "$NJOB jsons for $NPICK picked structures -- some chain lacks an MSA"
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
