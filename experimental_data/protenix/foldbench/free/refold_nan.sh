#!/bin/bash
# Re-fold 8xnh free seeds 180 and 193. Their original folds produced ENTIRELY NaN coordinates
# (17,661/17,661 atoms, all 5 poses, both seeds), so OST refused them ("No residues found") and they
# never reached results.csv -- which left 8xnh with 48 matched seeds for 50 donors and blocked
# build_varseed_queue.py, which is all-or-nothing by design.
# The NaN folds are preserved under .nan_folds_20261004/, not deleted.
# Config copied verbatim from the original free_compute_matched run (run.json argv): dataset foldbench,
# nsample 5, --overlap-scoring, defaults for ncycle (10) and dtype (bf16). FoldBench free folds are
# template-free (paths: free_templates=False), which the dataset handles.
set -u
GPU=$1; SEED=$2
M=/home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free_compute_matched
cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2
printf 'free\t8xnh\t*\tfree\t%s\t\n' "$SEED" > $M/queue_refold_$SEED.tsv
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  /home/ebrahim/Protenix/.venv/bin/python folder.py \
    --queue $M/queue_refold_$SEED.tsv --out $M --csv $M/results.csv \
    --claim-dir $M/claims_refold_$SEED \
    --dataset foldbench --nsample 5 --overlap-scoring 2>&1 | tee -a $M/refold_nan_$SEED.log
echo "REFOLD $SEED EXIT $? $(date -Is)" | tee -a $M/refold_nan_$SEED.log
