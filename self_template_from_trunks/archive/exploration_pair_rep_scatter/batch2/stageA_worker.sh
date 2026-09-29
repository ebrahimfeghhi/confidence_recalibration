#!/bin/bash
# usage: stageA_worker.sh <gpu>
GPU=$1; B=/home/ebrahim/tts_abag/pair_rep_scatter/batch2
cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2 && CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRUNK_DUMP_DIR=$B/trunks PROTENIX_DUMP_TRUNK_BF16=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 /home/ebrahim/Protenix/.venv/bin/python folder.py --queue $B/queue_free.tsv --out $B/free --csv $B/free/results.csv --claim-dir $B/free/claims --dataset foldbench --ncycle 10 --nsample 5 --dtype bf16 2>&1 | tee -a $B/free/w$GPU.log
