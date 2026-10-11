#!/bin/bash
# Generate 10 NEW free seeds for every complex in FoldBench then PXM-2025, on two GPUs.
#
# SEPARATE ARM, NOT THE POOL. Output goes to <dataset>/free_extra10/, never into the canonical free
# pool: the pool is the baseline every past analysis quotes as "free 50", and silently making it 60
# would retroactively change published numbers. Merging is a later, explicit decision.
#
# SEED BLOCKS are chosen clear of everything in use (FoldBench 101-200, PXM 300000-300129), so these
# folds can never be confused with, or collide against, an existing arm:
#   FoldBench 501-510        PXM-2025 300501-300510
#
# SEQUENTIAL BY DATASET, both GPUs on one queue via atomic claims -- not one dataset per GPU. Two
# GPUs on one queue finishes both in ~31 h; splitting by dataset leaves one card on the longer
# dataset for ~37 h.
set -u
C=/home/ebrahim/tts_abag/diffusion_commit_clean_v2
PY=/home/ebrahim/Protenix/.venv/bin/python
ROOT=/home/ebrahim/tts_abag/experimental_data/protenix
GPUS="${GPUS:-0 1}"
LOG=$ROOT/free_extra10_chain.log
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait for the pose-branch run to release the cards. It is 95/120 and resumable from its claims, so
# we never kill it; we queue behind it.
while tmux has-session -t pb2_0 2>/dev/null || tmux has-session -t pb2_1 2>/dev/null; do sleep 60; done
say "pose-branch drained; starting generation"

for DS in foldbench pxm2025; do
  E=$ROOT/$DS/free_extra10
  TOTAL=$(wc -l < $E/queue.tsv)
  say "PHASE $DS: $TOTAL folds -> $E"
  for g in $GPUS; do
    tmux new-session -d -s fx_${DS}_$g "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$g \
      LAYERNORM_TYPE=openfold $PY $C/folder.py --queue $E/queue.tsv --out $E/poses \
      --csv $E/results.csv --claim-dir $E/claims --dataset $DS --ncycle 10 --nsample 5 \
      --dtype bf16 --overlap-scoring >> $E/w$g.log 2>&1"
    say "  launched worker on GPU $g"
  done
  while :; do
    alive=0
    for g in $GPUS; do tmux has-session -t fx_${DS}_$g 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 60
  done
  say "PHASE $DS complete: $(find $E/poses -name run.json 2>/dev/null | wc -l)/$TOTAL folds"
done
say "ALL PHASES COMPLETE"
touch $ROOT/free_extra10_DONE
