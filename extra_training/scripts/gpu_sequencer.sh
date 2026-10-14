#!/bin/bash
# Hand the three cards from one job to the next with no idle gap.
#   stage 1: wait for the cached-validation rescoring to finish
#   stage 2: FoldBench completion top-off (2,607 folds) -- resumable, claim-based
# The training fold is launched separately once phase H and the input gates pass; this script's job
# is only to keep the cards busy until then. Killing a fold worker is safe: completed folds persist
# and claimed-but-unfinished units are released by reconcile_stale_claims.py.
set -u
R=/home/ebrahim/tts_abag
A=$R/experimental_data/protenix/foldbench/free_top100b
LOG=$R/extra_training/train_prep/sequencer.log
say() { echo "[$(date '+%H:%M:%S')] $*" >> $LOG; }

say "waiting for rescore_0/1/3 to finish"
while tmux ls 2>/dev/null | grep -q '^rescore_'; do sleep 30; done
say "rescoring done; starting the FoldBench top-off ($(wc -l < $A/queue.tsv) folds)"

for G in 0 1 3; do
  tmux new-session -d -s fbtop_$G "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$G LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python $R/diffusion_commit_clean_v2/folder.py --queue $A/queue.tsv --out $A/poses --csv $A/results.csv --claim-dir $A/claims --dataset foldbench --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $A/w$G.log 2>&1"
  say "launched fbtop_$G on GPU $G"
done
say "top-off running; it yields to the training fold when that is ready"
