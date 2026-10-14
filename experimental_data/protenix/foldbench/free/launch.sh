#!/bin/bash
# The FoldBench free pool AND its compute-matched extras, one directory. $1 = GPU.
#
# free/ holds exactly the two seed blocks the experiment uses: the pool (101-150, whose scores live in
# ../pool.csv) and the matched extras (151-200, scored into this directory's results.csv). Every other
# seed that used to live here is in ../free_other_seeds/ -- see its README; keeping them apart is what
# stops matched_seeds() from reaching undocumented leftovers.
#
# queue.tsv is the remaining matched work: seeds 151-200 for complexes holding an interface below free
# top-ipTM 0.8, as many seeds per complex as the method spends donor folds there (capped at 50),
# decoy-containing complexes first then nocov, size ascending within each group. folder.py skips an
# (iface, seed, arm) already in results.csv, so a relaunch resumes.
#
# Settings match the pool's and the earlier matched batches': ncycle 10, nsample 5, bf16, no templates.
set -u
GPU=$1
M=/home/ebrahim/tts_abag/experimental_data/protenix/foldbench/free
# GPU HOLD, built in rather than remembered -- and LIVING ONLY AS LONG AS THIS WORKER. A shared
# machine reclaims an idle card within minutes, and this run pauses for CPU work (scoring, merges)
# often enough that the gap is real. But a holder with nothing folding behind it is squatting: an
# earlier version of this block put the holder in its own tmux session so it would OUTLIVE the
# worker, and `hold0/1/2` then sat on three idle cards for hours after the sweeps were stopped.
# The trap ties it to this process instead, so stopping the run releases the card in the same breath.
# 500 MB is a signal, not a reservation -- the real working set is 15-37 GB.
HOLDER=""
if [ "${HOLD:-1}" = 1 ]; then
    cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2
    CUDA_VISIBLE_DEVICES=$GPU /home/ebrahim/Protenix/.venv/bin/python hold_gpus.py \
        --gpus $GPU --gb 0.5 --minutes 0 >/dev/null 2>&1 &
    HOLDER=$!
    trap '[ -n "$HOLDER" ] && kill $HOLDER 2>/dev/null' EXIT INT TERM
fi
cd /home/ebrahim/tts_abag/diffusion_commit_clean_v2
CUDA_VISIBLE_DEVICES=$GPU LAYERNORM_TYPE=openfold PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PROTENIX_TIMING=1 \
  /home/ebrahim/Protenix/.venv/bin/python folder.py \
    --queue $M/queue.tsv --out $M --csv $M/results.csv --claim-dir $M/claims \
    --dataset foldbench --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring 2>&1 | tee -a $M/w$GPU.log
echo "WORKER $GPU EXIT $? $(date -Is)" | tee -a $M/w$GPU.log
