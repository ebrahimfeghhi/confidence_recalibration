#!/usr/bin/env bash
# Overnight: at13 (single late splice, 13 recycles) and periodic 2/5/8 (10 recycles) on 45 complexes
# x 15 seeds, covering 20 confident decoys, 20 solved and 5 no-coverage interfaces.
#
# ONE FOLD SCORES EVERY INTERFACE of its complex, so the queue lists complexes, not interfaces.
#
# at13 runs with NO observation decodes (--hook-observe ""): only at10 (the splice decode, which is
# the free baseline) and at13 (the fold's own final output) are wanted, so the at11/at12 decodes and
# their confidence-head calls are pure cost -- about 30% of the fold.
#
# PERIODIC FIRST at 50 seeds: it is the arm that beat both alternatives on 9uzh (0.541/0.559 against
# 0.433/0.422 for the late splice and 0.176/0.213 plain), and it raised the ORACLE too, so it is
# generating better structures rather than only reranking. at13 follows if there is time; its at10
# decode is the shared baseline, but the pool already provides a 10-recycle free baseline per seed.
set -u
O=/home/ebrahim/tts_abag/experimental_data/overnight
PY=/home/ebrahim/Protenix/.venv/bin/python
G=${G:?set G to the GPU index}
cd /home/ebrahim/tts_abag/diffusion_commit || exit 1
for arm in per258 at13; do
  if [ "$arm" = at13 ]; then HOOK=(--ncycle 13 --hook self --hook-after 9 --hook-observe "")
  else                       HOOK=(--ncycle 10 --hook self --hook-splice 2,5,8); fi
  echo "$(date +%H:%M:%S) gpu$G starting $arm"
  CUDA_VISIBLE_DEVICES=$G LAYERNORM_TYPE=openfold $PY fold.py \
    --queue "$O/queue_${arm}.tsv" --out "$O/$arm" --csv "$O/${arm}.csv" \
    --claim-dir "$O/.claims_${arm}" --dtype bf16 "${HOOK[@]}" --drop-cifs \
    >> "$O/${arm}_gpu${G}.log" 2>&1
  echo "$(date +%H:%M:%S) gpu$G finished $arm (rc=$?)"
done
