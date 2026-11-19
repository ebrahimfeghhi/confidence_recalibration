#!/usr/bin/env bash
# THE IN-SAMPLE (hard 280) METRICS PASS FOR ONE CHECKPOINT, outside a training run.
#
# WHY IT EXISTS: the trainer fires this pass itself at --extra-eval's `every`, but comparing two
# arms at epoch 3 when both were launched for more epochs needs it on demand. Every path and flag
# is taken from what the trainer's own pass recorded, not retyped: the architecture flags are the
# `pathway_flags` list each shard's meta.json stores, which is how a pass states which head it
# built.
#
# --pae-loss MAKES IT ALL FOUR METRICS in one forward: mean DockQ, frac acceptable, rho, and the
# PAE loss split good/bad/total. run_logging/topk_table.py then adds top-1 and top-5 from the same
# rows with no further GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CKPT:?set CKPT to the head checkpoint}"
: "${OUT:?set OUT to the shard directory}"
CARD="${CARD:-4}"
SHARD="${SHARD:-0}"
NSHARDS="${NSHARDS:-3}"
POOL="${POOL:-training/slice280/pool.csv}"
FOLD="${FOLD:-training/fold}"
ROLES="${ROLES:-training/slice280/chain_roles.csv}"
SEED_BASE="${SEED_BASE:-500000}"
BUDGET="${BUDGET:-15}"
FLAGS="${FLAGS:---frame-direction --pose-perblock --pose-perblock-copy --pose-readout --pose-readout-post-zero --pose-readout-prenorm}"

mkdir -p "$OUT"
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') in-sample pass: $(basename "$CKPT") shard $SHARD/$NSHARDS card $CARD"
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES="$CARD" \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_eval.py \
  --ckpt "$CKPT" --budget "$BUDGET" --out "$OUT/metric_shard${SHARD}.json" \
  --fold "$FOLD" --results "$FOLD/results.csv" \
  --pool "$POOL" --roles "$ROLES" --seed-base "$SEED_BASE" \
  --pair-iptm-only --pool-mean --pae-loss \
  --rows-out "$OUT/rows_s${SHARD}.csv" --shard "$SHARD" --nshards "$NSHARDS" \
  $FLAGS "$@"
