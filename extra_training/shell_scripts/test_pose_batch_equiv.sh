#!/bin/bash
# EQUIVALENCE GATE FOR --pose-batch: batched and unbatched must train the SAME objective.
#
# WHY END TO END AND NOT A UNIT TEST ON THE HEAD. The head is already batched over samples, so a
# unit test on it would assert that torch broadcasting works. What can actually go wrong is
# everything AROUND it: the label slice (_cc[i] vs _cc[i:i+nb]), the PAE mask, the loss reduction
# across the sample axis, and whether a per-pose quantity got applied to a whole group. Only the
# real path exercises those, so this runs the real trainer twice on identical data and compares
# what it computed.
#
# THE COMPARISON IS THE LOSS AND THE GRADIENT, not the wall clock. A faster run that trains a
# slightly different objective is the worst possible outcome and would look like success.
#
# ONE CARD, NEVER 4/5/7.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
CARD="${CARD:-0}"
EPOCHS="${EPOCHS:-0.025}"
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

case ",4,5,7," in *",$CARD,"*) say "REFUSING: card $CARD is the live arm's"; exit 2;; esac
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2>/dev/null)
case "$U" in ''|*[!0-9]*) say "REFUSING: card $CARD unreadable"; exit 2;; esac
[ "$U" -gt 2000 ] && { say "REFUSING: card $CARD holds ${U} MiB"; exit 2; }

TRAIN_ONLY='^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.'

run_one () {
  rm -rf "$2"; mkdir -p "$2"
  CUDA_VISIBLE_DEVICES="$CARD" LAYERNORM_TYPE=openfold \
  $PY -I phases/phase_k_train.py \
    --crop-pool training/crop_final --minimal-items --epoch-cover-all \
    --pae-only --alpha-pae 1.0 --alpha-confidence 1.0 \
    --accum 240 --nsample 5 --pose-batch "$1" \
    --alignment-preflight training/alignment_preflight_sharded.json \
    --manifest-selection train_val_splits/cache_train_structures_extended.csv \
    --manifest-skip '4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m' \
    --fold training/fold --indices training/prep/all857_indices.csv \
    --pae-scored-pairs training/combined/pool.csv \
    --pae-scored-roles training/fold/chain_roles_all.csv \
    --lean-readouts --train-only "$TRAIN_ONLY" --expect-trainable 113 \
    --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single \
    --epochs "$EPOCHS" --lr 1e-3 --warmup-steps 0 --seed 0 --clip-norm 25.0 \
    --blocks-per-ckpt 1 --ckpt-token-threshold 1250 \
    --prefetch-units 2 --prefetch-workers 4 \
    --max-rss-gb 80 --preload-min-free-gb 100 --save-every 0 \
    --out "$2" --require-replay skip --allow-code-drift > "$2/log" 2>&1
  echo "  pose-batch $1 -> exit $?"
}

say "run A: --pose-batch 1 (the historical path)"
run_one 1 training/runs/pb_equiv_1
say "run B: --pose-batch 5 (one forward per unit)"
run_one 5 training/runs/pb_equiv_5

say "COMPARE"
$PY -I run_logging/compare_pose_batch.py
say "gate exit $?"
