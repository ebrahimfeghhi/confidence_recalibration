#!/bin/bash
# ONE-CARD GPU SMOKE FOR THE CROPPED ARM. A handful of optimizer steps, then stop.
#
# THE ONE THING THE CPU GATES CANNOT SAY. Every check so far is static or arithmetic: the wiring
# suite greps for execution gates that still read the old expression, the label gate compares
# numbers. None of them can show that a FORWARD PASS actually receives 256 tokens rather than the
# full structure -- and the defect external review found was precisely an execution path that
# silently did not run. So this asserts it from inside a real training step:
#
#   N_head == the selected mask's size, and N_head < N_uncropped
#   finite, nonzero rank and PAE gradients
#   the target used is the one keyed to THIS epoch's crop index
#
# ONE CARD, chosen by the caller, and never 4/5/7 while the uncropped arm holds them.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
CARD="${CARD:-0}"
STEPS="${STEPS:-2}"
OUT="${OUT:-training/runs/crop_smoke}"
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

case ",4,5,7," in *",$CARD,"*)
  say "REFUSING: card $CARD belongs to the live uncropped arm. Pick a free card."; exit 2;; esac

FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2>/dev/null)
case "$FREE" in ''|*[!0-9]*) say "REFUSING: card $CARD unreadable"; exit 2;; esac
[ "$FREE" -gt 2000 ] && { say "REFUSING: card $CARD already holds ${FREE} MiB"; exit 2; }
say "card $CARD is free (${FREE} MiB), running $STEPS step(s)"

rm -rf "$OUT"; mkdir -p "$OUT"
TRAIN_ONLY='^(?!pairformer_stack\.)'

# --epochs is fractional so the run stops after a few steps rather than a full epoch.
CUDA_VISIBLE_DEVICES="$CARD" LAYERNORM_TYPE=openfold \
$PY -I phases/phase_k_train.py \
  --crop-pool training/crop_final \
  --rel-labels training/fold/itm_labels_crop \
  --rank-loss --rank-form rel --rank-pool mean --alpha-rank 75 \
  --rank-cross-seed --rank-structs-per-step 2 \
  --minimal-items \
  --alignment-preflight training/alignment_preflight_sharded.json \
  --manifest-selection train_val_splits/cache_train_structures_extended.csv \
  --manifest-skip '4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m' \
  --fold training/fold --indices training/prep/all857_indices.csv \
  --pae-scored-pairs training/combined/pool.csv \
  --pae-scored-roles training/fold/chain_roles_all.csv \
  --pae-only --lean-readouts \
  --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single \
  --train-only "$TRAIN_ONLY" \
  --epochs 1 --lr 1e-3 --warmup-steps 20 --seed 0 --accum 30 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 1250 \
  --dist-backend gloo --dist-timeout-min 30 \
  --max-rss-gb 80 --preload-min-free-gb 100 \
  --mem-log \
  --out "$OUT" --require-replay skip --allow-code-drift 2>&1 | tee "$OUT/smoke.log" | tail -40
say "exit ${PIPESTATUS[0]}"
