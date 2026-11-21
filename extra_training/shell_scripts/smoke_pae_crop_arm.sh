#!/bin/bash
# ONE-CARD GPU SMOKE FOR THE PAE-ONLY CROPPED ARM. Hard 15-minute cap.
#
# THE ONE THING NO CPU GATE CAN SAY. Every check so far is static or arithmetic: the wiring suite
# greps execution gates, the coverage gate simulates the epoch plan, the crop/pair gate replays
# iptm_pooled's None branches. None of them can show that a real backward produces a nonzero
# finite PAE gradient on this configuration -- and "the loss computes" and "the gradient flows"
# are different claims. External review called this out and it is this project's recurring
# failure: an arm that logs happily while training nothing.
#
# ASSERTS, FROM INSIDE A REAL STEP:
#   --expect-trainable 113 accepted (the wrong-tensor-set failure, caught at startup)
#   the head receives 256 tokens, not the full structure
#   grad_norm finite and > 0 on a step that was actually taken
#   no STOP, no Traceback, no nonfinite flag
#
# ONE CARD, AND NEVER 4/5/7 WHILE THE UNCROPPED ARM HOLDS THEM.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
CARD="${CARD:-0}"
CAP="${CAP:-900}"
OUT="${OUT:-training/runs/pae_crop_smoke}"
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

case ",4,5,7," in *",$CARD,"*)
  say "REFUSING: card $CARD belongs to the live arm. Pick a free card."; exit 2;; esac
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2>/dev/null)
case "$U" in ''|*[!0-9]*) say "REFUSING: card $CARD unreadable"; exit 2;; esac
[ "$U" -gt 2000 ] && { say "REFUSING: card $CARD already holds ${U} MiB"; exit 2; }
say "card $CARD free (${U} MiB), hard cap ${CAP}s"

rm -rf "$OUT"; mkdir -p "$OUT"
# THE SAME MANIFEST AND SKIP LIST AS THE REAL ARM. A reduced manifest changes the digest and the
# alignment preflight refuses it -- which cost four failed launches in twenty minutes once.
TRAIN_ONLY='^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.'

CUDA_VISIBLE_DEVICES="$CARD" LAYERNORM_TYPE=openfold \
timeout "$CAP" $PY -I phases/phase_k_train.py \
  --crop-pool training/crop_final \
  --minimal-items --epoch-cover-all \
  --pae-only --alpha-pae 1.0 --alpha-confidence 1.0 \
  --accum 240 --nsample 5 \
  --alignment-preflight training/alignment_preflight_sharded.json \
  --manifest-selection train_val_splits/cache_train_structures_extended.csv \
  --manifest-skip '4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m' \
  --fold training/fold --indices training/prep/all857_indices.csv \
  --pae-scored-pairs training/combined/pool.csv \
  --pae-scored-roles training/fold/chain_roles_all.csv \
  --lean-readouts --train-only "$TRAIN_ONLY" --expect-trainable 113 \
  --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single \
  --epochs 0.02 --lr 1e-3 --warmup-steps 2 --seed 0 --clip-norm 25.0 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 1250 --mem-log \
  --prefetch-units 2 --prefetch-workers 4 \
  --max-rss-gb 80 --preload-min-free-gb 100 --save-every 0 \
  --out "$OUT" --require-replay skip --allow-code-drift > "$OUT/smoke.log" 2>&1
RC=$?
say "trainer exited $RC (124 = hit the ${CAP}s cap, which is a PASS if the gates below are green)"
