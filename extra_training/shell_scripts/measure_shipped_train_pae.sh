#!/bin/bash
# THE SHIPPED HEAD'S PAE LOSS OVER THE TRAINING SET -- the epoch-0 row that has always been blank.
#
# WHY IT MATTERS. Train PAE fell 2.7384 (ep1) -> 2.1007 (ep26) while held-out rose 2.6953 ->
# 3.0841. That is high bias AND high variance at once, which is NOT the regime where "reduce
# capacity" is the indicated fix. The 30-structure anchor reached train PAE 1.4324 in 2,254 steps
# with the SAME 19.8M trainable parameters; this run has taken 4,916 and sits at 2.1007. So the
# head is straining against 488 structures, not comfortably memorising them.
#
# WHAT THIS NUMBER DECIDES:
#   shipped train PAE ~2.95  ->  real fit was available and training took ~29% of it. The gap to
#                               held-out is memorisation, and restricting capacity is on-point.
#   shipped train PAE ~2.30  ->  2.1007 is near the floor. The mapping is not learnable from
#                               these features, and NEITHER capacity arm will help; the lever is
#                               pose features (the head sees only CB cdist -- no orientation) or
#                               a ranking loss.
# It also separates overfitting from distribution shift: if the shipped head scores much better on
# train than its 2.9756 on held-out, part of the "gap" was never overfitting at all, it was that
# the training structures are intrinsically easier.
#
# SETTINGS ARE THE TRAINING RUN'S, exactly -- same crop, same --pae-scored-pairs mask, same
# --pae-only stub, same alphas. A different crop would not be comparable to 2.1007.
# NO --resume, WHICH IS THE WHOLE POINT: phase_k_train.py:3258 reads `a.resume or "the shipped
# head"`, so an absent bundle measures the shipped weights.
#
# SINGLE PROCESS BY DESIGN (phase_k_train.py:630 refuses WORLD_SIZE > 1), so one card. At the
# 1.12 s/unit the epoch-26 held-out pass measured (300 units in 5.6 min), 4,880 units is ~90 min.
#
# DO NOT RUN THIS ALONGSIDE A TRAINER. It wants a card to itself; mid-epoch the cards sit at ~70
# of 97.9 GB and an OOM here would kill the RUN, not just the measurement.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
OUT=training/runs/shipped_train_pae
BUSY=$(ps -eo cmd | grep -cE "python3? -u .*phase_k_train\.py" || true)
if [ "$BUSY" -gt 0 ]; then
    echo "REFUSING: $BUSY training process(es) are live and this would share their cards."
    [ "${FORCE:-0}" = "1" ] || exit 1
fi
mkdir -p "$OUT"
LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES=7 \
/home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_train.py \
  --loss-only \
  --fold training/fold \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --alignment-preflight training/full_train/alignment_preflight.json --minimal-items \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --seeds 500000-500009 \
  --crop-margin 10 --crop-contact 5 --crop-size 3400 --accum 85 --nsample 5 \
  --alpha-pae 1.0 --alpha-confidence 1.0 --clip-norm 25.0 --blocks-per-ckpt 0 \
  --epochs 1 --save-every 0 --warmup-steps 0 \
  --shuffle structure --out "$OUT" \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift 2>&1 | tee "$OUT/measure.log"
echo "exit $?"
