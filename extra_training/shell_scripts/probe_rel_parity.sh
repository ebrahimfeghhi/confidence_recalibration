#!/bin/bash
# GRADIENT PARITY PROBE for --rank-form rel. Measures, does not train.
#
# WHAT IT REPORTS, from the trainer's own --grad-probe over EXACTLY ONE accumulation window at
# identical weights: |g_PAE|, |g_rel| (printed as |g_rank| -- rel reuses that path), their ratio,
# cos(PAE, rel), the combined norm against the clip threshold, and the alpha that would give
# parity. Per-pose rows are printed too, which is what answers whether the accumulated norm is
# small because the term is weak or because it CANCELS.
#
# WHY THE NORM AND NOT THE LOSS VALUE. rel's coefficients sum to exactly zero, so its accumulated
# gradient is a sum of terms that cancel by construction, and its loss value (order 1e-2) is ~180x
# below PAE's (order 2.0). Setting alpha from loss values would be meaningless; setting it from
# the accumulated norm is the measurement that matters. See phases/rank_loss.py.
#
# ONE WINDOW ONLY, and that is the point: a probe that runs past the step mixes gradients measured
# at two different parameter states, and the whole purpose is to compare the two terms at
# IDENTICAL weights.
#
# RUN IT SEVERAL TIMES and take the MEDIAN ratio -- a single window is one draw of structures.
# --seed is exposed for that; everything else is held at the launch configuration.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
SEED="${1:-0}"
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
$PY_BIN -I run_logging/test_rel_itm_loss.py || {
  echo "REFUSING: the rel-loss gate failed"; exit 1; }
echo "probing gradient parity for --rank-form rel, seed $SEED"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "GRADIENT PARITY PROBE for --rank-form rel (mean-centred true-iTM regression) against the PAE term, at identical weights over ONE accumulation window. Measures only -- no optimiser step, no checkpoint. Reports |g_PAE|, |g_rel|, their ratio, cos(PAE,rel), the combined norm against the clip threshold, the parity alpha, and per-pose rows. The per-pose rows are the point as much as the ratio: rel's coefficients sum to EXACTLY zero, so its accumulated gradient cancels by construction and a small accumulated norm does NOT imply small per-pose gradients. Pooling is MEAN over anchor rows in both pass A and pass B, with labels read from the g_mean column so the target matches the quantity produced. Everything else is the launch configuration: 368 winnable structures, crop 3400, accum 85, nsample 5, seeds 500000-500009, pose-copy injection, --frame-direction, no L2." \
  -- --grad-probe \
  --rank-loss --rank-form rel --rank-cross-seed --alpha-rank 1.0 \
  --rank-pool mean --rel-labels training/fold/itm_labels \
  --rank-structs-per-step 3 \
  --pose-perblock --pose-perblock-copy --frame-direction \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
  --manifest-selection training/full_train/structures_winnable.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 1 --lr 0.01 --warmup-steps 20 --seed "$SEED" --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "probe_rel_parity (seed $SEED) returned $rc"
exit $rc
