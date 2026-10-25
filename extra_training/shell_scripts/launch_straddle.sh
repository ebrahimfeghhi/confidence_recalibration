#!/bin/bash
# STRADDLE-ONLY RANKING LOSS ON THE FULL TRAINING FOLD.
#
# Usage:  ./shell_scripts/launch_straddle.sh <lambda>
#
# THE OBJECTIVE. A pair counts only when exactly one of its two poses has DockQ above 0.23:
# --rank-straddle 0.23. The reported criterion is the FRACTION of interfaces whose selected pose
# clears 0.23, and that number moves ONLY when selection crosses the threshold -- two acceptable
# poses, or two failures, can be reordered freely without changing it. So 87.5% of the pairs the
# loss used to spend its weight on cannot affect the metric, and this aims the gradient at the
# subspace that can.
#
# WHAT IT COSTS, stated rather than hidden. Mean DockQ is NOT invariant to the dropped pairs:
# promoting 0.61 over 0.38 raises the mean and leaves the fraction untouched, and that gradient
# is discarded. This arm optimises the FRACTION and treats mean DockQ as a free variable. Both
# are reported side by side because they can now move in opposite directions.
#
# AND 590 OF 1028 INTERFACES GO SILENT. Only 438 are usable: the rest have every pose on one
# side of 0.23 -- all acceptable, or all failures -- so they form no straddling pair and
# contribute nothing to the loss. That is the single biggest risk in this arm and the first
# thing to check if it fails.
#
# BOTH FAMILIES, cross-seed AND within-seed: --rank-pairs all at within_weight 1. The straddle
# restriction is orthogonal to the family split, and a within-seed pair that crosses 0.23 is
# exactly as selection-relevant as a cross-seed one.
#
# TAU IS 0.1429, the median |ipTM gap| over the pairs THIS ARM ACTUALLY TRAINS -- all-family,
# straddling, full 497-structure fold, after the DockQ-tie filter (measure_tau.py --pairs all
# --straddle 0.23). It is 4.7x DEFAULT_TAU's 0.0305 and that is not an error: poses on opposite
# sides of 0.23 are genuinely different structures and the shipped head already separates them
# more. Training these pairs at 0.0305 would put nearly all of them deep in saturation.
#
# NO ORIENTATION, NO PAE. --frame-direction is dropped, so the trainable set is 8 tensors /
# 40,960 parameters: linear_no_bias_d + pose_perblock, distance injected at all four pairformer
# blocks and nothing else. --alpha-pae 0: the PAE term is gone, which the oe6sru verdict
# supports (paired -0.0209, positive 0 of 8, against a +0.0116 threshold fixed in advance).
#
# L2-SP COVERS EVERY TRAINABLE TENSOR. There is no --l2-sp-exclude because the only thing
# zg5fnw excluded was frame_dir, and frame_dir is not trained here. The anchored set is
# therefore 40,960 parameters, the SAME count zg5fnw anchored, so l2sp_lambda.py's size factor
# is exactly 1 and lambda transfers on the measured gradient ratio alone.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
LAM="${1:-}"
[ -n "$LAM" ] || { echo "usage: $0 <lambda>   (measured by probe_straddle.sh, never defaulted)"; exit 1; }
BUSY=$($PY_BIN -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY process(es) occupy cards 4,5,7."; exit 1; }
echo "launching straddle arm: lambda=$LAM"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "STRADDLE-ONLY RANKING LOSS ON THE FULL TRAINING FOLD, lambda $LAM, 15 epochs. THE OBJECTIVE: --rank-straddle 0.23 keeps a pair only when exactly one of its two poses has DockQ above 0.23. The reported criterion is the FRACTION of interfaces whose selected pose clears 0.23, and that number moves ONLY when selection crosses the threshold -- two acceptable poses, or two failures, can be reordered freely without changing it, so 87.5% of the pairs the loss used to weight cannot affect the metric. This aims the gradient at the subspace that can. 143,680 of 1,146,184 pairs survive. WHAT IT COSTS, stated rather than hidden: mean DockQ is NOT invariant to the dropped pairs -- promoting 0.61 over 0.38 raises the mean and leaves the fraction untouched -- so this optimises the FRACTION and treats mean DockQ as a free variable; both are reported because they can now move in opposite directions. AND 590 OF 1028 INTERFACES GO SILENT: only 438 are usable, the rest having every pose on one side of 0.23, so they form no straddling pair and contribute nothing. That is the biggest risk here and the first thing to check if it fails. BOTH FAMILIES, cross-seed AND within-seed (--rank-pairs all at within_weight 1): the straddle restriction is orthogonal to the family split and a within-seed pair crossing 0.23 is exactly as selection-relevant as a cross-seed one. TAU IS 0.1429, the median |ipTM gap| over the pairs THIS ARM ACTUALLY TRAINS (measure_tau.py --pairs all --straddle 0.23, full 497-structure fold, after the DockQ-tie filter). It is 4.7x DEFAULT_TAU and that is not an error: poses on opposite sides of 0.23 are genuinely different and the shipped head already separates them more; training them at 0.0305 would put nearly all of them deep in saturation. NO ORIENTATION, NO PAE: --frame-direction is dropped so the trainable set is 8 tensors / 40,960 parameters -- linear_no_bias_d + pose_perblock, distance at all four pairformer blocks and nothing else -- and --alpha-pae 0 removes the PAE term, which the oe6sru verdict supports (paired -0.0209, positive 0 of 8, against a +0.0116 threshold fixed in advance). L2-SP COVERS EVERY TRAINABLE TENSOR, no --l2-sp-exclude, because the only tensors zg5fnw excluded were frame_dir and frame_dir is not trained here; the anchored set is 40,960 parameters, the SAME count zg5fnw anchored, so lambda transfers on the measured gradient ratio alone with a size factor of exactly 1. CONTROL: zg5fnw, this same fold and this same 44,032-parameter family (it additionally trained frame_dir), cross-seed at tau 0.0305 with alpha_pae 1 and lambda 1e-05, which reached paired +0.00052 positive 4 of 7 before being stopped at epoch 7 -- its training ranking loss never descended, 0.05525 at epoch 1 to 0.05865 at epoch 7. CRITERION: FRACTION ABOVE 0.23 over valfull's 92 interfaces is the primary number for this arm, against shipped 0.7391; selected mean DockQ against shipped 0.5831 is reported but is explicitly NOT what this objective optimises. NOTE 13 of valfull's 92 interfaces (14.1%) have NO acceptable pose at any rank and are censored in the rank metric. STEPS WITH NO TASK GRADIENT ARE NOT TAKEN. 52.3% of the fold's structures have no straddling pair at all, so with --rank-structs-per-step 2 about 27% of steps accumulate nothing from the ranking term, and with --alpha-pae 0 there is then no task gradient whatsoever. The first attempt at this arm (t7dvob) handed those steps to AdamW with only the L2-SP term, and because AdamW is scale-invariant that is a near-full lr-sized step straight back toward the anchor rather than a small one: 16 of its first 69 taken steps had grad_norm < 1e-3 against a typical 0.01-0.20, the late ones clustered at ~2.5e-4 and rising as the weights drifted. Those steps are now skipped and counted in the epoch line. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: fraction at or below shipped 0.7391 over epochs 10-15, which would say that concentrating the gradient on threshold-crossing pairs buys nothing even when it is the metric being optimised." \
  -- --pose-perblock --pose-perblock-copy --l2-sp "$LAM" \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs all --rank-straddle 0.23 --rank-tau 0.1429 \
  --train-only 'linear_no_bias_d,pose_perblock' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 15 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 0.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/valfull/structures.csv \
  --val-loss-pool training/valfull/pool.csv \
  --val-loss-preflight training/valfull/alignment_preflight.json \
  --val-loss-seeds 400000-400029 \
  --val-loss-prep validation/prep \
  --train-eval-pool training/valfull/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 30 \
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "launch_straddle returned $rc"
exit $rc
