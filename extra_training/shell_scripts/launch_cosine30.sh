#!/bin/bash
# zg5fnw's OBJECTIVE, overfit30 -> val10, with a COSINE LEARNING RATE 1e-2 -> 1e-4 over 15 epochs.
#
# Usage:  ./shell_scripts/launch_cosine30.sh <lambda>
#
# WHY THIS ARM. The straddle filter was worse, measured: over epochs 1-7 on the same 92 valfull
# interfaces, zg5fnw (normal cross-seed softplus) is paired +0.00052 and above shipped in 4 of 7
# epochs, peaking at frac 0.7609 / DockQ 0.5878 against shipped 0.7391 / 0.5832, while 8fkrsu
# (straddle, tau 0.1429) is paired -0.00733 and above shipped in 0 of 7. So the objective reverts
# to zg5fnw's exactly: --rank-pairs cross at DEFAULT_TAU 0.0305, --rank-weight-by-gap,
# --rank-form softplus, alpha_pae 1, orientation on, train_only linear_no_bias_d,pose_perblock,
# frame_dir with frame_dir excluded from the anchor.
#
# WHAT IS NEW IS THE SCHEDULE, and it is the one thing no arm in this project has ever varied.
# run_logging/weight_travel.py on both full-fold runs: the weights sit at a FIXED distance from the
# L2-SP anchor for five epochs -- 5.55 for 8fkrsu, ~20 for zg5fnw -- while still MOVING 3-12 per
# epoch, with gradient norms steady at 0.026 and the training loss flat. 8fkrsu's straightness is
# 0.227 against 0.462 for a free random walk of the same steps: confined and retracing, i.e. an
# optimiser oscillating at the limit its step size can resolve. lr was constant 0.01 from step 20
# to the end in every arm, because until now phase_k_train.py had no decay at all. The anchor is
# NOT the constraint -- 2*lambda*||W-W0|| is 5.55e-04 against a 2.6e-02 task gradient, 2.1%.
#
# THE POOL IS overfit30 -> val10, 30 training structures / 77 interfaces and 18 held-out
# interfaces, which puts ~17 optimiser steps in an epoch and so 255 across 15 -- the cosine
# horizon the startup banner prints. val10 and valfull are NEVER comparable; the control for this
# run is hronsr/6v6bbf/bjwcm5 on the same pool and the same trainable set.
#
# LAMBDA IS AN ARGUMENT, derived by probe_cosine30.sh + l2sp_lambda.py, never defaulted: the
# trainable set is zg5fnw's but the pool is 16x smaller, so zg5fnw's 1e-05 does not transfer.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
LAM="${1:-}"
[ -n "$LAM" ] || { echo "usage: $0 <lambda>   (measured by probe_cosine30.sh, never defaulted)"; exit 1; }
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching cosine arm: lambda=$LAM, lr 1e-2 -> 1e-4 cosine over 15 epochs"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "COSINE LR DECAY 1e-2 -> 1e-4 over 15 epochs, zg5fnw's objective, overfit30 -> val10, lambda $LAM. REVERTS THE STRADDLE FILTER, which was measured worse: over epochs 1-7 on the same 92 valfull interfaces zg5fnw (normal cross-seed softplus, DEFAULT_TAU 0.0305) is paired +0.00052 and above shipped in 4 of 7 epochs, peaking frac 0.7609 / DockQ 0.5878 against shipped 0.7391 / 0.5832, while 8fkrsu (straddle, tau 0.1429) is paired -0.00733 and above shipped in 0 of 7. THE ONE NEW VARIABLE IS THE SCHEDULE. weight_travel.py on both full-fold arms shows the weights parked at a fixed distance from the anchor for five epochs -- 5.55 for 8fkrsu, ~20 for zg5fnw -- while still moving 3-12 per epoch, gradient norms steady at 0.026, training loss flat; 8fkrsu's straightness is 0.227 against 0.462 for a free random walk of the same steps, so the trajectory is CONFINED and retracing, an optimiser oscillating at the limit its step size resolves. lr had been constant 0.01 after warmup in every arm ever run here because no decay existed until this commit. The anchor is not the constraint: 2*lambda*||W-W0|| = 5.55e-04 against a 2.6e-02 task gradient, 2.1% of the force, so lowering lambda cannot release anything. CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, from this run's own shards; Spearman is reported but is NOT the criterion. CONTROLS on the same pool and same trainable set: hronsr (+0.0116 paired over epochs 8-15, 8 of 8 positive), 6v6bbf, bjwcm5 -- all at constant lr 0.01, which is exactly what this arm changes. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: landing at or below the constant-lr controls on paired DockQ, which would say the oscillation diagnosed by weight_travel.py is not what limits these runs." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --l2-sp "$LAM" --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs cross --rank-weight-by-gap \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 15 --lr 0.01 --lr-final 0.0001 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 3 \
  --val-loss-every 1 \
  --val-loss-fold validation/fold \
  --val-loss-selection training/val10/structures.csv \
  --val-loss-pool training/val10/pool.csv \
  --val-loss-preflight training/val10/alignment_preflight.json \
  --val-loss-seeds 400000-400029 \
  --val-loss-prep validation/prep \
  --train-eval-pool training/val10/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 30 \
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "launch_cosine30 returned $rc"
exit $rc
