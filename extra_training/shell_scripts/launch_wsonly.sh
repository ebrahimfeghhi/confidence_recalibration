#!/bin/bash
# WITHIN-SEED PAIRS ONLY, the one pair family this project has never trained with a real loss.
#
# Usage:  ./shell_scripts/launch_wsonly.sh <lambda> <alpha_rank> <alpha_pae>
#
# WHY IT IS NEW. run_index.py answers this directly: no run has ever had rank_pairs='within', and
# no softplus or expected arm has ever run without --rank-cross-seed. Five runs DID train
# within-seed pairs -- ntwjfp, ppxxgm, qiodxz, zbbci6, wpg274 -- but all with the LINEAR form
# before --rank-form existed, so none of them had a tau and none of them is a test of this.
#
# TAU IS 0.0036, the within-seed median |ipTM gap| over the FULL 497-structure training set after
# the DockQ-tie filter (measure_tau.py --pairs within). Chosen deliberately over overfit30's own
# 0.0137 to follow the project's standing rule -- DEFAULT_TAU 0.0305 is likewise a full-training
# figure used on overfit30 arms -- so the knob is a property of the data, not of the pool.
#
# THE COST OF THAT CHOICE, stated once: against overfit30's own within-seed median of 0.0137, tau
# 0.0036 gives m/tau ~ 3.8, which puts ~69% of interfaces in the saturated zone where only
# outright inversions contribute. That is the sparse-signal end of the trade, and it is the
# deliberate consequence of taking tau from the full fold. 48.5% of within-seed pairs on overfit30
# ARE inverted, so the signal does not vanish; it concentrates.
#
# TRAINABLE SET IS hronsr's, 12 tensors / 44,032 parameters: linear_no_bias_d + pose_perblock +
# frame_dir, i.e. distance AND orientation injected at all four pairformer blocks. "With pose
# information" is the point of the arm -- the five pre-flag within-seed runs had no orientation
# input at all, so this is the first within-seed arm that can see the geometry it is ranking.
#
# ALL THREE WEIGHTS ARE ARGUMENTS, derived by probe_wsonly.sh and passed in, never defaulted:
#   lambda      L2-SP, transferred from 24u34d's 2.5e-4 at |g_rank| 4.0207e-02 / 40,960 params
#               against this arm's measured task gradient at the same anchored size.
#   alpha_rank  1 when PAE is off; gradient parity |g_PAE|/|g_rank| when it is on.
#   alpha_pae   0 or 1, decided by whether PAE helped oe6sru -- see handover_to_wsonly.sh.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
LAM="${1:-}"; AR="${2:-}"; AP="${3:-}"
[ -n "$LAM" ] && [ -n "$AR" ] && [ -n "$AP" ] || {
    echo "usage: $0 <lambda> <alpha_rank> <alpha_pae>   (all three are measured, none defaulted)"
    exit 1; }
BUSY=$($PY_BIN -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY process(es) occupy cards 4,5,7."; exit 1; }
echo "launching within-seed-only: lambda=$LAM alpha_rank=$AR alpha_pae=$AP"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "WITHIN-SEED PAIRS ONLY, lambda $LAM, alpha_rank $AR, alpha_pae $AP, 15 epochs. THE ONE PAIR FAMILY THIS PROJECT HAS NEVER TRAINED WITH A REAL LOSS: run_index.py reports no run has ever had rank_pairs='within' and no softplus or expected arm has ever run without --rank-cross-seed. Five runs did train within-seed pairs -- ntwjfp, ppxxgm, qiodxz, zbbci6, wpg274 -- all with the LINEAR form before --rank-form existed, so none had a tau and none is a test of this. TAU 0.0036 is the within-seed median |ipTM gap| over the FULL 497-structure training set after the DockQ-tie filter, chosen over overfit30's own 0.0137 to follow the standing rule that tau is a property of the DATA not the pool -- DEFAULT_TAU 0.0305 is likewise a full-training figure used on overfit30 arms. THE COST, stated rather than hidden: against overfit30's own within-seed median of 0.0137 this gives m/tau ~ 3.8, putting ~69% of interfaces in the saturated zone where only outright inversions contribute; 48.5% of within-seed pairs on overfit30 ARE inverted, so the signal concentrates rather than vanishes. TRAINABLE SET IS hronsr's 12 tensors / 44,032 parameters -- linear_no_bias_d + pose_perblock + frame_dir, distance AND orientation at all four pairformer blocks. That is the point: the five pre-flag within-seed runs had NO orientation input, so this is the first within-seed arm that can see the geometry it is ranking. WITHIN-SEED ORDERING IS WORTH MEASURING: on valfull at the shipped head, perfectly ordering within each seed while leaving every cross-seed relationship untouched is worth +0.0213 +- 0.0032 DockQ, improving 65 of 92 interfaces and harming none -- 19% of all remaining oracle headroom. CONTROLS: hronsr (same trainable set, same pool, pairs=all at L=8, paired +0.0116 over epochs 8-15, positive in 8 of 8) and 24u342 (same trainable set, cross-seed only at tau 0.0305, 50 epochs). CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, from this run's own shards. Spearman is reported but is NOT the criterion. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: landing at or below shipped on paired DockQ, which would say the within-seed family carries no usable signal even with orientation input and its own tau." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp "$LAM" --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank "$AR" --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs within --rank-tau 0.0036 \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 15 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae "$AP" --clip-norm 25.0 --crop-size 3400 \
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
echo "launch_wsonly returned $rc"
exit $rc
