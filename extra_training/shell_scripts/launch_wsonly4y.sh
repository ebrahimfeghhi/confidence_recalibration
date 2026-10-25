#!/bin/bash
# 4y3ve6's CONFIGURATION, WITHIN-SEED PAIRS ONLY, tau 0.0036, 20 epochs.
#
# Usage:  ./shell_scripts/launch_wsonly4y.sh
#
# ONE VARIABLE AGAINST 4y3ve6: --rank-pairs within (it ran the default, cross) and the tau that
# follows from that population, 0.0036 instead of DEFAULT_TAU 0.0305. Every other setting is
# copied from 4y3ve6's own args.json -- softplus, --rank-weight-by-gap, --rank-cross-seed as the
# STEP SHAPE, --alpha-pae 0, --train-only 'linear_no_bias_d' with --pose-perblock-share (2 tensors
# / 10,240 shared parameters), NO orientation, NO L2-SP, lr 1e-2 flat after a 20-step warmup,
# overfit30 -> val10.
#
# WHY 4y3ve6 IS THE BASE. docs/overfit30_val10_runs.md: of 31 runs on this pool it is the only one
# that FINISHES above shipped on both metrics, 0.3089 / 0.6111 against 0.2909 / 0.5556. It is also
# the only arm that gets its held-out ranking loss UNDER 2*tau*ln2 = 0.04228, the value the loss
# reads when a head scores every pose of an interface identically -- run_logging/zero_margin.py.
# Everything added since (orientation, per-block copies, an anchor) finishes worse.
#
# TAU IS 0.0036, the UNWEIGHTED median within-seed |ipTM gap| over the full 497-structure training
# set after the DockQ-tie filter (run_logging/measure_tau.py --pairs within) -- the same statistic
# DEFAULT_TAU uses, measured on the population this arm actually trains. At it, 48.3% of
# within-seed pairs are inverted and 78.8% of pairwise score-space gradient mass reaches inverted
# pairs, the highest share any tau measurement in this project has produced. The cost is stated
# rather than hidden: 20.3% of interfaces are "too TIGHT" at this tau, their pairs reading as ties.
#
# THE LANDMARK MOVES WITH TAU AND NOTHING FROM OTHER ARMS TRANSFERS. 2*tau*ln2 = 0.00499 here
# against 0.04228 at DEFAULT_TAU. A loss of 0.0040 in this run is BETTER than chance; the same
# number would be meaningless next to a cross-seed arm's 0.040. Do not put them on one axis.
#
# CEILING, measured on the 18 val10 interfaces from shipped scores: perfectly ordering within each
# seed while leaving every cross-seed relationship untouched is worth +0.0370 DockQ, improving 15
# of 18 and harming none, 12% of remaining oracle headroom. FRAC DOES NOT MOVE AT ALL under that
# perfect reordering -- it stays at 0.5556 -- so DockQ is the only criterion this arm can win on
# and frac is reported but cannot be the test.
#
# PAIR BUDGET is 8x smaller than cross-seed: 10 seeds x C(5,2) = 100 within-seed pairs per
# interface against 1,125 cross-seed. Expect a noisier training curve.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching within-seed-only on 4y3ve6's config: tau 0.0036, 20 epochs, no L2-SP"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "WITHIN-SEED PAIRS ONLY on 4y3ve6's CONFIGURATION, tau 0.0036, 20 epochs. ONE VARIABLE AGAINST 4y3ve6: --rank-pairs within where it ran the default cross, plus the tau that follows from that population. Everything else copied from 4y3ve6's own args.json: softplus, --rank-weight-by-gap, --rank-cross-seed as the STEP SHAPE not the objective, --alpha-pae 0 so ordering is the only term, --train-only 'linear_no_bias_d' with --pose-perblock-share = 2 tensors / 10,240 SHARED parameters read by all four pairformer blocks, NO orientation, NO L2-SP at all, lr 1e-2 flat after a 20-step warmup, overfit30 train, val10 held out every epoch at budget 30. WHY 4y3ve6 IS THE BASE: of 31 runs on this pool it is the only one that FINISHES above shipped on both metrics (0.3089 / 0.6111 against 0.2909 / 0.5556) and the only one whose held-out ranking loss gets UNDER 2*tau*ln2 = 0.04228, the value the loss reads when a head scores every pose of an interface identically; everything added since -- orientation, per-block copies, an anchor -- finishes worse. TAU 0.0036 is the UNWEIGHTED median within-seed |ipTM gap| over the full 497-structure training set after the DockQ-tie filter, the same statistic DEFAULT_TAU uses, measured on the population this arm trains: 48.3% of within-seed pairs are inverted and 78.8% of pairwise score-space gradient mass reaches inverted pairs, the highest share any tau measurement here has produced; the cost is that 20.3% of interfaces are too TIGHT at this tau and their pairs read as ties. THE LANDMARK MOVES WITH TAU: 2*tau*ln2 = 0.00499 here against 0.04228 at DEFAULT_TAU, so NO loss value from any other arm is comparable and they must never share an axis. CEILING measured on the 18 val10 interfaces from shipped scores: perfect within-seed ordering with every cross-seed relationship untouched is worth +0.0370 DockQ, improving 15 of 18 and harming none, 12% of remaining oracle headroom -- but FRAC DOES NOT MOVE AT ALL under it, staying at 0.5556, so DockQ is the only criterion this arm can win on and frac is reported but is NOT the test. PAIR BUDGET is 8x smaller than cross-seed, 100 within-seed pairs per interface against 1,125, so expect a noisier curve. CRITERION: paired (trained - shipped) DockQ over the 18 val10 interfaces from this run's own shards, against shipped 0.2909 and 4y3ve6's +0.018. Spearman is reported but is NOT the criterion. CONTROL: 4y3ve6, this arm with cross-seed pairs at tau 0.0305. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: landing at or below 4y3ve6's +0.018 paired DockQ, which would say the within-seed family carries no usable signal even with its own tau and the one configuration that has worked." \
  -- --pose-perblock --pose-perblock-share \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs within --rank-tau 0.0036 --rank-weight-by-gap \
  --train-only 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 20 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 0.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
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
echo "launch_wsonly4y returned $rc"
exit $rc
