#!/bin/bash
# L2-SP LAMBDA SWEEP, ARM "1e-4". The whole confidence head trains; distance from the shipped
# weights is priced:   loss = masked PAE + 1e-4 * ||W - W_shipped||^2
#
# WHY A SWEEP AND NOT A DERIVED LAMBDA. The one previous run in this cell (wpg274, 1e-5, 50
# epochs) had its lambda transferred through l2sp_lambda.py's per-parameter-drift formula, sized
# for an ASSUMED drift of 67,052. The drift that actually occurred was 3,801 -- 18x smaller -- so
# the penalty landed at 0.0380 against a PAE loss near 2.0, under 2% of the objective. The
# lambda was sized on a trajectory the lambda itself prevented. Circular, and at the user's
# explicit direction (2026-10-26) that machinery is not used again.
#
# SO READ THE MEASURED NUMBERS, printed every epoch as "l2-sp: ||W-W0||^2 = ... penalty = ...".
# Those two numbers against the PAE loss are the whole result:
#   penalty near 0 and drift near 67,052  -> too weak, the penalty is decoration
#   penalty a sizeable fraction of ~2.0   -> the anchor is actually binding
#   drift near 0                          -> the head is pinned; this is shipped with extra steps
# Do not predict which of those happens from lambda. That prediction is what failed last time.
#
# WHAT IT IS TESTING: whether ANY lambda makes train AND val PAE fall together on the full head.
# wpg274 drove train PAE to 1.7854 while val PAE ended at 3.0990, having bottomed at 2.6912 by
# epoch 22. Its best epoch of 50 was still BELOW shipped on paired DockQ (-0.0020 +- 0.0292).
#
# ONE VARIABLE against wpg274: identical 30 structures, seeds, crop 10/5 at 3400, accum 85,
# nsample 5, lr 1e-2, warmup 20, masked PAE-only, per-epoch val10 + DockQ at budget 30, fresh
# from the shipped head, checkpointing off. Only --l2-sp changes, and --epochs is 15 not 50.
#
# val10 IS 18 INTERFACES: one interface is 1/18 of the mean and the paired s.e. was +-0.0292 on
# wpg274, so DockQ here cannot resolve a working head from a broken one. The PAE curves and the
# drift are what this arm is read on. val10 and valfull are NEVER comparable.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "L2-SP LAMBDA SWEEP arm 1e-4: full confidence head trains, 15 epochs, fresh from the shipped head, overfit30 -> val10. Second and third samples of LAMBDA after wpg274 at 1e-5, whose penalty reached only 0.0380 against a loss near 2.0 (under 2% of the objective) because its lambda had been sized for an assumed drift 18x larger than what occurred. LAMBDA IS NOT DERIVED HERE -- it is swept, and the arm is read on the measured ||W-W0||^2 and penalty printed each epoch, against train and val PAE. The question is whether any lambda makes BOTH PAE curves fall on the full head; wpg274 got train to 1.7854 and val to 3.0990. ONE VARIABLE against wpg274 apart from epoch count." \
  -- --l2-sp 1e-4 \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 15 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
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
# PROPAGATE IT. The chain runner stops on a non-zero arm, and a trailing bare `echo` always
# returns 0 -- which is why a broken arm reported success and the chain marched straight on to
# the next one.
echo "launcher returned $rc"
exit $rc
