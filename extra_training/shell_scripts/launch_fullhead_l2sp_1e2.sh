#!/bin/bash
# FULL CONFIDENCE HEAD + POSE COPY, ANCHOR ON EVERYTHING EXCEPT THE POSE PARAMETERS, LAMBDA 1e-2.
#
#   loss = masked PAE + 1e-2 * ||W - W_shipped||^2   over the 246 NON-POSE head tensors only
#   the 8 pose tensors train FREE, with no anchor at all
#
# ONE VARIABLE AGAINST tv6hue: lambda, 1e-3 -> 1e-2. Everything else is copied from
# launch_fullhead_l2sp_posefree.sh verbatim -- same fold, selection, skip list, crop, lr 1e-2 flat
# after 20 warmup steps, seeds, accum, nsample, valfull held-out set, 20 epochs, pose-copy
# injection, --numa-map 0:0,1:0,2:1, --prefetch-units 2 --prefetch-workers 4, no --train-only.
#
# WHY HIGHER AND NOT LOWER. --l2-sp-exclude leaves exactly 8 tensors unanchored and dntxmn trains
# exactly 8 tensors -- the same ones. So lambda -> infinity on this arm IS the copy arm, literally:
# the 246 anchored tensors freeze at shipped and only the pose injection moves. That makes the
# sweep an interpolation between two runs we have measured, and the better endpoint is the top:
#
#   lambda 0     the old unconstrained full head (wpg274): ||W-W0||^2 reached 3,801 over 50 epochs,
#                val PAE 2.7463 -> 3.0990 while train fell to 1.7854, never beat shipped
#   lambda 1e-3  tv6hue: rho 0.2120 / 0.2305 / 0.2253 at epochs 1/2/3
#   lambda inf   dntxmn: rho 0.2386 / 0.2466 / 0.2495 at the SAME epochs 1/2/3
#
# tv6hue is behind BOTH pose arms on rho and on selected DockQ at every matched epoch (share
# j583i8 ran 0.2430 / 0.2397 / 0.2437). Every direction we have evidence for points up.
#
# THE PRE-LAUNCH ARGUMENT AGAINST 1e-2 WAS WRONG, and the correction is the reason this is worth
# running rather than skipping to the copy arm. launch_fullhead_l2sp_posefree.sh's header argued
# 1e-2 would price the drift at 38.0 against a PAE loss near 2.6 -- 14x the objective -- and so
# would PIN the 246 tensors and reduce the arm to dntxmn with 19.7M idle parameters. That used
# wpg274's drift of 3,801, which is the drift of an UNANCHORED head. The drift that actually
# occurs under an anchor is 77x smaller: tv6hue measured ||W-W0||^2 = 59.2 / 48.2 / 49.2 at
# epochs 1/2/3, settling near 49 rather than running away. At that drift 1e-2 prices the penalty
# at 0.49 against a PAE loss of 2.63 -- about 19% of the objective, and less once the stronger
# anchor pulls the drift in further. That is a real constraint, not a pin.
#
# LAMBDA IS NOT DERIVED. 1e-2 is one rung up at the user's direction, chosen because 1e-1 would
# sit so close to the lambda -> infinity endpoint that it could not distinguish "tighter helps"
# from "the full head is the wrong lever". l2sp_lambda.py's transfer formula is deliberately not
# used; on wpg274 it produced 1e-5, whose penalty reached 0.0380 against a loss near 2.0 -- under
# 2% of the objective -- because it sized lambda for an assumed drift 18x larger than the one that
# occurred. Read the MEASURED "l2-sp: ||W-W0||^2 = ... penalty = ..." line each epoch against
# train and val PAE. Do NOT predict which regime 1e-2 lands in; that prediction is what failed.
#   drift near 0        -> the 246 are pinned; this is the copy arm with 19.7M idle parameters
#   penalty >> the loss -> same thing, and the gradient is going into the penalty
#   both PAE curves fall together -> the thing none of the six previous arms achieved
#
# THE EXPECTED OUTCOME IS NOT A WIN, and it was stated before launch so it cannot be reframed
# afterwards: rho most likely improves MONOTONICALLY in lambda and never beats dntxmn. If that is
# what happens, the payoff is not a better arm -- it is that 19.8M free parameters is the wrong
# lever on 487 structures, and the 40,960-parameter pose arm is what to push on instead.
#
# WHAT WOULD FALSIFY THAT: rho above dntxmn's matched epoch at any epoch, i.e. > 0.2495 by epoch
# 3 or > 0.2500 by epoch 4. Then the interior of the sweep beats both endpoints and lambda is a
# real knob rather than a dial between two known arms.
#
# THE LAUNCH LOG MUST PRINT "8 tensor(s) stay TRAINABLE but are OUT of the anchor, 246 anchored".
# If it says 2 excluded, the pose copies are being anchored to their init and the arm is not what
# it claims. The trainer refuses outright if the pattern matches no trainable parameter, or all.
#
# THE ANCHOR IS THE SHIPPED HEAD, snapshotted at init to <out>/l2sp_anchor.pt. This is a FRESH
# launch, not a resume, so the anchor is built from the shipped weights rather than inherited.
#
# CRITERION: selected DockQ and fraction above 0.23 over valfull's 92 interfaces against shipped
# DERIVED FROM THIS RUN'S OWN SHARDS, plus mean within-interface Spearman(ipTM, DockQ). VAL10 AND
# VALFULL ARE NOT COMPARABLE.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching FULL HEAD + pose-COPY, L2-SP 1e-2 on the 246 non-pose tensors, full_train -> valfull, 20 epochs"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "FULL CONFIDENCE HEAD + POSE COPY, L2-SP 1e-2 ANCHORING EVERYTHING EXCEPT THE POSE PARAMETERS. ONE VARIABLE AGAINST tv6hue: lambda 1e-3 -> 1e-2, every other flag copied verbatim. The 8 pose tensors (block 0's linear_no_bias_d and linear_no_bias_d_wo_onehot plus three copies of each, 40,960 parameters) train FREE with no anchor; the other 246 head tensors are trainable-but-priced at 1e-2 * ||W - W_shipped||^2. The exclusion is an unanchored regex selecting exactly the set dntxmn trained, and the launch log must print 8 excluded / 246 anchored or the arm is not what it claims. WHY HIGHER AND NOT LOWER: lambda -> infinity on this arm IS the copy arm, since the 8 unanchored tensors are exactly dntxmn's 8 trainable ones, so the sweep interpolates between two measured runs and the better endpoint is the top. lambda 0 is the unconstrained full head (wpg274), which drifted to ||W-W0||^2 = 3,801 over 50 epochs with val PAE rising 2.7463 -> 3.0990 while train fell to 1.7854, and never beat shipped. lambda 1e-3 is tv6hue at rho 0.2120/0.2305/0.2253 for epochs 1/2/3. lambda infinity is dntxmn at 0.2386/0.2466/0.2495 on the SAME epochs, and share (j583i8) ran 0.2430/0.2397/0.2437, so tv6hue is behind BOTH pose arms on rho and selected DockQ at every matched epoch. THE PRE-LAUNCH ARGUMENT AGAINST 1e-2 WAS WRONG: launch_fullhead_l2sp_posefree.sh claimed 1e-2 would price the drift at 38.0 (14x the loss) and pin the 246 tensors, but that used wpg274's UNANCHORED drift of 3,801. The drift that actually occurs under an anchor is 77x smaller -- tv6hue measured 59.2/48.2/49.2 at epochs 1/2/3, settling near 49 rather than running away -- so 1e-2 prices it at 0.49 against a PAE loss of 2.63, about 19% of the objective and less once the tighter anchor pulls the drift in. A real constraint, not a pin. LAMBDA IS NOT DERIVED: 1e-2 is one rung up at the user's direction, chosen over 1e-1 because 1e-1 sits so near the infinity endpoint it could not separate 'tighter helps' from 'the full head is the wrong lever'. l2sp_lambda.py's transfer formula is deliberately not used, since on wpg274 it produced a penalty of 0.0380 against a loss near 2.0 by sizing lambda for an assumed drift 18x larger than what occurred. Read the measured ||W-W0||^2 and penalty each epoch against train and val PAE. EXPECTED OUTCOME, STATED BEFORE LAUNCH SO IT CANNOT BE REFRAMED AFTER: rho most likely improves monotonically in lambda and never beats dntxmn, in which case the payoff is not a better arm but the finding that 19.8M free parameters is the wrong lever on 487 structures and the 40,960-parameter pose arm is what to push on. FALSIFIED BY rho above dntxmn's matched epoch -- > 0.2495 by epoch 3 or > 0.2500 by epoch 4. FRESH launch, so the anchor is built from the shipped weights, not inherited. TEN-EPOCH MINIMUM does not apply -- the user retracted it. VAL10 AND VALFULL ARE NOT COMPARABLE." \
  -- --pose-perblock --pose-perblock-copy \
  --l2-sp 1e-2 --l2-sp-exclude 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 20 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
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
echo "launch_fullhead_l2sp_1e2 returned $rc"
exit $rc
