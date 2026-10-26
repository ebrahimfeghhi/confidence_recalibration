#!/bin/bash
# FULL CONFIDENCE HEAD + POSE COPY, WITH THE ANCHOR ON EVERYTHING EXCEPT THE POSE PARAMETERS.
#
#   loss = masked PAE + 1e-3 * ||W - W_shipped||^2   over the 246 NON-POSE head tensors only
#   the 8 pose tensors train FREE, with no anchor at all
#
# WHY. Every capacity increase so far went straight into the generalisation gap: the full head
# (19.8M, wpg274) ended 50 epochs with val PAE 3.0990 against 2.7463 at epoch 1 while train fell
# to 1.7854; share (10,240) and copy (40,960) kept val falling but converted train gains to val
# at roughly half rate, and copy's gap was already wider than share's by +0.0067 +- 0.0017 over
# 12 matched epochs. What that suggests is not "less capacity" but "capacity only where it is
# new": there is no reason to want the 246 pretrained tensors far from shipped, and every reason
# to want the pose injection free, since it is the only part that has to learn something the
# shipped head never knew.
#
# SO THIS IS THE EXACT GENERALISATION OF THE CURRENT COPY ARM. --l2-sp-exclude is an unanchored
# regex, so the pattern below selects the SAME 8 tensors dntxmn trained -- block 0's
# linear_no_bias_d and linear_no_bias_d_wo_onehot plus the three copies of each. Those stay free.
# The other 246 go from FROZEN to trainable-but-priced: a hard constraint becomes a soft one,
# exactly where we have no reason to want motion.
#
# THE LAUNCH LOG MUST PRINT "8 tensor(s) stay TRAINABLE but are OUT of the anchor, 246 anchored".
# If it says 2 excluded, the pose copies are being anchored to their init and the arm is not what
# it claims. The trainer refuses outright if the pattern matches no trainable parameter, or every
# one of them.
#
# LAMBDA IS 1e-3 AT THE USER'S DIRECTION AND IS NOT DERIVED. It was 1e-2 when this arm was first
# queued and the user lowered it before launch. The arithmetic that supports the change, from the
# MEASURED drift rather than a formula: wpg274's full head ended 50 epochs at ||W-W0||^2 = 3,801,
# which 1e-2 would price at 38.0 against a PAE loss near 2.6 -- about 14x the objective, strong
# enough that the 246 tensors would likely be PINNED and the arm would reduce to dntxmn with 19.7M
# idle parameters. 1e-3 prices the same drift at 3.8, roughly 1.4x the loss: enough to bend the
# solution without freezing it. That is a reason, NOT a prediction -- read the measured penalty.
# l2sp_lambda.py's transfer formula is
# deliberately not used: on wpg274 it produced 1e-5, whose penalty reached 0.0380 against a PAE
# loss near 2.0 -- under 2% of the objective -- because the lambda had been sized for an assumed
# drift 18x larger than the one that occurred. Read the MEASURED "l2-sp: ||W-W0||^2 = ...
# penalty = ..." line each epoch against train and val PAE instead, and do not predict which
# regime 1e-3 lands in; that prediction is what failed last time.
#   drift near 0        -> the 246 are pinned; this is the copy arm with 19.7M idle parameters
#   penalty >> the loss -> same thing, and the gradient is going into the penalty
#   both PAE curves fall together -> the thing none of the five previous arms achieved
#
# ONE VARIABLE AGAINST dntxmn: identical fold, selection, skip list, crop, lr 1e-2 flat after
# warmup, seeds, accum, nsample, valfull held-out set, 20 epochs, pose-copy injection. What
# changes is that the trainable restriction is GONE and the anchor is in, so the same 8 tensors
# are free while 246 more may now move at a price. dntxmn through epoch 14 is the control: paired
# dDockQ +0.0016 mean, rho 0.2478 vs shipped 0.2218, val PAE low 2.7580 at epoch 13 then 2.7609.
#
# THE ANCHOR IS THE SHIPPED HEAD, snapshotted at init to <out>/l2sp_anchor.pt. install() runs
# long before the anchor is built, so the pose copies exist as named parameters when the
# exclusion is applied and never enter the anchor at all.
#
# CRITERION: selected DockQ and fraction above 0.23 over valfull's 92 interfaces against shipped
# DERIVED FROM THIS RUN'S OWN SHARDS, plus mean within-interface Spearman(ipTM, DockQ). VAL10 AND
# VALFULL ARE NOT COMPARABLE.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching FULL HEAD + pose-COPY, L2-SP 1e-3 on the 246 non-pose tensors, full_train -> valfull, 20 epochs"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "FULL CONFIDENCE HEAD + POSE COPY, L2-SP 1e-3 ANCHORING EVERYTHING EXCEPT THE POSE PARAMETERS. The 8 pose tensors (block 0's linear_no_bias_d and linear_no_bias_d_wo_onehot plus three copies of each, 40,960 parameters) train FREE with no anchor; the other 246 head tensors go from FROZEN to trainable-but-priced at 1e-3 * ||W - W_shipped||^2. The exclusion is an unanchored regex, so it selects exactly the set dntxmn trained, and the launch log must print 8 excluded / 246 anchored or the arm is not what it claims. WHY: every capacity increase so far went into the generalisation gap. The unconstrained full head (wpg274, 50 ep) ended with val PAE 3.0990 against 2.7463 at epoch 1 while train fell to 1.7854, and its best epoch of 50 was still below shipped on paired DockQ (-0.0020 +- 0.0292 on val10's 18 interfaces). share and copy kept val falling but converted train gains to val at about half rate, and copy's gap was already wider than share's by +0.0067 +- 0.0017 over 12 matched epochs. So the hypothesis is not less capacity but capacity ONLY WHERE IT IS NEW: no reason to want the 246 pretrained tensors far from shipped, every reason to want the pose injection free. LAMBDA IS NOT DERIVED -- 1e-3 is the user's choice (lowered from 1e-2 before launch, because 1e-2 would price wpg274's measured drift of 3,801 at 38.0 against a loss near 2.6 and likely pin the 246 tensors; 1e-3 prices it at 3.8) and l2sp_lambda.py's transfer formula is deliberately not used, since on wpg274 it produced a penalty of 0.0380 against a loss near 2.0 by sizing lambda for an assumed drift 18x larger than what occurred. Read the measured ||W-W0||^2 and penalty printed each epoch against train and val PAE. ONE VARIABLE AGAINST dntxmn: same fold, selection, skip list, crop, lr, seeds, accum, nsample, valfull, 20 epochs, pose-copy injection; the trainable restriction removed and the anchor added. CONTROL: dntxmn through epoch 14 ran paired dDockQ +0.0016 mean on valfull's 92 interfaces, rho 0.2478 vs shipped 0.2218, val PAE bottoming at 2.7580 at epoch 13. TEN-EPOCH MINIMUM does not apply -- the user retracted it. VAL10 AND VALFULL ARE NOT COMPARABLE." \
  -- --pose-perblock --pose-perblock-copy \
  --l2-sp 1e-3 --l2-sp-exclude 'linear_no_bias_d' \
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
echo "launch_fullhead_l2sp_posefree returned $rc"
exit $rc
