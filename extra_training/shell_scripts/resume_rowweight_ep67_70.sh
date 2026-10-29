#!/bin/bash
# RESUME nskyne 67 -> 70. Every flag is verbatim from launch_rowweight_cosine.sh except --resume
# and --epochs; read that script's header for what the arm is and what it cannot answer.
#
# WHY THIS EXISTS SEPARATELY from resume_rowweight_ep56_70.sh: that run (d32kqp) was stopped at
# epoch 66, mid-67, to free cards 4,5,7 for the mean-pooling evaluation, at the user's direction.
# This finishes the originally intended 70 epochs from resume_ep66.pt so there is more time to
# decide the next arm. Epoch 67's partial poses were discarded; nothing was deleted.
#
# WHY 70. The arm was stopped at epoch 55 to score it. In-sample it beat the shipped ipTM baseline
# (+0.0131 +- 0.0052) but NOT the matched control (+0.0063 +- 0.0055, t=1.14); held out on 92
# valfull interfaces at budget 100 both arms sat BELOW baseline and were indistinguishable from
# each other (+0.0017 +- 0.0137). So 15 more epochs are being run while the user considers other
# ideas, not because a trend predicts they will help. Nothing here is expected to change the
# verdict on its own.
#
# THE LR IS FLAT AT 1e-4 AND THAT IS VERIFIED, NOT ASSUMED. The decay window is keyed on
# --lr-decay-end-epoch 50, not on --epochs, so raising the horizon 60 -> 70 does NOT re-stretch it:
# lr_at_step returns 1.000e-04 at epochs 50, 55, 56, 60, 65 and 70 under both epochs=60 and
# epochs=70. Checked because the previous horizon bug (205 assumed against 184 actual) put lr 8x
# over intent at epoch 50 and was caught only by review.
#
# AdamW MOMENTS ARE RESTORED and the index keying is valid: the trainable tensor set is unchanged
# at 258, so no tensors were inserted mid-order the way they were at the epoch-48 resume.
#
# THE LAUNCH LOG MUST STILL PRINT: the row-weight line with mask FLOAT and weights DETACHED, 368
# structures / 738 interfaces, 258 trainable tensors, steps/epoch 184, and NO "anchor SNAPSHOT".
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
$PY_BIN -I run_logging/test_pae_row_weight.py || {
  echo "REFUSING: the row-weight correctness gate failed"; exit 1; }
$PY_BIN -I run_logging/test_lr_schedule.py || {
  echo "REFUSING: the lr schedule gate failed"; exit 1; }
$PY_BIN -I run_logging/test_guard_fields.py || {
  echo "REFUSING: the resume-guard wiring gate failed"; exit 1; }
echo "resuming nskyne from resume_ep55.pt for epochs 67-70 at a flat 1e-4"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "RESUME of the row-weighted arm, epochs 67-70, flat lr 1e-4, no evals, checkpoints only. Identical configuration to launch_rowweight_cosine.sh -- row-weighted PAE at frac 0.2 / mass 0.8 / floor 10 per scored chain pair, 368 winnable structures, 738 interfaces, --frame-direction on at 258 trainable tensors -- resumed from nskyne/resume_ep55.pt rather than relaunched. WHY: the arm was stopped at 55 and scored. In sample on its own 738 interfaces at budget 10 it beat the shipped ipTM baseline by +0.0131 +- 0.0052 (t=2.51) while the matched unweighted control beat it by only +0.0067 +- 0.0060, but the PAIRED head-to-head was +0.0063 +- 0.0055 (t=1.14) -- not separable -- and on frac acceptable the control was marginally AHEAD (0.7317 vs 0.7304). Held out on the 92 valfull interfaces at budget 100 both arms fell BELOW the shipped baseline (row-weighted -0.0123 +- 0.0139, control -0.0140 +- 0.0149) and differed from each other by +0.0017 +- 0.0137, t=0.12. So the in-sample gain does not transfer and the weighting is not distinguishable from training alone; these 15 epochs are being run while other ideas are considered, NOT because a trend predicts improvement. THE LR IS FLAT 1e-4 THROUGHOUT and that was verified rather than assumed: the cosine window is keyed on --lr-decay-end-epoch 50, not on --epochs, so lr_at_step returns 1.000e-04 at epochs 50, 55, 56, 60, 65 and 70 under both the old 60-epoch horizon and the new 70, and steps/epoch still derives to 184 from the epoch builder. That was checked because the earlier horizon bug (205 assumed against 184 actual) left lr 8x over intent at epoch 50. AdamW moments restore cleanly since the trainable tensor set is unchanged at 258 -- no mid-order insertion like the epoch-48 resume. Still no L2, no --val-loss-* monitor and no --train-eval-* pass, so nothing in this run can see overfitting while it happens and the checkpoints must be scored afterwards; given the held-out numbers above are already below baseline, that caveat is doing real work here." \
  -- --resume training/runs/d32kqp/resume_ep66.pt \
  --resume-note "epochs 67-70 at a flat 1e-4, after scoring epoch 55 in sample and held out. No configuration change at all beyond the horizon: same row weighting, same 368-structure manifest, same cosine definition (its decay ended at epoch 50, so this stretch is flat), same accum 85, nsample 5, seeds 500000-500009, --blocks-per-ckpt 0." \
  --pose-perblock --pose-perblock-copy \
  --frame-direction \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
  --manifest-selection training/full_train/structures_winnable.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 70 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --lr-final 1e-4 --lr-decay-shape cosine \
  --lr-decay-start-epoch 20 --lr-decay-end-epoch 50 \
  --pae-row-frac 0.2 --pae-row-mass 0.8 --pae-row-floor 10 --pae-row-log-every 50 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "resume_rowweight_ep56_70 returned $rc"
exit $rc
