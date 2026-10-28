#!/bin/bash
# FULL CONFIDENCE HEAD + POSE COPY, NO ANCHOR, NO EVALUATION, 40 EPOCHS -- CHECKPOINTS ONLY.
#
# At the user's direction: no L2 decay, do not compute training or val metrics, save every
# checkpoint and they will evaluate afterwards. So this run trains, writes checkpoints, and does
# nothing else.
#
# WHAT WAS REMOVED AND WHY IT IS THE SLOW PART. --val-loss-* scored 53 structures x 30 seeds =
# 1,590 held-out units EVERY epoch, and --train-eval-* scored 92 interfaces at budget 30 on top of
# the 1,461 training units. That is why xsy8a2 ran about 52 minutes per epoch. Dropping both
# leaves only the training units.
#
# WHAT THIS COSTS, stated rather than discovered later: nothing in this run can see overfitting
# while it is happening. The val-loss monitor exists because a previous anchored run's training
# PAE fell 38% while its val PAE ROSE 51% and nothing in the run could see it. THIS ARM IS
# UNANCHORED -- the configuration where that divergence was measured -- so the checkpoints must
# actually be scored afterwards rather than assumed good.
#
# WHAT IS WRITTEN: head_ep<N>_step<S>.pt and resume_ep<N>.pt at every epoch boundary, plus
# intra-epoch saves every 250 optimiser steps. ~10,840 steps over 40 epochs.
#
# CLOSEST PRIOR ARMS, neither identical: wpg274 (lambda 1e-5, NO pose injection) ran val PAE
# 2.7463 -> 3.0990 over 50 epochs while train fell to 1.7854; p5yoza (lambda 0, no pose
# injection). Adding pose-copy injection to an unanchored full head is a new point.
#
# THE LAUNCH LOG MUST PRINT 254 trainable tensors and NO anchor line. An "anchor SNAPSHOT" line
# would mean --l2-sp survived and this is not the arm it claims to be.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
echo "launching FULL HEAD + pose-COPY, NO anchor, NO evals, 40 epochs, checkpoints only"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "FULL CONFIDENCE HEAD + POSE COPY, NO ANCHOR, NO EVALUATION, 40 EPOCHS, CHECKPOINTS ONLY. At the user's direction and in their words: no L2 decay, do not compute training or val metrics, save every checkpoint and they will evaluate afterwards. So this run trains and writes checkpoints and does nothing else -- no --l2-sp, no --val-loss-* held-out PAE monitor, no --train-eval-* selected-DockQ pass, no --eval-cards. THE POINT IS SPEED: the monitor this replaces scored 53 structures x 30 seeds = 1,590 units every epoch and the valfull eval scored 92 interfaces at budget 30 on top of 1,461 training units, which is why xsy8a2 took about 52 minutes per epoch. Removing both leaves only the 1,461 training units. CONSEQUENCE, STATED PLAINLY: nothing in this run can see overfitting while it happens. The val-loss monitor exists precisely because a previous anchor's training PAE fell 38% while its val PAE rose 51% with nothing in the run able to see it. This arm is unanchored, which is the configuration where that divergence was measured, so the checkpoints must actually be scored afterwards rather than assumed good -- every epoch writes head_ep<N>_step<S>.pt and resume_ep<N>.pt, and --save-every 250 adds intra-epoch saves. CLOSEST PRIOR ARMS, neither identical: wpg274 at lambda 1e-5 WITHOUT pose injection ran val PAE 2.7463 -> 3.0990 across 50 epochs while train fell to 1.7854; p5yoza at lambda 0 also without pose injection. This one adds pose-copy injection to the unanchored full head, so it is a new point rather than a repeat. Everything else is copied verbatim from launch_fullhead_l2sp_1e2.sh: same fold, selection, skip list, crop 3400, margin 10, contact 5, lr 1e-2 flat after 20 warmup steps, accum 85, nsample 5, seeds 500000-500009, --pae-only, --blocks-per-ckpt 0. THE LAUNCH LOG MUST PRINT 254 trainable tensors and NO anchor line at all; an 'anchor SNAPSHOT' line would mean --l2-sp survived and the arm is not what it claims. VAL10 AND VALFULL ARE NOT COMPARABLE." \
  -- --pose-perblock --pose-perblock-copy \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 40 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 --prefetch-workers 4 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "launch_fullhead_noanchor_40ep returned $rc"
exit $rc
