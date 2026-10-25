#!/bin/bash
# EXPECTED-DockQ LOSS at SOFTMAX TEMPERATURE tau = 0.05.  THE STARTING POINT: the temperature the experiment is specified around.
#
# THE OBJECTIVE, and the whole point of this arm:
#
#     L = - sum_i softmax(ipTM_i / 0.05) * DockQ_i        over EVERY pose of the interface
#
# It is the expected DockQ of the pose the model would SELECT. Three things follow.
#
# IT IS PAIR-FREE. Every previous arm used a pairwise loss, which spends gradient in proportion
# to how many pairs a pose appears in -- the same for the pose ranked 1st and the pose ranked
# 90th. Deployment takes the argmax and nothing else, so ordering the bottom of the list is work
# that cannot pay. The softmax is a differentiable argmax; tau sets how much mass sits on the
# leaders. phases/test_rank_loss.py measures the consequence directly: for a pose with no chance
# of selection, this form gives a gradient 8.75e-08 of the contender's, where the softplus gives
# that same hopeless pose a gradient ~7e11 times the contender's.
#
# --rank-weight-by-gap AND THE CROSS-SEED PAIR FILTER DO NOT APPLY. There are no pairs to weight
# or to filter. The softmax runs over all 50 poses of the interface by construction, which is the
# "all poses competing for selection" requirement. The banner prints this as inapplicable rather
# than letting a flag appear to be honoured.
#
# THE FLOOR IS NOT ZERO. L lies in [-max(DockQ), -min(DockQ)] for the group, so a perfect model
# scores -max(DockQ), and the log line says so instead of repeating the softplus's "0 is the
# FLOOR".
#
# WHY h7msge MOTIVATES IT. That arm (pairwise softplus + this same anchor) drove the held-out
# pairwise loss to a run best of 0.03679 and lifted Spearman to +0.075 over shipped -- paired
# per-interface +0.0803 +- 0.0416, 1.93 sigma -- while selected DockQ stayed flat-to-down and
# frac acceptable did not beat shipped once in 37 epochs. Better ordering throughout the list,
# same pose on top. The user's diagnosis, which this arm acts on: Spearman weights all pairs
# equally and so does the loss, so the pairs that decide the argmax are a vanishing share of the
# gradient.
#
# ONE VARIABLE AGAINST h7msge: the objective. --l2-sp 1e-3 is UNCHANGED and deliberately so --
# it is held fixed across all three temperatures, so the three arms differ from each other only
# in tau and from h7msge only in the loss. Same trainable set (linear_no_bias_d +
# linear_no_bias_d_wo_onehot = 10,240 params, pae_ln and linear_no_bias_pae frozen), same single
# pose injection site so step 0 is the shipped head bit-for-bit, same overfit30 train and val10
# held-out at budget 30 every epoch, 50 epochs, 100 poses/step, lr 1e-2, warmup 20.
#
# NOTE ON tau: --rank-tau is SHARED with the softplus form, where it means a margin scale in ipTM
# units (0.0305, the measured median cross-seed gap). Here it is a softmax temperature. They are
# not comparable and the default differs by form (DEFAULT_SOFTMAX_TAU = 0.05); the banner states
# which meaning is in force.
#
# WHAT DECIDES IT: selected DockQ and fraction of selected poses above 0.23, both already
# reported per epoch over the 18 val10 interfaces against the shipped head derived from the same
# shards (DockQ 0.2909, frac 0.5556). Spearman is reported but is NOT the criterion -- the point
# of this arm is that the two can move apart.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "EXPECTED-DockQ LOSS at softmax temperature tau 0.05 (THE STARTING POINT). L = -sum_i softmax(ipTM_i/0.05) * DockQ_i over EVERY pose of the interface -- the expected DockQ of the pose the model would SELECT. PAIR-FREE, which is the point: a pairwise loss spends gradient in proportion to how many pairs a pose appears in, the same for rank 1 and rank 90, while deployment takes the argmax and nothing else, so ordering the bottom of the list is work that cannot pay. test_rank_loss.py measures it: a pose with no chance of selection gets 8.75e-08 of the contender's gradient here, against ~7e11 TIMES the contender's under the softplus. --rank-weight-by-gap and the cross-seed pair filter are INAPPLICABLE -- no pairs to weight or filter -- and the softmax covers all 50 poses of the interface by construction. FLOOR IS -max(DockQ) per group, NOT 0. WHY: h7msge (pairwise softplus, same anchor, same trainable set) reached a held-out pairwise best of 0.03679 and Spearman +0.075 over shipped (paired +0.0803 +- 0.0416, 1.93 sigma) while selected DockQ stayed flat-to-down and frac acceptable never beat shipped in 37 epochs -- better ordering, same pose on top. ONE VARIABLE AGAINST h7msge: the objective. --l2-sp 1e-3 unchanged and held fixed across all three temperatures (0.02, 0.05, 0.10), so the three differ from each other only in tau. Trainable set linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 params, pae_ln and linear_no_bias_pae FROZEN, no --pose-perblock so step 0 is the shipped head bit-for-bit, overfit30 train, val10 held out at budget 30 every epoch, 50 epochs, 2 structures x 10 seeds x 5 poses = 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, accum 85, crop 10/5 at 3400, seeds 500000-500009. --rank-tau IS SHARED with the softplus form where it means a margin scale in ipTM units (0.0305 measured); here it is a softmax temperature, not comparable, and DEFAULT_SOFTMAX_TAU is 0.05. WHAT DECIDES IT: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped (DockQ 0.2909, frac 0.5556, oracle 0.5968/1.0000). Spearman is reported but is NOT the criterion." \
  -- --l2-sp 1e-3 \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form expected --rank-tau 0.05 --rank-cross-seed --rank-structs-per-step 2 \
  --train-only 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
# PROPAGATE IT. The chain runner stops on a non-zero arm, and a trailing bare `echo` always
# returns 0 -- which is why a broken arm reported success and the chain marched straight on to
# the next one.
echo "launcher returned $rc"
exit $rc
