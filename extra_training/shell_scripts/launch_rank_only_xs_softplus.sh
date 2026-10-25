#!/bin/bash
# RANKING-LOSS ARM. See phases/rank_loss.py for why, and docs/next_restart.md for the
# smoke test this has NOT had: nothing below has run on a GPU.
# POSE INJECTION SHARED ACROSS ALL FOUR PAIRFORMER BLOCKS.
#
# WHAT THE ARCHITECTURE DOES. The confidence head embeds the predicted pose ONCE -- a one-hot of
# cdist over the distogram representative atom through linear_no_bias_d, PLUS the raw distance
# through linear_no_bias_d_wo_onehot -- adds both into z_pair, then runs 4 pairformer blocks that
# know nothing about the pose. This re-applies block 0's OWN injection, both terms verbatim, at
# blocks 1, 2 and 3. One matrix pair, four depths, ZERO new parameters.
#
# VERIFIED ON CPU BEFORE LAUNCH, because this mode had never executed: install adds 0 parameters,
# block 0 stays unwrapped, and a wrapped block equals the unwrapped block fed pre-injected z to
# max|diff| 0.00e+00 -- so the injection is exactly z + d(onehot) + d_wo_onehot(dist).
#
# WHAT IT TESTS, AND WHAT IT CANNOT. It tests whether the pose signal DEGRADES crossing the stack,
# by handing every block the identical embedding. It adds ACCESS, not information: the one-hot is
# the same tensor each time and still carries NO ORIENTATION -- pose enters only as a distance
# between representative atoms, never as a backbone frame. If the head's deficit is orientation
# rather than attenuation, this arm cannot find it, and --frame-direction (pathway C) is the one
# that would. With only 4 blocks the attenuation story is thin on its face.
#
# TRAINABLE: linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 parameters,
# 0.052% of the head's 19.75M. Everything else frozen, as in the pose-focused arm.
#
# STEP 0 IS NOT BIT-IDENTICAL TO THE SHIPPED HEAD. Blocks 1-3 gain an injection they never had,
# using weights that are already non-zero, so epoch 0 of this arm is NOT the shipped head. That is
# inherent to sharing a pretrained matrix rather than adding a zero-init one, and it means the arm
# is a different ARCHITECTURE evaluated from the shipped weights, not a one-variable change to
# them. The --pose-perblock and --pose-perblock-tied modes keep bit-identity; this one cannot.

#
# ONE VARIABLE against 8eiek8: the added pairwise ranking term. No L2-SP in this arm --
# L2-SP is 0-for-2 here (4x8t44 null at 2.0% of the loss; wpg274 did not stop the ordering
# collapse) and bundling it would make a ranking result uninterpretable. Everything else
# matches the three
# capacity arms -- 30 structures, seeds 500000-500009, crop 10/5 at 3400, accum 85, nsample 5,
# K=6, lr 1e-2, warmup 20, masked PAE-only, per-epoch val10 + DockQ at budget 30, checkpointing
# off, single-worker prefetch, 15 steps/epoch, 90 units/rank, 1350 poses -- except 100 epochs.
#
# --pose-perblock ADDS NO TENSORS IN THIS MODE but still changes the forward, and the val-loss
# subprocess and the evaluator are both passed the flags (phase_k_train.py forwards them), which
# is the thing that cost ggfhj5 and m8rote their early held-out points when it was missed.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "RANKING LOSS ALONE, CROSS-SEED, SCALED SOFTPLUS. --alpha-pae 0: no PAE gradient is taken at all, so the ONLY objective is pose ordering. Trainable set is the two pose projections -- linear_no_bias_d (256x39) and linear_no_bias_d_wo_onehot (256x1), 10,240 parameters. pae_ln and linear_no_bias_pae stay FROZEN, deliberately: ntwjfp unfroze both and was the only arm ever to go ABOVE shipped val10, reaching epoch 18 at 3.0797 with spread tripling and held-out rank loss moving the wrong way. WHY THE LINEAR FORM WAS ABANDONED. L = -sum (s_i-s_j)(d_i-d_j)/sum|dd| is LINEAR in the scores, so it collapses to sum_i c_i s_i with coefficients from the DockQ labels alone. That is why it needed no scoring pass, and it is the same fact as the defect: a gradient constant in the scores has no stopping point and cannot tell an already-correct pair from an inverted one. Scaling all scores by lambda scales L by lambda, so widening the SPREAD pays without limit and pays better than reordering. Pure sign weighting does not fix it -- it only changes the constants. THE SOFTPLUS AT tau 0.0305. Per-pair gradient weight sigmoid(-m/tau) decays to 0 once a pair is comfortably correct and caps at 1 when badly inverted, so spreading buys a bounded amount and stops. tau is MEASURED, not chosen: the median cross-seed |ipTM gap| over the full training set (497 structures, 1,028 interfaces, 1,056,035 pairs after dropping DockQ ties; run_logging/measure_tau.py). The within-seed median is 0.0081 and does not transfer -- cross-seed pairs are 3.9x wider. At this tau inverted pairs are 43.7 percent of pairs by count but take 81.6 percent of the pairwise score-space gradient mass, against 40.4 percent under the DockQ-weighted linear loss. The 2*tau prefactor makes the tie gradient match the linear loss exactly, so the form change is not also a ~33x learning-rate change. CROSS-SEED GROUPING. Deployment selects argmax of ipTM over ALL poses of an interface, so cross-seed ordering IS the objective; ipTM is 82.3 percent between-seed while DockQ is 54.6 percent pose-level. 2 structures x all 10 seeds x 5 poses = 100 poses per step, which is what the previous schedule already used -- so this buys an EXACT ranking signal (~1,225 cross-seed pairs per interface, cosine 1.000 against the full-group gradient direction) by covering 2 structures per step instead of 6. At 3 seeds that cosine is 0.856 mean / 0.544 p10. No seed sits out, where the old schedule dropped one per structure per epoch. 20 units over 3 ranks is 7/7/6 so the short rank rotates, giving every rank 100 units over the 15 steps. alpha_rank 1 AND lr 1e-2. With no PAE term alpha and lr are redundant -- alpha scales the loss, lr scales the step, only the product matters -- so alpha stays at 1. Probe ev3d6u: |g_rank| 7.96e-02 at alpha 1 against |g_PAE| 1.64e-01, ratio 0.486, cos +0.3311, parity alpha 2.06, and every trainable tensor received finite non-zero ranking gradient THROUGH calculate_chain_based_ptm -- the one thing no offline test can establish. That |g_rank| is 9.2x the 8.61e-03 zbbci6 measured through the SAME frozen readout, which is the cross-seed pair count and the softplus mass redistribution showing up. AdamW normalises by gradient magnitude, so lr sets the step size largely regardless of |g|; 1e-2 is the same lr every other arm used. THE HELD-OUT MONITOR STILL MEASURES AT alpha_pae 1, or val10 PAE would be reported as 0.0 every epoch and the one metric comparable across all arms would vanish. WHAT WOULD FALSIFY THIS: spread running away anyway (watch above 0.05), or held-out rank loss improving while top-1 does not, which would mean margin expansion rather than discrimination. CONTROLS: ggfhj5+zcuiab (PAE only, 26,624 params, best val10 2.7510, rankloss -0.00178 at epoch 1, rho 0.24-0.28) and 8eiek8 (pose-share, no ranking term, 10,240 params -- the same trainable set as this arm)." \
  -- --pose-perblock --pose-perblock-share \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
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
# PROPAGATE IT. The chain runner stops on a non-zero arm, and a trailing bare `echo` always
# returns 0 -- which is why a broken arm reported success and the chain marched straight on to
# the next one.
echo "launcher returned $rc"
exit $rc
