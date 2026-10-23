#!/bin/bash
# RANKING LOSS + L2-SP, POSE INJECTED AT ALL FOUR PAIRFORMER BLOCKS WITH UNTIED WEIGHTS.
#
# ONE VARIABLE AGAINST h7msge: the injection architecture. Same objective (cross-seed scaled
# softplus at tau 0.0305, --alpha-pae 0 so ordering is the ONLY term), same data (overfit30 train,
# val10 held out every epoch at budget 30), same 50 epochs, same 100 poses/step, same lr 1e-2.
#
# WHAT CHANGES. h7msge injected the pose once, before the stack. This injects it at blocks 1, 2
# and 3 as well, and each depth gets its OWN COPY of the two projections rather than sharing one
# matrix. 30,720 new parameters, 40,960 trainable across 8 tensors. Each copy is initialised as an
# exact duplicate of block 0's weights and install verifies that against block 0 exactly.
#
# STEP 0 IS NOT THE SHIPPED HEAD. It is bit-identical to --pose-perblock-share's step 0, because
# blocks 1-3 gain an injection they never had, using weights that are already non-zero. Against a
# SHARE arm capacity is the only variable; against h7msge or the shipped head it is not. Stated
# because the single-site arms could claim shipped-head bit-identity and this one cannot.
#
# THE ANCHOR IS ALSO A SOFT TIE, which is the point of untying this way. Every copy is anchored to
# the SAME weights, so the penalty pulls each copy toward the shipped projection AND toward the
# other copies. The blocks differentiate only as far as the ranking loss pays for it, and
# --pose-perblock-share is the lambda -> infinity limit of this arm.
#
# WHY LAMBDA IS 2.5e-4 AND NOT 1e-3, measured by probe ygpn6t on THIS configuration:
#   |g_rank| = 4.0207e-02 across the 8 trainable tensors, against 7.96e-02 for the single-site
#   10,240 set (probe ev3d6u). All 8 tensors received a finite, non-zero ranking gradient through
#   calculate_chain_based_ptm -- the one thing no offline test can establish.
# Two effects compound in the same direction. The task gradient is 0.505x what it was, so a fixed
# lambda is ~2x stiffer relative to it. And 4x the parameters means ||dW|| is ~2x larger at the
# same PER-PARAMETER drift, while the anchor gradient is 2*lambda*||dW|| -- another 2x. At 1e-3
# the anchor would bite ~4x harder per site than it did in h7msge, pinning the copies to block 0
# and quietly reproducing --pose-perblock-share. 1e-3/4 = 2.5e-4 preserves the anchor strength
# that made h7msge's drift plateau in the 4-7 band.
#
# WHAT IT TESTS. h7msge established that the single-site pairwise arm improves its own objective
# (held-out loss 0.0741 -> 0.0368) without improving selection: frac acceptable never beat shipped
# in 43 epochs and 0 of 18 interfaces changed which pose won. Pairs involving the selected pose are
# only 2.29% of that loss, so halving it left selection untouched. This arm asks whether the deficit
# was CAPACITY -- one 256x39 matrix seeing the pose once, four blocks downstream of it -- rather
# than the objective. If it reproduces h7msge, capacity is not the limit either.
#
# WHAT WOULD FALSIFY IT: drift leaving the band (the anchor mis-scaled after all), or the copies
# failing to diverge from block 0 at all (lambda still too stiff -- check ||dW||^2 against 4x the
# single-site value, since 4x the tensors alone accounts for that much).
#
# CONTROLS: h7msge (same objective and anchor, single site, stopped ep43), 4y3ve6 (same objective,
# perblock SHARE, no anchor, overfit from ep3), 8eiek8 (perblock share, no ranking term).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "RANKING LOSS + L2-SP WITH THE POSE INJECTED AT ALL FOUR PAIRFORMER BLOCKS, UNTIED. ONE VARIABLE AGAINST h7msge: the injection architecture. Same objective -- cross-seed scaled softplus at tau 0.0305 with --alpha-pae 0, so pose ordering is the ONLY term -- same overfit30 train and val10 held-out every epoch at budget 30, same 50 epochs, same 2 structures x 10 seeds x 5 poses = 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, accum 85, crop 10/5 at 3400, seeds 500000-500009. ARCHITECTURE: --pose-perblock --pose-perblock-copy gives blocks 1-3 their OWN COPY of linear_no_bias_d (256x39) and linear_no_bias_d_wo_onehot (256x1) instead of sharing block 0's, 30,720 new parameters, 40,960 trainable over 8 tensors, each copy initialised as an exact duplicate of block 0's weights and verified against them at install. pae_ln and linear_no_bias_pae stay FROZEN. STEP 0 IS BIT-IDENTICAL TO --pose-perblock-share's STEP 0, NOT to the shipped head: blocks 1-3 gain an injection they never had, using already non-zero weights. Against a share arm capacity is the only variable; against h7msge it is not. THE ANCHOR IS ALSO A SOFT TIE: every copy is anchored to the SAME shipped weights, so the penalty pulls each copy toward the shipped projection and toward the other copies, letting the blocks differentiate only as far as the ranking loss pays for. --pose-perblock-share is the lambda -> infinity limit of this arm. LAMBDA 2.5e-4, NOT 1e-3, MEASURED BY PROBE ygpn6t ON THIS EXACT CONFIGURATION: |g_rank| 4.0207e-02 over the 8 trainable tensors against 7.96e-02 for the single-site 10,240 set (probe ev3d6u), |g_PAE| 8.5061e-02, cos +0.3298, and all 8 tensors received a finite non-zero ranking gradient through calculate_chain_based_ptm. Two effects compound: the task gradient is 0.505x what it was so a fixed lambda is ~2x stiffer relative to it, and 4x the parameters makes ||dW|| ~2x larger at the same per-parameter drift while the anchor gradient is 2*lambda*||dW|| -- another 2x. At 1e-3 the anchor would bite ~4x harder per site, pinning the copies to block 0 and quietly reproducing the share arm. WHAT IT TESTS: whether h7msge's failure was CAPACITY rather than the objective. That arm drove its held-out pairwise loss 0.0741 -> 0.0368 and rho to +0.075 over shipped while frac acceptable never beat shipped in 43 epochs and 0 of 18 interfaces changed which pose won; pairs involving the selected pose are only 2.29% of that loss, so halving it left selection untouched. If this reproduces h7msge, capacity is not the limit either. FALSIFIERS: drift leaving the band (anchor mis-scaled), or the copies not diverging from block 0 at all (lambda still too stiff -- compare ||dW||^2 against 4x the single-site value, since 4x the tensors alone accounts for that). CRITERION: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped derived from the same shards (DockQ 0.2909, frac 0.5556, rho 0.1958, oracle 0.5968/1.0000). Spearman is reported but is NOT the criterion. CONTROLS: h7msge (same objective and anchor, single site, stopped ep43), 4y3ve6 (same objective, perblock share, no anchor, overfit from ep3), 8eiek8 (perblock share, no ranking term)." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp 2.5e-4 \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir' \
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
