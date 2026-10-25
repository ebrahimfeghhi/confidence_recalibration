#!/bin/bash
# L8 PLUS BLOCK 3's pair_transition. One variable against hronsr: 786,944 more parameters.
#
# WHAT IS IDENTICAL TO hronsr (L8): --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270,
# --rank-topk-decay 0, softplus, --alpha-pae 0, distance AND orientation at all four pairformer
# blocks with untied per-block copies, overfit30 train, val10 held out every epoch at budget 30,
# 100 poses/step, lr 1e-2, warmup 20, --blocks-per-ckpt 0.
#
# WHAT CHANGES: --train-only gains 'pairformer_stack\.blocks\.3\.pair_transition', taking the
# trainable set from 12 tensors / 44,032 parameters to 17 / 830,976 -- 18.9x. Block 3 is the LAST
# pairformer block, so its output feeds pae_ln -> linear_no_bias_pae directly.
#
# WHY pair_transition IS WORTH 786,944 PARAMETERS. ipTM reads linear_no_bias_pae(pae_ln(z_pair)),
# and pair_transition acts on z_pair, so unlike the 73.1% of the head that cannot affect ipTM at
# all (docs/confidence_head_reachability.md) every one of these parameters is reachable. Across
# all four blocks pair_transition is 3,147,776 parameters = 59% of the entire 5,313,536-parameter
# reachable set; this arm takes one block's worth.
#
# LAMBDA IS 5e-05, NOT 2.5e-4, and that is the whole reason the probe ran first. L2-SP's strength
# relative to the task is 2*lambda*||dW|| / |g_task|, and both factors moved: the anchored set
# goes 40,960 -> 827,904 parameters (||dW|| grows as its square root) and |g_rank| measured
# 5.1398e-02 here against 4.0207e-02 at the old set (probe n24xyz vs ygpn6t). So
#     lambda_new = 2.5e-4 * (5.1398e-02/4.0207e-02) / sqrt(827904/40960)
#                = 2.5e-4 * 1.2783 / 4.4958 = 7.11e-05  ->  5e-05
# Keeping 2.5e-4 would have pinned the new weights roughly 3.5x harder than hronsr's were.
#
# THE NEW TENSORS ARE ANCHORED, the orientation ones still are not. L2-SP prices distance from the
# SHIPPED weights, which is a prior worth holding for a pretrained matrix -- pair_transition is
# pretrained, so anchoring is meaningful. frame_dir stays in --l2-sp-exclude because those four
# tensors are ZERO-INIT and anchoring them to their init is plain decay toward zero.
#
# WHAT WOULD FALSIFY IT: landing on hronsr's +0.0116 within error, which would say the extra
# 786,944 parameters buy nothing the 44,032 did not already provide. A WORSE result would echo
# nzqhdf, which trained all 5.3M reachable parameters -- mostly pair_transition -- and produced
# the lowest held-out ranking loss on record (0.03624) with the worst selection (DockQ 0.1816,
# frac 0.3333). That arm is the standing warning that capacity here buys loss, not selection.
#
# NOT A RESUME, so --l2-sp snapshots its anchor at init from the SHIPPED head; the log must say
# "anchor SNAPSHOT at init", which is the opposite of what a resume requires.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."; exit 1; }
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "L8 PLUS BLOCK 3's pair_transition, 50 epochs. ONE VARIABLE AGAINST hronsr: the trainable set goes from 12 tensors / 44,032 parameters to 17 / 830,976, an 18.9x increase, by adding pairformer_stack.blocks.3.pair_transition (layernorm1.weight+bias 512, linear_no_bias_a 1024x256, linear_no_bias_b 1024x256, linear_no_bias 256x1024 = 786,944). Everything else matches hronsr exactly: --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270, --rank-topk-decay 0, softplus, --alpha-pae 0, distance AND orientation injected at all four pairformer blocks with untied per-block copies, overfit30 train, val10 held out every epoch at budget 30, 2 structures x 10 seeds x 5 poses = 100 poses/step, lr 1e-2, warmup 20, accum 85, crop 10/5 at 3400, seeds 500000-500009, --blocks-per-ckpt 0. WHY THIS FAMILY: ipTM reads linear_no_bias_pae(pae_ln(z_pair)) and pair_transition acts on z_pair, so unlike the 73.1% of the head that cannot affect ipTM at all every one of these parameters is reachable; across all four blocks pair_transition is 3,147,776 parameters = 59% of the whole 5,313,536-parameter reachable set, and this arm takes the last block's share. LAMBDA IS 5e-05, NOT hronsr's 2.5e-4, derived rather than carried: L2-SP strength relative to the task is 2*lambda*||dW||/|g_task| and both factors moved, the anchored set going 40,960 -> 827,904 parameters (||dW|| grows as its square root) and |g_rank| measuring 5.1398e-02 here against 4.0207e-02 at the old set, so run_logging/l2sp_lambda.py gives 2.5e-4 * 1.2783 / 4.4958 = 7.11e-05, used as 5e-05. Keeping 2.5e-4 would have pinned the new weights ~3.5x harder than hronsr's. PROBE n24xyz also measured rank ALIGNMENT 0.1072 against a 1/sqrt(N) baseline of 0.1000, so the ranking gradient remains as good as mutually independent across poses even at 19x the parameters -- the ~91% cancellation tax is UNCHANGED by capacity. THE NEW TENSORS ARE ANCHORED and the orientation ones still are not: L2-SP prices distance from the SHIPPED weights, a prior worth holding for a pretrained matrix, and pair_transition is pretrained, while frame_dir stays in --l2-sp-exclude because those four tensors are ZERO-INIT and anchoring them to their init is plain decay toward zero. CONTROLS: hronsr (this arm minus the new tensors, 15 epochs, paired +0.0116 against shipped over epochs 8-15, positive in 8 of 8, final DockQ 0.3024 vs shipped 0.2909 at frac parity 0.5556) and nzqhdf (all 5.3M reachable parameters, mostly pair_transition, which drove the held-out ranking loss to the lowest on record at 0.03624 while selection collapsed to DockQ 0.1816 / frac 0.3333 -- the standing warning that capacity here buys loss, not selection). CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, derived from this run's own shards. Spearman is reported but is NOT the criterion. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: landing on hronsr's +0.0116 within error, which would say the extra 786,944 parameters buy nothing the 44,032 did not already provide." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp 5e-05 --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs all --rank-within-weight 8 --rank-tau 0.0270 --rank-topk-decay 0 \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir,pairformer_stack\.blocks\.3\.pair_transition' \
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
echo "launch_pt3_arm returned $?"
