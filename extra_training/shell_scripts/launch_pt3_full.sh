#!/bin/bash
# L8 PLUS BLOCK 3's pair_transition. One variable against hronsr: 786,944 more parameters.
#
# THE FULL-FOLD VERSION of xmz9ge: 487 structures / 977 interfaces training, valfull's 92
# interfaces held out every epoch at budget 30, 12 epochs. Objective identical to xmz9ge and to
# hronsr: --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270, --rank-topk-decay 0,
# softplus, --alpha-pae 0, distance AND orientation at all four pairformer blocks with untied
# per-block copies, 100 poses/step, lr 1e-2, warmup 20, --blocks-per-ckpt 0.
#
# LAMBDA MUST BE RE-DERIVED ON THIS FOLD AND IS A PLACEHOLDER UNTIL IT IS. The overfit30 value
# 5e-05 does NOT transfer: |g_rank| was measured 16.6x SMALLER on the full fold than on
# overfit30 at identical weights, which is why zg5fnw ran 1e-05 where 24u34d ran 2.5e-4 at the
# same 40,960 anchored parameters. Run shell_scripts/probe_pt3_full.sh, then
#   run_logging/l2sp_lambda.py --old-lambda 1e-05 --old-grad 2.4260e-03 --old-params 40960 \
#                              --new-grad <probe> --new-params 827904
# and substitute for LAMBDA_PLACEHOLDER. The script refuses to launch while the placeholder is
# still there, because a silent default here would be the whole experiment.
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
# THE SENTINEL IS BUILT FROM TWO PIECES ON PURPOSE. handover_to_full.sh substitutes the real
# lambda with `sed s/LAMBDA_PLACEHOLDER/<value>/`, which rewrites EVERY line containing that
# token -- including this guard. The guard then read `grep -q 5e-05 "$0"`, which matches the
# --l2-sp argument below and so refused to launch forever, exactly inverting its purpose: a
# check meant to catch an unsubstituted lambda instead blocked a correctly substituted one.
# Splitting the literal means sed cannot see it here, so this line survives substitution intact.
SENTINEL="LAMBDA_""PLACEHOLDER"
# READ FROM THE ARGUMENT LINE, not from anywhere the token appears. A plain
# `grep -oE -- '--l2-sp [^ ]+'` over the whole file matched the PROSE in a header comment
# ("--l2-sp snapshots its anchor at init") and reported the lambda as "snapshots" -- a guard
# that cannot see the real value is worse than none. The flag pair `--l2-sp <v> --l2-sp-exclude`
# occurs only in the command itself; the --note's prose says "via --l2-sp-exclude" with no value
# in between, so it cannot match.
LAM_IN_USE=$(grep -oE -- '--l2-sp [A-Za-z0-9._-]+ --l2-sp-exclude' "$0" | head -1 | awk '{print $2}')
if [ "$LAM_IN_USE" = "$SENTINEL" ] || [ -z "$LAM_IN_USE" ]; then
    echo "REFUSING: --l2-sp is still unsubstituted. Derive it on THIS fold first:"
    echo "  ./shell_scripts/probe_pt3_full.sh"
    echo "  python -I run_logging/l2sp_lambda.py --old-lambda 1e-05 --old-grad 2.4260e-03 \\"
    echo "         --old-params 40960 --new-grad <probe |g_rank|> --new-params 827904"
    echo "The overfit30 value does NOT carry: |g_rank| is 16.6x smaller on the full fold."
    exit 1
fi
echo "launching with --l2-sp $LAM_IN_USE"

LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "L8 PLUS BLOCK 3's pair_transition, ON THE FULL TRAINING FOLD, 12 epochs. The full-fold counterpart of xmz9ge: 487 structures / 977 interfaces training, valfull's 92 interfaces held out every epoch at budget 30. OBJECTIVE IDENTICAL to xmz9ge and hronsr -- --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270, --rank-topk-decay 0, softplus, --alpha-pae 0, distance AND orientation injected at all four pairformer blocks with untied per-block copies, 2 structures x 10 seeds x 5 poses = 100 poses/step, lr 1e-2, warmup 20, accum 85, crop 10/5 at 3400, seeds 500000-500009, --blocks-per-ckpt 0. TRAINABLE SET 17 tensors / 830,976 parameters: the 12 orientation+distance tensors plus pairformer_stack.blocks.3.pair_transition (layernorm1.weight+bias 512, linear_no_bias_a 1024x256, linear_no_bias_b 1024x256, linear_no_bias 256x1024 = 786,944), 18.9x the 44,032 that zg5fnw trained on this same fold. WHY THIS FAMILY: ipTM reads linear_no_bias_pae(pae_ln(z_pair)) and pair_transition acts on z_pair, so unlike the 73.1% of the head that cannot affect ipTM at all every one of these parameters is reachable; across all four blocks pair_transition is 3,147,776 parameters = 59% of the whole 5,313,536-parameter reachable set, and this arm takes the last block's share. LAMBDA IS DERIVED ON THIS FOLD, NOT CARRIED FROM overfit30. The overfit30 arm xmz9ge runs 5e-05, and that value would be badly wrong here: |g_rank| measures 16.6x SMALLER on the full fold than on overfit30 at identical shipped starting weights and the same 100-pose window, which is exactly why zg5fnw ran 1e-05 where 24u34d ran 2.5e-4 at the same 40,960 anchored parameters. The value in --l2-sp here comes from probe_pt3_full.sh plus l2sp_lambda.py against zg5fnw's 1e-05 at |g_rank| 2.4260e-03 and 40,960 anchored parameters, scaled to 827,904. THE NEW TENSORS ARE ANCHORED and the orientation ones are not: L2-SP prices distance from the SHIPPED weights, a prior worth holding for a pretrained matrix, and pair_transition is pretrained, while frame_dir stays in --l2-sp-exclude because those four tensors are ZERO-INIT and anchoring them to their init is plain decay toward zero. CONTROLS: zg5fnw (this fold, this objective's predecessor at --rank-pairs cross and tau 0.0305, 44,032 parameters, stopped at epoch 7 of 12 with DockQ 0.5848 against shipped 0.5831 and frac 0.7391 at parity) and xmz9ge (this exact objective and trainable set on overfit30/val10). WHY THE FULL FOLD MATTERS HERE: val10 has 18 interfaces so frac moves in steps of 0.0556 and a single interface dominates it, while valfull has 92 and moves in steps of 0.0109; the overfit30 arms separated on DockQ but gave contradictory frac readings, which 18 interfaces cannot resolve. NOTE 13 of valfull's 92 interfaces (14.1%) have NO acceptable pose at any rank and are censored in the rank metric; val10 had none. CRITERION: selected DockQ and fraction above 0.23 over valfull's 92 interfaces against shipped derived from this run's own shards. Spearman is reported but is NOT the criterion. TEN-EPOCH MINIMUM before any trend claim, which at 12 epochs leaves little margin. FALSIFIER: landing on zg5fnw's paired -0.0041 +- 0.0064 within error, which would say the 786,944 extra parameters and the within-seed weighting together buy nothing on real data." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp LAMBDA_PLACEHOLDER --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs all --rank-within-weight 8 --rank-tau 0.0270 --rank-topk-decay 0 \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir,pairformer_stack\.blocks\.3\.pair_transition' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 12 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 0.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/valfull/pool.csv --pae-only \
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
echo "launch_pt3_full returned $?"
