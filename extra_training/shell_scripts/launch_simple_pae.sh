#!/bin/bash
# SIMPLE RANKING LOSS + PAE + L2-SP, at the pair_transition-block-3 trainable set.
#
# WHAT THIS DROPS from xmz9ge: --rank-pairs all, --rank-within-weight 8 and --rank-tau 0.0270.
# The ranking term is back to plain cross-seed softplus at DEFAULT_TAU 0.0305.
#
# WHAT THIS ADDS: --alpha-pae 1. Every arm since h7msge has run --alpha-pae 0, making ordering
# the only term. The trainable set is unchanged at 17 tensors / 830,976 parameters.
#
# ALL THREE WEIGHTS ARE MEASURED AT THESE WEIGHTS, from probe jn2pg7:
#
#   alpha_pae  = 1          the reference every PAE arm uses; moving it only rescales the others.
#
#   alpha_rank = 5.347      GRADIENT PARITY AT INIT. The probe reports |g_PAE| 2.5742e-01 against
#                           |g_rank| 4.8142e-02 in the same accumulation window at the same
#                           weights, and prints the parity ratio itself: 5.347. The two terms
#                           then contribute equal gradient magnitude at step 0. Same method that
#                           set arm B's 2.058 from |g_PAE| 1.64e-01 vs |g_rank| 7.96e-02.
#
#   lambda     = 5e-04      ~10x the ranking-only value, because L2-SP strength is
#                           2*lambda*||dW|| / |g_task| and |g_task| is now the COMBINED gradient:
#                           |g_PAE + 5.347 g_rank| = 0.4225, against 0.0481 for ranking alone.
#                           l2sp_lambda.py transfers 24u34d's 2.5e-4 as
#                           2.5e-4 * (0.4225/0.0402) / sqrt(827904/40960) = 5.84e-4 -> 5e-04.
#
# THE TWO OBJECTIVES AGREE, which is the reason to expect anything from this at all:
#     cos(PAE, rank) = +0.3466
# The ranking term reinforces the PAE direction rather than fighting it. That matters because the
# ranking loss ALONE produces per-pose gradients that are as good as mutually independent --
# alignment 0.107 against a 1/sqrt(N) baseline of 0.100, i.e. ~90% of its magnitude destroyed by
# cancellation every step -- while PAE's per-pose gradients are nearly parallel. PAE supplies a
# coherent direction the ranking term can only sharpen.
#
# PARITY HOLDS AT INITIALISATION ONLY. In 4y3ve6 the ranking gradient decayed ~22x over 11 epochs
# as pairs became comfortably ordered, so the ranking term fades relative to PAE as training
# proceeds. That is an accidental annealing schedule and it points the right way, but alpha_rank
# is a STARTING ratio, not one that is maintained.
#
# THE LAMBDA TRANSFER COMPARES UNLIKE QUANTITIES, stated rather than hidden: the old side is a
# ranking-only gradient at alpha_pae 0, the new side is a combined one. The invariant preserved --
# anchor strength relative to the task gradient -- is still the right one, but 5e-04 is a
# principled starting point and not a tuned value.
#
# 25 EPOCHS, reported at 15 as well so it stays comparable to hronsr, bjwcm5 and 6v6bbf, which
# all ran 15.
#
# WHAT WOULD FALSIFY IT: the same 0.037 training-loss plateau xmz9ge hit at epoch 2, with
# selection still swinging. That would say PAE's coherent direction does not rescue a ranking
# signal that cancels itself, and the problem is not the objective's weighting at all.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."; exit 1; }
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "SIMPLE RANKING LOSS + PAE + L2-SP at the pair_transition-block-3 trainable set, 25 epochs. DROPS from xmz9ge: --rank-pairs all, --rank-within-weight 8, --rank-tau 0.0270 -- the ranking term is plain cross-seed softplus at DEFAULT_TAU 0.0305 again. ADDS --alpha-pae 1, where every arm since h7msge ran alpha_pae 0 and made ordering the only term. TRAINABLE SET UNCHANGED at 17 tensors / 830,976 parameters: the 12 orientation+distance tensors plus pairformer_stack.blocks.3.pair_transition. ALL THREE WEIGHTS MEASURED AT THESE WEIGHTS from probe jn2pg7, not guessed. alpha_pae 1 is the reference every PAE arm uses. alpha_rank 5.347 is GRADIENT PARITY AT INIT: the probe reports |g_PAE| 2.5742e-01 against |g_rank| 4.8142e-02 in one accumulation window at identical weights and prints the parity ratio itself, so the two terms contribute equal gradient magnitude at step 0 -- the same method that set arm B's 2.058. lambda 5e-04 is ~10x the ranking-only value because L2-SP strength is 2*lambda*||dW||/|g_task| and |g_task| is now COMBINED: |g_PAE + 5.347 g_rank| = 0.4225 against 0.0481 for ranking alone, so l2sp_lambda.py transfers 24u34d's 2.5e-4 as 2.5e-4 * (0.4225/0.0402) / sqrt(827904/40960) = 5.84e-4, used as 5e-04. THE TWO OBJECTIVES AGREE: cos(PAE, rank) = +0.3466, so the ranking term reinforces the PAE direction rather than fighting it. That is the reason to expect anything here -- the ranking loss ALONE produces per-pose gradients as good as mutually independent (alignment 0.107 against a 1/sqrt(N) baseline of 0.100, ~90% of its magnitude destroyed by cancellation every step) while PAE's are nearly parallel, so PAE supplies a coherent direction the ranking term can sharpen. PARITY HOLDS AT INITIALISATION ONLY: in 4y3ve6 the ranking gradient decayed ~22x over 11 epochs as pairs became ordered, so the ranking term fades relative to PAE as training proceeds -- an accidental annealing schedule that points the right way, but alpha_rank is a STARTING ratio. THE LAMBDA TRANSFER COMPARES UNLIKE QUANTITIES, stated rather than hidden: the old side is ranking-only at alpha_pae 0, the new side combined; the invariant preserved is still anchor strength relative to the task gradient, but 5e-04 is a principled starting point and not a tuned value. CONTROLS: xmz9ge (same trainable set, within-seed weighting, NO PAE -- stopped at epoch 9 having put 89% of its total training-loss fall into epoch 1->2 and then plateaued at 0.0365 while selection swung +-0.07 DockQ per epoch) and hronsr (44,032 parameters, within-seed weighting, no PAE, paired +0.0116 over epochs 8-15). CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, from this run's own shards. Spearman is reported but is NOT the criterion. TEN-EPOCH MINIMUM before any trend claim; reported at 15 as well so it stays comparable to the three 15-epoch arms. FALSIFIER: the same 0.037 plateau at epoch 2 with selection still swinging, which would say PAE's coherent direction does not rescue a ranking signal that cancels itself." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp 5e-04 --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank 5.347 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs cross \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir,pairformer_stack\.blocks\.3\.pair_transition' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 25 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
echo "launch_simple_pae returned $?"
