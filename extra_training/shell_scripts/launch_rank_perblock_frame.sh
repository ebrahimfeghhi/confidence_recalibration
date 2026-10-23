#!/bin/bash
# RANKING LOSS + L2-SP, WITH DISTANCE *AND* ORIENTATION INJECTED AT ALL FOUR PAIRFORMER BLOCKS.
#
# TWO VARIABLES AGAINST zfahpb/zfahp2, not one, and both are deliberate:
#   1. ORIENTATION IS ADDED. --frame-direction gives the head the unit direction of j in i's
#      backbone frame, three numbers per ordered pair, which no arm in this project has ever
#      trained. Every previous arm saw the pose only as cdist over one representative atom per
#      token, so the frame's orientation was absent from the input rather than merely implicit.
#   2. THE ANCHOR NO LONGER COVERS EVERY TRAINABLE TENSOR. --l2-sp-exclude 'frame_dir' leaves the
#      orientation weights trainable but out of the L2-SP penalty.
# Everything else matches zfahpb: cross-seed softplus at tau 0.0305, --alpha-pae 0, overfit30
# train, val10 held out every epoch at budget 30, 50 epochs, 100 poses/step, lr 1e-2, warmup 20.
#
# PARAMETERS, 44,032 trainable over 12 tensors:
#   ANCHORED at 2.5e-4   linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1)      10,240
#                        3 per-block copies of both                                           30,720
#   NOT ANCHORED         frame_dir_proj LinearNoBias(3 -> 256)                                   768
#                        3 per-block orientation copies                                        2,304
# pae_ln and linear_no_bias_pae stay FROZEN.
#
# WHY THE ORIENTATION WEIGHTS ARE EXCLUDED FROM THE ANCHOR. L2-SP prices distance from the
# SHIPPED weights, which is a prior worth holding for a pretrained matrix. These four tensors are
# ZERO-INIT by construction and verified exactly zero at install, so anchoring them to their init
# is plain decay toward zero -- it penalises the very term being added. AdamW also runs
# weight_decay=0, so they are genuinely unregularised; that is the intent, and the startup log
# names each excluded tensor so the reduced anchor is not something a reader has to infer.
#
# STEP 0 IS UNCHANGED FROM zfahpb's. The orientation projections contribute exactly zero at
# install, so this arm starts from the same function zfahpb did -- bit-identical to
# --pose-perblock-share's step 0 on the distance terms, NOT to the shipped head, because blocks
# 1-3 gain a distance injection they never had.
#
# LAMBDA STAYS 2.5e-4. It was transferred from h7msge's 1e-3 by run_logging/l2sp_lambda.py using
# probe ygpn6t (|g_rank| 4.02e-02 against 7.96e-02 at 10,240 params). The anchored set here is
# the SAME 8 tensors at the same 40,960 parameters as zfahpb, so the transfer is unchanged; the
# 3,072 orientation parameters are outside the anchor and so do not enter it at all.
#
# --blocks-per-ckpt 0 IS LOAD-BEARING, NOT INCIDENTAL. The orientation copies read
# pose_direction's direction cache, which the outer forward clears after the stack. A
# checkpointed block recomputes its forward during backward, after that clear, so the cache would
# be missing or hold a LATER sample's geometry -- pairing one pose's directions with another's
# gradient. pose_perblock.install now REFUSES per-block orientation with blocks_per_ckpt set
# rather than leaving that as a comment someone enables later.
#
# WHAT IT TESTS. Three arms have now failed to beat shipped on frac acceptable: h7msge (pairwise,
# single site), ctij23 (expected-DockQ, single site) and zfahpb/zfahp2 (pairwise, distance at
# four depths). The alignment probes showed the ranking gradient destroys ~91% of its magnitude
# to cancellation at every epoch while PAE destroys almost none, and that the parameter gradient
# collapse is lost SENSITIVITY rather than growing cancellation -- consistent with the trainable
# distance pathway simply having little leverage over ipTM. Orientation is the one input the head
# has never had, so this asks whether the deficit is INFORMATION rather than capacity or
# objective.
#
# WHAT WOULD FALSIFY IT: the orientation weights staying near zero (the loss finds no use for
# them -- check their norm against the anchored tensors' drift), or the same 0.038-0.042 held-out
# plateau and unchanged frac that the three previous arms reached.
#
# CONTROLS: zfahp2 (this arm minus orientation, running), h7msge (single site, same objective and
# anchor), 4y3ve6 (perblock share, no anchor), 8eiek8 (perblock share, no ranking term).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "RANKING LOSS + L2-SP WITH DISTANCE AND ORIENTATION INJECTED AT ALL FOUR PAIRFORMER BLOCKS. TWO VARIABLES AGAINST zfahpb/zfahp2, both deliberate: (1) --frame-direction ADDS ORIENTATION -- the unit direction of j in i's backbone frame, three numbers per ordered pair, which no arm in this project has ever trained, since every previous arm saw the pose only as cdist over one representative atom per token and so had the frame's orientation ABSENT from its input rather than merely implicit; and (2) --l2-sp-exclude 'frame_dir' means THE ANCHOR NO LONGER COVERS EVERY TRAINABLE TENSOR. Everything else matches zfahpb: cross-seed softplus at tau 0.0305, --alpha-pae 0 so ordering is the only term, overfit30 train, val10 held out every epoch at budget 30, 50 epochs, 2 structures x 10 seeds x 5 poses = 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, accum 85, crop 10/5 at 3400, seeds 500000-500009. 44,032 TRAINABLE OVER 12 TENSORS. ANCHORED at lambda 2.5e-4: linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240, plus 3 per-block copies of both = 30,720, so 40,960 over 8 tensors -- the SAME anchored set as zfahpb at the same size, which is why the lambda transfer from h7msge via run_logging/l2sp_lambda.py and probe ygpn6t (|g_rank| 4.02e-02 against 7.96e-02 at 10,240 params) is unchanged. NOT ANCHORED: frame_dir_proj LinearNoBias(3 -> 256) = 768 plus 3 per-block orientation copies = 2,304, so 3,072 over 4 tensors. pae_ln and linear_no_bias_pae stay FROZEN. WHY THE ORIENTATION WEIGHTS ARE OUT OF THE ANCHOR: L2-SP prices distance from the SHIPPED weights, a prior worth holding for a pretrained matrix, but these four tensors are ZERO-INIT by construction and verified exactly zero at install, so anchoring them to their init is plain decay toward zero -- penalising the very term being added. AdamW runs weight_decay=0 so they are genuinely unregularised, which is the intent; the startup log names each excluded tensor. STEP 0 IS UNCHANGED FROM zfahpb's, because the orientation projections contribute exactly zero at install -- so still bit-identical to --pose-perblock-share's step 0 on the distance terms, NOT to the shipped head, since blocks 1-3 gain a distance injection they never had. --blocks-per-ckpt 0 IS LOAD-BEARING: the orientation copies read pose_direction's direction cache, which the outer forward clears after the stack, and a checkpointed block recomputes its forward during backward after that clear -- so the cache would be missing or hold a LATER sample's geometry, pairing one pose's directions with another's gradient. pose_perblock.install REFUSES per-block orientation when blocks_per_ckpt is set. WHAT IT TESTS: whether the deficit is INFORMATION rather than capacity or objective. Three arms have failed to beat shipped on frac acceptable -- h7msge (pairwise, single site), ctij23 (expected-DockQ, single site), zfahpb/zfahp2 (pairwise, distance at four depths) -- and the alignment probes showed the ranking gradient destroys ~91% of its magnitude to cancellation at every epoch while PAE destroys almost none, with the parameter-gradient collapse being lost SENSITIVITY rather than growing cancellation, consistent with the distance pathway having little leverage over ipTM. Orientation is the one input the head has never had. FALSIFIERS: the orientation weights staying near zero (the loss finds no use for them -- compare their norm against the anchored tensors' drift), or the same 0.038-0.042 held-out plateau and unchanged frac the three previous arms reached. CRITERION: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped derived from the same shards (DockQ 0.2909, frac 0.5556, rho 0.1958, oracle 0.5968/1.0000, E[DockQ] 0.2342). Spearman is reported but is NOT the criterion. CONTROLS: zfahp2 (this arm minus orientation), h7msge (single site, same objective and anchor), 4y3ve6 (perblock share, no anchor), 8eiek8 (perblock share, no ranking term)." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp 2.5e-4 --l2-sp-exclude 'frame_dir' \
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
