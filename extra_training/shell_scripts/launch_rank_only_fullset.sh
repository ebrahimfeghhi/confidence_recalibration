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
  --cards 4,5,7 --note "ARM A ON THE FULL TRAINING AND VALIDATION SETS. Same objective and same architecture as 4y3ve6 (stopped at epoch 20); the ONLY changes are the data and the step shape. 24x the training structures, 5.1x the validation interfaces, and an even per-rank split. WHY: 4y3ve6 overfit on 20 structures, and the train-vs-held-out split on the SAME objective showed it unambiguously -- train xsloss fell monotonically 0.05401 -> 0.04026 over 12 epochs while held-out bottomed at epoch 3 (0.03773) and rose to 0.04012, closing the gap to nothing. Overfitting was the binding constraint, not the loss, so the fix is data. DATA. training/full_train_x = 487 structures (full_train's 488 minus 9dzj, which lacks DockQ for seeds 500000-500001 on both its interfaces -- the ranking coverage guard refuses to start without all 10 seeds, and that was caught by pre-check rather than at launch). 4,870 units per epoch against 300 before. Its alignment marker was DERIVED from full_train's passing 488-structure marker by phase_k_derive_preflight_subset.py: the parent digest 77fd7d86b4079a45 was reproduced from its own selection csv, containment confirmed, child digest 258f939fd53397bf over 4,870 units. No trunk was re-read, because every structure here was already checked individually by the parent. VALIDATION. training/valfull = all 53 structures / 92 interfaces of validation/fold, built here from validation/jobs/manifest.csv. Its preflight was generated fresh and PASSED: 53 structures, 1,590 units, 0 mismatches, 0 missing trunks. All 92 pool pairs resolve through chain_roles to in-range model asym indices, and DockQ coverage at budget 30 is complete (2,760 groups, 0 missing, 0 short). None of the 10 skipped training structures appear in the validation set, so the skip is a no-op there and the digest holds. WHY THE FULL VAL SET IS AFFORDABLE NOW. On overfit30 it would have cost +240 percent of an epoch; with 16x the training work it costs about +37 percent. The 18-interface set could not settle this arm's central number: selected-pose DockQ alternated between 0.308-0.311 (epochs 10-13, 15-17, 19) and 0.280-0.282 (epochs 14, 18), and on both dips the paired s.e. doubled from 0.018 to 0.037 -- one interface swinging decided the mean. 92 interfaces cuts that leverage roughly fivefold. STEP SHAPE. --rank-structs-per-step 3 = 3 structures x all 10 seeds x 5 poses = 150 poses per step, 30 units, which divides EVENLY as 10/10/10 across the ranks. That retires the 7/7/6 rotation and the straggler wait it caused in rank_gather (~0.9 min/epoch). 162 steps per epoch, 1,620 over the run. UNCHANGED FROM 4y3ve6, deliberately, so this is a data comparison and not a redesign: ranking loss ALONE (--alpha-pae 0), cross-seed scaled softplus at the measured tau 0.0305, alpha_rank 1 (with no PAE term alpha and lr are redundant, so lr 1e-2 carries it), the two pose projections as the only trainable tensors (10,240 params) with pae_ln and linear_no_bias_pae FROZEN, and --pose-perblock-share, block 0's own pretrained injection re-applied at all four depths with zero new parameters. Untying those (one zero-init matrix per block, 40,960 params, step 0 bit-identical to the shipped head) is a separate arm and was deliberately NOT bundled in here. 10 EPOCHS, NOT 50. One epoch of 487 structures sees more distinct structures than 24 epochs of overfit30 did, so the early turn should be far weaker; 10 is enough to see the shape at ~85 min/epoch. CONTROLS: 4y3ve6 itself (same objective, 20 structures, 18 val interfaces) is the direct comparison. 8eiek8 = the same 10,240 trainable set with NO ranking term, xsloss 0.052-0.055, rho to 0.2800. Shipped on the 18: DockQ 0.2909, frac 0.5556, rho 0.1958, xsloss 0.0803. Those references are 18-interface numbers and are NOT directly comparable to this run's 92-interface ones. ARM B (PAE + the same ranking term at alpha 2, parity from probe ev3d6u) follows this." \
  -- --pose-perblock --pose-perblock-share \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 3 \
  --train-only 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train_x/alignment_preflight.json \
  --manifest-selection training/full_train_x/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 10 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 0.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 81 \
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
# PROPAGATE IT. The chain runner stops on a non-zero arm, and a trailing bare `echo` always
# returns 0 -- which is why a broken arm reported success and the chain marched straight on to
# the next one.
echo "launcher returned $rc"
exit $rc
