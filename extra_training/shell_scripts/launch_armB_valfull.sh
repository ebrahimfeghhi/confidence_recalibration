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
  --cards 4,5,7 --note "ARM B: PAE PLUS THE CROSS-SEED SCALED SOFTPLUS, on the REDUCED training set, evaluated on the FULL validation set every 3 epochs. --alpha-pae 1 --alpha-rank 2. THE HYPOTHESIS. 4y3ve6 ran this exact objective MINUS the PAE term, on this exact training set and trainable set, and overfit from epoch 3: train xsloss fell monotonically 0.05401 -> 0.04026 over 12 epochs while held-out bottomed at epoch 3 (0.03773) and rose to 0.04012, closing the generalisation gap to nothing. Arm B asks whether PAE's presence is the regulariser that moves that turn later. If it is, the ranking term is usable on 20 structures and no data scaling is needed; if arm B simply reproduces 8eiek8 (xsloss 0.052-0.055, rho to 0.2800), then the ranking term adds nothing once PAE is there, which is a clean negative worth having. alpha_rank 2 IS MEASURED, NOT GUESSED, and needs no new probe: ev3d6u ran arm B's exact configuration -- same trainable set, same loss form, same cross-seed grouping, deliberately at alpha_pae 1 -- and reported |g_PAE| 1.64e-01 against |g_rank| 7.96e-02 at alpha 1, ratio 0.486, so parity is 2.058. cos(g_PAE, g_rank) = +0.3311: the two objectives AGREE, where with the readout unfrozen they opposed at -0.15. Parity holds at INITIALISATION only -- in 4y3ve6 the ranking gradient decayed ~22x over 11 epochs as pairs became comfortably ordered, which is the softplus working, so the ranking term fades relative to PAE as training proceeds. That is an accidental annealing schedule and it points the right way for this arm. WHY THE FULL VAL SET, AND WHY EVERY 3 EPOCHS. The 18-interface val10 set could not settle 4y3ve6's central number: selected-pose DockQ alternated between 0.308-0.311 (epochs 10-13, 15-17, 19) and 0.280-0.282 (epochs 14, 18), and on both dips the paired s.e. doubled from 0.018 to 0.037 -- a single interface swinging decided the mean. training/valfull is all 53 structures and 92 interfaces of validation/fold, which cuts that leverage about fivefold. At one eval per epoch the full set would cost ~25 min against 3.5 min of training; every 3 epochs amortises it to ~8 min/epoch, so 30 epochs is ~6 h rather than ~14. VAL ARTIFACTS, ALL VERIFIED BEFORE LAUNCH. valfull's preflight was generated fresh and PASSED: 53 structures, 1,590 units, 0 mismatches, 0 missing trunks. All 92 pool pairs resolve through chain_roles to in-range model asym indices. DockQ coverage at budget 30 is complete: 2,760 groups, 0 missing, 0 short. The label CACHE under validation/fold/labels had only val10's 10 structures and the first full-val launch refused with 1,290 of 1,590 missing -- results.csv (DockQ scores) and the label cache (per-pose features the loss reads) are DIFFERENT artifacts and checking the first establishes nothing about the second. The other 43 were built by shell_scripts/build_valfull_labels.sh. None of the 10 skipped training structures appear in the validation set, so the skip is a no-op there. STEP SHAPE. --rank-structs-per-step 3 = 3 structures x all 10 seeds x 5 poses = 150 poses, 30 units, which divides EVENLY as 10/10/10 across the ranks on the 300-unit reduced manifest. That retires the 7/7/6 rotation and its straggler wait in rank_gather (~0.9 min/epoch). 10 steps per epoch, 300 over the run -- comparable to 4y3ve6's 300 at 15 steps x 20 epochs. UNCHANGED FROM 4y3ve6 so the PAE term is the only variable: cross-seed scaled softplus at the measured tau 0.0305, the two pose projections as the only trainable tensors (10,240 params) with pae_ln and linear_no_bias_pae FROZEN, --pose-perblock-share (block 0's own pretrained injection re-applied at all four depths, zero new parameters), lr 1e-2. Untying the per-block injections (one zero-init matrix each, 40,960 params, step 0 bit-identical to the shipped head) is a separate arm and was deliberately not bundled here. WHAT WOULD FALSIFY IT: spread above 0.05 (the runaway the softplus exists to prevent); or xsloss improving while rho and top-1 do not, which is margin expansion rather than discrimination. NOTE ON COMPARABILITY: every reference number from earlier arms -- shipped DockQ 0.2909, frac 0.5556, rho 0.1958, xsloss 0.0803, 8eiek8's 0.052-0.055 -- is an 18-INTERFACE figure and is NOT directly comparable to this run's 92-interface ones. The shipped baseline must be recomputed on the 92 before any claim is made against it." \
  -- --pose-perblock --pose-perblock-share \
  --rank-loss --alpha-rank 2.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 3 \
  --train-only 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 30 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 30 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
  --val-loss-every 3 \
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
  --train-eval-every 3 \
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
