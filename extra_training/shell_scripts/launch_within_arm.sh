#!/bin/bash
# WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS. One variable pair against 24u34d/24u342.
#
# Usage:  ./shell_scripts/launch_within_arm.sh <L> [<K>]
#   L  --rank-within-weight, the within-seed share knob
#   K  --rank-topk-decay, 0 (default) disables it. The k=2 arm runs this on top of the better L.
#
# WHAT CHANGES, AND ONLY THIS:
#   --rank-pairs all          the loss forms BOTH cross-seed and within-seed pairs. 24u342 formed
#                             cross-seed only. Selection is argmax ipTM over all poses regardless
#                             of seed, so within-seed ordering is part of what decides the pick,
#                             and it has never entered this objective.
#   --rank-within-weight L    within-seed pairs scaled by L in numerator AND normaliser, so L is a
#                             SHARE knob and cannot act as a second --alpha-rank (asserted in
#                             test_rank_loss.py: L normalises away when every pair is within-seed).
#   --rank-tau 0.0270         the median |ipTM gap| over ALL training pairs of both families
#                             (1,146,184 pairs, 497 structures), replacing the cross-seed-only
#                             0.0305. Measured with measure_tau.py --pairs all.
# EVERYTHING ELSE IS 24u34d's, flag for flag.
#
# WHY TWO ARMS. L=1 is the control: both families present, natural weighting, within-seed left at
# the ~7.0% of pairwise gradient mass it earns by count. L=8 lifts that to 37.1%. Running L=1
# separately is what makes the comparison attributable to the WEIGHTING rather than to the mere
# presence of within-seed pairs, which also changes tau's population. Without it, a difference
# against 24u342 would confound three changes.
#
# WHY tau IS NOT RETUNED PER L. tau is fixed to a property of the DATA (the median gap over the
# training pairs), not to the knob. It does NOT equal the w_ij-weighted median -- that is 0.1329
# at L=1 and 0.0794 at L=8, both well above tau -- and there is no rule requiring it to. Freezing
# tau is what keeps the two arms a one-variable comparison.
#
# WHAT WOULD FALSIFY IT: L=8 landing on L=1 within error on selected DockQ and frac acceptable
# over the 18 val10 interfaces, which would say the within-seed family carries no usable signal at
# any weight this objective can give it.
#
# NOT A RESUME, so no anchor is carried: --l2-sp snapshots at init from the SHIPPED head, which is
# the correct anchor point for a fresh run. The log should say "anchor SNAPSHOT at init" here --
# the opposite of what a resume requires.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
L="${1:-}"
K="${2:-0}"
[ -n "$L" ] || { echo "usage: $0 <within-weight L> [topk-decay K]"; exit 1; }
BUSY=$($PY -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "REFUSING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    exit 1
fi

SHARE=$(case "$L" in 1) echo "7.0%";; 3) echo "18.3%";; 8) echo "37.1%";; 20) echo "59.3%";; *) echo "see run_logging/within_weight.py";; esac)

LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "WITHIN-SEED PAIRS ADDED TO THE RANKING LOSS, --rank-within-weight $L, --rank-topk-decay $K, 25 epochs. Identical to 24u34d/24u342 in every other setting, so this is a one-variable-pair comparison against the only arm that has ever beaten shipped on frac acceptable. WHAT CHANGES: (1) --rank-pairs all, so the loss forms BOTH cross-seed and within-seed pairs where 24u342 formed cross-seed only -- deployment selects argmax ipTM over ALL poses regardless of seed, so ordering inside a seed is part of what decides the pick and has never entered this objective; (2) --rank-within-weight $L scales within-seed pairs in the numerator AND the normaliser W, making L a SHARE knob that cannot act as a second --alpha-rank (test_rank_loss.py asserts L normalises away when every pair is within-seed, has no effect under --rank-pairs cross, leaves the pair count alone, and keeps coefficients summing to zero); (3) --rank-tau 0.0270, the median |ipTM gap| over ALL 1,146,184 training pairs of both families across 497 structures, replacing the cross-seed-only DEFAULT_TAU of 0.0305. At L=$L within-seed pairs carry $SHARE of the pairwise score-space gradient mass against 7.0% at L=1 (reproduce: run_logging/within_weight.py --tau 0.0270), while mass reaching INVERTED pairs falls only 75.0% to 71.6% between L=1 and L=8. WHY THE WEIGHT AND NOT A SECOND TAU: within-seed pairs were never short of gradient -- at tau 0.0270 a median within-seed pair on overfit30 (gap 0.0137) weights 0.376 correctly ordered against 0.624 inverted, real discrimination, and both directions push toward the better-DockQ pose so they ADD rather than cancel. What they lack is SHARE: 7.7% of pairs by count at the 10 training seeds. A multiplier is monotone in the quantity being targeted and can be set from a measurement; a second tau changes the weighting's SHAPE and its effect on mass share is not monotone. WHY TAU IS NOT RETUNED PER ARM: tau is fixed to a property of the data rather than to the knob, so the two arms differ in one thing. It does NOT equal the w_ij-weighted median (0.1329 at L=1, 0.0794 at L=8, both above tau) and no rule requires it to; DEFAULT_TAU has always been an unweighted median while the loss ran weight_by_gap=True. PARAMETERS, 44,032 over 12 tensors, unchanged from 24u34d: ANCHORED at lambda 2.5e-4, linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 plus 3 per-block copies = 30,720, so 40,960 over 8 tensors; NOT ANCHORED, frame_dir_proj LinearNoBias(3 -> 256) = 768 plus 3 per-block orientation copies = 2,304 over 4 tensors, excluded via --l2-sp-exclude 'frame_dir' because they are ZERO-INIT and anchoring them to their init would be plain decay toward zero, penalising the term being added. pae_ln and linear_no_bias_pae stay FROZEN. --blocks-per-ckpt 0 IS LOAD-BEARING: the per-block orientation copies read pose_direction's cache, which the outer forward clears after the stack, so a checkpointed block would pair one pose's directions with another's gradient. NOT A RESUME, so --l2-sp snapshots the anchor at init from the SHIPPED head; the log should say 'anchor SNAPSHOT at init' here, which is the opposite of what a resume requires. CONTROLS: 24u342 (this arm minus within-seed pairs, tau 0.0305, 50 epochs, final DockQ 0.2790 +- 0.0568, frac 0.5556, rho 0.2163, paired first-acceptable-rank negative in all 33 evaluated epochs 18-50 with mean -1.25) and the companion arm at the other L. CRITERION: selected DockQ and fraction above 0.23 over the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968, derived from this run's own shards. Spearman is reported but is NOT the criterion. TOP-k RANK DECAY K=$K: 0 means off; above 0 each pair is weighted by max(v(r_i), v(r_j)) with v(r)=exp(-(r-1)/K) over the current ipTM order, concentrating the loss on the pairs that can change the argmax -- at K=2, 39.3% of weight on pairs touching rank 1 against 1.6% undecayed, and 92.8% inside the top 5. Ranks come from DETACHED scores, so the weight is constant in the backward and this is not the exact gradient of a fixed objective. TEN-EPOCH MINIMUM before any trend claim. FALSIFIER: L=8 landing on L=1 within error on DockQ and frac, which would say the within-seed family carries no usable signal at any weight this objective can give it." \
  -- --pose-perblock --pose-perblock-copy --frame-direction --l2-sp 2.5e-4 --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --rank-pairs all --rank-within-weight "$L" --rank-tau 0.0270 \
  --rank-topk-decay "$K" \
  --train-only 'linear_no_bias_d,pose_perblock,frame_dir' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 25 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
echo "launch_within_arm L=$L K=$K returned $rc"
exit $rc
