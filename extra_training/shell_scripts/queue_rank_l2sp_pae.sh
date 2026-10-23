#!/bin/bash
# QUEUED: when h7msge finishes its 50 epochs, run THE SAME ARM WITH THE PAE LOSS ADDED.
#
# ONE VARIABLE AGAINST h7msge: the PAE term. Everything else is identical -- --l2-sp 1e-3, no
# --pose-perblock (pose enters once, where Protenix put it), --train-only 'linear_no_bias_d'
# (10,240 params, pae_ln and linear_no_bias_pae FROZEN), overfit30, val10 every epoch at budget
# 30, 50 epochs, cross-seed softplus at tau 0.0305, --rank-structs-per-step 2 = 100 poses/step,
# 15 steps/epoch, lr 1e-2, warmup 20, accum 85, nsample 5, crop 10/5 at 3400, seeds
# 500000-500009, checkpointing off, single-worker prefetch.
#
# ALPHAS: --alpha-pae 1 --alpha-rank 2, the same pair arm B (64trgt) ran. Parity came from probe
# ev3d6u -- |g_PAE| 1.64e-01 against |g_rank| 7.96e-02 at alpha 1, ratio 0.486, parity alpha 2.06,
# cos +0.3311 (the two terms AGREE in direction). CAVEAT, STATED BECAUSE IT IS NOT FREE: ev3d6u
# was measured WITH --pose-perblock-share, and this arm has none, so the measured parity is
# inherited rather than re-established. A fresh --grad-probe on this exact configuration would
# settle it; it is not run here because the request was the same arm plus PAE, and changing the
# alphas as well would make the comparison with h7msge two variables instead of one.
#
# THE ANCHOR IS WEAKER HERE, BY ARITHMETIC, AND LAMBDA IS STILL 1e-3. The L2-SP gradient is
# 2*lambda*||dW||, and what it competes against is the TOTAL task gradient. Rank-only that was
# |g_rank| ~ 7.96e-02; with PAE at alpha 1 and rank at alpha 2 it is roughly
# 1.64e-01 + 2*7.96e-02 ~ 3.2e-01, about 4x larger. So at the same drift the anchor prices
# departure about 4x less firmly than it does in h7msge. Holding lambda fixed keeps the
# one-variable comparison; scaling it to 4e-3 to preserve the anchor's RELATIVE strength would be
# a different and also defensible experiment, and is deliberately NOT what this is.
#
# WHAT IT ANSWERS. h7msge asks whether anchoring stops the rank-only overfit. This asks whether
# the ranking term still adds anything once PAE is present AND the departure is priced -- the
# question arm B answered NO to without an anchor (30 epochs, 92 interfaces, no metric above
# shipped, held-out ranking loss rising at nine of ten evaluated epochs). If this reproduces
# 8eiek8 it is a second clean negative for the ranking term.
#
# LAUNCHES ONLY ON exit_code 0. A crash means the cards may be wanted to diagnose h7msge.
# CANCEL: touch training/runs/CANCEL_PAE_ARM, or kill this script BY PID read off ps.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
W=training/runs/h7msge
LOG=training/runs/pae_arm_queue.log
exec >> "$LOG" 2>&1
echo "=== PAE arm armed $(date '+%Y-%m-%d %H:%M:%S %Z'), waiting on h7msge ==="
while :; do
    [ -f training/runs/CANCEL_PAE_ARM ] && { echo "CANCELLED by training/runs/CANCEL_PAE_ARM"; exit 0; }
    if [ -f "$W/status.json" ]; then
        EC=$($PY -I -c "import json;print(json.load(open('$W/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') h7msge ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT LAUNCHING: h7msge exited $EC; the cards may be wanted for it."; exit 1; }
        break
    fi
    sleep 120
done
# NOTHING MUST STILL BE READING THE CARDS. h7msge wrote status.json, so its evaluator shards are
# finished; checked rather than assumed. `grep -v ugrep` and NOT `grep -v " grep "` -- the latter
# does not exclude the Bash tool's ugrep.
LEFT=$(ps -eo pid,cmd | grep -E "python3? -u .*phase_k_train\.py|phase_k_eval" | grep -v ugrep | wc -l)
[ "$LEFT" = "0" ] || { echo "NOT LAUNCHING: $LEFT training/eval process(es) still alive"; ps -eo pid,cmd | grep -E "phase_k_train|phase_k_eval" | grep -v ugrep; exit 1; }
BUSY=$($PY -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "NOT LAUNCHING: $BUSY process(es) occupy cards 4,5,7 besides the holders."
    $PY -I run_logging/cards_busy.py
    exit 1
fi
echo "$(date '+%H:%M:%S') launching the PAE arm"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "RANKING LOSS + PAE + L2-SP ANCHOR, POSE INJECTION AT ITS SINGLE ORIGINAL SITE. ONE VARIABLE AGAINST h7msge: the PAE term. Everything else identical -- --l2-sp 1e-3, no --pose-perblock so step 0 is the shipped head bit-for-bit, --train-only 'linear_no_bias_d' = linear_no_bias_d (256x39) + linear_no_bias_d_wo_onehot (256x1) = 10,240 parameters with pae_ln and linear_no_bias_pae FROZEN, overfit30, val10 every epoch at budget 30, 50 epochs, cross-seed softplus at tau 0.0305, 2 structures x 10 seeds x 5 poses = 100 poses per step, 15 steps/epoch, lr 1e-2, warmup 20, accum 85, nsample 5, crop 10/5 at 3400, seeds 500000-500009. ALPHAS --alpha-pae 1 --alpha-rank 2, the pair arm B ran, from probe ev3d6u: |g_PAE| 1.64e-01 against |g_rank| 7.96e-02 at alpha 1, ratio 0.486, parity alpha 2.06, cos +0.3311 so the terms AGREE in direction. CAVEAT: ev3d6u was measured WITH --pose-perblock-share and this arm has none, so parity is INHERITED, not re-established; a fresh --grad-probe would settle it and was skipped to keep this one variable against h7msge. THE ANCHOR IS ~4x WEAKER HERE BY ARITHMETIC AND LAMBDA IS STILL 1e-3: the L2-SP gradient 2*lambda*||dW|| competes against the TOTAL task gradient, which is ~3.2e-01 with PAE at alpha 1 plus rank at alpha 2 against ~7.96e-02 rank-only, so the same lambda prices departure about 4x less firmly. Holding lambda fixed preserves the one-variable comparison; scaling to 4e-3 to preserve RELATIVE anchor strength is a different experiment and deliberately not this one. WHY THE ANCHOR AT ALL: the two pose projections have ||W_shipped||^2 = 18.18 and the unregularised rank-only control 4y3ve6 moved them to ||dW||^2 = 49.23 by epoch 3 -- 2.7x the squared norm of the weights being updated -- and 1379.21 by epoch 19, while its held-out ranking loss bottomed at epoch 3 and rose thereafter. WHAT THIS ANSWERS: whether the ranking term adds anything once PAE is present AND departure is priced. Arm B (64trgt) answered NO without an anchor: 30 epochs, 92 interfaces, no metric above shipped, rho ending 0.0181 BELOW shipped after peaking +0.0139 at ep18, held-out ranking loss rising at nine of ten evaluated epochs. CONTROLS: h7msge (this arm minus PAE), 4y3ve6 (rank-only, perblock share, no anchor), 8eiek8 (same trainable set, no ranking term), 64trgt (PAE+rank, perblock share, no anchor, valfull). L2-SP IS 0-FOR-2 IN THIS PROJECT: 4x8t44 null at 2.0% of the loss, wpg274 did not stop the ordering collapse. val10 is 18 interfaces and is the HARD TAIL; nothing from 64trgt's 92-interface numbers is comparable." \
  -- --l2-sp 1e-3 \
  --rank-loss --alpha-rank 2.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only 'linear_no_bias_d' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
rc=$?
echo "$(date '+%H:%M:%S') PAE arm launcher returned $rc"
exit $rc
