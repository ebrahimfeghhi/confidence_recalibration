#!/bin/bash
# QUEUED AFTER 24u34d: the same orientation arm, plus the LAST pairformer block's
# pair_transition unfrozen and ANCHORED.
#
# THREE STEPS, IN ORDER, AND IT STOPS RATHER THAN GUESSING:
#   1. wait for 24u34d to write status.json with exit_code 0
#   2. --grad-probe at the new trainable set, to measure |g_rank|
#   3. compute lambda with run_logging/l2sp_lambda.py from that probe, then launch 50 epochs
# If the probe fails, or lambda comes back non-finite, NOTHING LAUNCHES. A wrong lambda is worse
# than no run: it would look like a clean experiment and silently price weight motion wrong.
#
# WHAT IS BEING UNFROZEN. confidence_head.pairformer_stack.blocks.3.pair_transition, the last
# block's pair-wise feed-forward: layernorm1.weight/bias (256 each) plus three linears at
# 262,144 each -- 787,456 parameters over 5 tensors. Block 3 is the last of four.
#
#   trainable   44,032 -> 831,488 over 17 tensors
#   anchored    40,960 -> 828,416 over 13 tensors   (the 4 orientation tensors stay excluded)
#
# WHY LAMBDA CANNOT BE CARRIED OVER. The anchor gradient is 2*lambda*||dW||, and ||dW|| sums
# over every anchored parameter, so a 20.2x larger anchored set makes it sqrt(20.2) = 4.50x
# larger at the same PER-PARAMETER drift. On that alone lambda would go 2.5e-4 -> 5.6e-5. But
# unfreezing a 787k-parameter pretrained layer also changes |g_rank| -- that is the whole point
# of unfreezing it -- and in an unknown direction, so the second factor has to be measured. That
# is what step 2 is for. See docs/softmax_tau_choice.md for the same argument about tau.
#
# THE ORIENTATION TENSORS STAY OUT OF THE ANCHOR and the pair_transition goes IN, which is the
# right split: pair_transition is PRETRAINED, so distance from the shipped weights is a prior
# worth holding, while the orientation projections are zero-init and anchoring them would be
# plain decay on the term being added.
#
# CANCEL: touch training/runs/CANCEL_PAIRTRANS, or kill this script BY PID read off ps.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
PARENT=24u34d
LOG=training/runs/pairtrans_queue.log
TRAIN_ONLY='linear_no_bias_d,pose_perblock,frame_dir,blocks\.3\.pair_transition'
exec >> "$LOG" 2>&1
echo "=== pair_transition arm armed $(date '+%Y-%m-%d %H:%M:%S %Z'), waiting on $PARENT ==="

while :; do
    [ -f training/runs/CANCEL_PAIRTRANS ] && { echo "CANCELLED by training/runs/CANCEL_PAIRTRANS"; exit 0; }
    if [ -f "training/runs/$PARENT/status.json" ]; then
        EC=$($PY -I -c "import json;print(json.load(open('training/runs/$PARENT/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') $PARENT ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT PROCEEDING: $PARENT exited $EC; the cards may be wanted to diagnose it."; exit 1; }
        break
    fi
    sleep 120
done

# NOTHING MUST STILL BE READING THE CARDS. `grep -v ugrep`, NOT `grep -v " grep "` -- the latter
# does not exclude the Bash tool's ugrep.
LEFT=$(ps -eo pid,cmd | grep -E "python3? -u .*phase_k_train\.py|phase_k_eval" | grep -v ugrep | wc -l)
[ "$LEFT" = "0" ] || { echo "NOT PROCEEDING: $LEFT training/eval process(es) still alive"; exit 1; }
BUSY=$($PY -I run_logging/cards_busy.py)
[ "$BUSY" = "0" ] || { echo "NOT PROCEEDING: $BUSY process(es) occupy cards 4,5,7"; exit 1; }

# ---- step 2: probe the new trainable set ------------------------------------------------------
echo "$(date '+%H:%M:%S') probing |g_rank| at the pair_transition trainable set"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "GRADIENT PROBE for the pair_transition arm, NOT A TRAINING RUN. Measures |g_rank| with the last pairformer block's pair_transition unfrozen alongside the orientation arm's tensors, so run_logging/l2sp_lambda.py can transfer lambda honestly: the anchored set goes 40,960 -> 828,416 parameters, a sqrt factor of 4.50 on its own, and unfreezing a 787,456-parameter pretrained layer also changes |g_rank| in an unknown direction. Exits after one accumulation window and writes no head_*.pt." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only "$TRAIN_ONLY" \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --grad-probe --epochs 1 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 100000 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
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
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
PRC=$?
[ "$PRC" = "0" ] || { echo "NOT LAUNCHING: the probe exited $PRC"; exit 1; }
# `ls -t training/runs | head -1` IS WRONG and cost a probe: this script's own log,
# training/runs/fullhead_queue.log, is the newest entry because the script is writing to it, so
# PROBE came back as "fullhead_queue.log" and lambda could not be derived from it. Match only
# DIRECTORIES, and only 6-character run ids, so no stray file or helper dir can win the race.
PROBE=$(ls -td training/runs/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null \
        | grep -xE '[a-z0-9]{6}' | head -1)
[ -n "$PROBE" ] || { echo "NOT LAUNCHING: could not identify the probe run directory"; exit 1; }
echo "$(date '+%H:%M:%S') probe run $PROBE"

# ---- step 3: lambda from the probe, then launch ----------------------------------------------
LAM=$($PY -I run_logging/l2sp_lambda.py --from-run zfahpb --from-probe ygpn6t \
        --to-probe "$PROBE" --new-params 828416 2>/dev/null \
      | grep -oE "nearest value worth using: [0-9.e-]+" | awk '{print $NF}')
case "$LAM" in
    ""|*[!0-9.e-]*) echo "NOT LAUNCHING: could not derive lambda from probe $PROBE (got '$LAM')"; exit 1 ;;
esac
echo "$(date '+%H:%M:%S') lambda from probe $PROBE: $LAM  -- launching 50 epochs"

LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "ORIENTATION ARM + THE LAST PAIRFORMER BLOCK's pair_transition UNFROZEN AND ANCHORED. Everything from 24u34d -- cross-seed softplus ranking loss at tau 0.0305 with --alpha-pae 0, distance AND orientation injected at all four pairformer blocks, overfit30 train, val10 held out every epoch at budget 30, 50 epochs, 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, --blocks-per-ckpt 0 -- plus confidence_head.pairformer_stack.blocks.3.pair_transition: layernorm1.weight/bias (256 each) and three linears at 262,144 each, 787,456 parameters over 5 tensors. TRAINABLE 44,032 -> 831,488 over 17 tensors; ANCHORED 40,960 -> 828,416 over 13, since the 4 zero-init orientation tensors stay excluded via --l2-sp-exclude 'frame_dir' while pair_transition is PRETRAINED and so distance from the shipped weights is a prior worth holding. LAMBDA WAS MEASURED, NOT CARRIED: the anchor gradient is 2*lambda*||dW|| and ||dW|| sums over every anchored parameter, so a 20.2x larger anchored set is sqrt(20.2) = 4.50x stiffer at the same per-parameter drift, and unfreezing a 787k pretrained layer also moves |g_rank|; run_logging/l2sp_lambda.py transferred it from zfahpb/ygpn6t using a fresh probe at this exact trainable set. WHY: four arms have failed to beat shipped on frac acceptable, and the alignment probes showed the ranking gradient destroys ~91% of its magnitude to cancellation at every epoch while the parameter-gradient collapse is lost SENSITIVITY -- the trainable pose pathway having little leverage over ipTM. The frozen readout and stack were the shared suspect across both the 10,240 and 40,960 arms; this gives the loss a large pretrained layer close to the output to move. CRITERION: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped (DockQ 0.2909, frac 0.5556, oracle 0.5968/1.0000). Spearman is reported but is NOT the criterion. CONTROLS: 24u34d (this arm minus the unfrozen layer), zfahp2 (minus orientation too), h7msge (single site)." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --l2-sp "$LAM" --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only "$TRAIN_ONLY" \
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
echo "$(date '+%H:%M:%S') pair_transition arm launcher returned $rc"
exit $rc
