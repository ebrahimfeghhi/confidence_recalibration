#!/bin/bash
# RESUME zfahpb FROM resume_ep7.pt, carrying its L2-SP anchor across.
#
# WHY THE ANCHOR HAS TO BE COPIED. --l2-sp on a resume reads <out>/l2sp_anchor.pt and REFUSES to
# snapshot a new one, because at resume the weights are N epochs from shipped and snapshotting
# them would anchor the run to its own trained state instead of to the shipped head -- the
# penalty would then price distance from the wrong point and read as ~0 forever. So the original
# anchor must travel with the resume.
#
# AND IT CANNOT BE PRE-PLACED. launch.py refuses a non-empty run directory (a second launch once
# overwrote config.json while the first run's trajectory.csv stayed, leaving a run whose settings
# and records came from different launches). So the only window is after launch.py creates the
# directory and before the trainer reaches its L2-SP block -- which is minutes wide, because
# launch.py runs the ~2 min label preflight in between. This polls for the directory and copies
# the anchor into it immediately.
#
# VERIFIED AFTERWARDS, not assumed: the run log must say "anchor READ from", not "anchor
# SNAPSHOT at init". A SNAPSHOT line means the copy lost the race and the arm is anchored to the
# wrong point; this script reports that as a failure rather than letting it run for 43 epochs.
#
# EVERY OTHER SETTING MATCHES THE PARENT, so no --resume-note is needed: same objective, same
# lambda 2.5e-4, same --pose-perblock --pose-perblock-copy, same --train-only, same data, and
# --epochs 50 so it continues to the original target rather than restarting the count.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
PARENT=zfahpb
EP=7
RID="${1:-zfahp2}"
CK="$(pwd)/training/runs/$PARENT/resume_ep${EP}.pt"
ANC="training/runs/$PARENT/l2sp_anchor.pt"

[ -f "$CK" ] || { echo "STOP: no $CK"; exit 1; }
[ -f "$ANC" ] || { echo "STOP: no $ANC to carry across"; exit 1; }
if [ -d "training/runs/$RID" ] && [ -n "$(ls -A "training/runs/$RID" 2>/dev/null)" ]; then
    echo "STOP: training/runs/$RID exists and is not empty; pick another id"; exit 1
fi
BUSY=$($PY -I run_logging/cards_busy.py)
if [ "$BUSY" != "0" ]; then
    echo "STOP: $BUSY process(es) occupy cards 4,5,7 besides the holders"; exit 1
fi

# The copier runs in the background and wins the race against the ~2 min preflight.
(
  for _ in $(seq 600); do
      if [ -d "training/runs/$RID" ]; then
          cp "$ANC" "training/runs/$RID/l2sp_anchor.pt" && \
              echo "anchor copied into training/runs/$RID" && exit 0
      fi
      sleep 0.2
  done
  echo "ANCHOR COPY FAILED: training/runs/$RID never appeared"
) &
COPIER=$!

LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --run-id "$RID" \
  --note "RESUME of $PARENT from resume_ep${EP}.pt, continuing to epoch 50. Same arm in every setting: cross-seed softplus ranking loss at tau 0.0305 with --alpha-pae 0, pose injected at ALL FOUR pairformer blocks with UNTIED per-block copies (--pose-perblock --pose-perblock-copy), --train-only 'linear_no_bias_d,pose_perblock' = 8 tensors / 40,960 parameters with pae_ln and linear_no_bias_pae FROZEN, L2-SP anchor at lambda 2.5e-4, overfit30 train, val10 held out every epoch at budget 30, 100 poses/step. THE ANCHOR WAS COPIED FROM THE PARENT: --l2-sp on resume reads <out>/l2sp_anchor.pt and refuses to snapshot a new one, because the weights here are 7 epochs from shipped and snapshotting them would anchor the run to its own trained state. The log must say 'anchor READ from' -- a 'SNAPSHOT at init' line would mean the copy lost the race and the arm is anchored to the wrong point. WHY IT WAS INTERRUPTED: stopped at epoch 7 to run alignment probes (||sum_i g_i|| / sum_i ||g_i||) at epochs 1, 3 and 6, separating gradient CANCELLATION from lost SENSITIVITY after the parameter gradient fell 8x while 40% of pairs were still inverted. Parent's state at the stop: DockQ 0.2877 against shipped 0.2909, frac exactly shipped at 0.5556, rho +0.0636, held-out ranking loss 0.03865, drift 4.5, max grad_norm 0.0074." \
  -- --pose-perblock --pose-perblock-copy --l2-sp 2.5e-4 \
  --resume "$CK" \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only 'linear_no_bias_d,pose_perblock' \
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
wait "$COPIER" 2>/dev/null || true
echo "resume launcher returned $rc"
exit $rc
