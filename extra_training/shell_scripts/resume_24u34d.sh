#!/bin/bash
# RESUME 24u34d FROM resume_ep8.pt, carrying its L2-SP anchor across, continuing to epoch 50.
#
# WHY THIS ARM AND NOT nzqhdf. 24u34d is the only arm that has ever beaten shipped on the
# criterion metric: frac 0.6111 at epochs 5, 6 and 7 against shipped 0.5556, i.e. 11 of 18
# interfaces where every previous arm sat at 10. DockQ climbed to 0.2893 against shipped 0.2909,
# rho was above shipped at every epoch, and -- the part that matters most next to nzqhdf -- its
# training and held-out losses flattened TOGETHER at ~0.040, with no divergence. It was stopped at
# epoch 8, under the ten-epoch minimum, only to free the cards for the capacity experiment.
#
# WHAT THE CAPACITY EXPERIMENT SETTLED. nzqhdf trained the 5.3M reachable parameters at the same
# lambda on the same data and drove the held-out ranking loss to 0.03624 by epoch 2 -- through the
# 0.042 wall four arms had hit -- while selection went to the WORST values on record: DockQ 0.1816,
# frac 0.3333, rho 0.0165 at epoch 4, paired -0.0999 +- 0.0470 (2.13 sigma) against epoch 3. Its
# training loss kept falling while the held-out loss rose, which is the overfitting signature that
# is absent from 24u34d. So lowering this ranking loss further does not improve selection and may
# anti-correlate with it; the 44,032-parameter arm is the one worth finishing.
#
# WHY THE ANCHOR HAS TO BE COPIED. --l2-sp on a resume reads <out>/l2sp_anchor.pt and REFUSES to
# snapshot a new one, because at resume the weights are 8 epochs from shipped and snapshotting them
# would anchor the run to its own trained state instead of to the shipped head -- the penalty would
# then price distance from the wrong point and read as ~0 forever.
#
# AND IT CANNOT BE PRE-PLACED. launch.py refuses a non-empty run directory, so the only window is
# after launch.py creates the directory and before the trainer reaches its L2-SP block -- minutes
# wide, because the ~2 min label preflight runs in between. A background copier polls for it.
#
# VERIFIED AFTERWARDS, NOT ASSUMED: the run log must say "anchor READ from", not "anchor SNAPSHOT
# at init". A SNAPSHOT line means the copy lost the race and the arm is anchored to the wrong
# point; this script reports that as a FAILURE rather than letting it run for 42 epochs.
#
# THE ANCHOR COVERS 8 TENSORS, NOT 12. --l2-sp-exclude 'frame_dir' is applied before the anchor is
# built, so the 4 zero-init orientation tensors stay trainable and OUT of ||W - W0||^2. Their norms
# are read from the checkpoint instead, and they are the named falsifier: they started at exactly
# zero and reached 5.66 / 5.66 / 5.14 / 5.02 by epoch 8, so the loss is using orientation.
#
# --blocks-per-ckpt 0 IS LOAD-BEARING. The per-block orientation copies read pose_direction's
# direction cache, which the outer forward clears after the stack; a checkpointed block recomputing
# during backward would find it missing or holding a later sample's geometry. pose_perblock.install
# refuses the combination, which is what killed 2g65a6 -- loudly, as intended.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
PARENT=24u34d
EP=8
RID="${1:-24u342}"
CK="$(pwd)/training/runs/$PARENT/resume_ep${EP}.pt"
ANC="training/runs/$PARENT/l2sp_anchor.pt"

[ -f "$CK" ] || { echo "STOP: no $CK"; exit 1; }
[ -f "$ANC" ] || { echo "STOP: no $ANC to carry across"; exit 1; }
if [ -d "training/runs/$RID" ] && [ -n "$(ls -A "training/runs/$RID" 2>/dev/null)" ]; then
    echo "STOP: training/runs/$RID exists and is not empty; pick another id"; exit 1
fi
# The bracket keeps this grep from matching its own cmdline in ps; a SIGTERMed rank also takes a
# few seconds to unwind after releasing its GPU, so both checks poll rather than refuse instantly.
for _ in $(seq 60); do
    LEFT=$(ps -eo pid,cmd | grep -cE "python3? -u .*[p]hase_k_train\.py|[p]hase_k_eval" || true)
    BUSY=$($PY -I run_logging/cards_busy.py)
    [ "$LEFT" = "0" ] && [ "$BUSY" = "0" ] && break
    sleep 1
done
[ "$LEFT" = "0" ] || { echo "STOP: $LEFT training/eval process(es) still alive after 60 s"; exit 1; }
[ "$BUSY" = "0" ] || { echo "STOP: $BUSY process(es) occupy cards 4,5,7 after 60 s"; exit 1; }

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
  --note "RESUME of $PARENT from resume_ep${EP}.pt, continuing to epoch 50. Identical in every setting: cross-seed softplus ranking loss at tau 0.0305 with --alpha-pae 0 so ordering is the only term, distance AND orientation injected at ALL FOUR pairformer blocks with untied per-block copies, --train-only 'linear_no_bias_d,pose_perblock,frame_dir' = 12 tensors / 44,032 parameters, L2-SP at lambda 2.5e-4 over 8 anchored tensors with --l2-sp-exclude 'frame_dir' keeping the 4 zero-init orientation tensors trainable but out of the anchor, overfit30 train, val10 held out every epoch at budget 30, 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, --blocks-per-ckpt 0. THE ANCHOR WAS COPIED FROM THE PARENT and the log must say 'anchor READ from'; a 'SNAPSHOT at init' line would mean the copy lost the race and the arm is anchored to its own 8-epoch state instead of the shipped head. WHY IT WAS INTERRUPTED: stopped at epoch 8 to free cards 4,5,7 for nzqhdf, which tested whether the plateau was a capacity limit by training the 5.3M reachable head parameters. It is not: nzqhdf drove the held-out ranking loss to 0.03624 by epoch 2, through the 0.042 wall, while selection fell to the worst on record (DockQ 0.1816, frac 0.3333, rho 0.0165 at epoch 4) with the training loss still falling as the held-out loss rose. 24u34d is the only arm ever to beat shipped on frac acceptable -- 0.6111 at epochs 5-7 against 0.5556 -- with DockQ 0.2893 against 0.2909 and both losses flattening together rather than diverging. PARENT'S STATE AT THE STOP: DockQ 0.2893 +- 0.0531, frac 0.6111 +- 0.1182, rho 0.2150 +- 0.0673, held-out ranking loss 0.04026, E[DockQ] 0.2165 against shipped 0.2572, drift 3.8, orientation norms 5.66/5.66/5.14/5.02 from exact zero. CRITERION: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped DockQ 0.2909 / frac 0.5556 / oracle 0.5968. Spearman is reported but is NOT the criterion. THE OPEN DISAGREEMENT: frac improved while E[DockQ] moved monotonically AWAY from shipped every epoch, and the frac gain is one interface (9c7x_BC crossing 0.23 by 0.011) -- which is exactly why this needs to run past the ten-epoch minimum instead of being judged at 7." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --l2-sp 2.5e-4 --l2-sp-exclude 'frame_dir' \
  --resume "$CK" \
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
wait "$COPIER" 2>/dev/null || true
echo "resume launcher returned $rc"
if grep -q "anchor READ from" "training/runs/$RID/run.log" 2>/dev/null; then
    echo "ANCHOR OK: read from the carried file, not snapshotted"
elif grep -q "anchor SNAPSHOT at init" "training/runs/$RID/run.log" 2>/dev/null; then
    echo "ANCHOR WRONG: SNAPSHOT at init -- the copy lost the race; this arm is anchored to its own 8-epoch state. STOP IT."
fi
exit $rc
