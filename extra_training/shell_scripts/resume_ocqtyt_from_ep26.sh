#!/bin/bash
# RESUME ocqtyt from epoch 26 after a deliberate pause to measure the shipped head's train PAE.
#
# WHY THE PAUSE. The epoch-0 train-PAE row had never been measured, and it is the number that
# decides whether the capacity arms are even the right lever: train PAE 2.7384 -> 2.1007 over 26
# epochs is a WEAK fit, and the 30-structure anchor reached 1.4324 in half the steps with the same
# 19.8M parameters. High bias and high variance together is not the regime "reduce capacity" fixes.
# All three cards were busy, --loss-only refuses WORLD_SIZE > 1, so the trainer had to stop.
# Epoch 27 was ~6 minutes in and was discarded; resume_ep26.pt is the last complete bundle.
#
# THIS RESUME IS ALSO THE FIRST TRAINING RUN ON 446 CHANGED LINES. Five commits landed in
# phase_k_train.py while ocqtyt ran (it launched off 324ad1c6): the recorded resume overrides and
# the dataset-item deletion, the two capacity arms, the loss-only isolation fix, the remainder-group
# fix, and multi-worker prefetch. The trainer reads the file fresh from disk, so all of it goes
# live here. The loss-only pass that caused the pause already exercised the shared path -- manifest,
# preflight, the deleted item path, epoch construction -- and reached 4,880 units cleanly.
#
# --prefetch-workers IS DELIBERATELY NOT SET. Its default is 1, which is the single-worker path
# epochs 7-26 ran. The multi-worker path has never run and its worst failure is a silent boundary
# deadlock against --dist-timeout-min 240; it wants a two-epoch smoke test of its own, not a
# 500-epoch resume.
#
# WHAT THE CURVE LOOKS LIKE AT THE POINT OF RESUME. Held-out PAE has CROSSED the shipped head:
#   ep8  train 2.4318  held-out 2.6451   <- best held-out of the run
#   ep24 train 2.1238  held-out 2.8160
#   ep25 train 2.1152  held-out 2.9051
#   ep26 train 2.1007  held-out 3.0841   <- worse than no finetuning at all (shipped 2.9756)
# DockQ did NOT follow it down: ep26 posted the run's best, 0.3477 / 0.6111, against baseline
# 0.2909 / 0.5556. That disagreement is either 18-interface noise or the PAE loss and pose ranking
# coming apart, and it is the second time DockQ peaked on a bad-PAE epoch (ep16 too).
#
# SO THIS SEGMENT IS NOT EXPECTED TO HELP. It continues the curve because the user asked for the
# run to be resumed after the measurement. resume_ep8.pt is the checkpoint worth keeping.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BUSY=$(ps -eo cmd | grep -cE "python3? -u .*phase_k_train\.py" || true)
if [ "$BUSY" -gt 0 ]; then
    echo "REFUSING: $BUSY phase_k_train process(es) are live -- the measurement may still be running."
    exit 1
fi
LAYERNORM_TYPE=openfold /home/ebrahim/Protenix/.venv/bin/python -I run_logging/launch.py \
  --cards 4,5,7 --note "FULL SET segment 5, resumed from ocqtyt/resume_ep26.pt after a deliberate pause to measure the shipped head's train PAE over all 4,880 units (the epoch-0 row that had never been measured). No setting changed. First training run on the 446 lines that landed in phase_k_train.py since 324ad1c6; multi-worker prefetch left at its default 1 worker, which is the path epochs 7-26 ran. Held-out PAE crossed the shipped head's 2.9756 at epoch 26 (3.0841) while epoch 26 posted the run's best DockQ, 0.3477." \
  -- --resume training/runs/ocqtyt/resume_ep26.pt \
  --resume-note "resumed after the shipped-head train-PAE measurement, which needed a card and could not share one; epoch 27 was discarded mid-epoch" \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight.json \
  --manifest-selection training/full_train/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 500 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 244 \
  --pae-scored-pairs training/full_train/pool.csv --pae-only \
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
echo "launcher returned $?"
