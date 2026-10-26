#!/bin/bash
# EXTEND THE ARM'S HORIZON FROM 15 EPOCHS TO 20, resuming uhyypy at epoch 10.
#
# Usage:  ./shell_scripts/launch_e20.sh
#
# WHY A RESUME IS NEEDED AT ALL: --epochs is read once at launch and bounds the loop, so 20 epochs
# cannot be asked of a process started with 15. Nothing about the objective or the schedule
# changes -- lr_final is 0.0, which is NO DECAY, and the trajectory's lr column is a flat 0.01
# from the end of warmup through step 2108. The horizon is the only difference.
#
# TAKEN NOW RATHER THAN AT EPOCH 15 so there is one stop instead of two, and so the stop happens
# at a boundary we are watching rather than an unattended kill five hours from now.
#
# NO --resume-override THIS TIME, deliberately. uhyypy's own bundles record crop_all_poses=False,
# so the field MATCHES and naming it would be a stale override -- which the guard refuses by
# design, because a name left on a command line disarms that field on the next resume that
# reuses it.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
BUNDLE=training/runs/uhyypy/resume_ep9.pt
[ -f "$BUNDLE" ] || { echo "REFUSING: $BUNDLE does not exist"; exit 1; }
echo "RESUMING uhyypy at epoch 10, horizon extended 15 -> 20 epochs"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "RESUME OF j583i8 AT EPOCH 8, SPEED ONLY. Same arm, same data, same objective: PAE on the full training fold, --pose-perblock-share, 2 tensors / 10,240 parameters, 15 epochs total with cosine lr 1e-2 -> 0. WHAT IS DIFFERENT IS WALL CLOCK, NOT THE MODEL: (1) the held-out PAE pass shards over cards 4,5,7 instead of running on one while two sit at a barrier -- 12.8 min -> ~4.3 expected, same 7,950 poses pooled by count under a SHARD COVERAGE assert; (2) readahead is issued from the thread that reads, so the first pose of a unit does not wait on disk the GPU could have been given; (3) the training crop is built from pose 0 rather than the union over all five, which is the SAME MASK -- crop_pose_equivalence.py finds 0 differences and identical kept-token counts, because the per-pose label permutation is within-residue (symmetric atoms swapped inside one token, so the token's atom-position set and therefore its contact distances are unchanged) -- for 88.4 -> 19.1 ms/unit of CPU. PREDICTION: cycle 46 -> ~37 min, and the held-out PAE and DockQ curves CONTINUE across the 7/8 boundary with no step. A visible step falsifies one of the three claims and is the thing to look for first. WHERE IT STOOD AT EPOCH 7, all on valfull's 92 interfaces against shipped DERIVED FROM THIS RUN'S OWN SHARDS: held-out PAE 2.8305 -> 2.7719 over epochs 1-6 then 2.8016 at epoch 7; selected DockQ 0.5868 / 0.5876 / 0.5886 / 0.5834 / 0.5824 / 0.5832 / 0.5868 against shipped 0.5831; frac 0.7500 x3 then 0.7391 / 0.7283 / 0.7391 / 0.7391 against shipped 0.7391. Paired against shipped on the same interfaces the arm is level: +0.0053 +- 0.0037 at epoch 3, +0.0000 +- 0.0021 at epoch 6, while 37 of 92 interfaces select a DIFFERENT pose -- it reshuffles 40% of the picks and the wins and losses cancel. Spearman is the one steady gain, 0.243 vs shipped 0.2218. TEN-EPOCH MINIMUM still applies and epochs 1-7 carry over, so epoch 10 is the first verdict. GUARD: crop_all_poses drifts by design; resume_ep7.pt predates the flag and migrates to the pre-change True." \
  -- --pose-perblock --pose-perblock-share \
  --resume "$BUNDLE" \
  --train-only 'linear_no_bias_d,linear_no_bias_d_wo_onehot' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 20 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
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
echo "launch_e20 returned $rc"
exit $rc
