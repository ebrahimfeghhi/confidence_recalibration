#!/bin/bash
# RESUME j583i8 AT EPOCH 8 WITH THE THREE SPEED CHANGES LOADED.
#
# Usage:  ./shell_scripts/launch_resume_j583i8_fast.sh
#
# WHY A RESUME AND NOT A RELAUNCH: epochs 1-7 cost 5.5 hours and the arm is mid-verdict. The
# bundle restores the head, AdamW's moments and the step counter, so the cosine schedule and the
# per-parameter scales continue rather than restarting.
#
# WHAT CHANGES, AND WHY NONE OF IT MOVES THE CURVE:
#   1. 4f4c01b0 shards the held-out PAE pass over --eval-cards instead of running it on the
#      training card alone. Same 7,950 poses, pooled by count, with a SHARD COVERAGE assert that
#      the shards agree on the population and sum to it. 12.8 min -> ~4.3 expected.
#   2. 4f4c01b0 issues the readahead from the thread that performs the read. Scheduling only.
#   3. a16911de defines the training crop from pose 0 instead of the union over all five. The
#      masks are EQUAL -- run_logging/crop_pose_equivalence.py, 0 differences, identical kept
#      token counts -- so the model sees the same tensors; the saving is 88.4 -> 19.1 ms/unit of
#      CPU, ~1.7 min per rank per epoch, and partly hidden behind compute.
#
# THE ONE GUARD FIELD THAT DRIFTS is crop_all_poses, and deliberately. resume_ep7.pt was written
# by a process that loaded its code before a16911de existed, so the field is absent and migrates
# to the pre-change default True, while this run's default is False. 227fd007 is what makes that
# visible at all -- before it the field was saved but never compared, so this change would have
# been accepted in silence in either direction.
#
# EXPECTED READING: the cycle was 44.9 / 46.0 / 45.8 / 39.4 / 41.5 / 47.2 min for epochs 1-6. If
# the changes do what they are priced at, epoch 8 onward lands near 37 min. The curve itself
# should be continuous -- a step in held-out PAE or DockQ at the 7/8 boundary is NOT the expected
# outcome and would mean one of the three claims above is wrong.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
BUNDLE=training/runs/j583i8/resume_ep7.pt
[ -f "$BUNDLE" ] || { echo "REFUSING: $BUNDLE does not exist"; exit 1; }
echo "RESUMING j583i8 at epoch 8 with the three speed changes locked in"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --note "RESUME OF j583i8 AT EPOCH 8, SPEED ONLY. Same arm, same data, same objective: PAE on the full training fold, --pose-perblock-share, 2 tensors / 10,240 parameters, 15 epochs total with cosine lr 1e-2 -> 0. WHAT IS DIFFERENT IS WALL CLOCK, NOT THE MODEL: (1) the held-out PAE pass shards over cards 4,5,7 instead of running on one while two sit at a barrier -- 12.8 min -> ~4.3 expected, same 7,950 poses pooled by count under a SHARD COVERAGE assert; (2) readahead is issued from the thread that reads, so the first pose of a unit does not wait on disk the GPU could have been given; (3) the training crop is built from pose 0 rather than the union over all five, which is the SAME MASK -- crop_pose_equivalence.py finds 0 differences and identical kept-token counts, because the per-pose label permutation is within-residue (symmetric atoms swapped inside one token, so the token's atom-position set and therefore its contact distances are unchanged) -- for 88.4 -> 19.1 ms/unit of CPU. PREDICTION: cycle 46 -> ~37 min, and the held-out PAE and DockQ curves CONTINUE across the 7/8 boundary with no step. A visible step falsifies one of the three claims and is the thing to look for first. WHERE IT STOOD AT EPOCH 7, all on valfull's 92 interfaces against shipped DERIVED FROM THIS RUN'S OWN SHARDS: held-out PAE 2.8305 -> 2.7719 over epochs 1-6 then 2.8016 at epoch 7; selected DockQ 0.5868 / 0.5876 / 0.5886 / 0.5834 / 0.5824 / 0.5832 / 0.5868 against shipped 0.5831; frac 0.7500 x3 then 0.7391 / 0.7283 / 0.7391 / 0.7391 against shipped 0.7391. Paired against shipped on the same interfaces the arm is level: +0.0053 +- 0.0037 at epoch 3, +0.0000 +- 0.0021 at epoch 6, while 37 of 92 interfaces select a DIFFERENT pose -- it reshuffles 40% of the picks and the wins and losses cancel. Spearman is the one steady gain, 0.243 vs shipped 0.2218. TEN-EPOCH MINIMUM still applies and epochs 1-7 carry over, so epoch 10 is the first verdict. GUARD: crop_all_poses drifts by design; resume_ep7.pt predates the flag and migrates to the pre-change True." \
  -- --pose-perblock --pose-perblock-share \
  --resume "$BUNDLE" \
  --resume-override crop_all_poses \
  --resume-note "a16911de changed the crop default from the union over all nsample poses to pose 0; the two masks are equal on every unit measured (crop_pose_equivalence.py: 0 differences, identical kept-token counts), so this is a CPU saving and not a change of input, and the resume is taken deliberately to put the speed changes in flight without discarding epochs 1-7." \
  --train-only 'linear_no_bias_d,linear_no_bias_d_wo_onehot' \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json \
  --manifest-selection training/full_train/structures_no9dzj.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 15 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
echo "launch_resume_j583i8_fast returned $rc"
exit $rc
