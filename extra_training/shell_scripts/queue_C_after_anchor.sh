#!/bin/bash
# When the lr 1e-2 anchor (335psh) finishes its 50 epochs, run the SAME arm plus pathway C.
#
# ONE VARIABLE. Identical to 335psh in every setting -- lr 1e-2, warmup 20, 50 epochs, masked
# PAE-only, accum 85, same 30 structures, same crop, same seeds, val10 eval every 2 epochs at
# budget 100 -- with --frame-direction added. Fresh from the shipped head, because C adds a
# trainable tensor and no existing bundle contains it.
#
# WHAT C IS: for each ordered pair (i, j), the UNIT direction of j in i's backbone frame, through
# one zero-initialised LinearNoBias(3 -> c_z) added into z_pair beside the two distance paths.
# Zero-init means step 0 is bit-identical to the head without it. The head's only existing view of
# the pose is cdist over the DISTOGRAM representative atom, which is CB, so no backbone frame atom
# is in its input at all and the frame's orientation is absent rather than implicit -- while PAE is
# defined on exactly that orientation. docs/pose_pathways.md has the argument and the corrections.
#
# THE VALIDATION BLOCKER IS GONE. The evaluator used to read frame_atom_index and
# pae_rep_atom_mask from the label npz, and validation/fold has 53 trunks and no labels, so C
# could train but never be scored on held-out structures. pose_direction.derive_frames now
# reconstructs both from atom_to_token_idx and atom_to_tokatom_idx, which the trunk already
# carries -- verified EXACT against the label npz on all 500 training structures, zero mismatches,
# and verified end to end on a val10 trunk that has no labels at all.
#
# LAUNCHES ONLY ON exit_code 0. A crash means the cards may be wanted to diagnose the anchor.
# CANCEL: touch training/runs/CANCEL_C_ARM, or kill this script BY PID read off ps.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
W=training/runs/335psh
LOG=training/runs/C_arm_queue.log
exec >> "$LOG" 2>&1
echo "=== C arm armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
while :; do
    [ -f training/runs/CANCEL_C_ARM ] && { echo "CANCELLED"; exit 0; }
    if [ -f "$W/status.json" ]; then
        EC=$($PY -I -c "import json;print(json.load(open('$W/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') anchor 335psh ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT LAUNCHING: anchor exited $EC; the cards may be wanted for it."; exit 1; }
        break
    fi
    sleep 120
done
# NOTHING MUST STILL BE READING THE CARDS. The anchor wrote status.json, so its evaluator shards
# are finished; checked rather than assumed.
LEFT=$(ps -eo pid,cmd | grep -E "python3? -u .*phase_k_train\.py|phase_k_eval" | grep -v ugrep | wc -l)
[ "$LEFT" = "0" ] || { echo "NOT LAUNCHING: $LEFT training/eval process(es) still alive"; ps -eo pid,cmd | grep -E "phase_k_train|phase_k_eval" | grep -v ugrep; exit 1; }
echo "$(date '+%H:%M:%S') launching the C arm"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "PATHWAY C: identical to the lr 1e-2 anchor 335psh plus --frame-direction. 30 structures, masked PAE-only, lr 1e-2, warmup 20, 50 epochs, early stop off, val10 eval every 2 epochs at budget 100, fresh from the shipped head. One variable against the anchor." \
  -- --frame-direction \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 18 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/val10/pool.csv \
  --train-eval-fold validation/fold \
  --train-eval-seed-base 400000 \
  --train-eval-budget 100 \
  --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "$(date '+%H:%M:%S') launcher returned $?"
