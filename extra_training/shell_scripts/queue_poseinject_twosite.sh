#!/bin/bash
# QUEUED AFTER THE 150-EPOCH HARD ARM: pose injection at BOTH readout sites, 150 epochs, with the
# lr drop built into the schedule instead of a stop-and-resume.
#
# TWO SITES, TWO SETS OF WEIGHTS.
#   BEFORE pae_ln, copy-initialised from block 0's pretrained injection: the layernorm divides the
#     magnitude back out, so this site can only turn the pair feature, not scale the readout.
#   AFTER pae_ln, zero-initialised: this one can scale it. Zero-init means step 0 is bit-identical
#     to the pre-norm-only arm, so whatever the second site earns is attributable to it.
# They do NOT share matrices. Sharing would add the pose twice through one weight at two scales,
# which is what pose_readout.py's single-site rule was protecting against.
#
# EXPECT_TRAINABLE 119, not 116: the second site adds pose_readout_post_d,
# pose_readout_post_d_wo_onehot and pose_readout_post_frame_dir.
#
# lr 1e-3 TO EPOCH 50, THEN 5e-4, via --lr-steps. That is the same shape the current chain reached
# by hand (ambgom at 1e-3 to 50, 55reqf at 5e-4 after), so the arms are comparable.
#
# THE POOL IS WHATEVER THE PREVIOUS ARM TRAINED ON, read at launch rather than frozen here: the
# 22 re-folded structures' qualifying interfaces are added to it before that arm resumes.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
OH=training/overfit_hard
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid(){ ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

WANT=${WANT:-150}
say "waiting for the hard arm to reach epoch $WANT"
for i in $(seq 1 2880); do
    RID=$(cat training/runs/hard150_rid 2>/dev/null || true)
    if [ -n "${RID:-}" ] && [ -s "training/runs/$RID/resume_ep$WANT.pt" ]; then
        say "$RID reached epoch $WANT"; break
    fi
    sleep 60
done
RID=$(cat training/runs/hard150_rid 2>/dev/null || true)
[ -n "${RID:-}" ] && [ -s "training/runs/$RID/resume_ep$WANT.pt" ] || {
    say "ABORT: epoch $WANT never arrived"; exit 2; }

for i in $(seq 1 120); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 30
done
[ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] || { say "ABORT: cards busy"; exit 3; }

PRE=$(newest_rid)
say "=== two-site pose injection, 150 epochs, lr 1e-3 -> 5e-4 at epoch 50 ==="
SUBSET_OK=1 \
PREFLIGHT=$OH/alignment_preflight.json \
MANIFEST=$OH/structures.csv \
SCORED_PAIRS=$OH/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 POSE_READOUT_POST_ZERO=1 \
EXPECT_TRAINABLE=119 \
EXTRA_EVAL="name=hard15,pool=$OH/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=$OH/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5;name=hardnew15,pool=$OH/pool_new15.csv,fold=training/fold,seed_base=500000,budget=15,every=5" \
CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=1e-3 LR_STEPS=50:5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=${DRY:-0} \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/poseinject_twosite_launch.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { echo "$N" > training/runs/twosite_rid
        echo "$N" > training/runs/hard150_rid   # the per-epoch plotter follows this
        say "launched as $N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:|not found" training/runs/poseinject_twosite_launch.log | head -5; exit 4; }
    sleep 20
done
say "DONE"
