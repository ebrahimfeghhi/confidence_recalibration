#!/bin/bash
# THE SAME OVERFIT ARM PLUS A POSE INJECTION BEFORE pae_ln, launched when ambgom finishes.
#
# ONE VARIABLE AGAINST ambgom. Identical hard-list pool, identical PAE-only objective with the
# per-list class balance at DockQ 0.23, identical 60 epochs / lr 1e-3 / no dropout / no held-out
# eval, identical extra-eval cadence. The only difference is --pose-readout
# --pose-readout-prenorm --pose-readout-copy-distance.
#
# WHY PRE-NORM. The readout is linear_no_bias_pae(pae_ln(z_pair)). The default --pose-readout site
# adds after the LN, straight into the normalised features, where it can bias the readout up or
# down wholesale. Pre-norm adds into z_pair and lets the LN renormalise, so it can only change the
# DIRECTION of the pair feature -- which is the asked-for site.
#
# WHY COPY-DISTANCE. It matches --pose-perblock-copy, which is how every per-block injection in
# this arm is initialised: block 0's pretrained linear_no_bias_d and linear_no_bias_d_wo_onehot.
# Pre-norm is the one site where that copy is semantically right, because those weights were
# fitted to add into z_pair with a normalisation downstream, and pae_ln is exactly that. The arm
# therefore does NOT start bit-identical to ambgom -- the injection starts active, by request.
#
# EXPECT_TRAINABLE 116, not 113: the site adds pose_readout_d, pose_readout_d_wo_onehot and
# pose_readout_frame_dir (orientation, since --frame-direction is on). The count is a fail-closed
# gate, so if it is wrong the trainer refuses and names the count it found.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid() { ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

PREV="${PREV:-ambgom}"
WANT="${WANT:-60}"
say "waiting for $PREV to reach epoch $WANT"
for i in $(seq 1 720); do
    E=$(ls training/runs/$PREV/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
    [ "${E:-0}" -ge "$WANT" ] && break
    if grep -qE "^STOP:" "training/runs/$PREV/run.log" 2>/dev/null; then
        say "$PREV refused at epoch ${E:-0}; launching anyway is not obviously right, so stopping here"
        grep -m1 "^STOP:" "training/runs/$PREV/run.log" | cut -c1-165
        exit 2
    fi
    sleep 60
done
say "$PREV is at epoch $(ls training/runs/$PREV/head_ep*.pt | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)"

say "waiting for the cards"
for i in $(seq 1 120); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 30
done
[ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] || { say "ABORT: cards busy"; exit 3; }

PRE=$(newest_rid)
say "=== pose injection BEFORE pae_ln, copy-init, 60 epochs on the 151 hard lists ==="
SUBSET_OK=1 \
PREFLIGHT=training/overfit_hard/alignment_preflight.json \
MANIFEST=training/overfit_hard/structures.csv \
SCORED_PAIRS=training/overfit_hard/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
POSE_READOUT=1 POSE_READOUT_PRENORM=1 POSE_READOUT_COPY_DISTANCE=1 \
EXPECT_TRAINABLE=116 \
EXTRA_EVAL="name=hard15,pool=training/overfit_hard/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=training/overfit_hard/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5" \
CARDS=4,5,7 EPOCHS=60 EVAL_EVERY=0 LR=1e-3 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/poseinject_launch.log 2>&1 &
LPID=$!
say "launcher backgrounded as pid $LPID"
RID=""
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { RID="$N"; break; }
    [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:|not found" training/runs/poseinject_launch.log | head -5; exit 4; }
    sleep 20
done
[ -n "$RID" ] || { say "ABORT: no run directory after 30 min"; exit 5; }
echo "$RID" > training/runs/poseinject_rid
echo 0 > "training/runs/${RID}_last_sent"
say "pose-injection arm is $RID"
grep -E "COPY-DISTANCE IS ON AT THE PRE-NORM SITE|pose_readout" "training/runs/$RID/run.log" 2>/dev/null | head -3 | cut -c1-170
