#!/bin/bash
# AFTER THE FOLD: rebuild the labels the new poses need, add the qualifying interfaces to the hard
# TRAINING pool, give the new ones their own eval pool, and resume the arm to epoch 150.
#
# THE STALE LABELS ARE THE SECOND COPY OF THE SAME TRAP. training/fold/labels/<pdb>/free_<seed>.npz
# survived the prune and holds the per-pose permuted ground truth AND the pose-to-sample-index
# mapping, which phase_j establishes BY COORDINATES. The new draws are different poses, so every
# one of those files is wrong for them. They are archived, not deleted, and rebuilt.
#
# THE EVAL POOLS hard15 AND hard1 ARE NOT TOUCHED, so the epoch 1-150 curve stays one series. The
# new interfaces get a THIRD pool, hardnew15, which is a third eval pass on the epochs that eval.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
LIST=docs/degenerate_regen_structures.txt
LO=500001; HI=500014
OH=training/overfit_hard
ARCH=training/fold/archive/stale_labels_degenerate_20261114
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

# ---------------------------------------------------------------- 1. stale labels out of the way
mkdir -p "$ARCH"
N=0
while read -r P; do
  for S in $(seq $LO $HI); do
    F=training/fold/labels/$P/free_$S.npz
    [ -f "$F" ] || continue
    mkdir -p "$ARCH/$P"; mv "$F" "$ARCH/$P/" && N=$((N+1))
  done
done < <(grep -v '^#' $LIST | awk 'NF')
say "archived $N stale label npz -> $ARCH"

# ---------------------------------------------------------------- 2. rebuild them
say "building labels for seeds $LO-$HI"
LIST=$LIST LO=$LO HI=$HI ./shell_scripts/build_extended_train_labels.sh || {
  say "STOPPING: label build failed"; exit 2; }

# ---------------------------------------------------------------- 3. who qualifies
$PY -I run_logging/extend_hard_pool.py --list "$LIST" --out "$OH" --apply || {
  say "STOPPING: pool extension refused"; exit 3; }

# ---------------------------------------------------------------- 4. preflight for the union
say "alignment preflight over the extended manifest"
$PY -I phases/phase_k_alignment_preflight.py --manifest "$OH/structures.csv" \
    --out "$OH/alignment_preflight.json" --workers 12 || { say "STOPPING: preflight failed"; exit 4; }

# ---------------------------------------------------------------- 5. resume
BUNDLE=$(ls -t training/runs/ctgfwj/resume_ep*.pt | head -1)
say "=== resuming from $(basename "$BUNDLE") to epoch 150, extended pool ==="
PRE=$(ls -td training/runs/*/ | head -1 | xargs basename)
SUBSET_OK=1 \
PREFLIGHT=$OH/alignment_preflight.json \
MANIFEST=$OH/structures.csv \
SCORED_PAIRS=$OH/pool.csv \
PAE_SPLIT_ACCEPT=0.23 PAE_BALANCE_ACCEPT=0.23 \
EXTRA_EVAL="name=hard15,pool=$OH/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=$OH/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5;name=hardnew15,pool=$OH/pool_new15.csv,fold=training/fold,seed_base=500000,budget=15,every=5" \
CARDS=4,5,7 EPOCHS=150 EVAL_EVERY=0 LR=5e-4 \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
RESUME="$R/${BUNDLE#./}" \
RESUME_OVERRIDE=manifest_digest,pae_scored_pairs \
RESUME_NOTE="paused at epoch 80 to re-fold 22 degenerate structures at seeds 500001-500014; the interfaces that now qualify as hard (>=1 acceptable, <=10% acceptable over 15 seeds) are added to the TRAINING pool. The hard15 and hard1 eval pools are unchanged so the curve stays one series; the new interfaces are evaluated separately as hardnew15." \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/hard150_resume_extended.log 2>&1 &
LPID=$!
for i in $(seq 1 90); do
  N2=$(ls -td training/runs/*/ | head -1 | xargs basename)
  [ "$N2" != "$PRE" ] && { echo "$N2" > training/runs/hard150_rid; say "resumed as $N2"; break; }
  [ -d "/proc/$LPID" ] || { say "ABORT: launcher exited before a run appeared"
      grep -E "REFUS|STOP:" training/runs/hard150_resume_extended.log | head -5; exit 5; }
  sleep 20
done
say "DONE"
