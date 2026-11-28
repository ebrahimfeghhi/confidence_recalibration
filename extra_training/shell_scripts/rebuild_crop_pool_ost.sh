#!/bin/bash
# REBUILD THE CROP POOL AGAINST THE PAIRS TRAINING ACTUALLY SCORES.
#
# WHY. The pool shipped to 3pdv7i was validated with interface_chain_pairs -- the STATIC
# chain_roles join -- while phase_k_train scores the PER-POSE OST join. On 90 of 846 structures
# the two name different chains. 9o4o is the one that fired 21 minutes into epoch 1: static calls
# its interfaces asym (0,3) and (1,3), OST calls them (0,2) and (1,2), so the pool certified a
# crop that keeps {0,1,3} and deletes chain 2 -- the chain training needs. Both interfaces scored
# None on every pose and the pass-A completeness guard stopped the run.
#
# WHAT CHANGED: build_crop_pool.py now enumerates the union of every pose's OST pair per
# interface and validates against all of them, because ONE crop is shared by every pose and OST
# resolves the same interface differently across poses of the same structure.
#
# OUTPUT GOES TO NEW DIRECTORIES. The old pools are left untouched -- they are the evidence for
# what 3pdv7i trained against, and nothing here deletes data.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

SIZES="${SIZES:-256 384 512 768}"
WORKERS="${WORKERS:-16}"          # x CROP_THREADS=3 = 48 threads, under the 64-core budget
export CROP_THREADS="${CROP_THREADS:-3}"
LOG=training/runs/crop_pool_ost
mkdir -p "$LOG" || exit 1

# A CORE BUDGET, ASSERTED. The box is shared and the standing limit is 64.
TOT=$(( WORKERS * CROP_THREADS ))
[ "$TOT" -gt 64 ] && { say "REFUSING: $WORKERS workers x $CROP_THREADS threads = $TOT > 64"; exit 2; }
say "rebuild: sizes [$SIZES], $WORKERS workers x $CROP_THREADS threads = $TOT core(s)"

for SZ in $SIZES; do
  OUT="training/crop_pool_ost_${SZ}"
  mkdir -p "$OUT"
  say "size $SZ -> $OUT"
  for w in $(seq 0 $(( WORKERS - 1 ))); do
    $PY -I run_logging/build_crop_pool.py \
        --crop-size "$SZ" --out "$OUT" \
        --workers "$WORKERS" --shard "$w" --skip-existing \
        > "$LOG/build_${SZ}_${w}.log" 2>&1 &
  done
  wait
  N=$(ls "$OUT"/*.npz 2>/dev/null | wc -l)
  say "size $SZ done: $N structure(s)"
  grep -ahE "Traceback|Error|REFUS" "$LOG"/build_${SZ}_*.log | head -5
done

say "ALL SIZES BUILT"
for SZ in $SIZES; do
  echo "  $SZ: $(ls training/crop_pool_ost_${SZ}/*.npz 2>/dev/null | wc -l) structure(s)"
done
