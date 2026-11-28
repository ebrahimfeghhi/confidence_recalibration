#!/bin/bash
# SELECT -> GATE -> LABELS -> MANIFEST -> GATE AGAIN, for the OST-pair crop pool.
#
# Runs after rebuild_crop_pool_ost.sh. Every step is here rather than typed by hand, because the
# last cropped arm died from an inconsistency between two steps that were run separately.
#
# THE GATE RUNS TWICE ON PURPOSE: once on the selected masks before any label work, so a bad
# selection costs seconds rather than an hour of label building, and once at the end on exactly
# what the launcher will read.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

POOLS="${POOLS:-training/crop_pool_ost_{}}"
FINAL="${FINAL:-training/crop_final_ost}"
LABELS="${LABELS:-training/fold/itm_labels_crop_ost}"
WORKERS="${WORKERS:-16}"
export CROP_THREADS="${CROP_THREADS:-3}"
LOG=training/runs/crop_pool_ost
mkdir -p "$LOG" || exit 1

TOT=$(( WORKERS * CROP_THREADS ))
[ "$TOT" -gt 64 ] && { say "REFUSING: $TOT cores > 64"; exit 2; }

for SZ in 256 384 512 768; do
  D=$(echo "$POOLS" | sed "s/{}/$SZ/")
  N=$(ls "$D"/*.npz 2>/dev/null | wc -l)
  say "pool $SZ: $N structure(s)"
  [ "$N" -lt 800 ] && { say "REFUSING: $D holds only $N -- rebuild is incomplete"; exit 3; }
done

say "SELECT -> $FINAL"
rm -rf "$FINAL"
$PY -I run_logging/select_crop_pool.py --pools "$POOLS" --out "$FINAL" \
    > "$LOG/select.log" 2>&1 || { say "REFUSING: selection failed"; tail -20 "$LOG/select.log"; exit 4; }
tail -6 "$LOG/select.log"
say "selected $(ls "$FINAL"/*.npz 2>/dev/null | wc -l) structure(s)"

say "GATE 1: every selected crop scorable for every pose's OST pair"
$PY -I run_logging/check_crop_pairs.py --pool "$FINAL" > "$LOG/gate1.log" 2>&1
RC=$?
tail -4 "$LOG/gate1.log"
[ "$RC" -eq 0 ] || { say "REFUSING: the NEW pool still has unscorable crops. Stopping before labels."; exit 5; }

say "LABELS -> $LABELS  ($WORKERS shard(s))"
rm -rf "$LABELS"; mkdir -p "$LABELS"
for w in $(seq 0 $(( WORKERS - 1 ))); do
  $PY -I run_logging/build_itm_labels_cropped.py \
      --pool "$FINAL" --out "$LABELS" \
      --workers "$WORKERS" --shard "$w" --skip-existing \
      > "$LOG/labels_${w}.log" 2>&1 &
done
wait
grep -ahE "Traceback|Error|STOP" "$LOG"/labels_*.log | head -5
say "label files: $(ls "$LABELS"/*.npz 2>/dev/null | wc -l)"

# THE MANIFEST IS ITS OWN STEP. A shard cannot know the others finished, and a manifest written
# by the last shard to exit once declared 791 structures complete while 829 were on disk.
say "MANIFEST"
$PY -I run_logging/build_itm_labels_cropped.py --pool "$FINAL" --out "$LABELS" --manifest-only \
    > "$LOG/manifest.log" 2>&1 || { say "REFUSING: manifest step failed"; tail -10 "$LOG/manifest.log"; exit 6; }
$PY -I -c "
import json;m=json.load(open('$LABELS/manifest.json'))
print('  manifest:', {k:m.get(k) for k in ('structures','rows','chain_mapping','d0')})"

say "GATE 2: the same check on exactly what the launcher will read"
$PY -I run_logging/check_crop_pairs.py --pool "$FINAL" > "$LOG/gate2.log" 2>&1
RC=$?
tail -3 "$LOG/gate2.log"
[ "$RC" -eq 0 ] || { say "REFUSING: gate 2 failed"; exit 7; }

say "READY. pool=$FINAL labels=$LABELS"
