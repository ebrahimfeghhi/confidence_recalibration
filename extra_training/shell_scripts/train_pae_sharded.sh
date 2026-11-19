#!/usr/bin/env bash
# THREE SHARDS OF AN UNCROPPED TRAINING PAE MEASUREMENT, AND WAIT FOR THEM.
#
# WHY A DRIVER AND NOT THREE BACKGROUNDED LAUNCHES: a bare `setsid ... &` from a tool call dies
# when that call returns -- twice today three shards were killed seconds after launch and the only
# evidence was an 848-byte log that stopped at the model build. This script stays in the
# foreground of whatever runs it, so the children outlive their launcher by construction.
#
# BUNDLE_RID is the arm whose checkpoint is measured; its architecture flags come from its own
# args.json through run_logging/valpae_flags.py, never retyped.
set -u
cd "$(dirname "$0")/.."
: "${BUNDLE:?set BUNDLE (empty for shipped)}"
: "${TAG:?set TAG, e.g. armE_ep1}"
: "${FLAGS_RID:?set FLAGS_RID, the arm whose args.json defines the architecture}"
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
mkdir -p "$SD"
/home/ebrahim/Protenix/.venv/bin/python -I run_logging/valpae_flags.py "$FLAGS_RID" >/dev/null
F=$(cat "training/runs/$FLAGS_RID/valpae_flags.txt")
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') uncropped TRAINING PAE, tag=$TAG, flags from $FLAGS_RID"
pids=""
for i in 0 1 2; do
    CARD=$((i==0?4:(i==1?5:7)))
    OUT="training/runs/trainpae_${TAG}_s$i"
    rm -rf "$OUT"
    CARD=$CARD BUNDLE="$BUNDLE" OUT="$OUT" FLAGS="$F" \
        ./shell_scripts/train_pae_once.sh \
        --manifest-selection "training/full_train/structures_bylabel.shard$i.csv" \
        > "$SD/trainpae_${TAG}_s$i.log" 2>&1 &
    pids="$pids $!"
done
echo "  shard pids:$pids"
rc=0
for p in $pids; do wait "$p" || rc=1; done
echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') all shards exited (rc=$rc)"
for i in 0 1 2; do
    printf "  shard %s: %s row(s)\n" "$i" \
        "$(wc -l < "training/runs/trainpae_${TAG}_s$i/trajectory.csv" 2>/dev/null || echo 0)"
done
