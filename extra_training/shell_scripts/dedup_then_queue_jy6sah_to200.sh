#!/bin/bash
# When jy6sah reaches ep150: clean results.csv FIRST, then continue the same arm to ep200.
#
# ORDER IS THE POINT. phase_j_drop_superseded_rows.py has to replace training/fold/results.csv in
# a window when nothing is reading it, and the only such window is between jy6sah's last train
# eval and the next run's first one. jy6sah's evals read that file, so swapping it mid-run would
# score part of one curve against each version. Doing it here makes the ordering deterministic
# instead of racing a 120 s poll.
#
# WHAT THE SWAP CHANGES, measured: 4 of the 77 overfit30 interfaces carry duplicated pose rows --
# 35uj_AC, 35uj_BC, 8dfh_AB, 8dfh_AC -- with 26 poses disagreeing by more than 0.05 DockQ and some
# by 0.79 (0.858 against 0.069). So the ep200 run's train evals may read those four differently
# from jy6sah's. That is a correction, not a regression: selecting on the resolved file reproduces
# ceiling.json's 0.0420 / 0.0000 over the 77 exactly.
#
# THIS IS A REPLACEMENT for queue_jy6sah_to200.sh, whose process was stopped by pid rather than
# edited: bash reads a script incrementally by byte offset, so editing a running one leaves it
# reading a deleted inode at shifted offsets -- that is how the plot daemon wedged silently.
#
# LAUNCHES ONLY ON exit_code 0, and only if the bundle exists. A crash means the cards may be
# wanted to diagnose jy6sah, and that is the user's call.
#
# CANCEL: touch training/runs/CANCEL_JY6SAH_200, or kill this script BY PID (ps -eo pid,cmd and
# read the pid off the row -- NEVER pkill a pattern, which matches the shell doing the killing and
# has self-killed this session twice).
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
WATCH=training/runs/jy6sah
CANCEL=training/runs/CANCEL_JY6SAH_200
LOG=training/runs/jy6sah_to200_queue.log
exec >> "$LOG" 2>&1
echo "=== dedup-then-queue armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

while :; do
    [ -f "$CANCEL" ] && { echo "$(date '+%H:%M:%S') CANCELLED by $CANCEL"; exit 0; }
    if [ -f "$WATCH/status.json" ]; then
        EC=$($PY -I -c "import json;print(json.load(open('$WATCH/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') jy6sah ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT LAUNCHING: jy6sah exited $EC; the cards may be wanted for it."; exit 1; }
        break
    fi
    sleep 120
done

BUNDLE=$WATCH/resume_ep150.pt
if [ ! -f "$BUNDLE" ]; then
    echo "NOT LAUNCHING: $BUNDLE does not exist. Highest bundle written:"
    ls -1 "$WATCH"/resume_ep*.pt 2>/dev/null | sed 's/.*resume_ep//; s/\.pt//' | sort -n | tail -1
    exit 1
fi

# NOTHING MAY BE READING results.csv HERE. jy6sah has written status.json, so its evaluator
# subprocesses are finished; checked rather than assumed, because a shard still running would be
# reading the file this is about to replace.
LEFT=$(ps -eo pid,cmd | grep phase_k_eval | grep -v " grep " | wc -l)
if [ "$LEFT" != "0" ]; then
    echo "NOT SWAPPING: $LEFT phase_k_eval process(es) still running; they read results.csv."
    ps -eo pid,cmd | grep phase_k_eval | grep -v " grep "
    echo "Launching the ep200 run WITHOUT the swap, so the run is not lost. Apply it by hand later:"
    echo "  $PY -I phases/phase_j_drop_superseded_rows.py --in-place"
else
    echo "$(date '+%H:%M:%S') resolving superseded rows in training/fold/results.csv"
    $PY -I phases/phase_j_drop_superseded_rows.py --in-place --seed-lo 500000 --seed-hi 500009
    echo "$(date '+%H:%M:%S') dedup exit $?"
fi

echo "$(date '+%H:%M:%S') launching the continuation to ep200 from $BUNDLE"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "CONTINUATION of jy6sah to ep200: same lr 1e-2, masked PAE-only, early stop off. The user asked for 200 epochs at 1:25 PM PT after ep121 reached a run-low 1.5424 and ep122-128 sat flat at ~1.556 -- the same shape as the two shelves this lineage already broke through, so it is not being treated as a floor. results.csv was deduplicated immediately before this run started, so 4 of the 77 train-eval interfaces may read differently from jy6sah's." \
  -- --resume "$BUNDLE" \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 200 --lr 0.01 --warmup-steps 0 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 17 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --train-eval-pool training/overfit30/pool.csv --train-eval-every 2 \
  --shuffle structure --seeds 500000-500009 \
  --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
echo "$(date '+%H:%M:%S') launcher returned $?"
