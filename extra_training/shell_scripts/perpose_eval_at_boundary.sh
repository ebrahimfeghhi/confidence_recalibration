#!/bin/bash
# PAUSE mieumn AT AN EPOCH BOUNDARY, RUN THE PER-POSE PREDICTED-CROP EVALS, THEN RESUME IT.
#
#   nohup ./shell_scripts/perpose_eval_at_boundary.sh 19 > training/runs/perpose_eval.log 2>&1 &
#
# THE EXPERIMENT. --crop-margin crops from label_coordinate; --crop-predicted-margin runs the
# IDENTICAL rule on ci_coordinate, per pose. The ground-truth reference already exists
# (training/runs/cropped_eval/valfull_{ours_ep30,shipped}.json: MAX 0.6178 vs 0.5802, +0.0376), so
# only the predicted passes are new. Same checkpoint, so the two differ only in which coordinates
# define the crop.
#
# PASS 0 IS THE CONTROL AND RUNS FIRST. Per-pose cropping forces one pose per forward pass instead
# of five, and if splitting the pose dimension moved a score on its own, every cropped number
# would be measuring the split. --pose-split-check runs the split UNCROPPED; its rows must
# reproduce training/runs/t4jtjr/valfull_eval_shards_ep30. If it does not match, the cropped
# passes mean nothing and the comparison script says so.
#
# THE RESUME IS THE POINT OF FAILURE THIS SCRIPT IS BUILT AROUND, exactly as in
# predcrop_eval_at_boundary.sh: it runs from an EXIT TRAP so it fires whether the evals succeed,
# fail, or the script is killed, and a sentinel makes it idempotent. Every eval is `|| true` so a
# failed eval cannot cost the run its cards.
#
# NEVER BY PATTERN. Stopping goes through stop_run.sh, which confirms each pid by its --out path.
#
# THE SHIPPED PASS CARRIES NO ARCHITECTURE FLAGS: --frame-direction/--pose-perblock would add
# untrained tensors and the result would not be the shipped head.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID=mieumn
EP=${1:-19}
OUT=training/runs/perpose_eval
CKPT=training/runs/t4jtjr/head_ep30_step3660.pt
SENTINEL=training/runs/.perpose_resumed
say() { echo "$(TZ=America/New_York date '+%F %I:%M:%S %p %Z') $*"; }

mkdir -p "$OUT"
rm -f "$SENTINEL"
echo $$ > training/runs/perpose_eval_pid

resume_run() {
  # IDEMPOTENT FIRST: the sentinel is written before anything else so a double-fire of the trap
  # can never launch two trainers. It marks the resume ATTEMPTED, not succeeded -- those are
  # different facts and the old script conflated them.
  [ -f "$SENTINEL" ] && return 0
  touch "$SENTINEL"
  local CP="training/runs/$RID/resume_ep${EP}.pt"
  if [ ! -f "$CP" ]; then
    say "CANNOT RESUME: $CP does not exist. $RID is STOPPED and needs a manual resume."
    return 1
  fi

  # CONFIRMED FREE CARDS, NOT AN EXPIRED WAIT. The old loop fell through into the launcher when
  # the wait ran out, which would start a trainer on cards another job holds.
  local BUSY=""
  for i in $(seq 1 30); do
    BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
    [ "$BUSY" = "0" ] && break
    say "  cards still busy ($BUSY), waiting 20s"; sleep 20
  done
  if [ "$BUSY" != "0" ]; then
    say "NOT RESUMING: cards still busy ($BUSY) after 10 minutes. Launching here would put a"
    say "              trainer on cards something else holds. $RID is STOPPED and needs a"
    say "              manual resume once the cards are free."
    return 1
  fi

  say "resuming $RID from resume_ep${EP}.pt"
  LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
    --cards 4,5,7 --numa-map 0:0,1:0,2:1 \
    --note "RESUME of the UNCROPPED beta-75 arm $RID from epoch $EP, paused only to free the cards for the per-pose predicted-crop evaluation of the epoch-30 CROPPED head. No training setting changed." \
    -- --resume "$CP" \
    --rank-loss --rank-form rel --rank-cross-seed --alpha-rank 75 \
    --rank-pool mean --rel-labels training/fold/itm_labels \
    --rank-structs-per-step 3 \
    --pose-perblock --pose-perblock-copy --frame-direction \
    --fold training/fold --crop-margin 0 \
    --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
    --manifest-selection training/full_train/structures_winnable.csv \
    --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
    --epochs 60 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
    --lr-final 1e-4 --lr-decay-shape cosine \
    --lr-decay-start-epoch 20 --lr-decay-end-epoch 50 \
    --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
    --save-every 250 \
    --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only \
    --shuffle structure --seeds 500000-500009 \
    --eval-pool training/valfull/pool.csv \
    --eval-pool-fold validation/fold --eval-pool-seed-base 400000 \
    --eval-pool-budget 30 --eval-pool-every 1 \
    --prefetch-units 2 --prefetch-workers 4 \
    --blocks-per-ckpt 1 --dist-backend gloo --dist-timeout-min 240 \
    --require-replay training/ft3_pilot_replay.log --allow-code-drift \
    --max-rss-gb 80 --preload-min-free-gb 260 \
    > training/runs/perpose_resume.log 2>&1
  local RC=$?
  # PROPAGATE THE FAILURE. The old line was `say "resume launcher exited $?"; tail ...`, so the
  # function returned the status of `tail` and a failed launch read as a clean finish.
  if [ $RC -ne 0 ]; then
    say "RESUME LAUNCHER FAILED (exit $RC). $RID is STOPPED. tail:"
    tail -6 training/runs/perpose_resume.log | sed 's/^/    /'
    return 1
  fi
  local NEW
  NEW=$(grep -oE "^run [a-z0-9]{6}" training/runs/perpose_resume.log | head -1 | awk '{print $2}')
  if [ -z "$NEW" ]; then
    say "RESUME NOT CONFIRMED: the launcher exited 0 but printed no run id. $RID is STOPPED."
    tail -6 training/runs/perpose_resume.log | sed 's/^/    /'
    return 1
  fi
  # ATTEMPTED IS NOT RESUMED. The erb7nd resume exited 1 after 0.0 minutes on a missing
  # --require-replay, and nothing noticed because nothing checked that training actually started.
  for i in $(seq 1 40); do
    if [ -s "training/runs/$NEW/trajectory.csv" ]; then
      say "RESUME CONFIRMED: $NEW is training (trajectory.csv has rows)"
      echo "$NEW" > training/runs/${RID}_chain_head
      return 0
    fi
    if grep -qE "^STOP|Traceback" "training/runs/$NEW/run.log" 2>/dev/null; then
      say "RESUME FAILED: $NEW stopped during startup:"
      grep -E "^STOP|Traceback" "training/runs/$NEW/run.log" | head -3 | sed 's/^/    /'
      return 1
    fi
    sleep 30
  done
  say "RESUME UNCONFIRMED: $NEW launched but wrote no trajectory row in 20 minutes. CHECK IT."
  return 1
}

resume_and_report() {
  local PRIOR=$?
  if resume_run; then
    [ $PRIOR -ne 0 ] && say "note: the evals exited $PRIOR, but $RID is back up"
    exit $PRIOR
  fi
  say "EXITING NONZERO: $RID was NOT confirmed back up. This is the failure that needs a human."
  exit 1
}
trap resume_and_report EXIT

# ---- 1. wait for epoch EP: train AND val ------------------------------------------------------
say "armed; waiting for $RID epoch $EP train + val"
while ! grep -qE "^\s*\[rank 0\] epoch ${EP} rank-loss" "training/runs/$RID/run.log" 2>/dev/null; do
  sleep 60
done
say "epoch $EP training closed"
while [ "$(grep -lE 'row\(s\) ->' training/runs/$RID/valfull_eval_shards_ep${EP}/shard*.log 2>/dev/null | wc -l)" -ne 3 ]; do
  sleep 20
done
say "epoch $EP eval done"
[ -f "training/runs/$RID/resume_ep${EP}.pt" ] || { say "REFUSING: no resume_ep${EP}.pt yet"; sleep 120; }
[ -f "training/runs/$RID/resume_ep${EP}.pt" ] || { say "ABORT: still no resume_ep${EP}.pt; NOT stopping the run"; trap - EXIT; exit 1; }

# ---- 2. stop, by --out path, never by pattern -------------------------------------------------
say "stopping $RID"
./shell_scripts/stop_run.sh "$RID" --yes 2>&1 | sed 's/^/  /'
for i in $(seq 1 30); do
  BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  sleep 10
done
BUSY=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { say "ABORT: cards still busy ($BUSY) after the stop"; exit 1; }
say "cards free"

# ---- 3. the control, then the two predicted-crop passes ---------------------------------------
BASE="--fold validation/fold --results validation/fold/results.csv --pool training/valfull/pool.csv --roles validation/fold/chain_roles.csv --seed-base 400000 --labels validation/fold/labels --join-cache training/valfull/join_cache.json --pair-iptm-only --prefetch --pool-mean --select-by mean"
ARCH="--frame-direction --pose-perblock --pose-perblock-copy"
MARGIN=10

REF=training/runs/t4jtjr/valfull_eval_shards_ep30
say "=== PASS 0 (CONTROL): OURS, UNCROPPED, one pose per forward pass ==="
say "    its rows must reproduce $REF"
# NO PIPE. `cmd | tail` reports tail's status, so the previous version could not see a failed
# pass at all -- in a mocked run PASS 0 exited 7, both cropped passes ran anyway and the script
# exited 0. Output goes to a file and the tail is printed separately.
LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py --cards 4,5,7 \
  --ckpt "$CKPT" --budget 30 --work "$OUT/split_check" --out "$OUT/split_check.json" \
  --extra "$BASE $ARCH --pose-split-check" > "$OUT/split_check.log" 2>&1
RC=$?
tail -8 "$OUT/split_check.log" | sed 's/^/    /'
if [ $RC -ne 0 ]; then
  say "ABORT: PASS 0 exited $RC. The cropped passes would be unverifiable, so they are NOT run."
  exit 1
fi

say "--- comparing the control against the batched reference ---"
$PY -I run_logging/compare_rows_identical.py "$OUT/split_check" "$REF" --tol 1e-5 \
  > "$OUT/split_check_compare.log" 2>&1
CMP=$?
cat "$OUT/split_check_compare.log" | sed 's/^/    /'
if [ $CMP -ne 0 ]; then
  say "ABORT: the pose-split control does NOT reproduce the batched scores (exit $CMP)."
  say "       Every cropped number would be measuring the split rather than the crop."
  exit 1
fi
say "control PASSED: splitting the pose dimension does not change a score"

say "=== PASS 1: OURS, PER-POSE crop from PREDICTED coords, margin ${MARGIN} A ==="
LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py --cards 4,5,7 \
  --ckpt "$CKPT" --budget 30 --work "$OUT/ours_pp${MARGIN}" --out "$OUT/ours_pp${MARGIN}.json" \
  > "$OUT/ours_pp${MARGIN}.log" 2>&1 \
  --extra "$BASE $ARCH --crop-predicted-margin ${MARGIN}" || say "PASS 1 exited $?"
tail -8 "$OUT/ours_pp${MARGIN}.log" | sed 's/^/    /'

say "=== PASS 2: SHIPPED (no --ckpt, no arch flags), SAME per-pose crops ==="
LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py --cards 4,5,7 \
  --budget 30 --work "$OUT/shipped_pp${MARGIN}" --out "$OUT/shipped_pp${MARGIN}.json" \
  > "$OUT/shipped_pp${MARGIN}.log" 2>&1 \
  --extra "$BASE --crop-predicted-margin ${MARGIN}" || say "PASS 2 exited $?"
tail -8 "$OUT/shipped_pp${MARGIN}.log" | sed 's/^/    /'

say "all three passes attempted; results:"
ls -1 "$OUT"/*.json 2>&1 | sed 's/^/  /'
# the EXIT trap resumes the run
