#!/usr/bin/env bash
# Health check + scoring for the self-templating sweep. Runs every INTERVAL seconds, REMEDIATES, and
# escalates rather than retrying blindly.
#
# WHAT IT CHECKS, each one because it has failed for real:
#   workers alive           a silent worker death halves throughput and nothing else notices
#   gpu busy                a worker can be alive and stuck in featurisation forever
#   arms scored advancing   a stalled csv means folds are running but producing nothing
#   template asserts        "templates consumed [0]" means the namespace is not being read correctly
#   claims vs scored        a claim with no rows and no worker is an arm that silently vanished
#   repeated relaunch       relaunching a process that cannot start is not remediation; escalate
#
# Stop with: touch <OUT>/WATCH_STOP
set -u
DRIVER=/home/ebrahim/tts_abag/diffusion_commit/fold.py
PY=/home/ebrahim/Protenix/.venv/bin/python
OUT=${OUT:?set OUT to the run root, e.g. .../artifacts/freespread -- there is no sane default}
QUEUE=${QUEUE:?set QUEUE to the queue tsv}
DTYPE=${DTYPE:-bf16}
NCYCLE=${NCYCLE:-10}
TEMPLATES=${TEMPLATES:-}
CSV=$OUT/results.csv
CLAIM=$OUT/.claims
LOG=$OUT/watch.log
STOP=$OUT/WATCH_STOP
ONLY=${ONLY:-all5}
INTERVAL=${INTERVAL:-300}
STALL_MIN=${STALL_MIN:-25}
GPUS=${GPUS:-"5 6"}
rm -f "$STOP"
last=-1; last_change=$(date +%s); relaunches=0

total=$(awk -F'\t' -v o="$ONLY" '{n=split($5,a," "); if ((o=="all5" && n>1) || (o=="perpose" && n==1) || o=="both") c++} END{print c+0}' "$MAN")

while :; do
  [ -f "$STOP" ] && { echo "[$(date +%H:%M:%S)] stop file seen" | tee -a "$LOG"; exit 0; }
  now=$(date +%s)
  arms=$(awk -F, 'NR>1{print $1"|"$2"|"$3}' "$CSV" 2>/dev/null | sort -u | wc -l)
  claimed=$(ls "$CLAIM" 2>/dev/null | wc -l)
  workers=$(pgrep -u "$USER" -f "fold.py --queue" | wc -l)
  [ "$arms" -ne "$last" ] && { last=$arms; last_change=$now; }
  stalled=$(( (now - last_change) / 60 ))

  msg="[$(date '+%m-%d %H:%M:%S')] $ONLY: $arms/$total arms | claimed $claimed | workers $workers"
  for G in $GPUS; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$G" 2>/dev/null)
    msg="$msg | gpu$G ${u}%"
  done
  [ "$stalled" -ge "$STALL_MIN" ] && msg="$msg | !! NO NEW ARMS ${stalled}min"

  # template asserts and hard errors are failures of the namespace contract, not transient
  tf=$(grep -ch "templates consumed" "$OUT"/gpu*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  [ "${tf:-0}" -gt 0 ] && msg="$msg | !! $tf TEMPLATE ASSERT FAILURES"
  tb=$(grep -l Traceback "$OUT"/gpu*.log 2>/dev/null | tr '\n' ' ')
  [ -n "$tb" ] && msg="$msg | !! Traceback in $tb"
  [ "$workers" -eq 0 ] && [ "$claimed" -gt "$arms" ] && \
    msg="$msg | !! $((claimed - arms)) orphaned claims"
  echo "$msg" | tee -a "$LOG"

  # SCORING REPORT -- the point of the watcher is to show results, not just liveness
  $PY "$DRIVER" --report --csv "$CSV" 2>&1 | tail -25 | tee -a "$LOG"

  if [ "$arms" -ge "$total" ]; then
    echo "[$(date +%H:%M:%S)] ALL $total $ONLY ARMS COMPLETE" | tee -a "$LOG"; exit 0
  fi

  # remediation: one worker per card, and never two on one card (fp32 needs ~90 GB of a 98 GB card)
  for G in $GPUS; do
    if ! pgrep -u "$USER" -f "CUDA_VISIBLE_DEVICES=$G.*fold.py" >/dev/null 2>&1; then
      mine=$(pgrep -u "$USER" -f "fold.py --queue" | wc -l)
      if [ "$mine" -lt 2 ]; then
        if [ "$relaunches" -ge 6 ]; then
          echo "[$(date +%H:%M:%S)] !! $relaunches relaunches with no progress -- NOT retrying, needs a human" \
            | tee -a "$LOG"
        else
          relaunches=$((relaunches+1))
          echo "[$(date +%H:%M:%S)] !! relaunching worker on gpu$G (attempt $relaunches)" | tee -a "$LOG"
          # DTYPE and QUEUE come from the environment: relaunching a bf16 sweep as fp32 -- which a
          # hardcoded --dtype did -- silently changes the precision half way through a dataset.
          CUDA_VISIBLE_DEVICES=$G LAYERNORM_TYPE=openfold nohup $PY "$DRIVER" \
            --queue "$QUEUE" ${TEMPLATES:+--templates "$TEMPLATES"} --out "$OUT" --csv "$CSV" \
            --claim-dir "$CLAIM" --dtype "$DTYPE" --ncycle "$NCYCLE" >> "$OUT/gpu$G.log" 2>&1 &
        fi
      fi
    fi
  done
  sleep "$INTERVAL"
done
