#!/bin/bash
# Block until <run>'s epoch <N> lands, then print everything worth reading AND BUILD THE PLOTS.
# Exiting notifies the session, which sends the plots and re-arms for N+1.
#
# ALSO WAITS FOR THE EVAL when that epoch has one, so a single notification carries both the
# training numbers and the held-out ones instead of two arriving minutes apart.
#
# IT BUILDS THE PLOTS ITSELF, which is the point of this version. The rule is one TRAIN plot every
# epoch and a VAL plot whenever that epoch has a held-out pass, and leaving that to be remembered
# turn by turn meant it was not done -- the user had to ask for plots that should have arrived
# unprompted. Generating them here makes the report and the figures the same artifact, so the
# only thing left to the caller is attaching the paths this script prints.
set -u
cd "$(dirname "$0")/.."
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: epoch_report.sh <run_id> <epoch> [eval_every] [insample_every] [outdir]}"
EP="${2:?}"
EV="${3:-5}"
IN="${4:-10}"
OUT="${5:-/tmp/epoch_plots}"
LOG="training/runs/$RID/run.log"
mkdir -p "$OUT"

until grep -qE "^\s*\[rank 0\] epoch $EP took" "$LOG" 2>/dev/null; do
  # give up only if the run is gone -- the health watcher owns that alarm, this just stops waiting
  grep -qE "Traceback|^STOP:" "$LOG" 2>/dev/null && { echo "run $RID FAILED before epoch $EP"; exit 0; }
  sleep 60
done
echo "=== $RID epoch $EP -- train done $(TZ=America/New_York date '+%-I:%M %p %Z') ==="
grep -E "^\s*\[rank 0\] epoch $EP took" "$LOG" | tail -1 \
  | tr ',' '\n' | grep -oE "epoch $EP took [0-9.]+ min|gpu_backward [0-9.]+m|cropped_tokens mean [0-9]+" | sed 's/^/  /'
grep -oE "epoch $EP token draw: kept [0-9.]+ .*added" "$LOG" | tail -1 | sed 's/^/  /'
grep -oE "epoch $EP memory: peak allocated [0-9.]+ GB" "$LOG" | tail -1 | sed 's/^/  /'
grep -oE "epoch $EP l2-sp: [^(]*" "$LOG" | tail -1 | sed 's/^/  /'
grep -oE "ckpt_threshold=[0-9]+ tok -> [0-9]+ unit\(s\) UNCHECKPOINTED, [0-9]+ checkpointed" "$LOG" | tail -1 | sed 's/^/  /'
echo "  nonfinite rows: $(awk -F, 'NR>1 && ($25=="nan"||$25=="1")' "training/runs/$RID/trajectory.csv" | wc -l)"

# the evals, when this epoch has them
HAS_VAL=0
if [ $((EP % EV)) -eq 0 ]; then
  echo "  waiting for the uncropped valfull pass..."
  for i in $(seq 1 60); do
    [ -f "training/runs/$RID/metric_valfull_ep${EP}.json" ] && break
    sleep 30
  done
  if [ -f "training/runs/$RID/metric_valfull_ep${EP}.json" ]; then
    HAS_VAL=1; echo "  VALFULL ep$EP ready"
  else
    echo "  valfull ep$EP did NOT appear"
  fi
fi
if [ $((EP % IN)) -eq 0 ]; then
  echo "  waiting for the uncropped in-sample pass..."
  for i in $(seq 1 90); do
    [ -f "training/runs/$RID/metric_insample_ep${EP}.json" ] && break
    sleep 30
  done
  [ -f "training/runs/$RID/metric_insample_ep${EP}.json" ] \
    && echo "  INSAMPLE ep$EP ready" || echo "  insample ep$EP did NOT appear"
fi

# ---- the plots. TRAIN every epoch; VAL only when this epoch actually has a held-out pass, so a
# ---- stale figure is never re-sent as if it were new.
echo "=== plots ==="
T="$OUT/${RID}_ep${EP}_train.png"
if $PY -I run_logging/plot_train_insample.py "$RID" "$T" >/dev/null 2>&1; then
  echo "  TRAIN_PLOT $T"
else
  echo "  TRAIN_PLOT FAILED"
fi
if [ "$HAS_VAL" -eq 1 ]; then
  V="$OUT/${RID}_ep${EP}_val.png"
  if $PY -I run_logging/plot_val_2x3.py "$RID" "$V" >/dev/null 2>&1; then
    echo "  VAL_PLOT $V"
  else
    echo "  VAL_PLOT FAILED"
  fi
else
  echo "  no held-out pass at epoch $EP -- train plot only"
fi
echo "=== end epoch $EP ==="
