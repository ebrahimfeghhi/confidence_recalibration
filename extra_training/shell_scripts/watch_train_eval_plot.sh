#!/bin/bash
# Regenerate the side-by-side training-pool plot whenever a new `epoch N TRAIN:` line appears.
#
# POLLS THE LOGS, does not hook the trainer. The runs it watches are already going, so a trainer
# change could not reach them -- and a reader that only reads run.log cannot perturb a run.
#
# STOP: touch the stop file, or kill this script BY PID.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
OUT=${OUT:-training/runs/train_eval_plot.png}
STOP=${STOP:-training/runs/STOP_PLOT_WATCH}
PY=/home/ebrahim/Protenix/.venv/bin/python
SPECS=("n3s8gb=unmasked PAE" "pykssn+x3888d+tgfiqx+89uwjx=PAE masked" "dapghi=PAE masked + PAE-only" "x5a4uj=PAE masked, lr 1e-2")
REFS=(--stock-dockq 0.0420 --stock-frac 0.0
      --ceiling-dockq 0.4216 --ceiling-frac 0.6883
      --oracle-dockq 0.5098 --oracle-frac 0.8831)
last=""
while :; do
    [ -f "$STOP" ] && { echo "$(date '+%H:%M:%S') stopped by $STOP"; exit 0; }
    now=$(grep -h -c "TRAIN:" training/runs/n3s8gb/run.log training/runs/pykssn/run.log training/runs/dapghi/run.log training/runs/x5a4uj/run.log training/runs/x3888d/run.log training/runs/tgfiqx/run.log training/runs/89uwjx/run.log 2>/dev/null | paste -sd,)
    if [ "$now" != "$last" ]; then
        if "$PY" -I run_logging/plot_train_eval.py "${SPECS[@]}" "${REFS[@]}" \
               --out "$OUT" --quiet 2>/dev/null; then
            echo "$(date '+%H:%M:%S') points=$now -> $OUT"
            last="$now"
        fi
    fi
    sleep 60
done
