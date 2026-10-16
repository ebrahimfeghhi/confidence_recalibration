#!/bin/bash
# WAIT FOR FOLDBENCH, VERIFY THE EVALUATOR, THEN RUN THE 3-EPOCH FT3 PILOT.
#
# WHY IT WAITS. FoldBench is using 70-83 GB of each card's 98 GB on large complexes. Training needs
# ~16 GB; starting it now risks an OOM that kills a fold worker, which is exactly how ~1,000 folds
# were lost on 2026-10-10. The GPU holders are never touched.
#
# WHY THE EVAL IS VERIFIED FIRST, with the SHIPPED head. The metric depends on three joins that can
# each be silently wrong -- interface name to chain pair, pose to sample, and (iface, seed, sample)
# to DockQ -- and the pose-to-sample one is REAL: 29tj/400000's pose 0 carries the stored ipTM of
# SAMPLE 4. phase_k_eval refuses to report a number unless the shipped head reproduces results.csv's
# own ipTM row by row. If that fails, a training curve would be uninterpretable, so the pilot does
# not start.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG="$R/ft3_pilot.log"
OUT="$R/training/ft3_pilot"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for FoldBench to finish (pgrep to WAIT, never to kill)"
while pgrep -f "[f]older.py.*foldbench_baseline" > /dev/null 2>&1; do sleep 120; done
say "FoldBench has exited"
sleep 60                      # let the cards settle before claiming memory

say "GPU state before starting:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tee -a "$LOG"

# GATE 1: the evaluator must reproduce the fold's own ipTM with the shipped head.
say "GATE 1: evaluator self-check with the SHIPPED head (full 100-seed budget)"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_k_eval.py" \
    --budget 100 --out "$R/training/ft3_pilot_baseline.json" >> "$LOG" 2>&1
rc=$?
say "GATE 1 exit=$rc"
if [ $rc -ne 0 ]; then
    say "THE EVALUATOR DOES NOT REPRODUCE results.csv. One of its three joins is wrong, so a"
    say "training curve would be uninterpretable. NOT starting the pilot. See $LOG."
    exit 1
fi

# GATE 2: the replay, which is what makes cached training equal to FT3 at all.
say "GATE 2: replay gate"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_j_replay.py" \
    --unit 9g13/free_500000 > "$R/training/ft3_pilot_replay.log" 2>&1
grep -q "^PASS:" "$R/training/ft3_pilot_replay.log" || {
    say "THE REPLAY DID NOT PASS. Cached training would be optimising something other than FT3."
    say "NOT starting the pilot. See $R/training/ft3_pilot_replay.log"; exit 1; }
say "GATE 2: PASS"

mkdir -p "$OUT"
say "starting the 3-epoch pilot: batch 256, lr 1e-3, warmup 20, eval after every epoch"
say "  ~292 optimiser steps (97/epoch); each epoch-end eval is a full 100-seed pass (~16.5 min)"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_k_train.py" \
    --fold "$R/training/fold" \
    --labels "$R/training/fold/labels" \
    --epochs 3 --accum 256 --lr 1e-3 --warmup-steps 20 \
    --eval-every-epoch --eval-budget 100 \
    --preload-items 1 \
    --grad-check-every 10 --save-every 100 \
    --require-replay "$R/training/ft3_pilot_replay.log" \
    --wandb-project confidence-recalibration --wandb-run ft3-pilot-3ep \
    --out "$OUT" >> "$LOG" 2>&1
say "pilot exit=$? -- trajectory $OUT/trajectory.csv, metrics $OUT/metric_ep*.json"
say "THE LOSS IS NOT THE RESULT: compare metric_ep*.json against mean_dockq 0.5783 / frac_acc 0.7283"
