#!/usr/bin/env bash
# STEP 1 of the parity ladder: does the organized trainer reach a forward and a backward?
#
# One card, the multi-seed fixture, NO OPTIMISER STEP (--loss-only). The point is to find the
# next failures, not to pass -- the CPU smoke needed four iterations to clear four stages, and
# everything past bootstrap has never executed a line.
set -uo pipefail
. "$(dirname "$0")/parity_common.sh"
LOG=$SD/gpu_smoke.log
require_idle_card || exit 3

mapfile -d '' -t ARGV < <(fixture_argv)
say "step 1: forward+backward on 8yvm (15 seeds x 5 poses), card $CARD, no optimiser step"
say "  ${#ARGV[@]} argv token(s); log -> $LOG"
rm -rf "$SD/gpu_smoke_run"
CUDA_VISIBLE_DEVICES=$CARD LAYERNORM_TYPE=openfold PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$ROOT/phases/trainer:$ROOT/phases:/home/ebrahim/Protenix \
  timeout 2400 $PY "$ROOT/phases/trainer/main.py" "${ARGV[@]}" \
    --limit-units 15 --loss-only --out "$SD/gpu_smoke_run" > "$LOG" 2>&1
RC=$?

say "--- stages reached ---"
grep -aE "accum:|replay gate|trainable:|steps/epoch|POSE |DEAD OPS|RESUMED|pose_total|rank-loss" "$LOG" | head -10
say "--- tail ---"; tail -8 "$LOG"
echo
if [ $RC -eq 0 ]; then say "GPU SMOKE PASS (exit 0) -- forward and backward ran"; exit 0; fi
say "GPU SMOKE FAIL (exit $RC). First unhandled error:"
grep -aE "^[A-Za-z_.]*Error|^Traceback|^STOP:" "$LOG" | tail -3
exit 1
