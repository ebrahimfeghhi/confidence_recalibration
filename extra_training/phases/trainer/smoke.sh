#!/usr/bin/env bash
# THE FIRST REAL EXECUTION of the organized trainer, on the live arm's own argv.
#
# IT REPORTS THREE OUTCOMES, NOT TWO. On a CPU-only invocation the Protenix head build cannot
# succeed -- it needs a card before any data, pose or loss is touched -- so "stopped at CUDA" is
# an EXPECTED BLOCKER, not a pass and not a regression. Collapsing those three into pass/fail is
# how a harness starts lying.
#
# THE EXIT STATUS IS THE TRAINER'S, NOT tail's. The first version piped straight into `tail`
# without `pipefail`, so it exited 0 while the trainer crashed -- verified: a CUDA failure
# reported success. Any automation reading that would have called a broken trainer healthy.
# Caught in review on issue #9.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
PY=/home/ebrahim/Protenix/.venv/bin/python
LIMIT="${LIMIT:-1}"
LOG="$SD/smoke.log"

# ARGV AS AN ARRAY, read NUL-delimited. The first version joined the fixture with spaces and
# expanded it unquoted, which loses empty strings, arguments containing whitespace, and anything
# the shell would glob. The 93-token fixture happens to survive that; the next one need not.
mapfile -d '' -t ARGV < <($PY -I -c "
import json, sys
a = json.load(open('run_logging/fixtures/cli_argv_87ja6a.json'))
out, skip = [], False
for x in a:
    if skip: skip = False; continue
    if x == '--out': skip = True; continue
    out.append(x)
sys.stdout.write('\0'.join(out))
")

echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') smoke: composed entrypoint, CPU, --limit-units $LIMIT"
echo "    ${#ARGV[@]} argv token(s) from the live fixture; full log -> $LOG"
CUDA_VISIBLE_DEVICES="" LAYERNORM_TYPE=openfold \
PYTHONPATH=phases/trainer:phases:/home/ebrahim/Protenix \
  $PY phases/trainer/main.py "${ARGV[@]}" --limit-units "$LIMIT" --out "$SD/smoke_run" \
  > "$LOG" 2>&1
RC=$?

echo "--- stages reached ---"
grep -aE "accum:|replay gate|RESUMED|trainable:|steps/epoch|POSE |DEAD OPS" "$LOG" | head -8
echo "--- last lines ---"
tail -6 "$LOG"
echo

if [ $RC -eq 0 ]; then
  echo "SMOKE PASS (exit 0)"; exit 0
fi
if grep -qa "No CUDA GPUs are available" "$LOG"; then
  echo "EXPECTED BLOCKER (exit $RC): the Protenix head build needs a card. Every stage before"
  echo "  bootstrap ran on the live configuration. This is NOT a pass for the trainer."
  exit 2
fi
echo "SMOKE FAIL (exit $RC) -- an unexpected failure, see $LOG"
exit 1
