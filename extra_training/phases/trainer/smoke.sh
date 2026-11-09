#!/usr/bin/env bash
# THE FIRST REAL EXECUTION OF THE ORGANIZED TRAINER. Not a parse, not an import.
#
# Runs the composed entrypoint on the LIVE ARM'S OWN ARGV (run_logging/fixtures/
# cli_argv_87ja6a.json) with the unit count capped, and reports exactly where it stops. The
# point is the FIRST FAILURE, not success: every stage that runs before it is a stage proven to
# work on the real configuration, and the traceback names the next thing to fix.
#
# CPU by default (CUDA_VISIBLE_DEVICES=""). Touches no GPU and no live run.
set -u
cd "$(dirname "$0")/../.." || exit 1
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
PY=/home/ebrahim/Protenix/.venv/bin/python
LIMIT="${LIMIT:-1}"

ARGV=$($PY -I -c "
import json
a=json.load(open('run_logging/fixtures/cli_argv_87ja6a.json'))
out=[]
skip=False
for i,x in enumerate(a):
    if skip: skip=False; continue
    if x=='--out': skip=True; continue
    out.append(x)
print(' '.join(out))
")

echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') smoke: composed entrypoint, CPU, --limit-units $LIMIT"
CUDA_VISIBLE_DEVICES="" LAYERNORM_TYPE=openfold \
PYTHONPATH=phases/trainer:phases:/home/ebrahim/Protenix \
  $PY phases/trainer/main.py $ARGV --limit-units "$LIMIT" \
      --out "$SD/smoke_run" 2>&1 | tail -40
