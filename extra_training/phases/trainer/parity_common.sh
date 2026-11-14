# Shared setup for the GPU parity harness. Sourced, not run.
#
# THE FIXTURE IS MULTI-SEED ON PURPOSE. 8yvm, 15 seeds x 5 poses = 75 poses, 1,004 tokens
# (measured 56.81 GB uncheckpointed, so it fits with room). A SINGLE-POSE fixture would prove
# wiring and certify NOTHING about the ranking term: `rel` is mean-centred within an interface
# ACROSS ITS SEEDS, so with one pose the coefficient is identically zero and the loss is vacuous.
# It would pass while testing the thing it exists to test -- the same shape as the zero-init
# orientation trap. Raised in review on issue #9.
ROOT=/home/ebrahim/confidence_recalibration/extra_training
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
PY=/home/ebrahim/Protenix/.venv/bin/python
FIX=$ROOT/run_logging/fixtures/parity_manifest.csv
PRE=$ROOT/run_logging/fixtures/parity_preflight.json
CARD="${CARD:-4}"

say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

# THE CARD MUST BE IDLE. This harness exists to be run while a 31-hour arm may be live; starting
# a second process on a card it holds would corrupt both. Refuses rather than competes.
require_idle_card() {
  local B
  B=$($PY -I "$ROOT/run_logging/cards_busy.py" 2>/dev/null) || B=fail
  if [ "$B" != "0" ]; then
    say "REFUSING: cards_busy reports '$B'. A training run is using the GPUs."
    return 3
  fi
  local F
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CARD" 2>/dev/null)
  case "$F" in ''|*[!0-9]*) say "REFUSING: GPU $CARD memory unreadable"; return 3;; esac
  if [ "$F" -lt 80000 ]; then say "REFUSING: GPU $CARD only ${F} MiB free"; return 3; fi
  say "card $CARD idle, ${F} MiB free"
  return 0
}

# The arm's own flags, with the fixture swapped in and the unit count capped.
fixture_argv() {
  $PY -I -c "
import json, sys
a = json.load(open('$ROOT/run_logging/fixtures/cli_argv_87ja6a.json'))
out, skip = [], False
DROP = {'--out', '--manifest-selection', '--alignment-preflight', '--manifest-skip'}
for x in a:
    if skip: skip = False; continue
    if x in DROP: skip = True; continue
    out.append(x)
out += ['--manifest-selection', '$FIX', '--alignment-preflight', '$PRE', '--manifest-skip', '']
sys.stdout.write('\0'.join(out))
"
}
