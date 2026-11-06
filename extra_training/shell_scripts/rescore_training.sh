#!/bin/bash
# Recover DockQ rows for training units that folded while their reference was missing.
#
# WHY A SCRIPT AND NOT A REMEMBERED COMMAND LINE. The scope, seed range, reference directory and
# scorer all have to be right together, and getting any of them wrong is silent: the wrong --refs
# scores against another structure's reference, and leaving out --worklist makes phase_i_check
# enumerate every job json in training/jobs -- 856 structures times 10 seeds, 8,560 candidate units
# instead of the 3,560 queued -- so a recovery reaches into older incomplete work. --expect-units
# makes that a hard stop rather than something to notice afterwards.
#
# IT WAITS FOR THE FOLD WORKERS, BY DESIGN. Rescoring re-reads the csv immediately before
# appending, which narrows the duplicate window but cannot close it: a worker can still append the
# same sample in between, and the duplicate double-weights that pose in every average over its
# seed. So this sequences rather than locks the live folding driver. Once the workers are gone it
# is safe alongside a VALIDATION fold, which writes a different results.csv.
#
# CPU IS BOUNDED so it cannot slow the validation fold's own ost scoring: nice 19, idle io, and a
# thread cap, with one scorer subprocess at a time.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration
THREADS="${RESCORE_THREADS:-4}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

say "waiting for the training fold workers to exit before touching training/fold/results.csv"
while [ "$(tmux ls 2>/dev/null | grep -c w1fold)" -gt 0 ]; do sleep 120; done
say "training fold workers are gone"

# belt and braces: the python refuses on its own if a writer is still there
BEFORE=$(wc -l < training/fold/results.csv)
say "results.csv rows before: $BEFORE"
cp -p training/fold/results.csv "training/fold/results.csv.before_rescore.$(date +%s)"
say "backup taken"

say "rescoring from the queue (3,560 units, seeds 500000-500009, refs training/refs, scorer ost)"
OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
nice -n 19 ionice -c 3 $PY -I phases/phase_i_check.py rescore \
  --e training/fold \
  --jobs training/jobs \
  --refs training/refs \
  --worklist training/fold/queue.tsv \
  --expect-units 3560 \
  --seeds 10 --seed0 500000 --nsample 5 \
  --scorer ost \
  --py $PY --code $R/run_model
RC=$?
AFTER=$(wc -l < training/fold/results.csv)
say "rescore exit $RC; rows $BEFORE -> $AFTER (added $((AFTER-BEFORE)))"

# EVERY STEP'S STATUS IS ACTED ON. Raised in review: this recorded $RC and then printed RESCORE
# DONE regardless, and ignored the drain and duplicate checks that follow, so a run with timeouts,
# scorer failures or unresolved units looked successful. Each step now sets FAIL, and the final
# line says which ones failed rather than announcing completion.
FAIL=""
[ $RC -ne 0 ] && FAIL="$FAIL rescore(exit=$RC)"

say "outstanding units after the rescore:"
$PY -I run_logging/training_queue_drained.py --show 5
DR=$?
say "drain check exit $DR (0 = the queue is complete in every respect)"
[ $DR -ne 0 ] && FAIL="$FAIL drain(exit=$DR)"

say "duplicate check (reports only; it never rewrites the live csv):"
$PY -I run_logging/dedupe_results.py --csv training/fold/results.csv
DD=$?
say "duplicate check exit $DD (0 = every duplicate resolved against its own chain pair)"
[ $DD -ne 0 ] && FAIL="$FAIL duplicates(exit=$DD)"

if [ -n "$FAIL" ]; then
  say "RESCORE INCOMPLETE --$FAIL"
  say "The recovered rows that DID land are in results.csv and are safe; what is listed above did"
  say "not recover and needs a decision. The canonical csv has not been replaced."
  exit 1
fi
say "RESCORE COMPLETE: queue fully scored and every duplicate resolved"
