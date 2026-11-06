#!/bin/bash
# Put every training score in ONE place, in order, with each step's result acted on.
#
# WHY A SCRIPT. Four tools have to run in a fixed order against the same file, and three of them
# refuse if the one before did not do its job: the merge gate will not touch a results.csv that
# still holds duplicate keys of its own, and the final verify is only meaningful once the recovered
# rows are actually in the file. A remembered command line would get the order right once.
#
# THE ORDER, AND WHY IT IS THIS ORDER:
#   1. WAIT for every fold session. dedupe --replace refuses while any w1fold/w1val/w2_ session
#      exists. That guard is broader than this file needs -- the live validation fold writes
#      validation/fold/results.csv, not this one -- but it is the existing guard and this does not
#      route around it.
#   2. DEDUPE the 10 pre-existing duplicate (iface,seed,arm,sample) rows. They predate the
#      recovery: 83,560 rows for 83,550 keys, each duplicate double-weighting one pose in every
#      average over its seed. All 10 are resolvable against the pose still on disk, identified by
#      OST's own chain_mapping. This must precede the merge, because the merge gate refuses a
#      canonical file that is not already clean -- otherwise its post-merge uniqueness check would
#      be asserting something it had already given up on.
#   3. MERGE the 850 recovered rows as one checked transaction. 17 interfaces x 10 seeds x 5
#      samples, from two producers: pair isolation for 15 interfaces and the full-complex route for
#      4zto's 2. Both counts are passed in so a short sweep cannot define its own success.
#   4. VERIFY the whole queue, and audit every pool pair for the reverse error. The verify is the
#      proof the gap is closed; the audit is the proof nothing non-contacting was scored to close
#      it.
#
# IT WRITES NOTHING WITHOUT A BACKUP. dedupe leaves results.csv.pre_dedupe and the merge leaves
# results.csv.pre_recovered_merge, both beside the file, and both swaps are os.replace within one
# filesystem so no reader sees a partial csv.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration
CSV=training/fold/results.csv
ISO="${ISO_ROWS:?set ISO_ROWS to the 750-row csv}"
FULL="${FULL_ROWS:?set FULL_ROWS to the 100-row csv}"
# THE SNAPSHOTS MUST SHARE A FILESYSTEM WITH THE CSV. Both tools finish with os.replace, which is
# atomic only within one filesystem and raises EXDEV across devices -- pointing SNAP_DIR at the
# session scratchpad produced exactly that, after the dedupe had done all of its work:
#   OSError: [Errno 18] Invalid cross-device link
# Nothing was damaged, because the failure is the swap itself, but the run was wasted. So the
# default sits beside the csv rather than in a temp dir, and an override is checked rather than
# trusted: a caller cannot pick a directory that makes the swap non-atomic.
SNAP="${SNAP_DIR:-$(dirname $CSV)/.consolidate}"
mkdir -p "$SNAP"
if [ "$(stat -c %d "$SNAP")" != "$(stat -c %d "$(dirname $CSV)")" ]; then
  echo "STOP: $SNAP is on a different filesystem from $CSV, so the atomic swap would fail with"
  echo "EXDEV after both tools had finished their work. Use a directory beside the csv."
  exit 1
fi
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
FAIL=""

say "step 1: waiting for any session that could write THIS csv"
# ONLY THE FOLDS THAT WRITE training/fold/results.csv. A live validation fold writes
# validation/fold/results.csv and is not a writer of this file; waiting on it would hold this
# repair for hours for no reason, which is the false alarm the guard inside dedupe_results.py and
# phase_l_merge_recovered.py was narrowed to stop. Those two derive the same mapping from the csv
# path, so all three agree about who counts as a writer.
while [ "$(tmux ls 2>/dev/null | grep -cE '^(w1fold|w2_)')" -gt 0 ]; do sleep 120; done
say "no session that writes $CSV remains"
OTHER=$(tmux ls 2>/dev/null | grep -cE '^w1val' || true)
[ "$OTHER" -gt 0 ] && say "note: $OTHER validation session(s) are running; they write a different csv"

say "rows before: $(wc -l < $CSV)"

say "step 2: resolving the 10 pre-existing duplicate rows"
OMP_NUM_THREADS=4 nice -n 19 ionice -c 3 $PY -I run_logging/dedupe_results.py \
  --csv $CSV --out "$SNAP/deduped.csv" --replace
RC=$?; say "dedupe exit $RC"
[ $RC -ne 0 ] && FAIL="$FAIL dedupe(exit=$RC)"

if [ -z "$FAIL" ]; then
  say "step 3: merging the 850 recovered rows"
  # --manifest IS NOT OPTIONAL HERE. Raised in review: the merge gate could name how MANY
  # interfaces to expect but not WHICH, so this documented workflow still admitted a different
  # set of 17 pooled interfaces carrying 850 otherwise valid rows. The routes are labelled
  # (iso=, full=) because the two producers are not interchangeable and a filename is not an
  # identity -- 4zto_DF cannot be pair-isolated at all.
  nice -n 19 ionice -c 3 $PY -I phases/phase_l_merge_recovered.py \
    --csv $CSV --rows "iso=$ISO" "full=$FULL" \
    --manifest "${MANIFEST:-docs/recovery_manifest_wave1.csv}" \
    --expect-rows 850 --expect-ifaces 17 --seeds 10 --seed0 500000 --nsample 5 \
    --out "$SNAP/merged.csv" --replace
  RC=$?; say "merge exit $RC"
  [ $RC -ne 0 ] && FAIL="$FAIL merge(exit=$RC)"
fi

say "rows after: $(wc -l < $CSV)"

if [ -z "$FAIL" ]; then
  say "step 4a: verifying the whole queue (3,560 units)"
  OMP_NUM_THREADS=4 nice -n 19 ionice -c 3 $PY -I phases/phase_i_check.py verify \
    --e training/fold --jobs training/jobs --refs training/refs \
    --worklist training/fold/queue.tsv --expect-units 3560 \
    --seeds 10 --seed0 500000 --nsample 5 --scorer ost \
    --py $PY --code $R/run_model
  RC=$?; say "verify exit $RC (0 = nothing left to fold or rescore)"
  [ $RC -ne 0 ] && FAIL="$FAIL verify(exit=$RC)"

  say "step 4b: auditing every pool pair for the reverse error"
  $PY -I run_logging/pool_contact_audit.py
  RC=$?; say "contact audit exit $RC (0 = every pair checked, none scored without contact)"
  [ $RC -ne 0 ] && FAIL="$FAIL audit(exit=$RC)"
fi

if [ -n "$FAIL" ]; then
  say "CONSOLIDATION INCOMPLETE --$FAIL"
  say "Backups are beside $CSV (.pre_dedupe, .pre_recovered_merge). Nothing is half-written:"
  say "each swap is atomic, so the file is either the old one or the new one."
  exit 1
fi
say "CONSOLIDATION COMPLETE: every queued training interface-sample is scored in $CSV"
