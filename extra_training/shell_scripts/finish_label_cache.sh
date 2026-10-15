#!/bin/bash
# WAIT FOR THE LABEL BUILD, THEN REPAIR WHATEVER MEASUREMENT SAYS NEEDS REPAIRING, THEN GATE.
#
# WHY NOT A HARDCODED LIST. The first version of this named 7stz, 8gpu, 8av9, 8pww -- the four I knew
# about. A list someone typed goes stale the moment a fifth structure needs rebuilding, and the
# reason a structure needs it is exactly the kind of thing nobody notices: review round 7's masking
# bug wrote labels that RETAINED atoms 2.4 A out, and nothing in the build's own output said so.
# So the list is DERIVED: phase_j_verify_labels.py re-checks every cached label on disk against its
# job cif and prints the structures that fail, and that is what gets rebuilt.
#
# A corrective pass that depends on someone remembering it is how the 2026-10-10 fold loss happened.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG="$R/label_cache_finish.log"
LIST="$R/training/fold/labels_to_repair.txt"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for any running phase_j_build_cache to exit"
# pgrep to WAIT, never to kill. Verified not to self-match: this script's own command line does not
# contain the pattern, so it cannot wait on itself forever.
while pgrep -f "[p]hase_j_build_cache" > /dev/null 2>&1; do sleep 60; done
say "the build has exited"

for attempt in 1 2 3; do
    say "pass $attempt: verifying every cached label on disk against its job cif"
    $PY -I "$R/phases/phase_j_verify_labels.py" --fold "$R/training/fold" \
        --out "$LIST" >> "$LOG" 2>&1
    if [ $? -eq 0 ]; then
        say "pass $attempt: ALL LABELS VERIFY. nothing to repair."
        break
    fi
    n=$(wc -l < "$LIST" 2>/dev/null || echo 0)
    if [ "$n" -eq 0 ]; then
        say "verification failed but named no structure -- STOPPING rather than guessing. Do NOT train."
        exit 1
    fi
    if [ "$attempt" -eq 3 ]; then
        say "STILL $n structure(s) failing after 3 passes. Do NOT train; see $LOG."
        exit 1
    fi
    say "pass $attempt: rebuilding $n structure(s): $(tr '\n' ',' < "$LIST" | sed 's/,$//')"
    CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_j_build_cache.py" \
        --only "$(tr '\n' ',' < "$LIST" | sed 's/,$//')" \
        --fold "$R/training/fold" --out-dir "$R/training/fold/labels" \
        --overwrite >> "$LOG" 2>&1
    say "pass $attempt: rebuild exit=$?"
done

say "asserting manifest completeness (501 structures x 10 seeds, 9mx1 excluded as unfoldable)"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_j_build_cache.py" \
    --fold "$R/training/fold" --out-dir "$R/training/fold/labels" \
    --require-complete --manifest-skip 9mx1 --limit 1 >> "$LOG" 2>&1
rc=$?
say "completeness exit=$rc"
[ $rc -ne 0 ] && { say "THE TRAINING SET IS INCOMPLETE. Do NOT train; see $LOG."; exit 1; }
say "LABEL CACHE FINISHED AND VERIFIED. Training may start."
