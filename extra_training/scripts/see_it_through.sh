#!/bin/bash
# Drive the whole validation run to completion: wait for the MSA prep, smoke-test one fold, then fold
# all 79 structures x 20 seeds, then hand the GPUs back to the PXM sweep.
#
# WHY THIS EXISTS: overnight.sh deliberately stops after prep, because the validation fold costs ~20 h
# of GPU and pauses PXM. This takes that last gate off the critical path WITHOUT removing the check --
# the smoke test still runs, but its three assertions are made here in code instead of by eye.
#
# WHAT IT WILL NOT DO
#   - it never kills a fold worker except through val_fold.sh, which kills BY PID (a pattern kill
#     would match this script)
#   - it never deletes a pose
#   - it stops and escalates rather than folding a set it cannot verify
#
# RESTARTABLE: every step is keyed on a state file or on counted output, so re-running continues
# instead of redoing. folder.py reconciles stale claims and skips scored units, so a relaunch is free.
#
# USAGE: tmux new-session -d -s seeit "bash see_it_through.sh"
set -u
R=/home/ebrahim/tts_abag
E=$R/extra_training/val_fold
P=/home/ebrahim/Protenix
S=$R/extra_training/scripts
LOG=$R/extra_training/see_it_through.log
PXM=$R/experimental_data/protenix/pxm2025/free_extra10
SEEDS=20

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
esc() { say "ESCALATE: $*"; say "PXM sweep left alone. Nothing further started."; exit 1; }

# Only the CURRENT chain run counts. overnight.log is appended to across restarts, and it already
# holds yesterday's STOP, so a plain grep would escalate on a failure that has since been fixed.
cur() { awk '/=== overnight chain starting/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
        "$R/extra_training/overnight.log" 2>/dev/null; }

say "=== watcher starting ==="

# ---------------------------------------------------------------- 1. wait for prep, RETRYING
# A WATCHER THAT ONLY PRINTS IS NOT WATCHING. This block used to escalate into a log file and exit;
# the one time a gate wrongly rejected a complete 100 GB download, that left the machine idle for
# 2.4 h until a human noticed. The prep chain is idempotent by construction -- every stage skips what
# it already finished -- so the correct response to a stop is to run it again.
#
# WHEN TO STOP RETRYING: when two attempts in a row die in the same stage with the same message.
# That is a real fault and relaunching into it would burn days. A different failure, or a later
# stage, means progress was made and earns another attempt. Six attempts is the hard ceiling.
prep_stage() { cur | grep -oE 'stage [0-9]+[a-c]?' | tail -1; }
prep_stop()  { cur | grep 'STOP:' | tail -1 | sed 's/^\[[^]]*\] //'; }

tick=0; attempt=0; last_fail=""
while ! cur | grep -q "PREP COMPLETE"; do
    if tmux has-session -t overnight 2>/dev/null; then
        tick=$((tick+1))
        if [ $((tick % 15)) -eq 0 ]; then
            say "  prep: $(tail -1 "$R/extra_training/overnight.log" | cut -c1-110)"
        fi
        sleep 60
        continue
    fi
    fail="$(prep_stage) :: $(prep_stop)"
    if [ "$attempt" -gt 0 ]; then
        say "prep attempt $attempt ended at: $fail"
        if [ "$fail" = "$last_fail" ]; then
            cur | tail -8 | tee -a "$LOG"
            esc "prep stopped twice in the same place: $fail"
        fi
    elif [ -n "$(prep_stop)" ]; then
        say "an earlier prep run had stopped at: $fail"
    fi
    if [ "$attempt" -ge 6 ]; then
        esc "prep still not complete after $attempt attempts"
    fi
    last_fail="$fail"
    attempt=$((attempt+1))
    say "launching prep attempt $attempt (finished stages are skipped, so it resumes)"
    tmux new-session -d -s overnight "bash $S/overnight.sh" \
        || esc "could not start the overnight session"
    sleep 30
done
say "prep complete after $attempt attempt(s) by this watcher"
NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
[ "$NJOB" -ge 70 ] || esc "only $NJOB validation job jsons"
say "prep complete: $NJOB job jsons, target $((NJOB*SEEDS)) units"

# ---------------------------------------------------------------- 2. smoke, then the full fold
# THE ASSERTIONS LIVE IN PHASE I NOW, not here. They are that phase's preconditions and postconditions
# -- that the cache can re-run the confidence head, that DockQ and ipTM are numeric, that every input
# path resolves -- and a watcher is the wrong place to decide whether a phase succeeded. Phase I also
# refuses to fold until smoke has passed, so the ordering this watcher used to enforce is enforced by
# the phase itself and cannot be bypassed by calling things in the wrong order.
PH=$R/extra_training/phases
I=$PH/phase_i_fold_validation.sh

if [ ! -f "$E/SMOKE_OK" ]; then
    say "phase I smoke: one complex, one seed (PXM untouched)"
    bash "$I" smoke >> "$LOG" 2>&1 \
        || esc "phase I smoke failed -- see $R/extra_training/phase_i.log. NOT folding."
    say "smoke OK"
else
    say "smoke already passed (SMOKE_OK present)"
fi

NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
TARGET=$((NJOB*SEEDS))
done_n() { find $E/poses -name run.json 2>/dev/null | wc -l; }
if [ "$(done_n)" -lt "$TARGET" ]; then
    say "phase I run: $(done_n)/$TARGET units. THIS PAUSES THE PXM SWEEP."
    bash "$I" run >> "$LOG" 2>&1 || esc "phase I run returned non-zero -- see $R/extra_training/phase_i.log"
else
    say "validation already complete at $(done_n)/$TARGET units"
fi

# ---------------------------------------------------------------- 3. the phase reports on itself
bash "$I" verify >> "$LOG" 2>&1
say "=== watcher done; phase I's own verification is in $R/extra_training/phase_i.log ==="
tail -8 "$R/extra_training/phase_i.log" | tee -a "$LOG"
