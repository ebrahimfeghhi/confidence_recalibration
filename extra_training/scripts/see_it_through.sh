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

# ---------------------------------------------------------------- 1. wait for prep
if ! cur | grep -q "PREP COMPLETE"; then
    say "waiting for the MSA/prep chain (session: overnight)"
    tick=0
    while tmux has-session -t overnight 2>/dev/null; do
        tick=$((tick+1))
        if [ $((tick % 15)) -eq 0 ]; then
            say "  prep: $(tail -1 "$R/extra_training/overnight.log" | cut -c1-110)"
        fi
        sleep 60
    done
fi
# The chain prints STOP on any failed gate.
if cur | grep -q "STOP:"; then
    say "prep chain stopped:"; cur | tail -6 | tee -a "$LOG"
    esc "prep did not complete"
fi
cur | grep -q "PREP COMPLETE" || esc "prep chain ended without PREP COMPLETE"
NJOB=$(ls $R/extra_training/val_jobs/*.json 2>/dev/null | wc -l)
[ "$NJOB" -ge 70 ] || esc "only $NJOB validation job jsons"
say "prep complete: $NJOB job jsons, target $((NJOB*SEEDS)) units"

# ---------------------------------------------------------------- 2. smoke test, asserted in code
if [ ! -f "$E/SMOKE_OK" ]; then
    say "smoke test: one complex, one seed (PXM untouched)"
    bash $S/val_fold.sh smoke >> "$LOG" 2>&1
    ncif=$(find $E/poses -name '*.cif' 2>/dev/null | wc -l)
    ntrunk=$(ls $E/trunks 2>/dev/null | wc -l)
    # a numeric dockq on at least one interface: the scorer ran AND OST produced a score rather than
    # discarding it, which is the failure mode we have hit before (--rigid-scores crashing)
    ndockq=$(awk -F, 'NR>1 && $7+0==$7 && $7!="" {n++} END{print n+0}' "$E/results.csv" 2>/dev/null)
    say "  cifs $ncif   trunk dumps $ntrunk   numeric dockq rows $ndockq"
    [ "$ncif"   -ge 1 ] || esc "smoke produced no pose cif"
    [ "$ntrunk" -ge 1 ] || esc "smoke produced no trunk dump -- PROTENIX_DUMP_TRUNK did not fire"
    [ "$ndockq" -ge 1 ] || esc "smoke produced no numeric DockQ -- scoring or the reference is wrong"
    touch "$E/SMOKE_OK"
    say "smoke OK on all three assertions"
else
    say "smoke already passed (SMOKE_OK present)"
fi

# ---------------------------------------------------------------- 3. the full fold
TARGET=$((NJOB*SEEDS))
done_n() { find $E/poses -name run.json 2>/dev/null | wc -l; }
if [ "$(done_n)" -lt "$TARGET" ]; then
    say "starting the full fold: $(done_n)/$TARGET units. This pauses the PXM sweep."
    say "  pxm before pause: $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/1230"
    bash $S/val_fold.sh run >> "$LOG" 2>&1
else
    say "validation already complete at $(done_n)/$TARGET units"
fi

# ---------------------------------------------------------------- 4. verify and hand back
N=$(done_n)
ROWS=$(($(wc -l < $E/results.csv 2>/dev/null || echo 1)-1))
TRUNKS=$(ls $E/trunks 2>/dev/null | wc -l)
say "=== validation fold finished ==="
say "  units $N/$TARGET   scored rows $ROWS   trunk dumps $TRUNKS   cache $(du -sh $E/trunks 2>/dev/null | cut -f1)"
say "  quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"

# val_fold.sh run relaunches PXM itself; confirm it actually came back rather than assuming it
sleep 30
up=0
for g in 0 1; do tmux has-session -t fx_pxm2025_$g 2>/dev/null && up=$((up+1)); done
say "  pxm workers back up: $up/2   folds $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/1230"
[ "$up" -eq 2 ] || say "ESCALATE: only $up/2 pxm workers restarted -- check $PXM/w0.log and w1.log"

if [ "$N" -lt "$TARGET" ]; then
    say "ESCALATE: $((TARGET-N)) units short of the target. Poses kept; rerun val_fold.sh run to continue."
else
    say "=== COMPLETE: all $TARGET validation units folded, scored and cached ==="
fi
