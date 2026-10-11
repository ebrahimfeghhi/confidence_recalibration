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
    ntrunk=$(find $E/trunks -name '*.npz' 2>/dev/null | wc -l)
    # numeric dockq AND numeric iptm on at least one interface: the scorer ran, OST produced a score
    # rather than discarding it (the --rigid-scores failure we have hit), and the ipTM we will compare
    # against after recalibration is actually recorded.
    ndockq=$(awk -F, 'NR>1 && $7+0==$7 && $7!="" {n++} END{print n+0}' "$E/results.csv" 2>/dev/null)
    niptm=$(awk -F, 'NR>1 && $8+0==$8 && $8!="" {n++} END{print n+0}' "$E/results.csv" 2>/dev/null)
    nconf=$(find $E/poses -name '*summary_confidence_sample_*.json' 2>/dev/null | wc -l)
    say "  cifs $ncif   trunk npz $ntrunk   numeric dockq $ndockq   numeric iptm $niptm   conf json $nconf"
    [ "$ncif"   -ge 1 ] || esc "smoke produced no pose cif"
    [ "$ntrunk" -ge 1 ] || esc "smoke produced no trunk npz -- TRUNK_DUMP_DIR did not fire"
    [ "$ndockq" -ge 1 ] || esc "smoke produced no numeric DockQ -- scoring or the reference is wrong"
    [ "$niptm"  -ge 1 ] || esc "smoke produced no numeric ipTM"
    [ "$nconf"  -ge 1 ] || esc "smoke wrote no summary_confidence json -- per-interface ipTM lost"

    # OPEN the npz. Counting files proves nothing about whether the confidence head can be re-run from
    # it: the tensors it needs are what matter, and a dump that silently dropped them would look
    # identical on disk.
    $P/.venv/bin/python - "$E" <<'PYEOF' >> "$LOG" 2>&1
import glob, sys, numpy as np
f = sorted(glob.glob(f"{sys.argv[1]}/trunks/*/*.npz"))
assert f, "no npz found"
d = np.load(f[0], allow_pickle=True)
need = ["z_trunk", "s_trunk", "ci_s_inputs", "ci_s_trunk_f32", "ci_coordinate",
        "ci_contact_probs", "ci_feat_atom_to_token_idx", "ci_feat_asym_id", "ci_N_cycle"]
missing = [k for k in need if k not in d.files]
print(f"npz {f[0]}")
for k in d.files:
    a = d[k]
    print(f"   {k:28s} {str(a.dtype):9s} {a.shape}")
assert not missing, f"MISSING CONFIDENCE-HEAD TENSORS: {missing}"
assert str(d["dtype"]) == "bf16", f"z/s stored as {d['dtype']}, expected bf16"
print("NPZ_OK all confidence-head tensors present")
PYEOF
    grep -q "NPZ_OK" "$LOG" || esc "the trunk npz is missing tensors the confidence head needs (see the log above)"
    touch "$E/SMOKE_OK"
    say "smoke OK: pose, per-fold npz with every confidence-head tensor, DockQ, ipTM, confidence json"
else
    say "smoke already passed (SMOKE_OK present)"
fi

# ---------------------------------------------------------------- 3. the full fold
TARGET=$((NJOB*SEEDS))
done_n() { find $E/poses -name run.json 2>/dev/null | wc -l; }
if [ "$(done_n)" -lt "$TARGET" ]; then
    say "starting the full fold: $(done_n)/$TARGET units. This pauses the PXM sweep."
    say "  pxm before pause: $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/$(wc -l < $PXM/queue.tsv 2>/dev/null)"
    bash $S/val_fold.sh run >> "$LOG" 2>&1
else
    say "validation already complete at $(done_n)/$TARGET units"
fi

# ---------------------------------------------------------------- 4. verify and hand back
N=$(done_n)
ROWS=$(($(wc -l < $E/results.csv 2>/dev/null || echo 1)-1))
TRUNKS=$(find $E/trunks -name '*.npz' 2>/dev/null | wc -l)
say "=== validation fold finished ==="
say "  units $N/$TARGET   scored rows $ROWS   trunk dumps $TRUNKS   cache $(du -sh $E/trunks 2>/dev/null | cut -f1)"
say "  quota $(quota -s 2>/dev/null | awk '/nvme/{print $2" of "$4}')"

# val_fold.sh run relaunches PXM itself; confirm it actually came back rather than assuming it
sleep 30
up=0
for g in 0 1; do tmux has-session -t fx_pxm2025_$g 2>/dev/null && up=$((up+1)); done
say "  pxm workers back up: $up/2   folds $(find $PXM/poses -name run.json 2>/dev/null | wc -l)/$(wc -l < $PXM/queue.tsv 2>/dev/null)"
[ "$up" -eq 2 ] || say "ESCALATE: only $up/2 pxm workers restarted -- check $PXM/w0.log and w1.log"

# one npz per folded unit. If these diverge, dumps were overwritten or failed silently, and the
# cache cannot be used to re-run the head on every pose.
[ "$TRUNKS" -eq "$N" ] || say "ESCALATE: $TRUNKS trunk npz for $N folded units -- they must match"

if [ "$N" -lt "$TARGET" ]; then
    say "ESCALATE: $((TARGET-N)) units short of the target. Poses kept; rerun val_fold.sh run to continue."
else
    say "=== COMPLETE: all $TARGET validation units folded, scored and cached ==="
fi
