#!/usr/bin/env bash
# Fires the Arm B cosine resume the moment the cards clear. Arm D is finishing its last epoch and
# its final evals; stopping it there would throw away the epoch-20 pass it is about to write, so
# this waits rather than pre-empting.
#
# THE LAUNCHER RUNS THE TRAINER SYNCHRONOUSLY, so this must NOT wait for the launcher process to
# exit before looking for the run id -- that bug cost Arm B's queue twenty minutes of blind time
# overnight. Snapshot the run dirs first, launch in the background, then poll for the new one.
set -euo pipefail
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
mkdir -p "$SD" || { echo "STOP: cannot create $SD"; exit 1; }
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python

# WAIT FOR ARM D'S EPOCH-20 VAL, THEN END ARM D. The user chose to skip Arm D's epoch-20 slice
# (rank-8 LoRA, already shown to add nothing) rather than let that ~28 min of in-sample scoring
# delay this arm. Val and slice run on the SAME three cards we hold, so the only way to start
# early is to stop Arm D's trainer between the two -- it spawns the slice itself once val returns.
#
# THE PIDS ARE DERIVED, NOT MATCHED BY PATTERN: each candidate comes from nvidia-smi's compute-app
# list and is kept only if its own /proc cmdline names phase_k_train.py AND this run's --out dir.
# pkill -f on a run id has self-killed this session twice; a pid that fails either test is left
# alone and the queue stops rather than guessing.
D_RID=$(cat training/runs/pose6_lora8_rid)
D_METRIC="training/runs/${D_RID}/metric_valfull83_ep20.json"
say "waiting for Arm D ($D_RID) epoch-20 VAL to complete; its slice will be skipped"
for i in $(seq 1 240); do          # 2 h ceiling
    [ -s "$D_METRIC" ] && break
    sleep 30
done
[ -s "$D_METRIC" ] || { say "NOT LAUNCHED: Arm D's epoch-20 val never completed"; exit 3; }
say "Arm D epoch-20 val is complete ($D_METRIC)"

D_PIDS=""
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
        *phase_k_train.py*"training/runs/${D_RID}"*) D_PIDS="$D_PIDS $pid" ;;
    esac
done
if [ -n "$D_PIDS" ]; then
    say "stopping Arm D's trainer ranks before it spawns the slice:$D_PIDS"
    kill $D_PIDS 2>/dev/null || true
    for i in $(seq 1 40); do
        alive=""
        for pid in $D_PIDS; do [ -d "/proc/$pid" ] && alive="$alive $pid"; done
        [ -z "$alive" ] && break
        sleep 3
    done
    [ -z "${alive:-}" ] || say "note: still up after 2 min:$alive -- the card wait below decides"
else
    say "Arm D's trainer ranks are already gone"
fi

say "waiting for the cards to clear"
for i in $(seq 1 120); do
    n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    case "$n" in ''|*[!0-9]*) n=99;; esac
    [ "$n" -eq 0 ] && break
    sleep 15
done
[ "${n:-99}" -eq 0 ] || { say "NOT LAUNCHED: cards still busy ($n)"; exit 3; }
say "cards clear"

SEEN=$SD/cosine_dirs_before.txt
ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename | sort > "$SEEN"
LOG=training/runs/pose6_cosine_launch.log
say "launching Arm B cosine resume"
DRY=0 nohup ./shell_scripts/launch_pose6_paeln_cosine.sh > "$LOG" 2>&1 &

RID=""
for i in $(seq 1 60); do
    grep -qE "REFUS|STOP: the preflight failed" "$LOG" 2>/dev/null && {
        say "NOT LAUNCHED: refused"; grep -E "REFUS|STOP: the preflight" "$LOG" | head -3; exit 4; }
    RID=$(ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename \
        | sort | comm -13 "$SEEN" - | head -1)
    [ -n "$RID" ] && break
    sleep 10
done
[ -n "$RID" ] || { say "NOT LAUNCHED: no new run directory"; exit 4; }
echo "$RID" > training/runs/pose6_cosine_rid
echo 0 > "training/runs/${RID}_last_plotted"
say "run $RID -- Arm B resumed at epoch 10, cosine 1e-3 -> 1e-4 by epoch 20"

for i in $(seq 1 60); do [ -s "training/runs/$RID/args.json" ] && break; sleep 10; done
[ -s "training/runs/$RID/args.json" ] || { say "WARNING: $RID has no args.json after 10 min"; exit 5; }
# FAIL CLOSED on the fields this arm exists to change, so a schedule that silently did not take
# is a stopped queue rather than a curve we read for ten epochs before noticing.
"$PY" -I - "$RID" <<'EOF'
import json, sys
rid = sys.argv[1]
a = json.load(open(f"training/runs/{rid}/args.json")); a = a.get("args", a)
want = {"lr": 0.001, "lr_final": 0.0001, "lr_decay_start_epoch": 10.0,
        "lr_decay_end_epoch": 20.0, "lr_decay_shape": "cosine", "epochs": 20.0,
        "expect_trainable": 20, "pae_split_accept": 0.23, "pair_dropout": 0.25}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
for k, v in want.items():
    print(f"    {k} = {a.get(k)!r}")
print(f"    resume = {a.get('resume')!r}")
print(f"    realised_seed_depths = {a.get('realised_seed_depths')!r}")
if bad:
    print("STOP: args.json does not match the intended arm:")
    for k, (got, exp) in bad.items():
        print(f"  {k}: got {got!r}, expected {exp!r}")
    raise SystemExit(6)
print("  all intended settings confirmed")
EOF
say "$RID is training"
