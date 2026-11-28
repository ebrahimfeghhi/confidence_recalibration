#!/usr/bin/env bash
# Fires ARM F once Arm E's epoch-20 val and slice passes free the cards.
#
# The base is Arm E on the user's instruction, not by the decision rule: val mean
# DockQ was 0.5940 (Arm B) against 0.5937 (Arm E), which the rule cannot separate.
#
# THE WAIT IS ON THE CARDS, NOT ON AN EPOCH COUNT. Arm E's last act is two eval
# passes that hold all three cards well after epoch 20's checkpoint lands, so a
# waiter keyed on the checkpoint would launch into a busy box and OOM.
set -euo pipefail
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
mkdir -p "$SD"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python

say "waiting for the cards (Arm E epoch-20 val + slice are finishing)"
for i in $(seq 1 240); do
    n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    case "$n" in ''|*[!0-9]*) n=99;; esac
    [ "$n" -eq 0 ] && break
    sleep 30
done
[ "${n:-99}" -eq 0 ] || { say "NOT LAUNCHED: cards still busy ($n) after 2 h"; exit 3; }
say "cards clear"

SEEN=$SD/armF_dirs_before.txt
ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename | sort > "$SEEN"
LOG=training/runs/armF_mlp_launch.log
say "launching Arm F"
DRY=0 nohup ./shell_scripts/launch_armF_mlp.sh > "$LOG" 2>&1 &

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
echo "$RID" > training/runs/armF_mlp_rid
# THE WATCHER FOLLOWS THE NEW ARM, and the reference it is drawn against becomes
# Arm E -- the base Arm F is built on, which is the only informative comparison.
echo "$RID" > training/runs/pose6_norms_rid
echo "$(cat training/runs/pose6_norms_rid.prev 2>/dev/null || echo 73rf4v)" > training/runs/plot_reference_rid
echo 0 > "training/runs/${RID}_last_plotted"
say "run $RID -- Arm E + the nonlinear pose MLP branch (68 tensors, 144,128 params)"

for i in $(seq 1 60); do [ -s "training/runs/$RID/args.json" ] && break; sleep 10; done
[ -s "training/runs/$RID/args.json" ] || { say "WARNING: $RID has no args.json after 10 min"; exit 5; }
"$PY" -I - "$RID" <<'PYEOF'
import json, sys
rid = sys.argv[1]
a = json.load(open(f"training/runs/{rid}/args.json")); a = a.get("args", a)
want = {"expect_trainable": 68, "lr": 0.001, "epochs": 20.0, "pair_dropout": 0.25,
        "pae_split_accept": 0.23, "pae_balance_accept": 0.0, "lora_rank": 0,
        "pose_readout_post_zero": True, "train_eval_every": 10,
        # THE WHOLE POINT OF THE ARM. A False here means the branch never installed
        # and the run is Arm E over again under a new name.
        "pose_mlp": True}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
for k in want: print(f"    {k} = {a.get(k)!r}")
print(f"    train_only = {a.get('train_only')!r}")
if bad:
    print("STOP: args.json does not match the intended arm:")
    for k, (got, exp) in bad.items():
        print(f"  {k}: got {got!r}, expected {exp!r}")
    raise SystemExit(6)
print("  all intended settings confirmed")
PYEOF
say "$RID is training"
