#!/usr/bin/env bash
# Relaunches Arm E once the shipped mean-pooled val pass frees the cards.
# Arm E was stopped one minute into its second start so the shipped baseline could run first;
# it has no epochs worth keeping, so this is a clean launch rather than a resume.
set -euo pipefail
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python

say "waiting for the cards (shipped mean-pooled val pass)"
for i in $(seq 1 120); do
    n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    case "$n" in ''|*[!0-9]*) n=99;; esac
    [ "$n" -eq 0 ] && break
    sleep 30
done
[ "${n:-99}" -eq 0 ] || { say "NOT LAUNCHED: cards still busy ($n)"; exit 3; }

# AND WAIT FOR THE MERGE, not just the cards: the shards release the GPUs before the metric json
# is written, and launching into that window would have Arm E's startup competing with the merge.
for i in $(seq 1 40); do
    [ -s "$SD/val30_shipped_mean2.json" ] && break
    grep -qE "ABORT|STOP" "$SD/shipped2.log" 2>/dev/null && break
    sleep 15
done
say "cards clear"

SEEN=$SD/armE_seen2.txt
ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename | sort > "$SEEN"
LOG=training/runs/pose6_norms_launch.log
say "launching Arm E"
DRY=0 nohup ./shell_scripts/launch_pose6_paeln_norms.sh > "$LOG" 2>&1 &

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
echo "$RID" > training/runs/pose6_norms_rid
echo 0 > "training/runs/${RID}_last_plotted"
say "run $RID -- Arm E, 53 tensors"

for i in $(seq 1 60); do [ -s "training/runs/$RID/args.json" ] && break; sleep 10; done
"$PY" -I - "$RID" <<'PYEOF'
import json, sys
rid = sys.argv[1]
a = json.load(open(f"training/runs/{rid}/args.json")); a = a.get("args", a)
want = {"expect_trainable": 53, "lr": 0.001, "epochs": 20.0, "save_init": ""}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
for k in want: print(f"    {k} = {a.get(k)!r}")
if bad:
    print("STOP: args.json does not match the intended arm:", bad); raise SystemExit(6)
print("  all intended settings confirmed")
PYEOF
say "$RID is training"
