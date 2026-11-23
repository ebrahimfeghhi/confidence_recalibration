#!/usr/bin/env bash
# ARM G FIRES ONCE THE SIX-ALPHA SWEEP SUPERVISOR EXITS.
#
# IT WAITS ON THE SUPERVISOR'S PID, NOT ON THE CARDS. The cards go free between every pair of
# sweep arms, so a card-only waiter would launch Arm G into the middle of the sweep and the
# launcher's own gate would then refuse one of the two -- losing an alpha or losing Arm G,
# depending which got there first.
#
# IT STILL WAITS FOR THE CARDS AFTERWARDS, because the last sweep arm's val and slice280 passes
# hold all three GPUs for ~35 min after its final checkpoint, and the supervisor only exits once
# those are done -- but a retry loop costs nothing and covers a neighbour taking a card first.
set -u
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python

WAIT_PID="${WAIT_PID:?set WAIT_PID to the sweep supervisor pid}"
say "Arm G queued; waiting for the alpha sweep (pid $WAIT_PID)"
while [ -d "/proc/$WAIT_PID" ]; do sleep 60; done
say "sweep supervisor exited"

n=99
for i in $(seq 1 480); do
    n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    case "$n" in ''|*[!0-9]*) n=99;; esac
    [ "$n" -eq 0 ] && break
    sleep 30
done
[ "$n" -eq 0 ] || { say "NOT LAUNCHED: cards busy ($n) after 4 h"; exit 3; }

SEEN=$SD/armG_before.txt
ls -d training/runs/*/ 2>/dev/null | grep -vE "/(latest|archive)/$" | xargs -r -n1 basename | sort > "$SEEN"
LOG=training/runs/armG_launch.log
say "launching Arm G"
DRY=0 nohup ./shell_scripts/launch_armG_lora_trimul.sh > "$LOG" 2>&1 &

RID=""
for i in $(seq 1 120); do
    RID=$(ls -d training/runs/*/ 2>/dev/null | grep -vE "/(latest|archive)/$" \
          | xargs -r -n1 basename | sort | comm -13 "$SEEN" - | head -1)
    [ -n "$RID" ] && break
    grep -qE "REFUS|STOP: the preflight failed" "$LOG" 2>/dev/null && {
        say "NOT LAUNCHED: $(grep -E 'REFUS|STOP:' "$LOG" | head -1)"; exit 4; }
    sleep 10
done
[ -n "$RID" ] || { say "NOT LAUNCHED: no new run directory"; exit 4; }
echo "$RID" > training/runs/armG_rid
echo "$RID" > training/runs/pose6_norms_rid
printf 'Arm G (tri_mul LoRA)' > "training/runs/${RID}_name"
echo 73rf4v > training/runs/plot_reference_rid
echo 0 > "training/runs/${RID}_last_autoplot"
say "run $RID -- Arm E + rank-8 LoRA on tri_mul of blocks 1-3, 20 epochs"

for i in $(seq 1 120); do [ -s "training/runs/$RID/args.json" ] && break; sleep 10; done
"$PY" -I - "$RID" <<'PYEOF'
import json, sys
rid = sys.argv[1]
a = json.load(open(f"training/runs/{rid}/args.json")); a = a.get("args", a)
want = {"lora_rank": 8, "epochs": 20.0, "lr": 0.001, "pair_dropout": 0.25,
        "pae_split_accept": 0.23, "pae_balance_accept": 0.0,
        "pose_readout_post_zero": True, "train_eval_every": 10, "rank_loss": False,
        "expect_trainable": 53,
        # THE TWO FIELDS THAT DEFINE THIS ARM. Without them a target pattern that lost its
        # trailing `$` -- adapting all 36 tri_mul modules, gates included -- passes verification.
        "lora_targets": r"blocks\.[123]\.tri_mul_(in|out)\.linear_(a_p|b_p|z)$"}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
print(f"    lora_targets        = {a.get('lora_targets')!r}")
print(f"    lora_keep_trainable = {str(a.get('lora_keep_trainable'))[:90]}...")
for k in want: print(f"    {k} = {a.get(k)!r}")
if bad:
    print("STOP: args.json does not match the intended arm:")
    for k, (got, exp) in bad.items(): print(f"  {k}: got {got!r}, expected {exp!r}")
    raise SystemExit(6)
print("  all intended settings confirmed")
PYEOF
[ $? -eq 0 ] || { say "Arm G: ARGS MISMATCH on $RID -- see the lines above. NOT continuing."; exit 6; }
say "Arm G is training; waiting for epoch 1 to check the adapters actually move"
for i in $(seq 1 180); do
    ls "training/runs/$RID"/head_ep1_*.pt >/dev/null 2>&1 && break
    sleep 30
done
if ls "training/runs/$RID"/head_ep1_*.pt >/dev/null 2>&1; then
    say "adapter movement check (vs Arm E, where these 18 weights are FROZEN):"
    "$PY" -I run_logging/check_adapters_moved.py \
        --base training/runs/73rf4v/head_ep20_step2900.pt \
        --after "$(ls "training/runs/$RID"/head_ep1_*.pt | head -1)" | tail -4
else
    say "WARNING: no epoch-1 checkpoint after 90 min; adapter movement UNVERIFIED"
fi
