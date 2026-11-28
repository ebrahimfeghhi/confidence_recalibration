#!/usr/bin/env bash
# SUPERSEDED by supervise_rel_alphas.sh -- DO NOT RUN.
# Why: a single transient card refusal exited the WHOLE sweep (exit 4) and silently dropped every
# remaining alpha, and its finish test -- cards_busy == 0 -- could not tell a completed arm
# from one that OOMed at epoch 2, printing "finished" either way.
# Kept for the record only; it refuses below.
echo "REFUSING: this script is superseded by supervise_rel_alphas.sh. See the header." >&2
exit 64
# THREE 10-EPOCH ARMS IN SEQUENCE: Arm E + the mean-centred iTM ranking term at alpha 100, then
# 200, then 50. One at a time, because each needs all three cards.
#
# ORDER IS DELIBERATE: parity (100) first, so the measured value has produced a result before
# either flank. If the box is lost overnight, the arm that exists is the one the measurement
# actually pointed at.
#
# EACH ARM WAITS FOR THE CARDS rather than for a fixed time. The previous arm's last act is a val
# pass plus a slice280 pass that hold all three cards well after the final checkpoint lands, so a
# waiter keyed on epoch count would launch into a busy box.
#
# THE CARDS ARE NOT TAKEN WITHOUT A FREE GATE EITHER WAY: launch_pae_crop_arm.sh runs cards_busy
# itself and refuses occupied cards, so a stray copy of this cannot displace a live arm.
set -u
cd "$(dirname "$0")/.."
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
mkdir -p "$SD"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
PY=/home/ebrahim/Protenix/.venv/bin/python

for ALPHA in ${ALPHAS:-100 200 50}; do
    say "alpha $ALPHA: waiting for the cards"
    n=99
    for i in $(seq 1 480); do
        n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
        case "$n" in ''|*[!0-9]*) n=99;; esac
        [ "$n" -eq 0 ] && break
        sleep 30
    done
    [ "$n" -eq 0 ] || { say "STOPPING: cards still busy ($n) after 4 h; alpha $ALPHA NOT launched"; exit 3; }

    SEEN=$SD/rel_${ALPHA}_before.txt
    ls -d training/runs/*/ 2>/dev/null | grep -vE "/(latest|archive)/$" | xargs -r -n1 basename | sort > "$SEEN"
    LOG=training/runs/rel_alpha${ALPHA}_launch.log
    say "launching alpha $ALPHA"
    ALPHA_RANK="$ALPHA" DRY=0 nohup ./shell_scripts/launch_armE_rel.sh > "$LOG" 2>&1 &

    RID=""
    for i in $(seq 1 60); do
        grep -qE "REFUS|STOP: the preflight failed" "$LOG" 2>/dev/null && {
            say "NOT LAUNCHED at alpha $ALPHA: refused"; grep -E "REFUS|STOP:" "$LOG" | head -3; exit 4; }
        RID=$(ls -d training/runs/*/ 2>/dev/null | grep -vE "/(latest|archive)/$" \
              | xargs -r -n1 basename | sort | comm -13 "$SEEN" - | head -1)
        [ -n "$RID" ] && break
        sleep 10
    done
    [ -n "$RID" ] || { say "NOT LAUNCHED at alpha $ALPHA: no new run directory"; exit 4; }
    echo "$RID" > "training/runs/rel_alpha${ALPHA}_rid"
    echo "$RID" > training/runs/pose6_norms_rid          # the table/figure pointer follows the live arm
    printf 'rel a%s' "$ALPHA" > "training/runs/${RID}_name"
    echo 73rf4v > training/runs/plot_reference_rid       # always drawn against Arm E, the base
    echo 0 > "training/runs/${RID}_last_autoplot"
    say "run $RID -- Arm E + rel at alpha $ALPHA"

    for i in $(seq 1 60); do [ -s "training/runs/$RID/args.json" ] && break; sleep 10; done
    [ -s "training/runs/$RID/args.json" ] || { say "WARNING: $RID has no args.json after 10 min"; exit 5; }
    "$PY" -I - "$RID" "$ALPHA" <<'PYEOF'
import json, sys
rid, alpha = sys.argv[1], float(sys.argv[2])
a = json.load(open(f"training/runs/{rid}/args.json")); a = a.get("args", a)
want = {"expect_trainable": 53, "lr": 0.001, "epochs": 10.0, "pair_dropout": 0.25,
        "pae_split_accept": 0.23, "pae_balance_accept": 0.0, "lora_rank": 0,
        "pose_readout_post_zero": True, "train_eval_every": 10,
        # THE WHOLE POINT OF THE ARM. rank_loss False, or the wrong alpha, is Arm E again.
        "rank_loss": True, "rank_form": "rel", "alpha_rank": alpha}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
for k in want: print(f"    {k} = {a.get(k)!r}")
print(f"    rel_labels = {a.get('rel_labels')!r}")
if bad:
    print("STOP: args.json does not match the intended arm:")
    for k, (got, exp) in bad.items(): print(f"  {k}: got {got!r}, expected {exp!r}")
    raise SystemExit(6)
print("  all intended settings confirmed")
PYEOF
    [ $? -eq 0 ] || { say "STOPPING: alpha $ALPHA args mismatch"; exit 6; }
    say "$RID is training; waiting for it to finish before the next alpha"

    # DONE = THE CARDS COME BACK. The run ends with two eval passes after the last checkpoint.
    sleep 120
    for i in $(seq 1 960); do
        n=$("$PY" -I run_logging/cards_busy.py 2>/dev/null | tail -1)
        case "$n" in ''|*[!0-9]*) n=99;; esac
        [ "$n" -eq 0 ] && break
        sleep 30
    done
    say "alpha $ALPHA finished ($RID)"
done
say "sweep complete: alphas ${ALPHAS:-100 200 50}"
