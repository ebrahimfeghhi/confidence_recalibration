#!/bin/bash
# QUEUE ARM B (six sites + pae_ln unfrozen) BEHIND ARM A (six sites), not merely behind the cards.
#
# WHY THAT DISTINCTION MATTERS. Arm A's queue is also waiting on free cards. If this one waited on
# the same condition the two would fire at the same instant when 4yu3s4 exits and race for the
# GPUs -- one would win the cards_busy check and the other would be refused by the launcher, at
# random. So this waits for Arm A to EXIST (its rid file) and then to be FINISHED (epoch-20 head
# on disk, or its processes gone), and only then for the cards.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-20}
NTRAIN=${NTRAIN:-20}
A_RID_FILE=${A_RID_FILE:-training/runs/pose6_paeonly_rid}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

say "waiting for Arm A to start (its rid file: $A_RID_FILE)"
for i in $(seq 1 720); do
    [ -s "$A_RID_FILE" ] && break
    [ "$i" = 720 ] && { say "GIVING UP: Arm A never started in 12 h"; exit 2; }
    sleep 60
done
A=$(cat "$A_RID_FILE")
say "Arm A is $A; waiting for it to finish $EPOCHS epochs"
for i in $(seq 1 720); do
    N=$(ls "training/runs/$A"/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' \
        | sort -n | tail -1)
    [ -n "$N" ] && [ "$N" -ge "$EPOCHS" ] 2>/dev/null && { say "Arm A reached epoch $N"; break; }
    # A DEAD ARM A ALSO RELEASES THE CARDS. Waiting only for epoch 20 would hang forever on a run
    # that died at epoch 3; the card check below is what actually gates the launch either way.
    if [ -f "training/runs/$A/run.log" ]; then
        AGE=$(( $(date +%s) - $(stat -c %Y "training/runs/$A/run.log") ))
    else
        AGE=0   # NO LOG YET IS NOT A QUIET RUN. stat on a missing file returned
    fi      # 0, making AGE the whole unix epoch and firing the fallback at once.
    [ "$AGE" -gt 3600 ] && { say "Arm A has been quiet for ${AGE}s at epoch ${N:-?}; proceeding"; break; }
    [ "$i" = 720 ] && { say "GIVING UP: Arm A still running after 12 h"; exit 2; }
    sleep 60
done

say "waiting for the cards"
for i in $(seq 1 240); do
    C=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    [ "$C" = "0" ] && { say "cards clear"; break; }
    [ "$i" = 240 ] && { say "GIVING UP: cards busy after 4 h"; exit 3; }
    sleep 60
done

SEEN=$SD/pose6_paeln_dirs_before.txt
ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename | sort > "$SEEN"
LOG=training/runs/pose6_paeln_launch.log
say "launching Arm B: 6 sites + pae_ln, lr $LR, $EPOCHS epochs, $NTRAIN trainable"
LR="$LR" EPOCHS="$EPOCHS" EXPECT_TRAINABLE="$NTRAIN" DRY=0 \
    nohup ./shell_scripts/launch_pose6_paeln.sh > "$LOG" 2>&1 &
LP=$!
for i in $(seq 1 60); do
    grep -qE "REFUS|STOP: the preflight failed" "$LOG" 2>/dev/null && {
        say "NOT LAUNCHED: refused"
        grep -E "REFUS|STOP: the preflight" "$LOG" | head -3; exit 4; }
    [ -d "/proc/$LP" ] || break
    sleep 20
done
RID=""
for i in $(seq 1 60); do
    RID=$(ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename \
        | sort | comm -13 "$SEEN" - | head -1)
    [ -n "$RID" ] && break
    sleep 10
done
[ -n "$RID" ] || { say "NOT LAUNCHED: no new run directory"; exit 4; }
echo "$RID" > training/runs/pose6_paeln_rid
say "run $RID -- 6 sites + pae_ln unfrozen, PAE-only, lr $LR, $EPOCHS epochs"

for i in $(seq 1 60); do
    [ -s "training/runs/$RID/args.json" ] && break
    sleep 10
done
if [ ! -s "training/runs/$RID/args.json" ]; then
    say "WARNING: $RID has no args.json after 10 min -- it may have died in its gates"
    tail -5 "$LOG"; exit 5
fi
$PY -I - "$RID" "$NTRAIN" <<'PYEOF'
import json, sys
rid, ntrain = sys.argv[1], int(sys.argv[2])
a = json.load(open(f"training/runs/{rid}/args.json"))
a = a.get("args", a)
want = {"lr": 0.001, "epochs": 20.0, "rank_loss": False, "pae_balance_accept": 0.0,
        "pae_split_accept": 0.23, "pair_dropout": 0.25, "expect_trainable": ntrain,
        "pose_readout": True, "pose_readout_prenorm": True, "pose_readout_post_zero": True,
        "train_eval_every": 10}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
for k in want:
    print(f"    {k} = {a.get(k)!r}")
# THE WHOLE POINT OF ARM B is in train_only, so it is asserted rather than printed.
if "pae_ln" not in (a.get("train_only") or ""):
    print("  MISMATCH: train_only does not contain pae_ln -- this is Arm A, not Arm B")
    bad["train_only"] = (a.get("train_only"), "must contain pae_ln")
if bad:
    print("  MISMATCH against the intended arm:", bad)
    sys.exit(1)
print("  all intended settings confirmed, including pae_ln in --train-only")
PYEOF
if [ $? != 0 ]; then
    say "STOPPING: $RID is NOT the arm that was queued -- read the dump above"
    exit 7
fi
sleep 60
if grep -qE "^STOP:|Traceback" "training/runs/$RID/run.log" 2>/dev/null; then
    say "NOT TRAINING: $RID stopped itself"
    grep -E "^STOP:" "training/runs/$RID/run.log" | head -3; exit 6
fi
say "$RID is training"
