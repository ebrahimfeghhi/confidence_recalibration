#!/bin/bash
# QUEUE ARM C THEN ARM D, each behind the one before it.
#
# ARM C = Arm B + rank-4 LoRA on blocks.3.pair_transition. ARM D = the same at rank 8.
#
# CHAINED, NOT PARALLEL-WAITING. Four queues all waiting on "cards free" would fire together the
# instant the GPUs cleared and race; each link here waits for the PREVIOUS arm's rid file, then
# for that arm to finish, then for the cards.
#
# THE GATE NEEDS A MATCHING REFERENCE HEAD. run_logging/test_lora_targets.py reads the real head
# tensor NAMES from a checkpoint, so it must be given one with this architecture -- Arm B's own
# head_ep20, which exists by the time Arm C starts. Without LORA_GATE_REF it falls back to the
# newest head on disk, which may be a 5-site one and would not see the readout tensors.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
EPOCHS=${EPOCHS:-20}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

# wait_for <rid-file> <label> -- the previous arm must exist and be done
PREV_RID=""
wait_for(){
    local f=$1 lab=$2 rid n age
    say "waiting for $lab to start ($f)"
    for i in $(seq 1 1440); do
        [ -s "$f" ] && break
        [ "$i" = 1440 ] && { say "GIVING UP: $lab never started in 24 h"; exit 2; }
        sleep 60
    done
    rid=$(cat "$f")
    say "$lab is $rid; waiting for it to finish $EPOCHS epochs"
    for i in $(seq 1 1440); do
        n=$(ls "training/runs/$rid"/head_ep*.pt 2>/dev/null \
            | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
        [ -n "$n" ] && [ "$n" -ge "$EPOCHS" ] 2>/dev/null && { say "$lab reached epoch $n"; break; }
        # A MISSING LOG IS NOT A QUIET RUN: stat on an absent file used to return 0, making the age
        # the whole unix epoch and firing this fallback on the first poll.
        if [ -f "training/runs/$rid/run.log" ]; then
            age=$(( $(date +%s) - $(stat -c %Y "training/runs/$rid/run.log") ))
        else
            age=0
        fi
        [ "$age" -gt 3600 ] && { say "$lab quiet for ${age}s at epoch ${n:-?}; proceeding"; break; }
        [ "$i" = 1440 ] && { say "GIVING UP: $lab still running after 24 h"; exit 2; }
        sleep 60
    done
    PREV_RID="$rid"
}

wait_cards(){
    say "waiting for the cards"
    local c
    for i in $(seq 1 240); do
        c=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
        [ "$c" = "0" ] && { say "cards clear"; return 0; }
        [ "$i" = 240 ] && { say "GIVING UP: cards busy after 4 h"; exit 3; }
        sleep 60
    done
}

# launch_arm <launcher> <ridfile> <label> <ntrain> <gate-ref-head>
launch_arm(){
    local sh=$1 ridfile=$2 lab=$3 ntrain=$4 ref=$5 log seen rid
    log=training/runs/$(basename "$ridfile" _rid)_launch.log
    seen=$SD/$(basename "$ridfile")_dirs_before.txt
    ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename \
        | sort > "$seen"
    say "launching $lab via $sh (gate ref: ${ref:-newest on disk})"
    LORA_GATE_REF="$ref" EPOCHS="$EPOCHS" EXPECT_TRAINABLE="$ntrain" DRY=0 \
        nohup "./$sh" > "$log" 2>&1 &
    local lp=$!
    rid=""
    for i in $(seq 1 90); do
        grep -qE "REFUS|STOP: the preflight failed" "$log" 2>/dev/null && {
            say "NOT LAUNCHED ($lab): refused"
            grep -E "REFUS|STOP: the preflight" "$log" | head -3; exit 4; }
        rid=$(ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename \
            | sort | comm -13 "$seen" - | head -1)
        [ -n "$rid" ] && break
        sleep 20
    done
    [ -n "$rid" ] || { say "NOT LAUNCHED ($lab): no new run directory"; tail -5 "$log"; exit 4; }
    echo "$rid" > "$ridfile"
    say "$lab is run $rid"
    for i in $(seq 1 60); do
        [ -s "training/runs/$rid/args.json" ] && break
        sleep 10
    done
    [ -s "training/runs/$rid/args.json" ] || {
        say "WARNING ($lab): no args.json after 10 min -- it may have died in its gates"
        tail -5 "$log"; exit 5; }
    $PY -I - "$rid" "$ntrain" "$6" <<'PYEOF'
import json, sys
rid, ntrain, rank = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = json.load(open(f"training/runs/{rid}/args.json")); a = a.get("args", a)
want = {"lr": 0.001, "epochs": 20.0, "rank_loss": False, "pae_balance_accept": 0.0,
        "pae_split_accept": 0.23, "pair_dropout": 0.25, "expect_trainable": ntrain,
        "pose_readout": True, "pose_readout_prenorm": True, "pose_readout_post_zero": True,
        "train_eval_every": 10, "lora_rank": rank}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
for k in want:
    print(f"    {k} = {a.get(k)!r}")
print(f"    lora_targets = {a.get('lora_targets')!r}")
print(f"    train_only   = {a.get('train_only')!r}")
if "pae_ln" not in (a.get("train_only") or ""):
    bad["train_only"] = (a.get("train_only"), "must contain pae_ln")
if "pair_transition" not in (a.get("lora_targets") or ""):
    bad["lora_targets"] = (a.get("lora_targets"), "must target a pair transition")
if bad:
    print("  MISMATCH against the intended arm:", bad); sys.exit(1)
print("  all intended settings confirmed")
PYEOF
    if [ $? != 0 ]; then
        say "STOPPING ($lab): $rid is NOT the arm that was queued"; exit 6
    fi
    sleep 60
    if grep -qE "^STOP:|Traceback" "training/runs/$rid/run.log" 2>/dev/null; then
        say "NOT TRAINING ($lab): $rid stopped itself"
        grep -E "^STOP:" "training/runs/$rid/run.log" | head -3; exit 7
    fi
    say "$lab ($rid) is training"
}

wait_for training/runs/pose6_paeln_rid "Arm B"
B="$PREV_RID"
wait_cards
launch_arm shell_scripts/launch_pose6_paeln_lora4.sh training/runs/pose6_lora4_rid \
    "Arm C (rank 4)" 26 "$(ls -t training/runs/$B/head_ep*.pt 2>/dev/null | head -1)" 4

wait_for training/runs/pose6_lora4_rid "Arm C"
C="$PREV_RID"
wait_cards
launch_arm shell_scripts/launch_pose6_paeln_lora8.sh training/runs/pose6_lora8_rid \
    "Arm D (rank 8)" 26 "$(ls -t training/runs/$C/head_ep*.pt 2>/dev/null | head -1)" 8
say "ARM C AND ARM D BOTH LAUNCHED"
