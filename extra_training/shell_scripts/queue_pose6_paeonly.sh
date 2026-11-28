#!/bin/bash
# QUEUE THE SIX-SITE PAE-ONLY ARM, behind 4yu3s4. THE SIXTH SITE IS THE ONLY VARIABLE.
#
# ONE VARIABLE. Everything is 4yu3s4 -- same PAE-only loss with no ranking term, same lr 1e-3,
# same pair-dropout 0.25, same PAE balance OFF with split 0.23 for the tally, 20 epochs with val and slice at 10 and 20 -- plus a SIXTH injection site and 18 trainable tensors instead of
# 15. A difference against 4yu3s4 is the sixth site and nothing else.
#
# cards_busy IS THE GATE, NOT A LOG LINE. It prints the count of foreign compute processes on
# stdout and always exits 0 when it can establish occupancy, so the count is what has to be read
# -- gating on the exit status passes immediately whatever is running, which is how an earlier
# queue in this project would have launched on top of a live arm.
#
# IT DOES NOT CARE WHETHER 4yu3s4 SUCCEEDED. It waits for the cards, not for an outcome: a run
# that dies at epoch 3 frees them just as a finished one does, and this arm is independent of it.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-20}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
# EXCLUDE THE `latest` SYMLINK. launch.py repoints it at every launch and `ls -td` with a
# trailing slash dereferences it, so their mtimes tie and GNU ls breaks the tie by NAME -- for
# any run id sorting after "latest" the newest directory comes back as "latest" and that string
# gets recorded as the run id. autoplot.sh has always filtered it; this copy had dropped the
# filter.
newest(){ ls -td training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | head -1 \
    | xargs -r basename; }

say "waiting for the cards (4yu3s4 holds them to epoch 20)"
for i in $(seq 1 480); do
    N=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    [ "$N" = "0" ] && { say "cards clear"; break; }
    [ "$i" = 480 ] && { say "GIVING UP: cards busy after 8 h"; exit 2; }
    sleep 60
done

SEEN=$SD/pose6_dirs_before.txt
ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename | sort > "$SEEN"
LOG=training/runs/pose6_paeonly_launch.log
say "launching the PAE-only SIX-site arm at lr $LR, $EPOCHS epochs"
LR="$LR" EPOCHS="$EPOCHS" DRY=0 nohup ./shell_scripts/launch_pose6_paeonly.sh > "$LOG" 2>&1 &
LP=$!
# POLL FOR THE RUN DIRECTORY, NOT FOR THE LAUNCHER TO EXIT. launch.py runs the trainer with
# subprocess.run, so the launcher blocks for the WHOLE run and a /proc liveness test never fires;
# waiting on it burned the full timeout before recording anything.
# BOTH REFUSAL SPELLINGS. launch_pae_crop_arm.sh prints "REFUSING"; launch.py prints
# "STOP: the preflight failed (exit N); nothing launched" -- grepping only REFUS missed the latter.
RID=""
for i in $(seq 1 90); do
    grep -qE "REFUS|STOP: the preflight failed" "$LOG" 2>/dev/null && {
        say "NOT LAUNCHED: refused"; grep -E "REFUS|STOP: the preflight" "$LOG" | head -3; exit 3; }
    RID=$(ls -d training/runs/*/ 2>/dev/null | grep -vE "/latest/$" | xargs -r -n1 basename \
        | sort | comm -13 "$SEEN" - | head -1)
    [ -n "$RID" ] && break
    sleep 20
done
[ -n "$RID" ] || { say "NOT LAUNCHED: no new run directory"; tail -5 "$LOG"; exit 3; }
echo "$RID" > training/runs/pose6_paeonly_rid
say "run $RID -- 6 sites (post-LN zero-init), PAE-only, lr $LR, $EPOCHS epochs"

# WAIT FOR args.json, THEN VERIFY IT. The directory appears up to two minutes before args.json
# does -- 4yu3s4's were 41 s apart -- so a dump fired immediately raises and, swallowed, printed
# nothing: the one check that would have caught a wrong architecture.
for i in $(seq 1 60); do
    [ -s "training/runs/$RID/args.json" ] && break
    sleep 10
done
if [ ! -s "training/runs/$RID/args.json" ]; then
    say "WARNING: $RID has no args.json after 10 min -- it may have died in its gates"
    tail -5 "$LOG"
    exit 4
fi
$PY -I - "$RID" <<'PYEOF'
import json, sys
rid = sys.argv[1]
a = json.load(open(f"training/runs/{rid}/args.json"))
a = a.get("args", a)
want = {"lr": 0.001, "epochs": 20.0, "rank_loss": False, "pae_balance_accept": 0.0,
        "pae_split_accept": 0.23, "pair_dropout": 0.25, "expect_trainable": 18,
        "pose_readout": True, "pose_readout_prenorm": True, "pose_readout_post_zero": True,
        "train_eval_every": 10}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
for k, v in want.items():
    print(f"    {k} = {a.get(k)!r}")
if bad:
    print("  MISMATCH against the intended arm:", bad)
    sys.exit(1)
print("  all intended settings confirmed")
PYEOF
if [ $? != 0 ]; then
    say "STOPPING: $RID is NOT the arm that was queued -- read the dump above"
    exit 6
fi

# THE TRAINER'S OWN STOP IS NOT IN THIS LOG. launch.py sends trainer stdout to the run's run.log,
# so --expect-trainable and pose_readout refusals are invisible here unless looked for.
sleep 60
if grep -qE "^STOP:|Traceback" "training/runs/$RID/run.log" 2>/dev/null; then
    say "NOT TRAINING: $RID stopped itself"
    grep -E "^STOP:" "training/runs/$RID/run.log" | head -3
    exit 5
fi
say "$RID is training"
