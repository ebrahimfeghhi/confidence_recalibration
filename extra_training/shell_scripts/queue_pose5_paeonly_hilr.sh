#!/bin/bash
# QUEUE THE SAME PAE-ONLY ARM AT A HIGHER lr, behind k5x24w. LR is the only variable.
#
# ONE VARIABLE. Everything is launch_pose5_paeonly_lr1e4.sh's environment with LR=3e-4: same five
# injection sites, same 15 trainable tensors with everything else frozen, same pair-dropout 0.25,
# same PAE balance OFF and split 0.23 for the tally, same 20 epochs with val and slice at 10 and
# 20, no ranking loss. So a difference against k5x24w is the learning rate and nothing else.
#
# cards_busy IS THE GATE, NOT A LOG LINE. It prints the count of foreign compute processes on
# stdout and always exits 0 when it can establish occupancy, so the count is what has to be read
# -- gating on the exit status passes immediately whatever is running, which is how an earlier
# queue in this project would have launched on top of a live arm.
#
# IT DOES NOT CARE WHETHER k5x24w SUCCEEDED. It waits for the cards, not for an outcome: a run
# that dies at epoch 3 frees them just as a finished one does, and this arm is independent of it.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
LR=${LR:-1e-3}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest(){ ls -td training/runs/*/ 2>/dev/null | head -1 | xargs -r basename; }

say "waiting for the cards (k5x24w holds them to epoch 20, about 5:50 PM EST)"
for i in $(seq 1 480); do
    N=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
    [ "$N" = "0" ] && { say "cards clear"; break; }
    [ "$i" = 480 ] && { say "GIVING UP: cards busy after 8 h"; exit 2; }
    sleep 60
done

BEFORE=$(newest)
LOG=training/runs/pose5_paeonly_hilr_launch.log
say "launching the PAE-only 5-site arm at lr $LR, 20 epochs"
LR="$LR" DRY=0 nohup ./shell_scripts/launch_pose5_paeonly_lr1e4.sh > "$LOG" 2>&1 &
LP=$!
for i in $(seq 1 90); do
    grep -qE "LAUNCH|REFUS|STOP:" "$LOG" 2>/dev/null && break
    [ -d "/proc/$LP" ] || break
    sleep 20
done
if grep -qE "REFUS|STOP:" "$LOG"; then
    say "NOT LAUNCHED: the launcher refused"; grep -E "REFUS|STOP:" "$LOG" | head -3; exit 3
fi
# A DIRECTORY THAT DID NOT EXIST BEFORE THIS STEP, so a refusal cannot be recorded as the new run.
RID=""
for i in $(seq 1 30); do
    N=$(newest)
    [ -n "$N" ] && [ "$N" != "$BEFORE" ] && { RID="$N"; break; }
    sleep 10
done
[ -n "$RID" ] || { say "NOT LAUNCHED: no new run directory"; exit 3; }
echo "$RID" > training/runs/pose5_paeonly_hilr_rid
say "run $RID -- 5 sites, PAE-only, lr $LR, 20 epochs"
$PY -I -c "
import json
a=json.load(open('training/runs/$RID/args.json')); a=a.get('args',a)
for k in ('lr','epochs','rank_loss','pae_balance_accept','pae_split_accept','pair_dropout','expect_trainable','train_eval_every'):
    print(f'    {k} = {a.get(k)!r}')" 2>/dev/null || true
