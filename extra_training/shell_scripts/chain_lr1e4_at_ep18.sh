#!/bin/bash
# DROP THE LEARNING RATE TO 1e-4 AT THE EPOCH-18 BOUNDARY, by stopping and resuming.
#
# WHY A STOP AND NOT A FLAG. The lr is read at startup and baked into the optimiser schedule, so
# there is no way to change it in a live process. The run stops at a boundary -- after the bundle
# is on disk, so nothing is lost -- and resumes from it with the new lr.
#
# WHY --resume-override lr IS REQUIRED AND CORRECT. phase_k_train's resume guard carries `lr`
# (line ~3724) and refuses a bundle whose lr differs, deliberately: lr changes the update at every
# step after warmup, so a silent change blends two objectives into one curve and the trajectory
# csv gives no sign of it. The override is NAMED, never --resume-allow-drift, so only this one
# field may move and any OTHER accidental difference still hard-stops.
#
# THE REASON FOR THE DROP, recorded so the curve is readable later. Training PAE has sat in a
# 0.032 band for eight epochs with no trend -- 2.7884, 2.7692, 2.7984, 2.7741, 2.8012, 2.7929,
# 2.7824, 2.7875 over epochs 10-17 -- and held-out selection has been flat across three evals
# (max-pooled 0.5845 / 0.5896 / 0.5846 at epochs 5/10/15). Flat training AND flat held-out is the
# stalled case rather than the dropout-noise-floor case, which is why this is an lr change and not
# a longer wait.
#
# IT STOPS NOTHING UNTIL THE BUNDLE EXISTS. Waiting on resume_ep18.pt rather than a clock means a
# slow epoch cannot cost the epoch.
#
# EPOCH 18 IS NOT A MULTIPLE OF 5, so no held-out eval follows it and there is no risk of killing
# a run mid-eval.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${RID:-7roo28}"
EP="${EP:-18}"
NEWLR="${NEWLR:-1e-4}"
LOG=training/runs/lr1e4_chain.log
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*" | tee -a "$LOG"; }

B="training/runs/$RID/resume_ep${EP}.pt"
say "waiting for $B before stopping $RID (lr -> $NEWLR)"
for i in $(seq 1 240); do
    [ -s "$B" ] && break
    sleep 15
done
[ -s "$B" ] || { say "FAULT: $B never appeared; NOT stopping anything"; exit 3; }
# the bundle can be mid-write; wait for its size to settle
S1=$(stat -c %s "$B"); sleep 10; S2=$(stat -c %s "$B")
[ "$S1" = "$S2" ] || { sleep 20; }
say "epoch $EP bundle on disk ($(stat -c %s "$B") bytes); stopping $RID"

./shell_scripts/stop_run.sh "$RID" --yes 2>&1 | tail -3 | tee -a "$LOG"

say "relaunching from epoch $EP at lr $NEWLR"
RESUME="$B" LR="$NEWLR" \
  RESUME_OVERRIDE=lr \
  RESUME_NOTE="lr 1e-3 -> $NEWLR at the epoch $EP boundary: training PAE flat in a 0.032 band for epochs 10-17 and held-out selection flat across the epoch 5/10/15 evals" \
  PAIR_DROPOUT=0.25 L2_SP=1e-3 EPOCHS=50 PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
  ./shell_scripts/launch_pae_crop_arm.sh >> training/runs/lr1e4_launch.log 2>&1
say "launcher returned $?"

NEW=$(ls -td training/runs/*/ 2>/dev/null | head -1 | xargs basename)
say "new run id: $NEW"
echo "$NEW" > training/runs/lr1e4_rid
for i in $(seq 1 60); do [ -f "training/runs/$NEW/args.json" ] && break; sleep 10; done
GOTLR=$($PY -I -c "import json;print(json.load(open('training/runs/$NEW/args.json')).get('lr'))" 2>/dev/null)
GOTPD=$($PY -I -c "import json;print(json.load(open('training/runs/$NEW/args.json')).get('pair_dropout'))" 2>/dev/null)
GOTL2=$($PY -I -c "import json;print(json.load(open('training/runs/$NEW/args.json')).get('l2_sp'))" 2>/dev/null)
say "args.json  lr=$GOTLR  pair_dropout=$GOTPD  l2_sp=$GOTL2"
[ "$GOTLR" = "0.0001" ] || { say "FAULT: lr is '$GOTLR', not 0.0001"; exit 4; }
[ "$GOTPD" = "0.25" ] || { say "FAULT: pair_dropout is '$GOTPD' -- the other terms did not carry over"; exit 4; }
[ "$GOTL2" = "0.001" ] || { say "FAULT: l2_sp is '$GOTL2' -- the other terms did not carry over"; exit 4; }
for i in $(seq 1 90); do
    grep -qa "continuing at epoch" "training/runs/$NEW/run.log" 2>/dev/null && break
    grep -qaE "STOP:|Traceback" "training/runs/$NEW/run.log" 2>/dev/null && { say "FAULT: died at startup"; exit 5; }
    sleep 10
done
say "$(grep -a 'continuing at epoch' training/runs/$NEW/run.log | head -1 | cut -c1-150)"
say "$(grep -a 'anchor' training/runs/$NEW/run.log | grep -a -iE 'INHERITED|READ' | head -1 | cut -c1-150)"
say "LR DROP COMPLETE: $NEW running at lr $NEWLR from epoch $((EP+1))"
