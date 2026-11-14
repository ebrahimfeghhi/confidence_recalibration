#!/bin/bash
# Continue the 1e-3 arm (3h3y37) from epoch 5 to epoch 20. Authorised by the owner 2026-11-08.
# Flags copied verbatim from lr_sweep.sh's run_arm, with EPOCHS 5 -> 20 and --resume added.
#
# THE GATES RETRY RATHER THAN REFUSE ON FIRST LOOK. GPU memory is released some seconds after the
# epoch-5 ranks exit, so an instantaneous check can see the cards still held and refuse -- which
# would cost the whole night for a transient. Each gate is re-checked for 10 minutes before the
# launcher gives up. A gate that is still failing after 10 minutes is a real condition, not a lag.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
RID=3h3y37
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
RANKS="2925941 2925942 2925943 2925824"

say "waiting for $RID epoch 5 + its held-out eval"
for _ in $(seq 1 240); do
  [ -s "training/runs/$RID/metric_valfull83_ep5.json" ] && break
  sleep 30
done
[ -s "training/runs/$RID/metric_valfull83_ep5.json" ] \
  && say "epoch-5 eval landed" || say "no ep5 metric json -- continuing to the exit check anyway"

say "waiting for the epoch-5 trainer to exit"
for _ in $(seq 1 120); do
  A=0; for p in $RANKS; do kill -0 $p 2>/dev/null && A=$((A+1)); done
  [ $A -eq 0 ] && break
  sleep 30
done
A=0; for p in $RANKS; do kill -0 $p 2>/dev/null && A=$((A+1)); done
[ $A -ne 0 ] && { say "REFUSING: $A epoch-5 process(es) still alive after 60 min. Not launching."; exit 3; }
say "epoch-5 run has exited; letting the cards settle for 60s"
sleep 60

# ---- gates, retried for 10 minutes ----
OK=0
for i in $(seq 1 40); do
  G=1
  B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py 2>/dev/null) || B=fail
  [ "$B" = "0" ] || G=0
  if [ $G -eq 1 ]; then
    for c in 4 5 7; do
      F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $c 2>/dev/null)
      case "$F" in ''|*[!0-9]*) G=0;; *) [ "$F" -lt 80000 ] && G=0;; esac
    done
  fi
  [ $G -eq 1 ] && { OK=1; break; }
  [ $(( i % 8 )) -eq 0 ] && say "  gates not clear yet (cards_busy='$B'), attempt $i/40"
  sleep 15
done
[ $OK -eq 1 ] || { say "REFUSING: cards did not come free within 10 min (cards_busy='$B')"; exit 3; }
say "gates clear"

RES="$(pwd)/training/runs/$RID/resume_ep5.pt"
[ -s "$RES" ] || { say "REFUSING: no resume bundle at $RES"; exit 3; }
[ -f "$SD/ep20_fired" ] && { say "REFUSING: this launcher already fired once."; exit 3; }
date > "$SD/ep20_fired"

say "LAUNCH: 1e-3 continuation, epochs 6-20, resume $RES"
SKIP="9mx1,4zto,9q5m,9dzf,9dzg" EPOCHS=20 NOEVAL=0 DRY=0 CARDS=4,5,7 \
  INBLOCK=1 LEAN_SINGLE=1 MEMLOG=1 TRUNK_CACHE_GB=10 TRUNK_MEMMAP=1 \
  EVAL_POOL=training/valfull83/pool.csv LR=1e-3 \
  EXTRA="--skip-dead-ops --lr-steps 20:1e-3,40:1e-4 --eval-pool-every 5 --prefix-cache-gb 10 --resume $RES" \
  ./shell_scripts/launch_default_arm.sh > $SD/arm_ep20.log 2>&1
say "launcher exit $?"
NEW=$(grep -oE "^run [a-z0-9]{6}" $SD/arm_ep20.log | head -1 | awk '{print $2}')
[ -z "$NEW" ] && { say "DID NOT START:"; grep -E "REFUS|STOP|Traceback" $SD/arm_ep20.log | tail -5; exit 4; }
echo "$NEW" > $SD/ep20_rid
say "continuation = run $NEW"
