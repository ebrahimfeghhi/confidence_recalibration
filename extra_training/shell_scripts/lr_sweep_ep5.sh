#!/bin/bash
# LR sweep to epoch 5, SEQUENTIALLY on cards 4,5,7.
#
# Sequential because each arm needs all three cards: world=3 deals a cross-seed step's units
# across ranks, so an arm on fewer cards has a different step composition and is not comparable
# to qip9ta. Cards 0,1,2,6 belong to other users.
#
# EPOCHS=5 rather than killing at 5: the run ends cleanly and fires its own held-out eval, and
# the LR at every step is identical to EPOCHS=60 because the staircase uses
# spe = horizon/epochs = steps_per_epoch, independent of the total.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
R=$(pwd)
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
TORCHRUN=2956691; LAUNCHER=2956511; RANKS="2956770 2956771 2956772"

# ---- 1. let qip9ta finish epoch 5 AND its eval ----
say "waiting for qip9ta epoch 5 + its held-out eval"
for _ in $(seq 1 240); do
  [ -s training/runs/qip9ta/metric_valfull83_ep5.json ] && break
  grep -qE "^STOP:|Traceback" training/runs/qip9ta/run.log 2>/dev/null && { say "qip9ta STOPPED:"; grep -E "^STOP:" training/runs/qip9ta/run.log | tail -2; break; }
  kill -0 $TORCHRUN 2>/dev/null || { say "qip9ta exited"; break; }
  sleep 30
done
if [ -s training/runs/qip9ta/metric_valfull83_ep5.json ]; then
  say "qip9ta epoch-5 eval landed"
  /home/ebrahim/Protenix/.venv/bin/python -I -c "
import json; d=json.load(open('training/runs/qip9ta/metric_valfull83_ep5.json'))
print('  qip9ta 3e-3  mean %.4f  max %.4f  frac %.4f  orc %.4f' % (d['mean_dockq'],
  d.get('mean_dockq_select_max',float('nan')), d['frac_acc'], d['oracle_mean_dockq']))"
else
  say "no metric json; continuing anyway so the sweep is not blocked"
fi

# ---- 2. stop qip9ta, by verified pid ----
for p in $TORCHRUN $LAUNCHER; do
  if kill -0 $p 2>/dev/null; then
    C=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$C" in */bin/torchrun*--nproc_per_node*|*run_logging/launch.py*--cards*)
        say "SIGTERM $p"; kill -TERM $p 2>/dev/null;;
      *) say "REFUSING to signal $p [${C:0:60}]"; exit 3;; esac
  fi
done
for i in $(seq 1 60); do A=0; for p in $TORCHRUN $RANKS; do kill -0 $p 2>/dev/null && A=$((A+1)); done
  [ $A -eq 0 ] && break; sleep 5; done
A=0; for p in $TORCHRUN $RANKS; do kill -0 $p 2>/dev/null && A=$((A+1)); done
[ $A -ne 0 ] && { say "$A alive; SIGKILL those exact pids"
  for p in $TORCHRUN $RANKS; do kill -0 $p 2>/dev/null && kill -KILL $p 2>/dev/null; done; sleep 20; }
say "qip9ta stopped at epoch 5"

# ---- 3. the two new arms, in turn ----
run_arm () {
  local LRV="$1" TAG="$2"
  for i in $(seq 1 36); do
    OK=1
    for c in 4 5 7; do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $c 2>/dev/null)
      case "$F" in ''|*[!0-9]*) OK=0;; *) [ "$F" -lt 80000 ] && OK=0;; esac; done
    [ $OK -eq 1 ] && break; sleep 10
  done
  B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py) || B=fail
  [ "$B" = "0" ] || { say "REFUSING $TAG: cards_busy '$B'"; return 3; }
  say "LAUNCH $TAG (lr $LRV), 5 epochs, eval at 5"
  SKIP="9mx1,4zto,9q5m,9dzf,9dzg" EPOCHS=5 NOEVAL=0 DRY=0 CARDS=4,5,7 \
    INBLOCK=1 LEAN_SINGLE=1 MEMLOG=1 TRUNK_CACHE_GB=10 TRUNK_MEMMAP=1 \
    EVAL_POOL=training/valfull83/pool.csv LR="$LRV" \
    EXTRA="--skip-dead-ops --lr-steps 20:1e-3,40:1e-4 --eval-pool-every 5 --prefix-cache-gb 10" \
    ./shell_scripts/launch_default_arm.sh > $SD/arm_${TAG}.log 2>&1
  say "$TAG launcher exit $?"
  NEW=$(grep -oE "^run [a-z0-9]{6}" $SD/arm_${TAG}.log | head -1 | awk '{print $2}')
  [ -z "$NEW" ] && { say "$TAG did not start:"; grep -E "REFUS|STOP" $SD/arm_${TAG}.log | tail -3; return 4; }
  echo "$NEW" > $SD/${TAG}_rid
  say "$TAG = run $NEW; waiting for its 5 epochs + eval"
  for _ in $(seq 1 900); do
    [ -s "training/runs/$NEW/metric_valfull83_ep5.json" ] && break
    pgrep -u "$(id -u)" -f "runs/$NEW" >/dev/null 2>&1 || { sleep 30
      [ -s "training/runs/$NEW/metric_valfull83_ep5.json" ] || { say "$TAG ended without a metric"; break; }; }
    sleep 30
  done
  if [ -s "training/runs/$NEW/metric_valfull83_ep5.json" ]; then
    /home/ebrahim/Protenix/.venv/bin/python -I -c "
import json; d=json.load(open('training/runs/$NEW/metric_valfull83_ep5.json'))
print('  $TAG $LRV  mean %.4f  max %.4f  frac %.4f  orc %.4f' % (d['mean_dockq'],
  d.get('mean_dockq_select_max',float('nan')), d['frac_acc'], d['oracle_mean_dockq']))"
  fi
  say "$TAG done"
}

run_arm 1e-3 lr1e3
run_arm 8e-4 lr8e4
say "SWEEP COMPLETE -- 3e-3 (qip9ta), 1e-3, 8e-4 all at epoch 5"
