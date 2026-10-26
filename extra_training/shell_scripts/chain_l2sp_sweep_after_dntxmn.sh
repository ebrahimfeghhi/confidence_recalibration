#!/bin/bash
# WAIT FOR dntxmn TO FINISH ITS 20 EPOCHS, THEN RUN THE TWO L2-SP LAMBDA SWEEP ARMS IN ORDER.
#
# Usage:  nohup ./shell_scripts/chain_l2sp_sweep_after_dntxmn.sh \
#              > training/runs/l2sp_sweep_chain.log 2>&1 &
#
# NO KILL ANYWHERE IN HERE, and that is the point. The user's first plan was to pause dntxmn
# after epoch 14 and run the sweeps in the gap; they then offered to let it finish instead, which
# costs the same total wall clock (dntxmn needs ~3.9h for epochs 15-20, each sweep arm ~2h) and
# removes the only step that required signalling a live training process. dntxmn has --epochs 20
# and exits on its own, so this chain can only ever fail to launch -- never kill the wrong thing.
#
# EACH LAUNCHER BLOCKS. launch.py runs the trainer under subprocess.run (launch.py:208), so a
# launcher does not return until its run is over. That is what makes a plain sequence correct
# here and why no pid bookkeeping is needed between arms.
#
# ORDER IS 1e-3 THEN 1e-4, as asked. Both are read on the MEASURED "l2-sp: ||W-W0||^2 = ...
# penalty = ..." line each epoch against the PAE loss -- not on a derived lambda. wpg274's 1e-5
# reached a penalty of 0.0380 against a loss near 2.0, under 2% of the objective, because its
# lambda had been sized for a drift 18x larger than the one that occurred.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
RANKS="1940634 1940635 1940636"

echo "$(date '+%F %T %Z') chain armed; waiting for dntxmn (ranks $RANKS) to reach epoch 20"
for i in $(seq 1 6000); do          # 50h ceiling at 30s
  alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && break
  sleep 30
done
alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
[ "$alive" = "1" ] && { echo "$(date '+%F %T') TIMEOUT: dntxmn still running, NOT launching"; exit 1; }

DONE=$(grep -cE "^  epoch [0-9]+ TRAIN:" training/runs/dntxmn/run.log 2>/dev/null)
DONE=${DONE:-0}
echo "$(date '+%F %T') dntxmn gone after $DONE evaluated epoch(s)"
# A RUN THAT DIED EARLY IS NOT A RUN THAT FINISHED. Launching over a crash would bury it under
# four hours of new output, and shell_scripts/resume_dntxmn_e20.sh exists for exactly that case.
if [ "$DONE" -lt 20 ]; then
  echo "$(date '+%F %T') REFUSING: expected 20 evaluated epochs, found $DONE."
  echo "  Look at dntxmn first; resume_dntxmn_e20.sh can finish it before the sweep runs."
  exit 1
fi

run_arm() {
  local script="$1" label="$2"
  for i in $(seq 1 60); do
    BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
    [ "$BUSY" = "0" ] && break
    sleep 10
  done
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] || { echo "$(date '+%F %T') REFUSING $label: cards occupied ($BUSY)"; return 1; }
  echo "$(date '+%F %T') launching $label"
  ./"$script"
  local rc=$?
  echo "$(date '+%F %T') $label launcher returned $rc"
  return $rc
}

run_arm shell_scripts/launch_l2sp_1e3_overfit30.sh "L2-SP lambda 1e-3 (full head, 15 ep, overfit30 -> val10)" \
  || { echo "$(date '+%F %T') STOPPING: the 1e-3 arm failed; not starting 1e-4."; exit 1; }
run_arm shell_scripts/launch_l2sp_1e4_overfit30.sh "L2-SP lambda 1e-4 (full head, 15 ep, overfit30 -> val10)"
echo "$(date '+%F %T') sweep chain done"
