#!/bin/bash
# WAIT FOR ip8ga4 TO FINISH ITS 20 EPOCHS, THEN LAUNCH THE copy x PAE ARM.
#
# Usage:  nohup ./shell_scripts/chain_copy_after_ip8ga4.sh > training/runs/copy_chain.log 2>&1 &
#
# NO KILL ANYWHERE IN HERE. ip8ga4 has --epochs 20 and exits on its own at the end of epoch 20,
# so the chain waits for its ranks to be gone and the cards to be free rather than taking it
# upon itself to stop a run. A chain that can kill is a chain that can kill the wrong thing at
# 3am; this one can only ever fail to launch.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
RANKS="930311 930312 930313"
echo "$(date '+%F %T %Z') chain armed; waiting for ip8ga4 (ranks $RANKS)"
for i in $(seq 1 6000); do          # 50h ceiling at 30s
  alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && break
  sleep 30
done
alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
[ "$alive" = "1" ] && { echo "$(date '+%F %T') TIMEOUT: ip8ga4 still running, NOT launching"; exit 1; }
DONE=$(grep -cE "^  epoch [0-9]+ TRAIN:" training/runs/ip8ga4/run.log 2>/dev/null || echo 0)
echo "$(date '+%F %T') ip8ga4 gone after $DONE evaluated epoch(s)"
# A RUN THAT DIED EARLY IS NOT A RUN THAT FINISHED, and launching the next arm over a crash
# would bury the crash under five hours of new output. 11 = epochs 10..20.
if [ "$DONE" -lt 11 ]; then
  echo "$(date '+%F %T') REFUSING: expected 11 evaluated epochs (10..20), found $DONE. Look at ip8ga4 first."
  exit 1
fi
for i in $(seq 1 60); do
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  sleep 10
done
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "$(date '+%F %T') REFUSING: cards still occupied ($BUSY)"; exit 1; }
echo "$(date '+%F %T') launching the copy x PAE arm"
./shell_scripts/launch_pae_copy_full.sh
echo "$(date '+%F %T') copy arm launcher returned $?"
