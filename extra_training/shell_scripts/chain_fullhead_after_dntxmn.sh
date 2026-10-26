#!/bin/bash
# WAIT FOR dntxmn TO FINISH ITS 20 EPOCHS, THEN LAUNCH THE FULL-HEAD + POSE-FREE-L2-SP ARM.
#
# Usage:  nohup ./shell_scripts/chain_fullhead_after_dntxmn.sh \
#              > training/runs/fullhead_chain.log 2>&1 &
#
# LAMBDA IS 1e-3: lowered from 1e-2 by the user before launch. See the launcher header.
#
# REPLACES the two overfit30 lambda sweep arms at the user's direction (2026-10-26): they were
# the cheap read on what lambda does to a full head, and the user chose to spend that time on
# the full-scale arm instead. launch_l2sp_1e3_overfit30.sh and launch_l2sp_1e4_overfit30.sh stay
# in the tree, unqueued, if the cheap read is ever wanted.
#
# NO KILL ANYWHERE IN HERE. dntxmn has --epochs 20 and exits on its own, so this chain waits for
# its ranks to be gone and the cards to be free rather than stopping a live run. It can only ever
# fail to launch.
#
# ORDER: wait for dntxmn -> ipTM join verification (informational, ~25-35 min) -> the arm. The
# arm does NOT get --val-loss-uncropped: that would make its val PAE a different number and lose
# the arm-to-arm val PAE comparison that is being read across arms 1, 2 and 3. The merge goes on
# at the start of the NEXT comparison, which is a clean boundary.
#
# EACH LAUNCHER BLOCKS -- launch.py runs the trainer under subprocess.run (launch.py:208) -- so a
# plain call is correct here and no pid bookkeeping is needed.
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
  echo "  Look at dntxmn first; resume_dntxmn_e20.sh can finish it before this arm runs."
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

# THE JOIN VERIFICATION GOES IN THE GAP, because this is the only moment the cards are free and
# the question blocks a ~3-4 min/epoch saving on every run after this one. IT IS INFORMATIONAL AND
# NEVER BLOCKS THE ARM: a failed join means the DockQ eval stays, not that the arm should wait. Its
# exit status is recorded and then ignored on purpose.
#
# IT RUNS AGAINST dntxmn's OWN EPOCH 16, whose eval shards already exist, so it compares an
# uncropped loss-only pass to the eval that this arm actually produced. It writes no checkpoint and
# does not touch dntxmn, which has already exited by this point.
#
# ~25-35 min: unsharded where the live pass is sharded over 3 cards (4 min), and uncropped where
# the live pass crops to ~31% of the pair matrix.
for i in $(seq 1 60); do
  BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
  [ "$BUSY" = "0" ] && break
  sleep 10
done
echo "$(date '+%F %T') running the ipTM join verification on dntxmn epoch 16 (informational)"
./shell_scripts/verify_iptm_join.sh dntxmn 16
JOINRC=$?
if [ "$JOINRC" = "0" ]; then
  echo "$(date '+%F %T') JOIN VERIFIED -- the eval merge can be wired for the arm after this one"
else
  echo "$(date '+%F %T') join NOT verified (exit $JOINRC) -- the separate DockQ eval stays"
fi

run_arm shell_scripts/launch_fullhead_l2sp_posefree.sh "FULL HEAD + pose-COPY, L2-SP 1e-3 on the 246 non-pose tensors (full_train -> valfull, 20 ep)"
echo "$(date '+%F %T') chain done"
