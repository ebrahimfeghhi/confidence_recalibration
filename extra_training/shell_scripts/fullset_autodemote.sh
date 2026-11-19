#!/bin/bash
# WATCH THE CHAIN'S EPOCH-2 GATE, AND IF IT TRIPS, RESTART THE WHOLE ARM AT 5e-4.
#
# The gate stops the chain with exit 10 when the epoch-2 easy bin is more than EASY_DROP points
# under shipped -- the protocol's reading of "the learning rate is too high". The user's standing
# instruction is to halve it and let it run rather than stop and wait, so that is what this does,
# ONCE: a second trip is a different problem and is left for a human.
#
# IT STARTS A FRESH ARM, not a resume. The epoch-1 and epoch-2 weights were taken at 1e-3 and
# carrying them forward would make the run neither one rate nor the other.
#
# AND THE 5e-4 RUN HAS NO GATE. The epoch-2 check exists to catch a rate that is too high; having
# already acted on it once, stopping the lower-rate run on the same test would just halt the arm
# the check was meant to produce. It runs to 60 and the tables are read afterwards.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
CHAIN_PID=${CHAIN_PID:?set CHAIN_PID}

while [ -d "/proc/$CHAIN_PID" ]; do sleep 60; done
RC=$(grep -cE "STOPPING at the epoch-2 check" training/runs/fullset_chain.log 2>/dev/null || echo 0)
if [ "$RC" = "0" ]; then
    say "the chain ended without tripping the epoch-2 gate; nothing to do"
    tail -3 training/runs/fullset_chain.log
    exit 0
fi
say "the epoch-2 gate tripped at lr 1e-3; relaunching the whole arm at 5e-4"
mv training/runs/fullset_chain.log training/runs/fullset_chain_1e-3.log 2>/dev/null || true
LR_RATE=5e-4 nohup ./shell_scripts/fullset_chain_lowlr.sh > training/runs/fullset_chain.log 2>&1 &
say "relaunched as pid $!; its log is training/runs/fullset_chain.log"
