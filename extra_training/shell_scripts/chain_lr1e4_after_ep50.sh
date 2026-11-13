#!/bin/bash
# WAIT FOR EPOCH 47, STOP THE RUN, PATCH THE BUNDLE, RESUME WITH ORIENTATION.
#
# THIS ONE DOES STOP A RUN, unlike every other chain here, because the user asked for exactly that:
# pause after 47, add orientation, continue. So it signals -- but only the pids named below, read
# from /proc, never derived by pattern. pgrep/pkill match the shell running them and self-killed
# this session twice on 2026-10-18.
#
# IT STOPS NOTHING UNTIL EPOCH 47'S BUNDLE EXISTS ON DISK. The whole point is to continue from
# epoch 50, so a stop before that bundle is written would lose the epoch and leave the run at 46.
#
# THE PATCH IS WHY THIS IS NOT A ONE-LINE FLAG CHANGE. phase_k_train's resume refuses a bundle
# missing any trainable tensor, and --frame-direction creates four, so the bundle gets them added
# as zeros first -- correct, not a fudge, because they are zero-init and install() verifies the
# output is exactly zero.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RANKS="3770725 3770726 3770727"
SRC=training/runs/j8axco/resume_ep50.pt
echo $$ > training/runs/lr1e4_chain_pid
echo "$(TZ=America/New_York date '+%F %-I:%M %p %Z') armed; waiting for $SRC"

for i in $(seq 1 240); do          # 2h ceiling at 30s
  [ -f "$SRC" ] && break
  alive=0; for P in $RANKS; do [ -d "/proc/$P" ] && alive=1; done
  if [ "$alive" = "0" ]; then
    echo "STOP: the run exited before epoch 50's bundle appeared. Nothing stopped, nothing"
    echo "      launched -- decide by hand."
    exit 1
  fi
  sleep 30
done
[ -f "$SRC" ] || { echo "STOP: timed out waiting for $SRC"; exit 1; }
echo "$(TZ=America/New_York date '+%-I:%M %p') epoch 50 bundle present"

# NO PATCH NEEDED. Only the lr changes, so the bundle already carries every
# trainable tensor and the resume accepts it as written. add_orientation_to_bundle.py
# would exit nonzero here (nothing to add) and abort the chain, so it is not called.
echo "$(TZ=America/New_York date '+%-I:%M %p') stopping j8axco (ranks $RANKS)"
PARENT=$(awk '/^PPid:/{print $2}' /proc/3770725/status 2>/dev/null || true)
for P in $PARENT $RANKS; do [ -n "$P" ] && [ -d "/proc/$P" ] && kill -TERM "$P" 2>/dev/null \
  && echo "  TERM -> $P"; done
for i in $(seq 1 40); do
  [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
  sleep 15
done
[ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] || {
  echo "STOP: cards still busy after 10 min; not launching on top of something."; exit 1; }
echo "$(TZ=America/New_York date '+%-I:%M %p') cards free; launching with orientation"
exec ./shell_scripts/resume_ep51_lr1e4.sh
