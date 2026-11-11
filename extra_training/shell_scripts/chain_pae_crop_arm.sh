#!/bin/bash
# QUEUE THE PAE-ONLY CROPPED ARM BEHIND WHATEVER IS ON THE CARDS NOW.
#
# At the user's direction: "get this run setup, queue it behind the current run".
#
# READINESS IS cards_busy == 0, NOT A LOG OR AN EPOCH COUNT. Three earlier attempts at this
# pattern keyed off run.log's mtime and fired early: the held-out eval runs as a CAPTURED
# subprocess, so for its ~13 minutes the parent prints nothing and the log looks idle while all
# three ranks are alive. cards_busy counts live compute processes by pid and is true exactly when
# launching is safe. NEVER pgrep -f / pkill -f: they match the shell running them and have
# self-killed this session twice.
#
# IT FIRES WHEN THE CARDS FREE, FOR WHATEVER REASON -- epoch 40 reached, or the run stopped
# early. That is the requested semantics ("queue it behind"), and it means a crash of the
# current arm does not leave three cards idle. It does NOT stop anything itself.
#
# ONE INSTANCE ONLY, via flock.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
LOGDIR=training/runs/chain_pae_crop
mkdir -p "$LOGDIR" || exit 1
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

exec 9>"$LOGDIR/.lock"
flock -n 9 || { say "another chain_pae_crop_arm already holds the lock; exiting"; exit 200; }

CEIL="${CEIL:-2880}"          # 48 h of 60 s polls
say "waiting for cards 4,5,7 to free (ceiling $((CEIL/60)) h)"
FREE=0
for _ in $(seq 1 "$CEIL"); do
  B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  if [ "$B" = "0" ]; then FREE=1; say "cards free"; break; fi
  sleep 60
done
[ "$FREE" = "1" ] || { say "REFUSING: cards still busy after the ceiling. Nothing launched."; exit 5; }

say "letting the cards settle for 60s"
sleep 60

# WHAT `latest` POINTED AT BEFORE WE LAUNCHED. Comparing against a hardcoded run id broke once
# when an intervening run had moved the symlink; compare against what was actually there.
PREV=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
say "latest currently points at '${PREV:-none}'"

say "LAUNCH: PAE-only cropped arm, 50 epochs"
DRY=0 CROP_THREADS=3 ./shell_scripts/launch_pae_crop_arm.sh > "$LOGDIR/launch.log" 2>&1

NEW=""
for _ in $(seq 1 60); do
  C=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
  if [ -n "$C" ] && [ "$C" != "${PREV:-}" ]; then NEW="$C"; break; fi
  sleep 10
done
if [ -z "$NEW" ]; then
  say "DID NOT START: training/runs/latest never moved off '${PREV:-none}'."
  grep -aE "REFUS|STOP|Traceback" "$LOGDIR/launch.log" | tail -10
  exit 4
fi
say "new run $NEW"
echo "$NEW" > "$LOGDIR/rid"

# SUCCESS IS ASSERTED, NOT ASSUMED. The launcher ends in a `say`, so its exit status is 0 even
# when launch.py dies, and launch.py prints its run id BEFORE the preflight gates.
for _ in $(seq 1 90); do
  grep -qaE "steps/epoch|STOP:|Traceback" "training/runs/$NEW/run.log" 2>/dev/null && break
  sleep 20
done
if grep -qaE "steps/epoch" "training/runs/$NEW/run.log" 2>/dev/null; then
  grep -aE "CROP POOL|steps/epoch|accum:|PAE term MASKED" "training/runs/$NEW/run.log" | head -5
  say "chain done -- $NEW is training"
  exit 0
fi
say "LAUNCHED BUT DID NOT REACH steps/epoch:"
grep -aE "STOP:|Traceback|REFUS" "training/runs/$NEW/run.log" 2>/dev/null | tail -10
exit 6
