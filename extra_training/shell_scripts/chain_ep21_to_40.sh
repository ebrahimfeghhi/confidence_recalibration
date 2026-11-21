#!/bin/bash
# WAIT FOR 87ja6a TO FINISH EPOCH 20 AND ITS HELD-OUT EVAL, THEN RESUME TO EPOCH 40.
#
# At the user's direction: "let's extend this run to 40 epochs", then "let's set up the resume at
# epoch 20".
#
# WHY A RESUME AND NOT A LIVE CHANGE. --epochs is read once at launch and drives both the epoch
# loop and the lr horizon, so a running trainer cannot be told to go further. The arm has to reach
# 20, exit, and come back from resume_ep20.pt.
#
# THE SCHEDULE DOES NOT CHANGE, which is the thing that had to be checked before agreeing to this.
# lr_decay_start_epoch and lr_decay_end_epoch are both 0.0, and the trainer reads 0/0 as a
# WHOLE-RUN cosine -- so stretching the horizon from 20 to 40 could have rewritten the lr at every
# remaining step and spliced two schedules into one curve. It does not, because lr_final is 0.0,
# which means no decay at all. Evaluated directly through phases/lr_schedule.lr_at_step:
#
#   end of epoch 15:  epochs=20 -> 1.000e-03   epochs=40 -> 1.000e-03
#   end of epoch 20:  epochs=20 -> 1.000e-03   epochs=40 -> 1.000e-03
#
# Flat 1e-3 after 20 warmup steps either way, so epochs 21-40 continue the identical schedule and
# the curve is one experiment rather than two.
#
# NO DRIFT OVERRIDE IS PASSED. `epochs` is not among the fields the resume guard compares, so
# --epochs 40 is accepted on its own. If the guard refuses anyway, something genuinely changed and
# that is worth stopping for rather than overriding.
#
# WHAT IT COSTS: 20 epochs at ~95 min on cards 4,5,7, about 32 hours. Evals at 25, 30, 35, 40.
#
# ---------------------------------------------------------------------------------------------
# FIVE BUGS AN ADVERSARIAL REVIEW FOUND IN THE FIRST VERSION, all in detection rather than in the
# launch itself. Recorded because each one would have failed silently.
#
# 1. READINESS WAS READ FROM run.log's mtime. The epoch-20 eval runs as a CAPTURED subprocess, so
#    for its ~13 minutes the parent prints nothing and run.log looks idle while all three ranks
#    are alive. The "trainer has exited" test would have passed on its first iteration, the
#    launcher's cards_busy gate would then have refused, and the chain would have exited for good
#    with the cards left idle all night. Readiness is now cards_busy == 0, polled, which counts
#    live ranks by pid and cannot be fooled by a quiet log.
# 2. THE NEW RUN ID WAS READ FROM THE WRONG FILE AND THE WRONG PATTERN, and launch.py's stdout is
#    block-buffered into a log nothing flushes until it exits ~32 hours later. A perfectly good
#    launch would have reported "DID NOT START". The id now comes from training/runs/latest,
#    which launch.py writes before torchrun starts.
# 3. $SD IS A /tmp SCRATCHPAD WITH NO mkdir. The launch IS the redirect, so a missing directory
#    meant nothing ran and the script blamed the launcher. Created explicitly, and the log lives
#    beside the run instead.
# 4. THE WRAPPERS ALWAYS EXIT 0 -- both end in a `say` -- so "launcher exit $?" is not a signal.
#    Success is now asserted from the new run's own log reaching steps/epoch.
# 5. NO LOCK. Two instances would both pass the gates and put 6 ranks on 3 cards sharing one
#    MASTER_PORT. flock makes a second instance exit immediately.
# ---------------------------------------------------------------------------------------------
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
RID=87ja6a
PY=/home/ebrahim/Protenix/.venv/bin/python
LOGDIR=training/runs/chain_ep21_to_40
mkdir -p "$LOGDIR" || exit 1
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

# ONE INSTANCE ONLY. 200 is "another copy already holds it", which is a clean no-op, not an error.
exec 9>"$LOGDIR/.lock"
flock -n 9 || { say "another chain_ep21_to_40 already holds the lock; exiting"; exit 200; }

RESUME="training/runs/$RID/resume_ep20.pt"

say "waiting for $RID epoch 20 and its held-out eval (ceiling 10h)"
for _ in $(seq 1 600); do
  [ -s "training/runs/$RID/metric_valfull83_ep20.json" ] && break
  sleep 60
done
[ -s "training/runs/$RID/metric_valfull83_ep20.json" ] \
  || { say "REFUSING: epoch 20 eval json never appeared. Nothing launched."; exit 2; }
say "epoch 20 and its eval are done"

# THE BUNDLE MUST EXIST BEFORE THE CARDS ARE CLAIMED. resume_ep<N>.pt is written at the epoch
# boundary, BEFORE the eval runs, so by here it cannot legitimately be missing.
[ -s "$RESUME" ] || { say "REFUSING: $RESUME is missing or empty. Nothing launched."; exit 3; }
say "resume bundle $RESUME ($(stat -c %s "$RESUME") bytes)"

# WAIT FOR THE RANKS TO ACTUALLY RELEASE THE CARDS. cards_busy counts compute processes by pid
# and excludes our three 778 MB holders, so 0 means the epoch-20 ranks are gone -- the one signal
# that is true exactly when launching is safe. NEVER pgrep -f / pkill -f: they match the shell
# running them and have self-killed this session twice.
say "waiting for the epoch-20 ranks to release cards 4,5,7"
FREE=0
for _ in $(seq 1 80); do
  B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
  if [ "$B" = "0" ]; then FREE=1; say "cards free"; break; fi
  sleep 30
done
[ "$FREE" = "1" ] || {
  say "REFUSING: cards still busy ('${B:-?}') after 40 min. Nothing launched."; exit 5; }
say "letting the cards settle for 60s"
sleep 60

# WHAT `latest` POINTED AT BEFORE WE LAUNCHED, captured here rather than assumed to be $RID.
# The original compared against $RID, which was only true when 87ja6a was the last run to start.
# A run in between -- the cropped arm 3pdv7i -- left latest pointing at IT, so the loop below
# would have matched on its first poll and adopted a dead run's id as the new one, then asserted
# success by grepping that dead run's log for steps/epoch. Compare against what was actually
# there a moment ago and the test means what it says.
PREV=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
say "latest currently points at '${PREV:-none}'; waiting for it to move off that"

say "LAUNCH: $RID epochs 21-40, resume $RESUME"
EPOCHS=40 EXTRA_APPEND="--resume $RESUME" DRY=0 \
  ./shell_scripts/launch_fullhead_5site.sh > "$LOGDIR/launch.log" 2>&1

# THE RUN ID COMES FROM training/runs/latest, not from the launcher's stdout: launch.py writes the
# symlink before torchrun starts, while its own "run <rid>" line sits in a block-buffered pipe for
# the whole 32 hours. Resolve it only AFTER it changes, so a stale symlink is never mistaken for
# the new run.
NEW=""
for _ in $(seq 1 60); do
  C=$(readlink training/runs/latest 2>/dev/null | xargs -r basename)
  if [ -n "$C" ] && [ "$C" != "${PREV:-$RID}" ]; then NEW="$C"; break; fi
  sleep 10
done
if [ -z "$NEW" ]; then
  say "DID NOT START: training/runs/latest never moved off $RID."
  grep -aE "REFUS|STOP|Traceback" "$LOGDIR/launch.log" | tail -8
  exit 4
fi
say "new run $NEW"
echo "$NEW" > "$LOGDIR/rid"

# SUCCESS IS ASSERTED, NOT ASSUMED. Both wrapper scripts end in a `say`, so their exit status is 0
# even when launch.py dies; and launch.py prints its run id BEFORE the preflight gates, so a run
# directory is not evidence either. steps/epoch is printed by the trainer itself, after the model
# is built and the epoch plan exists.
for _ in $(seq 1 90); do
  grep -qaE "steps/epoch|STOP:|Traceback" "training/runs/$NEW/run.log" 2>/dev/null && break
  sleep 20
done
if grep -qaE "steps/epoch" "training/runs/$NEW/run.log" 2>/dev/null; then
  grep -aE "RESUMED from|continuing at epoch|steps/epoch" "training/runs/$NEW/run.log" | head -3
  say "chain done -- $NEW is training epochs 21-40"
  exit 0
fi
say "LAUNCHED BUT DID NOT REACH steps/epoch:"
grep -aE "STOP:|Traceback|REFUS" "training/runs/$NEW/run.log" 2>/dev/null | tail -8
exit 6
