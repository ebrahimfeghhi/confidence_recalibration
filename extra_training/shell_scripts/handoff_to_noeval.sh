#!/bin/bash
# WAIT FOR EPOCH 1 AND ITS EVAL, PLOT, THEN HAND THE ARM OVER TO AN EVAL-FREE RESUME.
#
# THE OWNER'S PLAN B, 2026-11-06: skip evaluation from epoch 2 on, but pay epoch 1's eval rather
# than discard the 2.3 h already trained. Eval flags cannot change in a live process, so the only
# way to honour it is to stop at a clean boundary and resume without them.
#
# WHY THIS IS A SCRIPT. It stops a live 3-card run and starts another, in a fixed order, with the
# pids derived at the moment of the kill rather than hardcoded. A remembered command line would get
# that right once.
#
# IT NEVER KILLS BY PATTERN. pgrep/pkill match the shell running them and have self-killed this
# session twice. The GPU holders come from run_logging/cards_busy.py, the torchrun parent from
# their ppid, and the launcher from ITS ppid -- three explicit pids, SIGTERM, then verified gone.
#
# IT STOPS ONLY AFTER THE BOUNDARY IS SAFE: the epoch-1 resume bundle exists (the trainer writes it
# BEFORE the evaluation), the epoch-1 objective accounting is in the log, and the eval shards pass
# the completeness gate. Epoch 2 will have started by then and loses a few steps, which is the
# price of the clean boundary.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${RID:?set RID}"
EPOCHS="${EPOCHS:-10}"
R="training/runs/$RID"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
say "handoff armed for $RID: wait for epoch 1 + eval, plot, stop, resume to $EPOCHS with no eval"

while [ ! -s "$R/resume_ep1.pt" ]; do
  if grep -qE 'STOP:|Traceback' "$R/run.log" 2>/dev/null; then
    say "ABORTING: $R/run.log holds a STOP or traceback"; exit 1; fi
  sleep 120
done
say "epoch-1 bundle written"
while ! grep -qE '\[rank 0\] epoch 1 rank-loss' "$R/run.log" 2>/dev/null; do sleep 60; done
say "epoch 1 objective accounting present"

say "waiting for the epoch-1 eval to pass the shard-coverage gate"
for i in $(seq 1 120); do
  OK=$(RID=$RID $PY -I - <<'EOF'
import glob, importlib.util as u, os
sp = u.spec_from_file_location("m", "run_logging/plot_uncropped_progress.py")
m = u.module_from_spec(sp); sp.loader.exec_module(m)
d = glob.glob(f"training/runs/{os.environ['RID']}/valfull_eval_shards_ep1")
print("1" if d and m.shards_complete(d[0])[0] else "0")
EOF
)
  [ "$OK" = "1" ] && break
  sleep 60
done
say "eval complete=$OK (0 means it never passed the gate; the plot will say so)"
$PY -I run_logging/plot_uncropped_progress.py --run "$RID" || true

# ---- stop, by pids derived now ----
say "deriving the process tree"
H=$($PY -I run_logging/cards_busy.py 2>/dev/null | grep -oE 'pid [0-9]+' | awk '{print $2}' | sort -u)
[ -z "$H" ] && { say "no GPU holder found; is the run already gone? NOT resuming blind."; exit 1; }
TR=""
for p in $H; do PP=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$PP" ] && TR="$PP"; done
[ -z "$TR" ] && { say "could not derive the torchrun parent"; exit 1; }
LP=$(ps -o ppid= -p "$TR" 2>/dev/null | tr -d ' ')
say "holders [$H] -> torchrun $TR -> launcher ${LP:-none}"
kill $TR ${LP:-} 2>/dev/null
for i in $(seq 1 20); do
  B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
  [ "${B:-1}" = "0" ] && break
  sleep 3
done
B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
if [ "${B:-1}" != "0" ]; then say "REFUSING to resume: cards still report '$B' after SIGTERM."; exit 1; fi
say "stopped cleanly, cards free"

RID=$RID FROM_EP=1 EPOCHS=$EPOCHS DRY=0 ./shell_scripts/resume_noeval_to_10.sh
say "handoff exit $?"
