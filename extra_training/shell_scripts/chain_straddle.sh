#!/bin/bash
# probe -> derive lambda -> launch the straddle arm -> arm the per-epoch watcher. No idle cards.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=training/runs/chain_straddle.log
exec >> "$LOG" 2>&1
echo "=== straddle chain armed $(date '+%F %T %Z') ==="

before=$(ls training/runs | grep -E '^[a-z0-9]{6}$' | sort)
echo "$(date '+%T') probing |g_rank| for the straddle objective"
./shell_scripts/probe_straddle.sh >> training/runs/probe_straddle.log 2>&1
after=$(ls training/runs | grep -E '^[a-z0-9]{6}$' | sort)
PR=$(comm -13 <(echo "$before") <(echo "$after") | head -1)
[ -n "$PR" ] || { echo "STOP: probe produced no new run dir."; exit 1; }
G=$(grep -oE '\|g_rank\|[[:space:]]+[0-9.]+e?-?[0-9]*' "training/runs/$PR/run.log" 2>/dev/null \
    | tail -1 | awk '{print $NF}')
[ -n "$G" ] || { echo "STOP: no |g_rank| in probe $PR; not launching on a guessed lambda."; exit 1; }
echo "$(date '+%T') probe $PR gave |g_rank| = $G"

# zg5fnw: lambda 1e-05 at |g_rank| 2.4260e-03 over 40,960 anchored parameters on this same fold.
# n_new == n_old == 40960, so the size factor is exactly 1 and this is the gradient ratio alone.
LAM=$($PY -I run_logging/l2sp_lambda.py --old-lambda 1e-05 --old-grad 2.4260e-03 \
        --old-params 40960 --new-grad "$G" --new-params 40960 2>/dev/null \
      | grep -oE 'nearest value worth using: [0-9.e-]+' | awk '{print $NF}')
[ -n "$LAM" ] || { echo "STOP: l2sp_lambda.py produced no lambda."; exit 1; }
echo "$(date '+%T') lambda = $LAM"

before=$(ls training/runs | grep -E '^[a-z0-9]{6}$' | sort)
nohup ./shell_scripts/launch_straddle.sh "$LAM" >> training/runs/straddle_launch.log 2>&1 &
LP=$!
for i in $(seq 1 120); do
  after=$(ls training/runs | grep -E '^[a-z0-9]{6}$' | sort)
  RID=$(comm -13 <(echo "$before") <(echo "$after") | head -1)
  [ -n "$RID" ] && break
  sleep 5
done
[ -n "${RID:-}" ] || { echo "STOP: launcher started no run dir (pid $LP)."; exit 1; }
echo "$RID" > training/runs/straddle_rid
echo 0 > "training/runs/${RID}_last_plotted"
echo "$(date '+%T') straddle arm is $RID -- watcher arming"
nohup ./shell_scripts/watch_arm.sh "$RID" >> "training/runs/watch_$RID.log" 2>&1 &
echo "$(date '+%T') watcher pid $!"
