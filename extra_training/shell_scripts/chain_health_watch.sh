#!/bin/bash
# HEALTH WATCH for the seed chain: one status line every INTERVAL, and it EXITS when something
# needs a human.
#
# WHY IT EXITS RATHER THAN JUST LOGGING. The previous watcher wrote "CHAIN PROCESS GONE" to its log
# at 10:49 PM and nothing read it for 49 minutes, so the chain sat stopped while work that depended
# on it waited. A watcher that only writes is only as good as whoever reads it. This one terminates
# on any terminal condition, which is what gets the session's attention.
#
# WHAT COUNTS AS TERMINAL:
#   - the chain log reaches CHAIN COMPLETE / STOPPING / REFUSING   (it finished or gave up)
#   - the chain process is gone without one of those lines         (it died)
#   - workers are up but no worker log has been written in STALL_MIN minutes
#   - quota headroom falls under MIN_HEADROOM_GB                   (a seed costs ~106 GiB)
#   - MemAvailable falls under MIN_MEM_GB                          (the box is shared)
# A seed boundary is NOT terminal: between seeds there are legitimately no workers and no log
# writes for a couple of minutes, so the stall test requires workers to be UP, and the
# process-gone test is what covers a chain that dies between seeds.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
TG=training/fold
CHAIN_PID="${CHAIN_PID:?pass CHAIN_PID}"
CHAIN_LOG="${CHAIN_LOG:?pass CHAIN_LOG}"
INTERVAL="${INTERVAL:-600}"
STALL_MIN="${STALL_MIN:-25}"
MIN_HEADROOM_GB="${MIN_HEADROOM_GB:-150}"
MIN_MEM_GB="${MIN_MEM_GB:-250}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

headroom(){ $PY -I -c "
import subprocess
o = subprocess.run(['quota','-u','ebrahim'],capture_output=True,text=True).stdout.split()
for i,t in enumerate(o):
    if t.startswith('/dev/'):
        print(int((int(o[i+3])-int(o[i+1]))/2**20)); break
" 2>/dev/null; }

say "watching chain pid $CHAIN_PID every ${INTERVAL}s (stall ${STALL_MIN}m, floors ${MIN_HEADROOM_GB}G/${MIN_MEM_GB}G)"
while true; do
  TERM_LINE=$(grep -hoE 'CHAIN COMPLETE|STOPPING:|REFUSING:' "$CHAIN_LOG" 2>/dev/null | tail -1)
  ALIVE=$(ps -o pid= -p "$CHAIN_PID" 2>/dev/null | wc -l)
  UP=$(tmux ls 2>/dev/null | grep -c w1fold || true)
  H=$(headroom); M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  # PROGRESS PER SEED, from the canonical csv, so the line says what actually landed.
  PROG=$($PY -I - <<'PYEOF' 2>/dev/null
import csv, collections
want = {l.strip() for l in open("docs/extra_seed_structures.txt") if l.strip() and l[0] != "#"}
g = collections.defaultdict(set)
for r in csv.DictReader(open("training/fold/results.csv")):
    if r["seed"] >= "500011":
        g[r["seed"]].add(r["iface"].split("_")[0])
print(" ".join(f"{s}:{len(g[s] & want)}/{len(want)}" for s in sorted(g)) or "none yet")
PYEOF
)
  # NEWEST worker log write, in minutes. `find -newermt` silently matches nothing on a bad spec,
  # which nearly made an earlier check report a healthy fold as stalled, so the age is computed
  # from the mtime directly.
  NEW=$(ls -t $TG/w1fold_seed5000*_c*.log 2>/dev/null | head -1)
  if [ -n "$NEW" ]; then AGE=$(( ( $(date +%s) - $(stat -c %Y "$NEW") ) / 60 )); else AGE=-1; fi
  say "chain alive=$ALIVE workers=$UP quota_headroom=${H}G mem=${M}G last_write=${AGE}m  $PROG"

  if [ -n "$TERM_LINE" ]; then say "TERMINAL: chain log says '$TERM_LINE'"; exit 0; fi
  if [ "$ALIVE" -eq 0 ]; then
    say "TERMINAL: chain pid $CHAIN_PID is gone with no CHAIN COMPLETE/STOPPING/REFUSING line"
    tail -6 "$CHAIN_LOG" 2>/dev/null
    exit 1
  fi
  if [ "$UP" -gt 0 ] && [ "$AGE" -ge "$STALL_MIN" ]; then
    say "TERMINAL: $UP worker(s) up but no log write in ${AGE}m (>= ${STALL_MIN}m)"
    exit 1
  fi
  if [ -n "${H:-}" ] && [ "${H:-0}" -lt "$MIN_HEADROOM_GB" ]; then
    say "TERMINAL: quota headroom ${H}G under the ${MIN_HEADROOM_GB}G floor"; exit 1
  fi
  if [ "${M:-0}" -lt "$MIN_MEM_GB" ]; then
    say "TERMINAL: MemAvailable ${M}G under the ${MIN_MEM_GB}G floor"; exit 1
  fi
  sleep "$INTERVAL"
done
