#!/bin/bash
# ONE SCREEN OF HEALTH for a live arm. Everything that has gone wrong in this project before.
#
# WHY A SCRIPT: a status check assembled by hand each time omits whatever was not on the author's
# mind that minute. These are the specific failures this project has actually hit -- a GPU holder
# dying and handing the card to another user, memory creeping toward an OOM, MemAvailable falling
# below what other users of the box need, /home filling, and a run that is alive but not stepping.
#
# Usage: run_health.sh <rid>
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: run_health.sh <rid>}"
RUNDIR="$(pwd)/training/runs/$RID"
echo "=== $RID at $(TZ=America/New_York date '+%-I:%M %p %Z') ==="

# PROCESSES, by the --out contract, never by pattern.
N=0
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u); do
  case "$P" in ""|*[!0-9]*) continue ;; esac
  case " 306042 306045 306050 " in *" $P "*) continue ;; esac
  [ -r "/proc/$P/cmdline" ] || continue
  case "$(tr '\0' ' ' < "/proc/$P/cmdline")" in *"--out $RUNDIR "*|*"--out $RUNDIR/"*) N=$((N+1)) ;; esac
done
echo "  ranks alive: $N/3"
H=0; for p in 306042 306045 306050; do [ -d /proc/$p ] && H=$((H+1)); done
echo "  GPU holders: $H/3$([ "$H" -lt 3 ] && echo '   *** FLAG: a holder is gone ***')"

# PROGRESS AND PACE, from the trajectory the run writes.
$PY -I -c "
import csv, collections, os, time
p = 'training/runs/$RID/trajectory.csv'
if not os.path.exists(p):
    print('  progress: no trajectory.csv yet (startup)'); raise SystemExit
rows = list(csv.DictReader(open(p)))
c = collections.Counter()
for r in rows:
    e = r.get('epoch')
    if e not in (None, ''): c[int(float(e))] += 1
done = max([k for k, v in c.items() if v >= 5500] or [0])
cur = max(c) if c else 0
age = (time.time() - os.path.getmtime(p)) / 60
print(f'  progress: {done} epoch(s) complete, epoch {cur} in flight at {c.get(cur,0)}/6100 poses')
if len(rows) > 1:
    t = [float(r['wall_s']) for r in rows]
    n = min(200, len(t) - 1)
    print(f'  pace: {(t[-1]-t[-1-n])/n:.2f} s/pose over the last {n}  ->  {122*50*(t[-1]-t[-1-n])/n/60:.0f} min/epoch')
print(f'  last trajectory write: {age:.1f} min ago' + ('   *** FLAG: stalled? ***' if age > 15 else ''))
" 2>/dev/null

# MEMORY, the live device view on this project's three cards.
echo "  GPU free (MiB):$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==4||$1==5||$1==7{printf "  card%s %s",$1,$2}')"
awk '/MemAvailable/{printf "  host MemAvailable: %.0f GB%s\n",$2/1048576,($2/1048576<250?"   *** FLAG: under 250 GB ***":"")}' /proc/meminfo
df -h /home | awk 'NR==2{printf "  /home: %s used of %s (%s)%s\n",$3,$2,$5,(0+substr($5,1,length($5)-1)>90?"   *** FLAG: over 90% ***":"")}'

# FAILURES, if any have been written.
B=$(grep -cE "OutOfMemory|CUDA out of memory" "$RUNDIR/run.log" 2>/dev/null); B=${B:-0}
S=$(grep -cE "^STOP:|Traceback" "$RUNDIR/run.log" 2>/dev/null); S=${S:-0}
FLAG=""; { [ "$B" -gt 0 ] || [ "$S" -gt 0 ]; } && FLAG="   *** FLAG ***"
echo "  run.log: $B OOM line(s), $S stop/traceback line(s)$FLAG"
[ -f "training/runs/${RID}_watchdog_state" ] && echo "  watchdog: $(cat "training/runs/${RID}_watchdog_state")"
