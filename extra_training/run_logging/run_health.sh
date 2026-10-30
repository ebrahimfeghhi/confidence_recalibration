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
HOLDERS=" $( [ -r training/runs/gpu_holder_pids ] && tr '\n' ' ' < training/runs/gpu_holder_pids || echo '306042 306045 306050' ) "
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:?usage: run_health.sh <rid>}"
RUNDIR="$(pwd)/training/runs/$RID"
echo "=== $RID at $(TZ=America/New_York date '+%-I:%M %p %Z') ==="

# PROCESSES, by the --out contract, never by pattern.
N=0
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u); do
  case "$P" in ""|*[!0-9]*) continue ;; esac
  case "$HOLDERS" in *" $P "*) continue ;; esac
  [ -r "/proc/$P/cmdline" ] || continue
  case "$(tr '\0' ' ' < "/proc/$P/cmdline")" in *"--out $RUNDIR "*|*"--out $RUNDIR/"*) N=$((N+1)) ;; esac
done
echo "  ranks alive: $N/3"
H=0; for p in $HOLDERS; do [ -d /proc/$p ] && H=$((H+1)); done
echo "  GPU holders: $H/3$([ "$H" -lt 3 ] && echo '   *** FLAG: a holder is gone ***')"

# PROGRESS AND PACE, from the trajectory the run writes.
# A QUOTED HEREDOC, so the shell expands NOTHING inside it and the run id arrives as argv.
# The previous form was $PY -I -c "...$RID..." in DOUBLE quotes, which left every backslash in a
# regex at the mercy of shell quoting -- the progress line simply vanished, silently, because the
# whole block was wrapped in 2>/dev/null.
$PY -I - "$RID" <<'PYEOF' 2>&1
import collections, csv, os, re, sys, time
rid = sys.argv[1]
tp, lg = f"training/runs/{rid}/trajectory.csv", f"training/runs/{rid}/run.log"
if not os.path.exists(tp):
    print("  progress: no trajectory.csv yet (startup)"); raise SystemExit
rows = list(csv.DictReader(open(tp)))
c = collections.Counter()
for r in rows:
    e = r.get("epoch")
    if e not in (None, ""):
        c[int(float(e))] += 1
# COMPLETE MEANS THE RUN SAID SO. A pose-count threshold (>=5500) was inherited from an arm with a
# different epoch size and reported epoch 1 complete at 5600 of 6100 poses. The trainer prints
# "[rank 0] epoch N rank-loss" once an epoch has actually closed; the pose count stays, as
# progress WITHIN the epoch in flight.
txt = open(lg, errors="replace").read() if os.path.exists(lg) else ""
eps = [int(m) for m in re.findall(r"\[rank 0\] epoch (\d+) rank-loss", txt)]
done = max(eps) if eps else 0
cur = max(c) if c else 0
print(f"  progress: {done} epoch(s) complete, epoch {cur} in flight at {c.get(cur, 0)}/6100 poses")
if len(rows) > 1:
    t = [float(r["wall_s"]) for r in rows]
    n = min(200, len(t) - 1)
    print(f"  pace: {(t[-1]-t[-1-n])/n:.2f} s/pose over the last {n}"
          f"  ->  {6100*(t[-1]-t[-1-n])/n/60:.0f} min/epoch")
age = (time.time() - os.path.getmtime(tp)) / 60
print(f"  last trajectory write: {age:.1f} min ago" + ("   *** FLAG: stalled? ***" if age > 15 else ""))
PYEOF

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
