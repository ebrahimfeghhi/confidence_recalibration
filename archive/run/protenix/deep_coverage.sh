#!/usr/bin/env bash
# Deep Protenix sampling on COVERAGE FAILURES ONLY — the interfaces where more seeds is the only
# lever that can possibly work.
#
# WHY THIS REPLACES deep_traps.sh
#
# deep_traps.sh ran two phases: NO-POSE (coverage failures) and then TRAP (selection failures).
# Running the TRAP phase is a category error. A TRAP means an acceptable pose is ALREADY IN THE
# POOL and Protenix's confidence failed to pick it. Adding seeds cannot fix that -- it is a
# selection problem, and depth just enlarges the haystack. (ESMFold2 proved it: 51 of 55 traps
# survived all the way to 1,024 seeds.) So: coverage failures only.
#
# THE TARGET SET is computed from the OST ground truth, NOT hardcoded. deep_traps.sh's baked-in
# 17-complex list was stale -- it included 8f60 and 8r4u, which DO have a pose above 0.23.
#
#   FAIL_BELOW=0.40 (default): no pose anywhere in the pool reaches DockQ 0.40 -- Protenix never
#                              produces even a MEDIUM-quality structure. 33 interfaces, 22 complexes.
#   FAIL_BELOW=0.23          : no ACCEPTABLE pose at all. 21 interfaces, 15 complexes.
#
# SEEDS: 128, not 1024. At Protenix's measured ~32 structures/hour on 4 GPUs, 1,024 seeds over 22
# complexes is ~29 DAYS. This is exactly the mistake the original deep run made (estimated 8 days,
# actually 23). 128 seeds is a 2.5x depth increase over the current 50 and lands in ~3.7 days. If
# 128 seeds finds nothing, 1,024 almost certainly will not either -- and you learn that in four days
# instead of a month.
#
# SHARED BOX. naboo runs john's jobs on GPUs 3/4/6/7. NEVER touch them. OMP_NUM_THREADS is capped --
# unset, torch grabs every core (8 procs once became 220 of 288 and starved another user).
# Resume-safe: skips any (complex, seed) already on disk, so the 7,669 seed dirs already produced
# are free progress.
set -u
cd "$(dirname "$0")/../.." || exit 1

GPUS="${GPUS:-0 1 2 5}"                # naboo: 3/4/6/7 are john's
NSEEDS="${NSEEDS:-128}"
FAIL_BELOW="${FAIL_BELOW:-0.40}"
LOG=steer/experiments/logs/protenix_deep_coverage.log
mkdir -p steer/experiments/logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"

# --- the target set, derived from OST ground truth ---------------------------------------------
CX=$(~/esm/.venv/bin/python - "$FAIL_BELOW" <<'PY'
import csv, sys, collections
import numpy as np
T = float(sys.argv[1])
by = collections.defaultdict(list)
for r in csv.DictReader(open('results/csv_files/dockq_ost_protenix_full.csv')):
    if r.get('dockq_status') not in (None, '', 'ok'):
        continue
    try:
        dq = float(r['dockq'])
    except (TypeError, ValueError):
        continue
    if not np.isnan(dq):
        by[(r['assembly_id'][:4], r['chain1'], r['chain2'])].append(dq)
# an interface is a COVERAGE failure if NO pose in its pool reaches T
fail = {k[0] for k, v in by.items() if max(v) < T}
print(' '.join(sorted(fail)))
PY
)
n_cx=$(echo $CX | wc -w)
echo "=== $(date '+%F %T')  Protenix deep sampling — COVERAGE FAILURES ONLY ===" | tee -a "$LOG"
echo "  criterion : best DockQ in pool < $FAIL_BELOW" | tee -a "$LOG"
echo "  complexes : $n_cx" | tee -a "$LOG"
echo "  $CX" | fold -s -w 100 | sed 's/^/    /' | tee -a "$LOG"
echo "  seeds     : 0..$((NSEEDS-1))   GPUs: [$GPUS]" | tee -a "$LOG"
[ "$n_cx" -eq 0 ] && { echo "!! empty target set"; exit 1; }

# --- refuse to start on somebody else's GPU ------------------------------------------------------
for g in $GPUS; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v G="$g" '$1==G{print $2}')
  if [ "${used:-0}" -gt 500 ]; then
    echo "!!! GPU $g is in use (${used} MiB) — somebody else's. Refusing to launch." | tee -a "$LOG"
    exit 1
  fi
done

check_limits() {          # a config number is NOT what you consume — MEASURE it
  local n m c
  n=$(ps -u "$USER" --no-headers | wc -l)
  m=$(ps -u "$USER" -o rss= | awk '{s+=$1} END{printf "%d", s/1048576}')
  c=$(ps -u "$USER" -o pcpu= --no-headers | awk '{s+=$1} END{printf "%d", s/100}')
  echo "  [limits] $n procs (max 128) | ${m} GB (max 512) | ${c} cores (max 64) | load $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOG"
  if [ "$n" -gt 128 ] || [ "$m" -gt 512 ] || [ "$c" -gt 64 ]; then
    echo "  *** OVER CEILING — KILLING NOW ***" | tee -a "$LOG"
    pkill -9 -f run_benchmark_parallel; pkill -9 -f "runner/inference.py"
    exit 1
  fi
}

~/Protenix/.venv/bin/python run/protenix/run_benchmark_parallel.py \
    --gpus $GPUS --seeds $(seq 0 $((NSEEDS-1))) --complexes $CX >> "$LOG" 2>&1 &
pid=$!
sleep 20
check_limits
wait $pid
echo "=== $(date '+%F %T')  DONE ===" | tee -a "$LOG"
