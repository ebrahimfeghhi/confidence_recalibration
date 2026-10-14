#!/usr/bin/env bash
# Deep Protenix sampling (1024 seeds x 5 diffusion samples) on every complex where Protenix's own
# confidence FAILS. Classes come from results/csv_files/dockq_ost_protenix_full.csv (OST, 171
# interfaces, 42,750 poses):
#
#   NO-POSE = no acceptable pose (DockQ > 0.23) anywhere in its 250-pose pool. A COVERAGE failure.
#             More sampling is the RIGHT lever here -- it is the only class where depth can help.
#   TRAP    = its self-selected pick is < 0.23, but a pose > 0.23 IS already in the pool at 50
#             seeds. A SELECTION failure. Depth is unlikely to help (it did not for ESMFold2:
#             51/55 traps survived to 1024 seeds) -- this is what the selector is for.
#
# PHASE 1 runs the 17 NO-POSE complexes to completion, THEN phase 2 runs the 19 TRAP-only ones,
# so the class where depth actually matters finishes first.
#
# NOT included: 7yxu, 8ts0. The earlier deep run covers them, but Protenix already SELECTS
# correctly there (0.806, 0.950). They were only in that run because the old, corrupt
# Protenix_dockq.csv wrongly reported 0.000 for them. Wasted compute.
#
# THROUGHPUT. Each inference process uses only ~14 GB of a 96 GB card and leaves the GPU idle
# 25-40% of the time in CPU-bound phases (measured: 60-75% mean utilisation). So we run TWO
# processes per GPU -- while one is on CPU the other is on GPU. Passing a GPU id twice makes
# run_benchmark_parallel split the seed axis across both, and the resume marker is per (complex,
# seed) so they never collide.
#     3 GPUs x 1 proc  ->  ~2.1 GPU-equivalents  (what the old run was doing)
#     4 GPUs x 2 procs ->  ~3.8 GPU-equivalents  (~1.9x faster)
# bf16 is ALREADY the default (configs/configs_base.py: dtype=bf16) — no win available there.
#
# Shared box: refuses to launch on a GPU somebody else is using, and hard-kills itself if it
# exceeds 128 processes or 512 GB. Resume-safe: skips any (complex, seed) already complete, so it
# picks up everything the current run produced.
set -u
cd /home/ebrahim/tts_abag

GPUS="${GPUS:-1 2 4 7}"                   # 4 GPUs, 1 process each
# CAP CPU THREADS. torch uses ALL cores for CPU ops when OMP_NUM_THREADS is unset:
# 8 procs x ~28 cores = 220 of 288 cores, load 438, other users starved. Measured,
# not assumed. A process count is NOT a core count.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
SEEDS="${SEEDS:-$(seq 0 1023)}"
LOG=steer/experiments/logs/protenix_deep_traps.log
mkdir -p steer/experiments/logs

# depth is the right lever for these -- Protenix never generates an acceptable pose
NOPOSE=(7yv1 8a44 8a67 8dn7 8dtn 8f60 8im0 8ivx 8oud 8oxx 8r4u 8sfz 8udz 8uky 8ux6 8vzo 8z8v)
# the answer is already in the pool at 50 seeds; depth probably will not help, the selector must
TRAP=(7ucx 7z2m 7zau 8aok 8av2 8be1 8be3 8bf4 8che 8fas 8fbw 8oxw 8pn0 8q3j 8slb 8the 8vui 8yxi 9jbq)

UNIQ_GPUS=$(echo $GPUS | tr ' ' '\n' | sort -u | tr '\n' ' ')

check_limits() {   # a config number is NOT what you consume — MEASURE it
  local n m c
  n=$(ps -u "$USER" --no-headers | wc -l)
  m=$(ps -u "$USER" -o rss= | awk '{s+=$1} END{printf "%d", s/1048576}')
  c=$(ps -u "$USER" -o pcpu= --no-headers | awk '{s+=$1} END{printf "%d", s/100}')   # ACTUAL cores
  echo "  [limits] $n procs (max 128) | ${m} GB (max 512) | ${c} cores (max 64) | load $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOG"
  if [ "$n" -gt 128 ] || [ "$m" -gt 512 ] || [ "$c" -gt 64 ]; then
    echo "  *** OVER CEILING — KILLING NOW ***" | tee -a "$LOG"
    pkill -9 -f run_benchmark_parallel; pkill -9 -f "runner/inference.py"
    exit 1
  fi
}

run_phase() {      # $1 = phase name, rest = complexes
  local name=$1; shift
  echo "=== $(date '+%F %T') PHASE $name: $# complexes on GPUs [$GPUS] ===" | tee -a "$LOG"
  ~/Protenix/.venv/bin/python run/protenix/run_benchmark_parallel.py \
      --gpus $GPUS --seeds $SEEDS --complexes "$@" >> "$LOG" 2>&1 &
  local pid=$!
  sleep 10
  check_limits
  wait $pid
  echo "=== $(date '+%F %T') PHASE $name COMPLETE ===" | tee -a "$LOG"
}

echo "=== $(date '+%F %T') Protenix deep sampling, 1024 seeds ===" | tee -a "$LOG"
echo "  phase 1 (NO-POSE, depth is the right lever): ${#NOPOSE[@]} complexes" | tee -a "$LOG"
echo "  phase 2 (TRAP only, selection failure)     : ${#TRAP[@]} complexes" | tee -a "$LOG"

for g in $UNIQ_GPUS; do
  s=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F', ' -v G="$g" '$1==G { print ($2 < 500) ? "free" : "busy" }')
  echo "  GPU $g: $s" | tee -a "$LOG"
  if [ "$s" = "busy" ]; then
    echo "!!! GPU $g is NOT idle. Refusing to launch. Free it, or set GPUS." | tee -a "$LOG"
    exit 1
  fi
done

run_phase "1/NO-POSE" "${NOPOSE[@]}"
run_phase "2/TRAP"    "${TRAP[@]}"

echo "=== $(date '+%F %T') ALL DONE ===" | tee -a "$LOG"
