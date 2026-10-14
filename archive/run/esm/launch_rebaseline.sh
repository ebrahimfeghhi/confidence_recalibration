#!/usr/bin/env bash
# RE-BASELINE: base ESMFold2 + MSA over the full FoldBench AbAg benchmark.
#
# WHY THIS IS THE MOST IMPORTANT JOB IN THE REPO
#
# Every ESMFold2 number in tts_abag was produced by ESMFold2-**Fast** run SINGLE-SEQUENCE. Measured
# overnight 2026-07-14 (docs/overnight_2026_07_14_summary.md), on the same complexes and seeds:
#
#                                 d TOP-QUARTILE   selector   trap rate
#   ESMFold2-Fast, no MSA            -0.139          40%        60%      <- what the repo benchmarks
#   base ESMFold2, no MSA            -0.200          25%        60%
#   base ESMFold2, WITH MSA          +0.327          62%        33%
#   Protenix,      WITH MSA          +0.568          86%        13%
#
# Running single-sequence is what CAUSES the confident-decoy inversion (confidence discriminates
# across the pool, then points at decoys exactly where argmax reads). An MSA cures it: selector
# 25% -> 62% on the SAME checkpoint (+38%, 95% CI [+15%,+62%], P=0.998), and the trap rate HALVES.
#
# So the 47-interface confident-decoy set, the 6/22 clustering result, and every scaling curve in
# this repo are measured on a handicapped configuration. NOTHING should be compared against them
# until this run exists.
#
# THE 8the PROBLEM, AND WHY THE LIST IS SORTED
#
# The runner walks its complex list IN ORDER. 8the (1,230 residues) takes >20 min for a single fold
# while everything else takes ~192s, so if it sits early in the list it blocks everything behind it
# -- which is exactly what happened during the ablation, and which I initially misdiagnosed as
# "base ESMFold2 stalls on the five largest complexes" (it does not; the others were never reached).
#
# So: SORT BY SIZE, SMALLEST FIRST. Every GPU then chews through the cheap complexes before it ever
# reaches an expensive one, and a run that is cut short still yields a usable, balanced dataset.
#
# Shared box: NEVER touch a GPU another user is on. OMP_NUM_THREADS capped -- unset, torch grabs
# every core (8 procs once became 220 of 288 and starved another user).
set -u
cd /home/ebrahim/tts_abag

GPUS="${GPUS:-1 2 3 4 6 7}"          # CHECK FIRST. 0 and 5 were shawn's and kao's.
NSEEDS="${NSEEDS:-50}"               # 50 x 5 samples = 250 poses, matching Protenix
MSA_DEPTH="${MSA_DEPTH:-512}"        # 1024 (the processor default) gives ~0 folds/hour on 3-4 chains
OUT="${OUT:-/home/ebrahim/esm/foldbench_predictions/esmfold2_msa_full}"
LOG=steer/experiments/logs/rebaseline
mkdir -p "$LOG"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
PY=/home/ebrahim/esm/.venv/bin/python

# every complex that has a prepped Protenix MSA, SORTED BY SIZE (smallest first)
CX=$($PY - <<'EOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, '.')
from paths import JSON_DIR
sizes = {}
for p in Path(JSON_DIR).glob("*updated*.json"):
    cx = p.name[:4]
    if not os.path.isdir(f"/home/ebrahim/Protenix/foldbench_abag/prep/{cx}/msa"):
        continue                                   # no MSA prepped -> cannot run this arm
    try:
        d = json.loads(p.read_text()); d = d[0] if isinstance(d, list) else d
        L = sum(len(e["proteinChain"]["sequence"]) * e["proteinChain"].get("count", 1)
                for e in d["sequences"] if "proteinChain" in e)
    except Exception:
        continue
    if L:
        sizes[cx] = L
print(" ".join(c for c, _ in sorted(sizes.items(), key=lambda kv: kv[1])))
EOF
)
n_cx=$(echo $CX | wc -w)
echo "complexes with a prepped MSA, smallest first: $n_cx"
echo "  $CX" | fold -s -w 100 | sed 's/^/  /'
echo

for g in $GPUS; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v G="$g" '$1==G{print $2}')
  if [ "${used:-0}" -gt 500 ]; then
    echo "!!! GPU $g is in use (${used} MiB) — somebody else's. Refusing to launch."; exit 1
  fi
done

ngpu=$(echo $GPUS | wc -w); i=0
for g in $GPUS; do
  seeds=$(python3 -c "print(' '.join(str(s) for s in range($NSEEDS) if s % $ngpu == $i))")
  echo "  GPU $g: seeds [$seeds]"
  CUDA_VISIBLE_DEVICES=$g nohup $PY run/esm/run_msa_ablation.py \
      --out_dir "$OUT" --complexes $CX --seeds $seeds --msa_max_depth "$MSA_DEPTH" \
      > "$LOG/gpu${g}.log" 2>&1 &
  i=$((i+1))
done

sleep 25
echo
echo "=== MEASURED, 25s after launch (hard rule: <=128 procs, <=512 GB, <=64 cores) ==="
n=$(ps -u "$USER" --no-headers | wc -l)
m=$(ps -u "$USER" -o rss= | awk '{s+=$1} END{printf "%d", s/1048576}')
c=$(ps -u "$USER" -o pcpu= --no-headers | awk '{s+=$1} END{printf "%d", s/100}')
echo "  procs=$n/128   mem=${m}GB/512   cores=${c}/64   load=$(cut -d' ' -f1 /proc/loadavg)"
if [ "$n" -gt 128 ] || [ "$m" -gt 512 ] || [ "$c" -gt 64 ]; then
  echo "  *** OVER CEILING — KILLING NOW ***"; pkill -9 -f run_msa_ablation; exit 1
fi
echo
echo "It is resume-safe: rerun this exact script after any interruption and it skips what exists."
echo "Then score it:"
echo "  taskset -c 0-31 env OST_WORKERS=12 OMP_NUM_THREADS=1 nice -n 19 \\"
echo "    ~/esm/.venv/bin/python eval/eval_ost_foldbench.py \\"
echo "      --pred_dir $OUT --out_csv results/csv_files/dockq_ost_esmfold2_msa_full.csv"
echo "  (OST_WORKERS LOW: each worker forks a multithreaded ost subprocess. OST_WORKERS=128 once"
echo "   produced ~600 processes and drove load to 774 on this box.)"
