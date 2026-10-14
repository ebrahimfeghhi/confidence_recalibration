#!/usr/bin/env bash
# The MSA ablation, with its CONTROL.
#
# THE DESIGN FLAW THIS FIXES. My first version ran base biohub/ESMFold2 WITH MSAs and compared it
# to ESMFold2-Fast (single-sequence). But those are DIFFERENT CHECKPOINTS (1.3 GB vs 721 MB), so
# that comparison varies the MSA *and* the model, and nothing could be attributed to the MSA.
# Both arms now run the SAME base checkpoint; the MSA is the only variable.
#
#   arm B   base ESMFold2 + Protenix's MSAs   -> esmfold2_msa      (SLOW: the MSA encoder is
#                                                                   expensive, so it gets 4 GPUs)
#   arm A'  base ESMFold2, --no_msa           -> esmfold2_nomsa    [THE CONTROL]  (fast, 2 GPUs)
#
# MSA_MAX_DEPTH: 1024 (the processor default) turned out to be far too slow -- ~0 folds/hour with
# 3-4 chains. 512 still gives a real MSA (AF2-class models subsample to 128-512 per pass anyway)
# and is what makes the experiment finish. Documented because it weakens the treatment slightly.
#
# Shared box: NEVER touch GPU 0 (shawn) or GPU 5 (kao). OMP_NUM_THREADS capped -- unset, torch
# grabs every core (that is how 8 procs became 220 of 288 and starved another user).
set -u
cd /home/ebrahim/tts_abag

NSEEDS="${NSEEDS:-8}"
MSA_DEPTH="${MSA_DEPTH:-512}"
MSA_GPUS="${MSA_GPUS:-1 2 3}"
NOMSA_GPUS="${NOMSA_GPUS:-4 6 7}"
# 8the/8ts0/8uky/8w84/8w90 dropped: base ESMFold2 (NOT the -Fast checkpoint) stalls on them --
# both arms completed every complex up to 8the and then produced nothing for 20+ min at 100% GPU.
# The remaining 9 fold in ~140s each and are the SAME set in both arms, so the contrast is clean.
CX="7txw 7zoz 8che 8fxv 8ido 8oni 8opr 8q6r 8qz3"
LOG=steer/experiments/logs/msa_ablation
mkdir -p "$LOG"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
PY=/home/ebrahim/esm/.venv/bin/python

launch() {                                 # $1=arm  $2=out_dir  $3=extra  $4...=gpus
  arm=$1; out=$2; extra=$3; shift 3
  ngpu=$#; i=0
  for g in "$@"; do
    seeds=$(python3 -c "print(' '.join(str(s) for s in range($NSEEDS) if s % $ngpu == $i))")
    [ -z "$seeds" ] && { i=$((i+1)); continue; }
    echo "  [$arm] GPU $g: seeds [$seeds]"
    CUDA_VISIBLE_DEVICES=$g nohup $PY run/esm/run_msa_ablation.py \
        --out_dir "$out" --complexes $CX --seeds $seeds \
        --msa_max_depth "$MSA_DEPTH" $extra \
        > "$LOG/${arm}_gpu${g}.log" 2>&1 &
    i=$((i+1))
  done
}

echo "arm B  (base ESMFold2 + MSA, depth $MSA_DEPTH) -> GPUs $MSA_GPUS"
launch msa   /home/ebrahim/esm/foldbench_predictions/esmfold2_msa   ""         $MSA_GPUS
echo "arm A' (base ESMFold2, NO MSA)  [CONTROL]      -> GPUs $NOMSA_GPUS"
launch nomsa /home/ebrahim/esm/foldbench_predictions/esmfold2_nomsa "--no_msa" $NOMSA_GPUS

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
echo "  running: $(ps -u "$USER" -o args= | grep -c '[r]un_msa_ablation') processes"
