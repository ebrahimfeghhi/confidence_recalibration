#!/usr/bin/env bash
# Watch a self-templating sweep: score it, check health, and FIX what it can.
#
#   watch.sh <experiment-dir> <reporter.py> [more_reporters.py ...] [--interval N]
#
# ONE SCRIPT, NOT ONE PER EXPERIMENT. There used to be four of these, each made by sed-ing the last,
# ~90% identical and differing only in a directory name and a reporter. They drifted: a fix applied to
# one silently left the others stale, and patching by line number once restarted an unmodified copy.
#
# A watcher that only prints is useless overnight -- by morning it has logged the same failure sixteen
# times and nothing has run. This restarts a dead worker on its own GPU, which also releases that
# worker's stale claims, since fold.py reconciles claims at startup. What it cannot fix it prefixes
# ESCALATE so one grep finds it.
set -u
T=${1:?usage: watch.sh <experiment-dir> <reporter.py> [...]}
shift
INT=1800
REPORTERS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INT=$2; shift 2 ;;
    *) REPORTERS+=("$1"); shift ;;
  esac
done
CODE=/home/ebrahim/tts_abag/diffusion_commit_clean_v2   # the driver moved here
# reporters live beside this script; hardcoding the parent dir broke when it moved
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=/home/ebrahim/Protenix/.venv/bin/python
GPUS=(1 5 6)
NAME=$(basename "$T")
# worker tmux sessions are named <experiment>w<i>; the restart branch below recreates them by that name

while true; do
  echo "================ $(date '+%F %H:%M')  [$NAME] ================"

  # COUNT ONLY CLAIMS THIS QUEUE ASKS FOR. A run that carries results over from an earlier sweep
  # pre-creates a claim per reused donor, so a raw `ls claims | wc -l` exceeds the queue length and
  # the restart branch below becomes unreachable -- a dead worker would idle its GPU all night.
  total=$(wc -l < "$T/queue.tsv")
  awk -F'\t' 'NF>=6{print $2"_"$3"_"$4"_"$5}' "$T/queue.tsv" | sort -u > "$T/.qtags"
  ls "$T/claims" 2>/dev/null | sort -u > "$T/.ctags"
  claimed=$(comm -12 "$T/.qtags" "$T/.ctags" | wc -l)
  carried=$(comm -13 "$T/.qtags" "$T/.ctags" | wc -l)
  if [ "$claimed" -lt "$total" ]; then
    for i in 0 1 2; do
      # TWO NAMING CONVENTIONS EXIST for the same session: this script and hand-launches use the
      # index (w0/w1/w2), call_folder.py uses the GPU id (w1/w5/w6). Checking only one would declare
      # live workers dead and start a second set on the same cards. Accept either until one launcher
      # owns this -- call_folder.py should, and this branch should call it instead of rolling its own.
      if ! tmux has-session -t "${NAME}w$i" 2>/dev/null \
         && ! tmux has-session -t "${NAME}w${GPUS[$i]}" 2>/dev/null; then
        echo "REMEDIATE: worker $i (GPU ${GPUS[$i]}) gone -- restarting"
        tmux new-session -d -s "${NAME}w$i" \
          "cd $CODE && CUDA_VISIBLE_DEVICES=${GPUS[$i]} LAYERNORM_TYPE=openfold $PY folder.py \
           --queue $T/queue.tsv --out $T/out --csv $T/results.csv --templates $T/ns \
           --claim-dir $T/claims --drop-cifs 2>&1 | tee -a $T/w$i.log"
      fi
    done
  fi
  echo "workers=$(pgrep -u "$USER" -f "$NAME/queue.tsv" | wc -l)  claimed=$claimed/$total (+$carried carried)  rows=$(( $(wc -l < "$T/results.csv" 2>/dev/null || echo 1) - 1 ))"

  for i in 0 1 2; do
    # grep -c PRINTS 0 and EXITS 1 when nothing matches, so `|| echo 0` yields "0\n0"
    n=$(grep -cE "Traceback|OutOfMemory|CUDA out of memory" "$T/w$i.log" 2>/dev/null; true)
    [ "${n:-0}" -gt 0 ] && echo "ESCALATE: worker $i has $n error lines (grep -nE 'Traceback|out of memory' $T/w$i.log)"
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,5,6 | sed 's/^/  gpu /'

  : > "$T/table.txt"
  for r in "${REPORTERS[@]}"; do (cd "$HERE" && $PY "$r" >> "$T/table.txt" 2>&1); done
  cat "$T/table.txt"

  sleep "$INT"
done
