#!/bin/bash
# Append a timestamped status line every 10 minutes, so the morning report rests on a record of
# what actually happened rather than on whatever the last log line happens to be.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
while :; do
  T=$(TZ=America/New_York date '+%m-%d %H:%M')
  TF=$(tmux ls 2>/dev/null | grep -c w1fold)
  VF=$(tmux ls 2>/dev/null | grep -c w1val)
  TU=$(grep -hc 'interfaces in' training/fold/w2_c4.log training/fold/w2_c5.log training/fold/w2_c7.log 2>/dev/null | paste -sd+ | bc)
  VU=$(grep -hc 'interfaces in' validation/fold/w1val_c*.log 2>/dev/null | paste -sd+ | bc)
  TR=$(ls training/fold/trunks/*/*.npz 2>/dev/null | wc -l)
  RW=$(wc -l < training/fold/results.csv 2>/dev/null)
  MEM=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  echo "$T  w1fold=$TF w1val=$VF train_scored=${TU:-0} val_scored=${VU:-0} train_trunks=$TR results_rows=$RW mem=${MEM}G"
  sleep 600
done
