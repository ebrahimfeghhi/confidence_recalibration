#!/bin/bash
# Ends ONLY folder.py workers of the skip-the-MSA-module arms (output path contains /resume/nomsa_). MSA-on arms have /resume/msa_.
D=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot
while tmux ls 2>/dev/null | grep -q "^br_g"; do
  for pid in $(pgrep -f "folder.py .*--out $D/resume/nomsa_"); do
    echo "[$(date +%T)] ending skip-module worker pid $pid ($(tr '\0' ' ' < /proc/$pid/cmdline | grep -o 'resume/nomsa_c[0-9]*'))"; kill $pid
  done
  sleep 3
done
echo "[$(date +%T)] watcher exiting"
