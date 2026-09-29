#!/bin/bash
# Recreate the four FoldBench free top-up workers from the exact commands captured before they were stopped.
D=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot
for s in fb_freew0 fb_freew1 fb_freew2 fb_freew5; do
  if pgrep -f "folder.py .*free_compute_matched" >/dev/null && tmux has-session -t "=$s" 2>/dev/null; then echo "$s already running; skipping"; continue; fi
  tmux kill-session -t "=$s" 2>/dev/null
  cmd=$(sed 's/^bash -c //' $D/sweep_launch/$s.cmd)
  tmux new-session -d -s $s "$cmd" && echo "relaunched $s"
done
