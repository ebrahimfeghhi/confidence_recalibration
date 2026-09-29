#!/bin/bash
D=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot
sleep 20
while tmux ls 2>/dev/null | grep -q "^br_g"; do sleep 10; done
echo "[$(date +%T)] all analysis workers ended; relaunching the sweep"
bash $D/relaunch_sweep.sh
sleep 90
echo "[$(date +%T)] sweep workers now:"; pgrep -af "folder.py .*free_compute_matched" | wc -l
