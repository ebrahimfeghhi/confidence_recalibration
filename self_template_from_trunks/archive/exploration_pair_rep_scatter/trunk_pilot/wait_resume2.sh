#!/bin/bash
D=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot
trap 'echo "[$(date +%T)] resuming sweep"; bash $D/resume_sweep.sh; echo "[$(date +%T)] chain done"' EXIT
sleep 20
while tmux ls 2>/dev/null | grep -q "^br_g"; do sleep 10; done
echo "[$(date +%T)] all workers ended"
