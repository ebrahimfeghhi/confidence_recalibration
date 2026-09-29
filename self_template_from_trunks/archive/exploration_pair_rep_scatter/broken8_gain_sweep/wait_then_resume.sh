#!/bin/bash
D=/home/ebrahim/tts_abag/pair_rep_scatter/broken8_gain_sweep
trap 'echo "[$(date +%T)] resuming sweep"; bash $D/resume_sweep.sh; echo "[$(date +%T)] chain done"' EXIT
sleep 20
while tmux ls 2>/dev/null | grep -q "^bg_g"; do sleep 10; done
echo "[$(date +%T)] all 4 workers ended"
