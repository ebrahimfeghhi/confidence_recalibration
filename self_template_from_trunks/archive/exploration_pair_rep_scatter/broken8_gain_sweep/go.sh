#!/bin/bash
D=/home/ebrahim/tts_abag/pair_rep_scatter/broken8_gain_sweep
bash $D/pause_sweep.sh || { echo "pause refused; nothing launched"; exit 1; }
for spec in "g2 0" "g3 1" "g4 2" "g5 5"; do set -- $spec; tmux new-session -d -s bg_$1_$2 "bash $D/launch_$1_gpu$2.sh"; done
echo "launched 4 workers"
nohup bash $D/wait_then_resume.sh > $D/chain.log 2>&1 &
