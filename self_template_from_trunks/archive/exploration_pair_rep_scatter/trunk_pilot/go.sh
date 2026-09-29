#!/bin/bash
D=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot
bash $D/pause_sweep.sh || { echo "pause refused; nothing launched"; exit 1; }
for g in 0 1 2 5; do tmux new-session -d -s bp_g$g "bash $D/launch_gpu$g.sh"; done
echo "launched 4 workers"
nohup bash $D/wait_then_resume.sh > $D/chain.log 2>&1 &
