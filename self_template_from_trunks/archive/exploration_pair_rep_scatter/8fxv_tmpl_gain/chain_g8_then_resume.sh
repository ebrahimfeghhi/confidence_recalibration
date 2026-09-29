#!/bin/bash
D=/home/ebrahim/tts_abag/pair_rep_scatter/8fxv_tmpl_gain
log(){ echo "[$(date +%T)] $*"; }
resume(){ log "resuming sweep"; bash $D/resume_sweep.sh; log "chain done"; }
trap resume EXIT
declare -A S=( [0]=tg_g3_0 [1]=tg_g3_1 [2]=tg_g5_2 [5]=tg_g5_5 )
for g in 0 1 2 5; do
  ( while tmux has-session -t "${S[$g]}" 2>/dev/null; do sleep 5; done
    log "gpu $g free (${S[$g]} ended); starting g8 worker"
    tmux new-session -d -s tg_g8_$g "bash $D/launch_g8_gpu$g.sh" ) &
done
wait
sleep 15
while tmux ls 2>/dev/null | grep -q "^tg_g8_"; do sleep 5; done
log "all g8 workers ended"
