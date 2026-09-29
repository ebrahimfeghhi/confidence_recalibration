#!/bin/bash
N=/home/ebrahim/tts_abag/pair_rep_scatter/ownseed_nodrop
log(){ echo "[$(date +%T)] $*"; }
log "nodrop validation start"
used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '($1==0||$1==1||$1==2||$1==5)&&$2>500{print $1":"$2"MiB"}')
[ -z "$used" ] || { log "ABORT: GPUs 0,1,2,5 not empty: $used"; exit 1; }
for g in 0 1 2 5; do tmux new-session -d -s nd_g$g "bash $N/run_gpu$g.sh"; done
while tmux ls 2>/dev/null | grep -q '^nd_g'; do sleep 10; done
python3 $N/verify.py && log "complete" || log "incomplete"
python3 $N/verify.py | sed 's/^/  /'; log "done"
