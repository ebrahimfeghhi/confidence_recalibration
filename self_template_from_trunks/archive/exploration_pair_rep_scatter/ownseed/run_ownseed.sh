#!/bin/bash
O=/home/ebrahim/tts_abag/pair_rep_scatter/ownseed; T=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot; STOPPED=0
log(){ echo "[$(date +%T)] $*"; }
relaunch(){ if [ $STOPPED = 1 ]; then log "relaunching the FoldBench sweep from its saved commands"; bash $T/relaunch_sweep.sh; sleep 90; log "sweep folder.py processes up: $(pgrep -f '^/home/ebrahim/Protenix/.venv/bin/python folder.py .*free_compute_matched' | wc -l)"; else log "sweep was not running at the start; not relaunching"; fi; log "driver finished"; }
trap relaunch EXIT
log "ownseed driver start"
pids=$(pgrep -f '^/home/ebrahim/Protenix/.venv/bin/python folder.py .*free_compute_matched'); n=0; gpus=""
for pid in $pids; do g=$(tr '\0' '\n' < /proc/$pid/environ | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2); gpus="$gpus $g"; n=$((n+1)); done
if [ $n -eq 0 ]; then
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '($1==0||$1==1||$1==2||$1==5)&&$2>500{print $1":"$2"MiB"}')
  [ -z "$used" ] || { log "ABORT: no sweep running but GPUs are in use by someone else: $used"; exit 1; }
  log "no sweep workers running and GPUs 0,1,2,5 are empty; nothing to stop"
elif [ $n -eq 4 ] && [ "$(echo $gpus | tr ' ' '\n' | sort | tr '\n' ' ')" = "0 1 2 5 " ]; then
  for s in fb_freew0 fb_freew1 fb_freew2 fb_freew5; do [ -s $T/sweep_launch/$s.cmd ] || { log "ABORT: saved command for $s missing"; exit 1; }; done
  for pid in $pids; do kill -9 $pid; done; STOPPED=1; sleep 8; log "sweep stopped"
else log "ABORT: found $n sweep workers on:$gpus (expected 0 or exactly 4 on GPUs 0 1 2 5). Nothing stopped."; exit 1; fi
for pass in 1 2; do
  log "run pass $pass"; for g in 0 1 2 5; do tmux new-session -d -s os_g$g "bash $O/run_gpu$g.sh"; done
  while tmux ls 2>/dev/null | grep -q '^os_g'; do sleep 10; done
  python3 $O/verify_own.py && { log "all jobs complete"; break; } || log "incomplete after pass $pass"
done
python3 $O/verify_own.py | sed 's/^/  /'
log "done"
