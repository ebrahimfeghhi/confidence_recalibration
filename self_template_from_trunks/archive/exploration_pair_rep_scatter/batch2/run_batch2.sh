#!/bin/bash
B=/home/ebrahim/tts_abag/pair_rep_scatter/batch2; T=/home/ebrahim/tts_abag/pair_rep_scatter/trunk_pilot
log(){ echo "[$(date +%T)] $*"; }
relaunch(){ log "relaunching the FoldBench sweep from its saved commands"; bash $T/relaunch_sweep.sh; sleep 90; log "sweep folder.py processes up: $(pgrep -f '^/home/ebrahim/Protenix/.venv/bin/python folder.py .*free_compute_matched' | wc -l)"; log "driver finished"; }
trap relaunch EXIT
log "batch2 driver start"
# 1. full stop of the sweep (guarded: exactly the 4 expected workers, GPUs 0,1,2,5)
pids=$(pgrep -f '^/home/ebrahim/Protenix/.venv/bin/python folder.py .*free_compute_matched'); n=0; gpus=""
for pid in $pids; do g=$(tr '\0' '\n' < /proc/$pid/environ | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2); gpus="$gpus $g"; n=$((n+1)); done
if [ $n -ne 4 ] || [ "$(echo $gpus | tr ' ' '\n' | sort | tr '\n' ' ')" != "0 1 2 5 " ]; then log "ABORT: expected 4 sweep workers on GPUs 0 1 2 5, found $n on:$gpus. Nothing stopped."; exit 1; fi
for s in fb_freew0 fb_freew1 fb_freew2 fb_freew5; do [ -s $T/sweep_launch/$s.cmd ] || { log "ABORT: saved command for $s missing"; exit 1; }; done
for pid in $pids; do kill -9 $pid; done; sleep 8; log "sweep workers stopped; GPU memory: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '1p;2p;3p;6p' | tr '\n' ';')"
# 2. stage A: free folds with trunks
for pass in 1 2; do
  log "stage A pass $pass"; for g in 0 1 2 5; do tmux new-session -d -s b2a_g$g "bash $B/stageA_worker.sh $g"; done
  while tmux ls 2>/dev/null | grep -q '^b2a_'; do sleep 15; done
  python3 $B/verify.py A && { log "stage A complete"; break; } || log "stage A incomplete after pass $pass"
done
python3 $B/verify.py A > /dev/null || { log "ABORT: stage A incomplete"; exit 1; }
log "bf16 dump check (z): $(grep -h 'DUMP_TRUNK_BF16 z' $B/free/w*.log | sed 's/.*n_diff=\([0-9]*\)\/.*fp16 would clip (|x|>65504): \([0-9]*\), underflow (0<|x|<6.1e-5): \([0-9]*\).*/\1 \2 \3/' | awk '{if($1>a)a=$1; if($2>b)b=$2; if($3>c)c=$3; n++} END{print n\" dumps; max n_diff=\"a\"; max fp16 clip=\"b\"; max fp16 underflow=\"c}')"
# 3. stage B (CPU)
bash $B/stageB.sh > $B/stageB.log 2>&1 || { log "ABORT: stage B failed, see stageB.log"; exit 1; }
log "stage B done: $(grep '^donors' $B/stageB.log | tr '\n' ' ')"
python3 $B/make_stageC.py > $B/stageC_plan.log 2>&1 || { log "ABORT: make_stageC failed"; exit 1; }; log "stage C plan: $(cat $B/stageC_plan.log | tr '\n' ' ')"
# 4. stage C: the 8 arms
for pass in 1 2; do
  log "stage C pass $pass"; for g in 0 1 2 5; do tmux new-session -d -s b2c_g$g "bash $B/stageC_gpu$g.sh"; done
  while tmux ls 2>/dev/null | grep -q '^b2c_'; do sleep 15; done
  python3 $B/verify.py C && { log "stage C complete"; break; } || log "stage C incomplete after pass $pass"
done
python3 $B/verify.py C | sed 's/^/  /'
log "all stages ended"
