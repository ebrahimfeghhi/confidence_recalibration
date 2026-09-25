#!/usr/bin/env bash
# Wait for the in-flight workers to drain, then relaunch on the FULL 7-interface queue.
#
# WHY WAIT rather than restart now: the running workers read queue.tsv at startup, so they cannot see
# the 4 interfaces appended to it. Restarting would discard the folds in flight. The relaunch skips
# everything already in n13.csv, so the 75 done units cost nothing the second time.
#
# THE TRIGGER is the absence of worker processes, not a fold count: a worker exits only when it finds
# no unclaimed unit left, which is the real definition of drained. Counting folds can fire while the
# last three are still running.
set -u
D=/home/ebrahim/tts_abag/experimental_data/inmem
PY=/home/ebrahim/Protenix/.venv/bin/python
cd /home/ebrahim/tts_abag/diffusion_commit || exit 1
while ps -u "$USER" -o cmd= | grep -qE "^[^ ]*python[0-9.]* .*[f]old\.py --queue $D/queue.tsv"; do sleep 60; done
echo "$(date +%H:%M:%S) in-flight workers drained; relaunching on all 7 interfaces"
# stale claims from any worker killed mid-fold are released by fold.py itself on startup
for G in 1 5 7; do
  tmux kill-session -t im13_g$G 2>/dev/null
  tmux new-session -d -s im13_g$G "cd /home/ebrahim/tts_abag/diffusion_commit && CUDA_VISIBLE_DEVICES=$G LAYERNORM_TYPE=openfold $PY fold.py --queue $D/queue.tsv --out $D/n13 --csv $D/n13.csv --claim-dir $D/.claims13 --dtype bf16 --ncycle 13 --hook self --hook-after 9 --hook-observe auto >> $D/n13_g$G.log 2>&1"
  sleep 5
done
echo "$(date +%H:%M:%S) relaunched"
