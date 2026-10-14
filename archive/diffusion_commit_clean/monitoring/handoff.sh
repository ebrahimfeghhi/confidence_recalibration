#!/usr/bin/env bash
# When the single-template run finishes, hand the GPUs straight to the method run.
#
# WHY A SCRIPT. The alternative is noticing by hand, which means the cards sit idle however long it
# takes someone to look -- and idle cards on this box get taken. This polls cheaply and swaps within a
# minute of the last fold landing.
#
# It stops singlewatch FIRST. That watcher restarts dead workers by design, so launching the method
# run while it is alive risks it resurrecting single-template workers onto cards the method is using.
set -u
CODE=/home/ebrahim/tts_abag/diffusion_commit
PY=/home/ebrahim/Protenix/.venv/bin/python
S=/home/ebrahim/tts_abag/experimental_data/single
M=/home/ebrahim/tts_abag/experimental_data/method
GPUS=(1 5 6)

while pgrep -u "$USER" -f "single/queue.tsv" > /dev/null; do sleep 60; done
echo "$(date '+%F %H:%M') single-template run finished: $(ls $S/claims | wc -l)/$(wc -l < $S/queue.tsv) claimed"

tmux kill-session -t singlewatch 2>/dev/null
for i in 0 1 2; do tmux kill-session -t singw$i 2>/dev/null; done
sleep 5

for i in 0 1 2; do
  tmux new-session -d -s "methw$i" \
    "cd $CODE && CUDA_VISIBLE_DEVICES=${GPUS[$i]} LAYERNORM_TYPE=openfold $PY fold.py \
     --queue $M/queue.tsv --out $M/out --csv $M/results.csv --templates $M/ns \
     --claim-dir $M/claims --drop-cifs 2>&1 | tee -a $M/w$i.log"
done
sleep 10
tmux new-session -d -s methwatch \
  "/home/ebrahim/tts_abag/diffusion_commit_clean/watch_method.sh 1800 2>&1 | tee -a $M/watch.log"
echo "$(date '+%F %H:%M') method run resumed: $(ls $M/claims | wc -l)/$(wc -l < $M/queue.tsv) already claimed"
