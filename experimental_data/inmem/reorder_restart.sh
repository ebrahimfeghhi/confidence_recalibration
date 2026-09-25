#!/usr/bin/env bash
# Let the three in-flight 9vor folds finish, THEN restart the workers on the reordered queue.
#
# WHY WAIT: each in-flight fold is ~12 min of GPU on a 1862-token complex and is most of the way
# done. Killing now throws that away; killing after they score banks them.
#
# WHY RESTART AT ALL: the workers read queue.tsv once at startup, so they still hold the old order and
# would keep claiming 9vor units -- about 4 hours of the least informative interface in the set
# (0 creations, mean delta -0.003 in the full-refold data) before touching 9uzh, which is the
# strongest case for the mechanism.
#
# Stale claims left by the kill are released by fold.py itself on startup, so the 9vor units it was
# holding become available again when 9vor's turn comes.
set -u
D=/home/ebrahim/tts_abag/experimental_data/inmem
PY=/home/ebrahim/Protenix/.venv/bin/python
start=$(grep -hcE "interfaces in" "$D"/n13_g*.log | paste -sd+ | bc)
target=$(( start + 3 ))
echo "$(date +%H:%M:%S) waiting for the 3 in-flight folds ($start -> $target)"
for _ in $(seq 1 60); do
  cur=$(grep -hcE "interfaces in" "$D"/n13_g*.log | paste -sd+ | bc)
  [ "$cur" -ge "$target" ] && break
  sleep 60
done
echo "$(date +%H:%M:%S) at $(grep -hcE 'interfaces in' "$D"/n13_g*.log | paste -sd+ | bc) folds; restarting on the reordered queue"
for G in 1 5 7; do
  tmux kill-session -t im13_g$G 2>/dev/null
done
sleep 10
for G in 1 5 7; do
  tmux new-session -d -s im13_g$G "cd /home/ebrahim/tts_abag/diffusion_commit && CUDA_VISIBLE_DEVICES=$G LAYERNORM_TYPE=openfold $PY fold.py --queue $D/queue.tsv --out $D/n13 --csv $D/n13.csv --claim-dir $D/.claims13 --dtype bf16 --ncycle 13 --hook self --hook-after 9 --hook-observe auto >> $D/n13_g$G.log 2>&1"
  sleep 5
done
echo "$(date +%H:%M:%S) restarted; 9vor deferred to last"
