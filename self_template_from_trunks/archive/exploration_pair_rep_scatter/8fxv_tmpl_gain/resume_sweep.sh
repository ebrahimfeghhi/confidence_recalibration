#!/bin/bash
# SIGCONT the FoldBench free top-up workers on GPUs 0,1,2,5. Refuses unless each pid is still a folder.py on the expected GPU.
declare -A W=( [1555311]=0 [1555315]=1 [1555320]=2 [1555326]=5 )
for pid in "${!W[@]}"; do
  g=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
  c=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  if [ "$g" != "${W[$pid]}" ] || [[ "$c" != *folder.py* ]]; then echo "REFUSING: pid $pid is not the expected folder.py on GPU ${W[$pid]} (gpu=$g)"; exit 1; fi
  if [ -n "$(pgrep -P $pid)" ]; then echo "NOTE: pid $pid has child processes: $(pgrep -P $pid | tr '\n' ' ')"; fi
done
for pid in "${!W[@]}"; do kill -CONT $pid && echo "resumed $pid (gpu ${W[$pid]})"; done
