#!/bin/bash
# Block until <rid>'s held-out metrics gain a row beyond <have>, then exit, so the session can be
# re-invoked and SEND the plot -- which the committing watcher does not do.
#
#   ./shell_scripts/wait_epoch.sh <rid> <have> <pid> [pid...]
#
# The pids are the TRAINING ranks, passed explicitly. Never derived by pattern: pgrep/pkill match
# the shell running them and self-killed this session twice on 2026-10-18. An empty $P would make
# /proc/$P/stat collapse to /proc/stat, so each is tested for non-emptiness first.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
RID="${1:?rid}"; HAVE="${2:?have}"; shift 2
PIDS=("$@")
[ "${#PIDS[@]}" -gt 0 ] || { echo "REFUSING: no training pids given"; exit 1; }
# COUNT THE EVAL SHARD DIRECTORIES, NOT rank_rho_b30_v*.csv. The csv is written by
# run_logging/rank_rho.py, which only runs when a plot is generated -- i.e. when the SESSION asks
# for it. Waiting on the csv is therefore circular: it cannot grow until the thing it is supposed
# to wake up has already run. train_eval_shards_ep<N>/ is written by the training loop itself at
# every epoch boundary, so it is the signal that actually fires on its own.
rows() {
  ls -d "training/runs/$RID"/train_eval_shards_ep*/ 2>/dev/null | wc -l
}
for i in $(seq 1 1400); do   # ~7h ceiling at 18s
  n=$(rows)
  if [ "$n" -gt "$HAVE" ] 2>/dev/null; then echo "EPOCH_READY rid=$RID rows=$n"; exit 0; fi
  alive=0
  for P in "${PIDS[@]}"; do [ -n "$P" ] && [ -d "/proc/$P" ] && alive=1; done
  [ "$alive" = "0" ] && { echo "RUN_GONE rid=$RID rows=$n"; exit 0; }
  sleep 18
done
echo "TIMEOUT rid=$RID rows=$(rows)"
