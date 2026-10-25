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
rows() {
  local f
  f=$(ls -1 "training/runs/$RID"/rank_rho_b30_v*.csv 2>/dev/null | sort -V | tail -1)
  [ -n "$f" ] && [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0
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
