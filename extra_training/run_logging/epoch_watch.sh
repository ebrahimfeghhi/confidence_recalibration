#!/usr/bin/env bash
# Wait for the next epoch of an arm, refresh the figures, and push the results.
#
# ONE SCRIPT INSTEAD OF A RE-TYPED LOOP. The waiter was being rebuilt by hand at every epoch
# boundary, which is how the push step would eventually get forgotten.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
RID="${1:?usage: epoch_watch.sh <rid>}"
N=$(ls "training/runs/$RID"/head_ep*.pt 2>/dev/null | wc -l)
while [ "$(ls "training/runs/$RID"/head_ep*.pt 2>/dev/null | wc -l)" -le "$N" ]; do sleep 60; done
E=$(ls "training/runs/$RID"/head_ep*.pt | sed -E 's/.*head_ep([0-9]+)_.*/\1/' | sort -n | tail -1)
echo "$RID EPOCH $E $(TZ=America/New_York date '+%-I:%M %p %Z')"

# On a multiple of 5 the held-out eval runs after the checkpoint; wait for its json so the
# pushed figure and table carry the eval rather than showing the epoch without it.
if [ $(( E % 5 )) -eq 0 ]; then
  for _ in $(seq 1 180); do
    [ -s "training/runs/$RID/metric_valfull83_ep${E}.json" ] && {
      echo "eval json present $(TZ=America/New_York date '+%-I:%M %p %Z')"; break; }
    sleep 60
  done
fi

./run_logging/refresh_plots.sh "$RID" figures 2>&1 | tail -3
# the comparison against the 15-epoch restricted arm, kept current
/home/ebrahim/Protenix/.venv/bin/python -I run_logging/plot_rel_arm.py \
  peae97,"$RID" figures/fullhead_vs_restricted.png >/dev/null 2>&1 || true
grep -aoE "epoch ${E} took [0-9.]+ min" "training/runs/$RID/run.log" | head -1
./run_logging/push_epoch.sh "$RID" "$E"
