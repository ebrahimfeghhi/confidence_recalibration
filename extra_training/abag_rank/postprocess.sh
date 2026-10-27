#!/bin/bash
# After a PAE dump: merge the per-shard sums, then extract OST's per-pose chain mapping. CPU only.
# Usage: ./abag_rank/postprocess.sh <dump dir> <fold> <pool csv> [workers]
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
DUMP="${1:?dump dir}"; FOLD="${2:?fold}"; POOL="${3:?pool csv}"; W="${4:-8}"
echo "$(TZ=America/New_York date '+%-I:%M %p') merging PAE sums in $DUMP"
$PY -I abag_rank/merge_pae_mean.py "$DUMP" || { echo "merge refused; not continuing"; exit 1; }
echo "$(TZ=America/New_York date '+%-I:%M %p') extracting OST chain maps ($W workers)"
for s in $(seq 0 $((W-1))); do
  LAYERNORM_TYPE=openfold $PY -I abag_rank/chain_map.py --dump "$DUMP" --fold "$FOLD" \
    --pool "$POOL" --shard $s --nshards $W > /tmp/claude-1004/cm_$s.log 2>&1 &
done
wait
grep -h "resolved" /tmp/claude-1004/cm_*.log | awk '{ok+=$3; bad+=$6} END {print "  totals: "ok" resolved, "bad" unresolved"}'
echo "$(TZ=America/New_York date '+%-I:%M %p') postprocess done for $DUMP"
