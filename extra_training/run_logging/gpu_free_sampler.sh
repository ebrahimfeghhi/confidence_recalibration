#!/bin/bash
# MINIMUM FREE DEVICE MEMORY on cards 4,5,7, sampled until the marker file disappears.
#
# WHY NOT total - torch_reserved: torch counts only its own allocator. The CUDA context, the
# project's GPU holder and any other resident process are invisible to it, so that arithmetic
# overstates what is actually available. nvidia-smi's free column is the device's own view.
#
# Usage:  gpu_free_sampler.sh <marker-file> <out-csv> [interval_s]
#         (sampling stops when <marker-file> is removed)
set -u
MARK="$1"; OUT="$2"; IV="${3:-5}"
echo "t,card4_free_mib,card5_free_mib,card7_free_mib" > "$OUT"
while [ -e "$MARK" ]; do
  R=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' '$1==4{a=$2} $1==5{b=$2} $1==7{c=$2} END{print a","b","c}')
  echo "$(date +%s),$R" >> "$OUT"
  sleep "$IV"
done
