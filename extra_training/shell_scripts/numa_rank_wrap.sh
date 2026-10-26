#!/bin/bash
# Bind ONE torchrun rank to ONE NUMA node, then exec the real command.
#
# torchrun --no-python launches this once per rank in place of python, so each rank reaches
# numactl with its own LOCAL_RANK already set and can be sent to a different socket:
#
#   torchrun --nproc_per_node=3 --no-python numa_rank_wrap.sh <python> -u <train.py> ...
#   NUMA_RANK_MAP=0:0,1:0,2:1
#
# WHY PER RANK AND NOT ONE numactl AROUND torchrun. Wrapping torchrun binds all three ranks to the
# same socket, which fixes the mismatch but makes three processes share one socket's memory
# bandwidth. The bottleneck IS memory bandwidth -- cpucrop_slice runs 16.5 min on the fast ranks
# against 14.6 min of GPU work, so the CPU is the critical path for every rank -- so splitting the
# ranks across both sockets, each with its own memory local, gives us more of the scarce thing.
#
# WHAT IT FIXES, measured on dntxmn at epoch 15. The ranks were unpinned (taskset reported 0-287)
# so the kernel scattered them and memory followed first touch:
#
#   rank 0  cores socket 0, 74.6% of pages on socket 0   cpucrop_slice 16.5 min
#   rank 1  cores socket 0, 88.0% of pages on socket 0   cpucrop_slice 16.7 min
#   rank 2  cores socket 1, 41.8% of pages on socket 0   cpucrop_slice 27.5 min   <-- mismatched
#
# Cross-socket distance on this box is 32 against 10 local. Rank 2 was the only rank reaching
# across for a large share of its data, and it ran identical work -- all three ranks report the
# same 14,912,212 kept tokens and the same 14.6 min of GPU time -- 11 minutes slower. The barrier
# then converted that into 9.7 min of idle on the two fast ranks. Because nothing was pinned, WHICH
# rank suffered changed between epochs, which is why the straggler moved from rank 0 early in the
# run to rank 2 after epoch 10 and why the cost ranged 1.7-9.7 min.
#
# THE PREFETCH WORKER IS THE REAL TARGET. cpucrop_slice happens in each rank's loader worker, a
# child process. Children inherit both CPU affinity and memory policy, so binding the rank at
# startup binds the worker with it. Binding anything later -- inside the trainer, after the worker
# is spawned -- would pin the wrong process.
#
# --membind IS DELIBERATELY HARD, not --preferred. A soft policy may silently fall back to the
# other socket, and then a tight spread across ranks would not tell us the binding was what did
# it. Each socket has 515 GB and the run's cap is --max-rss-gb 80, so there is room; if an
# allocation ever fails under this, --preferred is the softer fallback and the failure will say so
# rather than being absorbed.
set -u

MAP="${NUMA_RANK_MAP:?numa_rank_wrap.sh: NUMA_RANK_MAP is not set}"
LR="${LOCAL_RANK:?numa_rank_wrap.sh: LOCAL_RANK is not set (run me under torchrun --no-python)}"
[ "$#" -ge 1 ] || { echo "numa_rank_wrap.sh: nothing to exec" >&2; exit 2; }

# REFUSE RATHER THAN RUN UNPINNED. A rank with no entry in the map would otherwise run with the
# default policy while the record says the run was pinned, which is the quiet failure this whole
# change exists to remove.
NODE=""
OLD_IFS="$IFS"; IFS=','
for pair in $MAP; do
  case "$pair" in
    "$LR":*) NODE="${pair#*:}" ;;
  esac
done
IFS="$OLD_IFS"
if [ -z "$NODE" ]; then
  echo "numa_rank_wrap.sh: LOCAL_RANK=$LR has no entry in NUMA_RANK_MAP='$MAP'" >&2
  exit 2
fi
case "$NODE" in
  ''|*[!0-9]*) echo "numa_rank_wrap.sh: node '$NODE' for rank $LR is not a number" >&2; exit 2 ;;
esac
if [ ! -d "/sys/devices/system/node/node$NODE" ]; then
  echo "numa_rank_wrap.sh: NUMA node $NODE does not exist on this machine" >&2
  exit 2
fi
command -v numactl >/dev/null 2>&1 || {
  echo "numa_rank_wrap.sh: numactl not found" >&2; exit 2; }

echo "[rank $LR] NUMA: cores and memory bound to socket $NODE (map $MAP)" >&2
exec numactl --cpunodebind="$NODE" --membind="$NODE" "$@"
