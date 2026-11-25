#!/bin/bash
# Keep a GPU visibly occupied between inference runs, so a neighbour on this shared box does not
# launch into a card we are using.
#
# WHY. A batch run re-invokes inference once per work unit. Each invocation exits and frees its
# memory, and the next starts ~2 minutes later after template building and scoring -- both CPU-only.
# So for roughly 30% of the wall clock our GPUs read 0 MiB and 0% util and look free. On 2026-09-10 a
# neighbour reasonably read that as idle, took four cards with 73.5 GB jobs, and our next inference
# died with CUDA out of memory. The workers did not notice: they marked those units done and kept
# going, so donors were consumed with no data produced.
#
# WHAT IT DOES NOT DO. It holds a SMALL block -- 10 GiB of ~97 -- for the life of the run. That is
# deliberately not enough to stop anyone. Reserving our real 25-45 GiB working set would be
# antisocial, and would make our own inference OOM when it starts. The protection is a SIGNAL:
# nvidia-smi shows our process on the card, which is what a neighbour actually checks.
#
# SIZE IT AGAINST THE JOB. The 10 GiB default suits inference with a 25-45 GiB working set. A
# backprop job peaking near 90 GiB has no 10 GiB to spare and the holder would OOM the very run it
# protects -- pass --gib 0.5 there. Only nonzero matters.
#
# Shell rather than Python because it lives beside the other watchers; the allocation itself needs
# torch, so that part is inline.
#
#   CUDA_VISIBLE_DEVICES=0 gpu_hold.sh [--gib 10] [--stop /path/to/STOP] [--poll 20]
set -u
PY=${PY:-/home/ebrahim/Protenix/.venv/bin/python}
GIB=10; STOP=""; POLL=20
while [ $# -gt 0 ]; do
    case "$1" in
        --gib)  GIB=$2; shift 2 ;;
        --stop) STOP=$2; shift 2 ;;
        --poll) POLL=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "set CUDA_VISIBLE_DEVICES first" >&2; exit 2; }

GIB="$GIB" STOP="$STOP" POLL="$POLL" exec "$PY" - <<'PYEOF'
import os, time, torch
gib, stop, poll = float(os.environ["GIB"]), os.environ["STOP"], int(os.environ["POLL"])
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible")
# fp16 so the element count stays inside int64 and the allocation is one block
buf = torch.empty(int(gib * (1024 ** 3) // 2), dtype=torch.float16, device="cuda:0")
buf.fill_(0)
print(f"[gpu_hold] holding {gib:.1f} GiB on CUDA_VISIBLE_DEVICES="
      f"{os.environ.get('CUDA_VISIBLE_DEVICES','?')} (pid {os.getpid()}); "
      f"stop file: {stop or 'none'}", flush=True)
try:
    while not (stop and os.path.exists(stop)):
        time.sleep(poll)
    print("[gpu_hold] stop file seen, releasing", flush=True)
except KeyboardInterrupt:
    pass
finally:
    del buf
    torch.cuda.empty_cache()
PYEOF
