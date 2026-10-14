#!/usr/bin/env python
"""Keep a GPU visibly occupied between our inference runs, so a shared-box neighbour does not
launch into a card we are actively using.

WHY THIS EXISTS. Our batch runs re-invoke runner/inference.py once per work unit. Each invocation
exits and frees its memory, and the next starts ~2 minutes later after template building and PXMeter
scoring -- both CPU-only. So for roughly 30% of the wall clock our GPUs show 0 MiB and 0% util and
look completely free. On 2026-09-10 a neighbour reasonably read that as idle hardware, took GPUs
0/1/5/6 with 73.5 GB jobs, and our next inference died with
    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.39 GiB
The workers did not notice, marked those donors done, and kept going -- so donors were consumed with
no data produced.

WHAT THIS DOES AND DOES NOT DO. It holds a SMALL block (default 10 GiB of ~97 GiB) for the lifetime
of the run. That is deliberately not enough to stop anyone -- reserving our real 25-45 GiB working set
would be antisocial, and worse, would make our own inference OOM when it starts. The protection is a
SIGNAL: `nvidia-smi` shows an ebrahim process on the card, which is what a neighbour actually checks.

THE PROPER FIX is a persistent worker that loads the checkpoint once and loops over donors in-process,
so the real job holds its own memory continuously (and skips ~1 min of model load per unit). This is
a stopgap until that rewrite.

Exits when the stop file appears, so a run can release its cards cleanly.

Usage: CUDA_VISIBLE_DEVICES=2 python gpu_hold.py --gib 0.5 --stop /path/to/STOP

SIZE THE HOLD AGAINST THE JOB. The 10 GiB default suits inference with a 25-45 GiB working set. A
backprop job that peaks near 90 GiB of 94.97 has no 10 GiB to spare, and the holder would OOM the
very run it protects -- pass --gib 0.5 there. The hold is a SIGNAL for `nvidia-smi`, not a reservation,
so its size only has to be nonzero.
"""
import argparse
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=10.0)
    ap.add_argument("--stop", default="")
    ap.add_argument("--poll", type=int, default=20)
    A = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    # fp16 so the element count stays well inside int64, and the allocation is a single block
    n = int(A.gib * (1024 ** 3) // 2)
    buf = torch.empty(n, dtype=torch.float16, device="cuda:0")
    buf.fill_(0)
    name = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[gpu_hold] holding {A.gib:.1f} GiB on CUDA_VISIBLE_DEVICES={name} "
          f"(pid {os.getpid()}); stop file: {A.stop or 'none'}", flush=True)
    try:
        while True:
            if A.stop and os.path.exists(A.stop):
                print("[gpu_hold] stop file seen, releasing", flush=True)
                break
            time.sleep(A.poll)
    except KeyboardInterrupt:
        pass
    finally:
        del buf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
