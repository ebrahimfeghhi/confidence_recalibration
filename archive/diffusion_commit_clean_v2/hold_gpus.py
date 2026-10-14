#!/usr/bin/env python
"""Occupy GPUs so they are not taken while a sweep is paused.

SUMMARY
    A shared machine reclaims an idle card within minutes. Allocating a small tensor is enough to show
    up in nvidia-smi as in use. One process per GPU, launched with CUDA_VISIBLE_DEVICES set, so each
    holder is killable on its own and the device is always index 0 inside the process.

INPUTS
    --gpus:    which device this process reports holding, for the log line. Required.
    --gb:      how much to allocate. Default 1.0.
    --minutes: how long to hold. Default 0, meaning until killed.

    CUDA_VISIBLE_DEVICES must be set by the caller: it is what actually selects the card.

OUTPUTS
    an allocation on the visible device, held until the process exits. Kill the tmux session to release.
    Holders must be stopped before real work is relaunched on the same card.

EXAMPLE
    tmux new-session -d -s hold1 "CUDA_VISIBLE_DEVICES=1 hold_gpus.py --gpus 1 --gb 1"
    CUDA_VISIBLE_DEVICES=5 hold_gpus.py --gpus 5 --minutes 30

FUNCTIONS
    hold():  allocate the tensor
    main():  hold the visible device, then wait
"""
import argparse
import time

import torch


def hold(gb=1.0):
    """in: gigabytes to occupy.  out: the tensor, kept alive by the caller."""
    n = int(gb * (1024 ** 3) / 4)
    return torch.zeros(n, dtype=torch.float32, device="cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="comma-separated device ids")
    ap.add_argument("--gb", type=float, default=1.0, help="per GPU")
    ap.add_argument("--minutes", type=float, default=0, help="0 = until killed")
    A = ap.parse_args()

    # ONE PROCESS PER GPU, launched with CUDA_VISIBLE_DEVICES set, so each holder is killable on its
    # own and the device is always index 0 here. A single process calling set_device(g) fails outright
    # when the caller's shell exports an empty CUDA_VISIBLE_DEVICES, which is how this first broke.
    held = [hold(A.gb)]
    print(f"  holding {A.gpus} with {A.gb} GB (device 0 of this process)", flush=True)
    deadline = time.time() + A.minutes * 60 if A.minutes else None
    while deadline is None or time.time() < deadline:
        time.sleep(30)
    print("  released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
