#!/usr/bin/env python
"""Average several epoch heads into one, elementwise, for a checkpoint-averaging test.

    python -I run_logging/average_heads.py --out <path.pt> <head1.pt> <head2.pt> ...

WHY. The val curve moves 0.005-0.015 between adjacent epochs while training improves smoothly, so
much of what we read per epoch is the selector re-picking among near-tied poses rather than the
head getting better. Averaging weights from nearby epochs is a temporal low-pass that costs one
eval instead of a training run: if the averaged head scores at or above the best of its members,
the per-epoch movement was noise around a better point than any single epoch showed.

IT IS NOT AN ENSEMBLE. One head goes to inference, so nothing about deployment changes. Averaging
weights only makes sense for checkpoints in the same basin -- consecutive epochs of one run, not
different arms or distant epochs -- and this refuses inputs that do not share a parameter set.

PICKING THE MEMBERS BY THEIR VAL SCORE IS SELECTION ON THE TEST SET. Averaging the three epochs
that happened to score best will look good for the same reason the best single epoch did. A
predeclared rule -- "the last three", "every consecutive triple" -- is what makes the number
mean something. This script does not choose; it averages what it is given, and the caller owns
that decision.

FLOAT64 ACCUMULATION, then cast back to each tensor's own dtype. The heads are bf16/fp32 mixes
and summing 3-5 bf16 tensors in bf16 loses bits that the average then cannot recover.
"""
import argparse
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("heads", nargs="+")
a = ap.parse_args()

if len(a.heads) < 2:
    sys.exit("STOP: averaging needs at least two heads.")

mats = []
for p in a.heads:
    if not os.path.exists(p):
        sys.exit(f"STOP: {p} does not exist.")
    d = torch.load(p, map_location="cpu", weights_only=False)
    # A HEAD FILE IS THE TENSOR DICT ITSELF in this project (head_ep*.pt), while a resume bundle
    # wraps it. Accept either rather than making the caller remember which.
    if isinstance(d, dict) and "head" in d and isinstance(d["head"], dict):
        d = d["head"]
    if not isinstance(d, dict):
        sys.exit(f"STOP: {p} is a {type(d).__name__}, not a state dict.")
    mats.append((p, d))

keys = set(mats[0][1])
for p, d in mats[1:]:
    if set(d) != keys:
        only_a = sorted(keys - set(d))[:4]
        only_b = sorted(set(d) - keys)[:4]
        sys.exit(f"STOP: {p} has a different parameter set from {mats[0][0]}.\n"
                 f"  only in the first: {only_a}\n  only in {os.path.basename(p)}: {only_b}\n"
                 f"  Averaging heads from different architectures is not defined.")

out = {}
n = float(len(mats))
for k in sorted(keys):
    ref = mats[0][1][k]
    if not torch.is_tensor(ref):
        # Non-tensor entries must AGREE, not be averaged: these are shapes, dtypes and flags, and
        # a mismatch means the heads are not the same function.
        vals = {repr(d[k]) for _p, d in mats}
        if len(vals) != 1:
            sys.exit(f"STOP: non-tensor entry {k!r} differs across heads: {sorted(vals)[:3]}")
        out[k] = ref
        continue
    for p, d in mats[1:]:
        if d[k].shape != ref.shape:
            sys.exit(f"STOP: {k} is {tuple(ref.shape)} in {mats[0][0]} and "
                     f"{tuple(d[k].shape)} in {p}.")
    acc = torch.zeros(ref.shape, dtype=torch.float64)
    for _p, d in mats:
        acc += d[k].detach().to(torch.float64)
    out[k] = (acc / n).to(ref.dtype)

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
torch.save(out, a.out)
print(f"averaged {len(mats)} head(s) over {len(keys)} key(s) -> {a.out}")
for p, _d in mats:
    print(f"  {p}")
# A SANITY NUMBER, so a silently-identical or silently-empty average is visible: the mean absolute
# difference between the average and its first member, over the trainable-looking float tensors.
_d0 = mats[0][1]
_deltas = [float((out[k].to(torch.float64) - _d0[k].to(torch.float64)).abs().mean())
           for k in sorted(keys)
           if torch.is_tensor(_d0[k]) and _d0[k].is_floating_point() and _d0[k].numel() > 1]
if _deltas:
    _m = sum(_deltas) / len(_deltas)
    print(f"  mean |avg - first| over {len(_deltas)} float tensor(s): {_m:.3e}")
    if _m == 0.0:
        sys.exit("STOP: the average equals its first member exactly; the inputs are identical.")
