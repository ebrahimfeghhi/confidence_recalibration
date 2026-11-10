#!/usr/bin/env python
"""Numerical gates for crop_itm_rank.itm_over_subsets, against a slow sliced reference.

The sgemm formulation evaluates 1000 crops at once and is not inspectable by eye -- the first
version pooled frame rows whose own token the crop had removed, which inflates the overlap between
cropped and uncropped iTM and biases the whole analysis toward "cropping is harmless". It was
found in external review, not here, so these gates exist to make the next one fail loudly.

    python -I run_logging/test_crop_itm_rank.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_itm_rank import itm_over_subsets          # noqa: E402


def reference(W, M, crop, row_tokens):
    """Slice first, then average. Unambiguous and obviously correct, at any speed."""
    out = []
    keep_set = set(crop.tolist())
    rows = [i for i, t in enumerate(row_tokens) if t in keep_set]
    for p in range(W.shape[0]):
        acc = []
        for i in rows:
            cols = [j for j in crop if M[i, j] > 0]
            if not cols:
                continue
            acc.append(float(np.mean([W[p, i, j] for j in cols])))
        out.append(float(np.mean(acc)) if acc else np.nan)
    return np.array(out)


def case(name, P, N, crop, seed, drop_rows=0):
    rng = np.random.default_rng(seed)
    W = rng.random((P, N, N)).astype(np.float32)
    M = (rng.random((N, N)) < .5).astype(np.float32)
    np.fill_diagonal(M, 0)
    row_tokens = np.arange(N)
    if drop_rows:                       # mimic has_frame=False on some tokens
        keep_rows = rng.permutation(N)[:N - drop_rows]
        keep_rows.sort()
        W = W[:, keep_rows]
        M = M[keep_rows]
        row_tokens = keep_rows
    keep = np.zeros((N, 1), dtype=np.float32)
    keep[crop] = 1
    got, nlive = itm_over_subsets(W, M, keep, row_tokens)
    ref = reference(W, M, crop, row_tokens)
    if int(nlive[0]) == 0:
        # A crop with no scored row is DEGENERATE, not wrong: the reference agrees by returning
        # nan. Agreement here means "both decline to produce a number", which is the contract.
        ok, d = bool(np.isnan(ref).all()), 0.0
    else:
        d = float(np.nanmax(np.abs(got[:, 0] - ref)))
        ok = d < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} max|diff| {d:.2e}  live rows {int(nlive[0])}")
    return ok


def main():
    print("crop_itm_rank numerical gates")
    good = []
    N = 12
    good.append(case("full mask is the uncropped value", 4, N, np.arange(N), 0))
    good.append(case("crop removes rows AND columns", 4, N, np.arange(6), 0))
    good.append(case("crop keeps one token only", 3, N, np.array([3]), 1))
    good.append(case("scattered crop", 5, N, np.array([0, 2, 5, 7, 11]), 2))
    good.append(case("mixed has_frame (3 rows dropped)", 4, N, np.arange(7), 3, drop_rows=3))
    good.append(case("mixed has_frame, scattered crop", 4, 16, np.array([1, 4, 9, 14]), 4,
                     drop_rows=5))
    good.append(case("larger, realistic shape", 6, 60, np.arange(31), 5, drop_rows=7))

    # A crop that keeps no scored row must report zero live rows rather than a confident 0.0.
    rng = np.random.default_rng(9)
    W = rng.random((2, 8, 8)).astype(np.float32)
    M = np.zeros((8, 8), dtype=np.float32)
    keep = np.zeros((8, 1), dtype=np.float32)
    keep[[0, 1]] = 1
    _v, nlive = itm_over_subsets(W, M, keep, np.arange(8))
    deg = int(nlive[0]) == 0
    print(f"  {'PASS' if deg else 'FAIL'}  {'degenerate crop reports 0 live rows':46s}")
    good.append(deg)

    # The bug that was shipped: pooling a row whose own token was cropped away. Assert the fix
    # actually changes the answer, so a regression cannot pass by accident.
    rng = np.random.default_rng(7)
    N = 14
    W = rng.random((3, N, N)).astype(np.float32)
    M = (rng.random((N, N)) < .6).astype(np.float32)
    np.fill_diagonal(M, 0)
    crop = np.arange(7)
    keep = np.zeros((N, 1), dtype=np.float32)
    keep[crop] = 1
    fixed = itm_over_subsets(W, M, keep, np.arange(N))[0][:, 0]
    WM = (W * M[None]).reshape(3 * N, N)
    num = (WM @ keep).reshape(3, N, -1)
    den = M @ keep
    t = num / np.clip(den, 1.0, None)[None]
    live_old = (den > 0).astype(np.float32)
    buggy = (t * live_old[None]).sum(1)[:, 0] / max(live_old.sum(0)[0], 1.0)
    differs = float(np.abs(fixed - buggy).max()) > 1e-4
    print(f"  {'PASS' if differs else 'FAIL'}  "
          f"{'fix changes the answer vs the old masking':46s} "
          f"max|diff| {float(np.abs(fixed - buggy).max()):.2e}")
    good.append(differs)

    print(f"\n{sum(good)}/{len(good)} gate(s) passed")
    return 0 if all(good) else 1


if __name__ == "__main__":
    sys.exit(main())
