#!/usr/bin/env python
"""Correctness gate for the row-weighted PAE term. CPU only -- safe to run while arms train.

THE FOUR CLAIMS THAT MATTER:

  1. A WEIGHTED MASK IS NOT THE UNWEIGHTED ONE. First, because the first version of this change
     cast the weights to the bool dtype of PAELoss's own pair_mask, so every positive weight became
     True, the term was an exact no-op, and the run would have logged itself as row-weighted for 60
     epochs while training a plain unweighted arm. The earlier gate only asserted the no-op
     DIRECTION -- that a uniform weight changes nothing -- which cannot catch a total no-op. This
     now runs the real PAELoss.calculate_label to get a real bool mask.

  2. t_i IS ipTM BEFORE THE MAX, per chain pair. max_i t_i over a pair's block must equal
     Protenix's own chain_pair_iptm for that pair -- same token_mask, same d0 over the two chains.
     Checked on a 3-chain crop, which is what 298 of our 368 training structures are.

  3. frac 1.0 IS THE UNWEIGHTED TERM. Uniform weights return None so the caller keeps its bool
     mask, making --pae-row-frac 1.0 an exact identity. That is the matched control arm.

  4. WEIGHT IS NOT SPENT WHERE THE LOSS CANNOT SEE IT. Rows with no scored targets get 0, so the
     realised top-set share is the requested mass rather than less.

Usage:  python -I run_logging/test_pae_row_weight.py
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import pae_row_weight as rw  # noqa: E402
from protenix.model.loss import PAELoss  # noqa: E402
from protenix.model.sample_confidence import calculate_iptm  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def real_pair_mask(n_tok):
    """A genuine bool pair_mask straight out of PAELoss.calculate_label.

    THE POINT IS THE DTYPE. The previous gate hand-rolled a float mask, which is exactly the
    difference that hid the bool-cast no-op.
    """
    pl = PAELoss()
    torch.manual_seed(1)
    true_c = torch.randn(n_tok, 3) * 12
    pred_c = true_c + torch.randn(n_tok, 3) * 0.6
    hf = torch.ones(n_tok, dtype=torch.bool)
    fai = torch.stack([torch.arange(n_tok), (torch.arange(n_tok) + 1) % n_tok,
                       (torch.arange(n_tok) + 2) % n_tok], -1)
    _, bins, pm = pl.calculate_label(
        pred_coordinate=pred_c.unsqueeze(0), true_coordinate=true_c,
        coordinate_mask=torch.ones(n_tok), rep_atom_mask=torch.ones(n_tok, dtype=torch.bool),
        frame_atom_index=fai, has_frame=hf)
    return pl, pm, bins, hf


def main():
    torch.manual_seed(0)
    N_BINS = 64
    # three chains: heavy / light / antigen, the shape 298 of 368 training structures have
    sizes = (70, 60, 110)
    asym = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])
    n_tok = int(asym.numel())
    has_frame = torch.ones(n_tok, dtype=torch.bool)
    has_frame[7] = has_frame[150] = False
    logits = torch.randn(1, n_tok, n_tok, N_BINS) * 2.0
    prob = torch.softmax(logits.float(), dim=-1)
    aa = asym
    fa = aa[has_frame]

    def block(x, y):
        return (((fa[:, None] == x) & (aa[None, :] == y))
                | ((fa[:, None] == y) & (aa[None, :] == x)))

    # ---- claim 2: per-pair t_i reproduces chain_pair_iptm -----------------------------------
    for x, y in ((0, 2), (1, 2), (0, 1)):
        tmask = (aa == x) | (aa == y)
        ref = float(calculate_iptm(pae_prob=prob, has_frame=has_frame, asym_id=asym,
                                   min_bin=0.0, max_bin=32.0, no_bins=N_BINS,
                                   token_mask=tmask).reshape(-1)[0])
        t, cnt = rw.row_t(logits, has_frame, block(x, y), int(tmask.sum()))
        got = float(t[cnt > 0].max())
        check(f"max t_i over block ({x},{y}) == chain_pair_iptm", abs(got - ref) < 1e-5,
              f"{got:.8f} vs {ref:.8f}  (|d| {abs(got-ref):.2e})")

    # the union, which the previous version ranked on, does NOT reproduce either pair's ipTM
    union = block(0, 2) | block(1, 2)
    t_u, c_u = rw.row_t(logits, has_frame, union, None)
    t_p, c_p = rw.row_t(logits, has_frame, block(0, 2), int(((aa == 0) | (aa == 2)).sum()))
    agree = int((torch.argsort(-t_u)[:20][:, None] == torch.argsort(-t_p)[:20][None, :]).any(-1).sum())
    check("union ranking DISAGREES with the per-pair ranking (why per-pair exists)", agree < 20,
          f"top-20 overlap {agree}/20")

    # ---- claim 1: a weighted mask differs from the unweighted one, through the REAL PAELoss ----
    pl, pm_real, bins_real, hf_real = real_pair_mask(n_tok)
    check("PAELoss.calculate_label returns a BOOL mask (the trap)", pm_real.dtype == torch.bool,
          f"{pm_real.dtype}")
    blocks = [(block(0, 2), int(((aa == 0) | (aa == 2)).sum())),
              (block(1, 2), int(((aa == 1) | (aa == 2)).sum()))]
    wm = rw.weighted_mask(logits, has_frame, blocks, 0.2, 0.8, 10)
    check("weighted_mask is float", wm is not None and wm.dtype == torch.float32,
          f"{None if wm is None else wm.dtype}")
    pm_u = pm_real[has_frame].to(torch.float32) * (union.to(torch.float32))
    pm_w = pm_real[has_frame].to(torch.float32) * wm
    ce = torch.rand(pm_w.shape) * 4.0
    lo = float((ce * pm_u).sum() / pm_u.sum())
    lw = float((ce * pm_w).sum() / pm_w.sum())
    check("the weighted loss DIFFERS from the unweighted loss", abs(lw - lo) > 1e-6,
          f"unweighted {lo:.8f} vs weighted {lw:.8f}  (d {lw-lo:+.2e})")
    bad = pm_real[has_frame] * wm.to(torch.bool)      # the shipped bug, reproduced
    check("the bool cast WOULD have been a no-op (regression guard)",
          bool((bad.to(torch.float32) == pm_u).all()),
          "bool(w) collapses every weight to True")

    # ---- claim 3: frac 1.0 is an exact identity ----------------------------------------------
    for n in (3, 7, 29, 332, 801):
        w, k = rw.row_weights(torch.rand(n), None, 1.0, 0.8, 10)
        if w is not None:
            check(f"frac 1.0 uniform at n={n}", False, f"got weights, k={k}")
    check("frac 1.0 -> None at every n tried", True)
    wm1 = rw.weighted_mask(logits, has_frame, blocks, 1.0, 0.8, 10)
    check("weighted_mask at frac 1.0 -> None (caller keeps its bool mask)", wm1 is None)

    # ---- claim 4: no weight on rows the loss cannot see ---------------------------------------
    b = block(0, 2)
    t, cnt = rw.row_t(logits, has_frame, b, int(((aa == 0) | (aa == 2)).sum()))
    w, k = rw.row_weights(t, cnt > 0, 0.2, 0.8, 10)
    nz = int((cnt > 0).sum())
    check("rows with no scored target get weight 0",
          bool((w[cnt == 0] == 0).all()), f"{int((cnt==0).sum())} such rows of {w.numel()}")
    check("k is taken over SCORED rows, not all rows", k == max(10, -(-nz // 5)),
          f"k={k}, scored rows={nz}, all rows={w.numel()}")
    top = torch.argsort(-torch.where(cnt > 0, t, torch.full_like(t, float('-inf'))),
                        stable=True)[:k]
    check("the requested mass lands on the top k", abs(float(w[top].sum() / w.sum()) - 0.8) < 1e-5,
          f"{float(w[top].sum()/w.sum()):.6f}")
    check("mean weight over scored rows is 1", abs(float(w[cnt > 0].mean()) - 1.0) < 1e-5,
          f"{float(w[cnt>0].mean()):.8f}")
    check("no scored row is zeroed", bool((w[cnt > 0] > 0).all()))

    # ---- determinism and hostile inputs -------------------------------------------------------
    tied = torch.zeros(40)
    w1, _ = rw.row_weights(tied, None, 0.2, 0.8, 4)
    w2, _ = rw.row_weights(tied.clone(), None, 0.2, 0.8, 4)
    check("tied t_i gives the SAME row set twice (3 ranks must agree)",
          bool((w1 == w2).all()) and bool((w1[:4] == w1.max()).all()),
          "stable sort, lowest index wins")
    nant = torch.rand(40)
    nant[3] = float("nan")
    wn, _ = rw.row_weights(nant, None, 0.2, 0.8, 4)
    check("a NaN row does NOT capture the top weight", float(wn[3]) < float(wn.max()),
          f"nan row weight {float(wn[3]):.4f} vs max {float(wn.max()):.4f}")
    check("empty t handled", rw.row_weights(torch.zeros(0), None, 0.2, 0.8, 10) == (None, 0))
    check("no frames -> None", rw.row_t(logits, torch.zeros(n_tok, dtype=torch.bool), None)[0] is None)
    check("no blocks -> None", rw.weighted_mask(logits, has_frame, [], 0.2, 0.8, 10) is None)
    check("ESS of the realised mask is below the row count",
          0 < rw.effective_sample_size(pm_w) < pm_w.shape[0],
          f"{rw.effective_sample_size(pm_w):.1f} of {pm_w.shape[0]} rows")

    # ---- memory and cost ----------------------------------------------------------------------
    big = torch.randn(1, 1473, 1473, N_BINS)          # the largest crop in our logs
    hf_b = torch.ones(1473, dtype=torch.bool)
    blk_b = torch.ones(1473, 1473, dtype=torch.bool)
    peak_full = 1473 * 1473 * N_BINS * 4 / 2**20
    peak_chunk = rw.ROW_CHUNK * 1473 * N_BINS * 4 / 2**20
    t0 = time.perf_counter()
    rw.row_t(big, hf_b, blk_b, 1473)
    dt = time.perf_counter() - t0
    print(f"\n  largest crop (1,473 tokens): chunked peak ~{peak_chunk:.0f} MB per chunk "
          f"against ~{peak_full:.0f} MB for a full softmax; {dt*1000:.0f} ms on CPU")

    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
