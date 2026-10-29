#!/usr/bin/env python
"""Correctness gate for the row-weighted PAE term. CPU only -- no GPU, safe while arms are running.

THE TWO CLAIMS THAT MATTER, because the whole change rests on them:

  1. t_i IS ipTM BEFORE THE MAX. max_i row_t(...) must equal Protenix's own calculate_iptm on the
     same logits. If it does not, the weighting is ranking rows by a quantity selection does not
     read, and the premise of the change is false.

  2. frac 1.0 IS THE UNWEIGHTED TERM, bit for bit. PAELoss reduces as
     sum(loss * pair_mask) / sum(pair_mask); the weighting rides that pair mask, so a uniform
     weight must be a no-op. This is the control arm: any arm can be re-run at frac 1.0 and must
     reproduce the unweighted number exactly.

Usage:  python -I run_logging/test_pae_row_weight.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "phases"))
import pae_row_weight as rw  # noqa: E402
from protenix.model.sample_confidence import calculate_iptm  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    torch.manual_seed(0)
    N_TOK, N_BINS = 120, 64
    # two chains, as an antibody-antigen crop has
    asym = torch.cat([torch.zeros(70, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    has_frame = torch.ones(N_TOK, dtype=torch.bool)
    has_frame[5] = has_frame[88] = False          # a couple of frameless tokens, as real crops have
    logits = torch.randn(1, N_TOK, N_TOK, N_BINS) * 2.0
    n = int(has_frame.sum())

    # ---- claim 1: t_i is ipTM before the max ------------------------------------------------
    diff_chain = (asym[None, :] != asym[:, None])[has_frame].float()   # [N_frame, N_token]
    t = rw.row_t(logits, has_frame, diff_chain)
    ref = float(calculate_iptm(pae_prob=torch.softmax(logits.float(), dim=-1),
                               has_frame=has_frame, asym_id=asym,
                               min_bin=0.0, max_bin=32.0, no_bins=N_BINS).reshape(-1)[0])
    got = float(t.max())
    check("max(row_t) == calculate_iptm", abs(got - ref) < 1e-5,
          f"row_t {got:.8f} vs protenix {ref:.8f}  (|d| {abs(got-ref):.2e})")
    check("row_t returns one value per FRAME row", tuple(t.shape) == (n,), f"{tuple(t.shape)} vs ({n},)")

    # ---- claim 2: frac 1.0 is uniform, hence an exact no-op ---------------------------------
    w1, k1 = rw.row_weights(t, frac=1.0, mass=0.8, floor=10)
    check("frac 1.0 -> None (uniform, exact no-op)", w1 is None, f"k={k1}, n={n}")
    check("frac 1.0 ESS == n", rw.effective_sample_size(w1, n) == float(n))

    # the reduction PAELoss performs, with and without a uniform weight
    ce = torch.rand(n, N_TOK) * 4.0
    pm = diff_chain.clone()
    base = float((ce * pm).sum() / pm.sum())
    for w in (None, torch.ones(n)):
        pmw = pm if w is None else pm * w.reshape(-1, 1)
        val = float((ce * pmw).sum() / pmw.sum())
        check(f"reduction unchanged by {'None' if w is None else 'all-ones'} weight",
              val == base, f"{val!r} vs {base!r}")

    # ---- the weighting does what the flag says ---------------------------------------------
    w, k = rw.row_weights(t, frac=0.2, mass=0.8, floor=10)
    top = torch.topk(t, k).indices
    bot = torch.tensor([i for i in range(n) if i not in set(top.tolist())])
    check("k == ceil(0.2n)", k == -(-n // 5), f"k={k}, n={n}")
    check("mean weight is 1 (term keeps its scale)", abs(float(w.mean()) - 1.0) < 1e-6,
          f"mean {float(w.mean()):.8f}")
    ratio = float(w[top[0]] / w[bot[0]])
    want = (0.8 * (n - k)) / (0.2 * k)
    check("top:bottom ratio matches mass and frac", abs(ratio - want) < 1e-4,
          f"{ratio:.3f} vs {want:.3f}")
    check("the top-k rows are the HIGHEST t_i", bool((t[top].min() >= t[bot].max()).item()))
    check("no row is zeroed (bottom keeps pull toward truth)", bool((w > 0).all().item()),
          f"min weight {float(w.min()):.4f}")
    check("mass lands on the top k", abs(float(w[top].sum() / w.sum()) - 0.8) < 1e-6,
          f"{float(w[top].sum()/w.sum()):.6f}")
    ess, pess = rw.effective_sample_size(w, n), rw.predicted_ess(n, 0.2, 0.8, 10)
    check("effective_sample_size matches predicted_ess", abs(ess - pess) < 1e-3,
          f"{ess:.2f} vs {pess:.2f} of {n}")

    # ---- the floor protects small structures ------------------------------------------------
    t_small = torch.rand(29)                      # our smallest structure
    w_s, k_s = rw.row_weights(t_small, frac=0.2, mass=0.8, floor=10)
    check("floor raises k on a 29-row structure", k_s == 10, f"k={k_s} (0.2*29 would be 6)")
    t_tiny = torch.rand(8)                        # fewer rows than the floor
    w_t, k_t = rw.row_weights(t_tiny, frac=0.2, mass=0.8, floor=10)
    check("floor above n -> uniform, not a crash", w_t is None and k_t == 8, f"k={k_t}")
    check("n == 0 handled", rw.row_weights(torch.zeros(0), 0.2, 0.8, 10) == (None, 0))
    check("row_t with no frames returns None",
          rw.row_t(logits, torch.zeros(N_TOK, dtype=torch.bool), None) is None)

    # ---- what it costs ----------------------------------------------------------------------
    big = torch.randn(1, 332, 332, N_BINS)
    hf_big = torch.ones(332, dtype=torch.bool)
    m_big = torch.ones(332, 332)
    for _ in range(3):
        rw.row_t(big, hf_big, m_big)
    t0 = time.perf_counter()
    for _ in range(20):
        tt = rw.row_t(big, hf_big, m_big)
        rw.row_weights(tt, 0.2, 0.8, 10)
    dt = (time.perf_counter() - t0) / 20
    print(f"\n  CPU cost at 332 rows: {dt*1000:.2f} ms/pose  "
          f"(a pose takes ~580 ms end to end on GPU; this runs on GPU in the real path, "
          f"where it is memory-bandwidth bound on ~28 MB)")

    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
