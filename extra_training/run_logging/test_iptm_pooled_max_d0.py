#!/usr/bin/env python
"""Gate for iptm_pooled's FIXED-d0 MAX pooling, added for the joint reranker (issue #16).

WHY IT EXISTS. The reranker scores a pose as `ipTM + learned correction`, and the baseline has to
be the ipTM that SELECTION deploys -- the max-pooled one. But under --crop-pool the vendor kernel
`calculate_chain_based_ptm` recomputes d0 from the asym_id it is handed, so the same pose scored
at 256 and at 512 tokens would be measured with different TM kernels (at 600 -> 256 pair tokens,
w(10 A) moves 0.423 -> 0.259). iptm_pooled used to REFUSE that combination outright. It now has a
real fixed-d0 max path built from the same differentiable row_t construction as the mean.

THE FOUR CLAIMS THIS PINS, all requested in review before launch:
  1. n_pair_tokens=None leaves the legacy max byte-identical -- every max number this project has
     reported came through iptm(), and this change must not move any of them.
  2. On an UNCROPPED fixture with n_pair_tokens set to the pair's own token count, the new path
     agrees with the legacy one. That is the consistency check: the two differ only in HOW d0 is
     obtained, so where they must agree, they do.
  3. The SAME uncropped d0 is used across crop sizes. Three nested crops of one interface give
     three results that each equal row_t at the PINNED count, and differ from the unpinned
     (crop-derived) value -- which is the bug the old refusal existed to prevent.
  4. The result is differentiable, so the ranking gradient reaches the head.

CPU ONLY, synthetic input: this checks the kernel and the plumbing, not the model.

Usage:  python -I run_logging/test_iptm_pooled_max_d0.py
"""
import os
import sys
import types

import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import pae_row_weight as rw                                      # noqa: E402
import rank_loss as RKL                                          # noqa: E402

FAIL = []
NB = 64


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _configs():
    """Minimal stand-in for the real configs object: only configs.loss.pae is read."""
    pae = types.SimpleNamespace(min_bin=0.0, max_bin=32.0, no_bins=NB)
    return types.SimpleNamespace(loss=types.SimpleNamespace(pae=pae))


def _feat(asym, keep=None):
    """A feat dict with the fields iptm_pooled reads, optionally restricted to a crop."""
    if keep is not None:
        asym = asym[keep]
    n = asym.shape[0]
    return {
        "asym_id": asym.clone(),
        "has_frame": torch.ones(n, dtype=torch.float32),
        "is_ligand": torch.zeros(n, dtype=torch.long),
        # one atom per token keeps the scatter_add ligand derivation trivial and exact
        "atom_to_token_idx": torch.arange(n, dtype=torch.long),
    }


def _row_t_max(logits, feat, npt, pair):
    """row_t's max over scored rows, computed here so the test does not trust iptm_pooled."""
    asym = feat["asym_id"].reshape(-1).long()
    hf = feat["has_frame"].reshape(-1).bool()
    i, j = pair
    tok = (asym == i) | (asym == j)
    arow = asym[hf]
    tm = (tok.unsqueeze(0) & (arow.unsqueeze(1) != asym.unsqueeze(0))
          & ((arow == i) | (arow == j)).unsqueeze(1))
    t, cnt = rw.row_t(logits, hf, tm, n_pair_tokens=npt, min_bin=0.0, max_bin=32.0)
    return float(t[cnt > 0].max())


def main():
    torch.manual_seed(0)
    N = 60
    asym = torch.cat([torch.zeros(25), torch.ones(20), torch.full((15,), 2.0)]).long()
    logits = torch.randn(1, N, N, NB) * 2.0
    cfg, pair = _configs(), (0, 1)
    feat_full = _feat(asym)
    full_npt = int(((asym == 0) | (asym == 1)).sum())

    # 1. legacy path untouched
    legacy = RKL.iptm_pooled(logits, feat_full, cfg, pair, pool="max", n_pair_tokens=None)
    direct = RKL.iptm(logits, feat_full, cfg, pair)
    check("n_pair_tokens=None returns iptm() unchanged",
          abs(float(legacy) - float(direct)) < 1e-9,
          f"{float(legacy):.8f} vs {float(direct):.8f}")

    # 2. uncropped agreement between the two ways of getting the same d0
    pinned = RKL.iptm_pooled(logits, feat_full, cfg, pair, pool="max", n_pair_tokens=full_npt)
    check("uncropped: fixed-d0 max agrees with the legacy max",
          abs(float(pinned) - float(legacy)) < 1e-5,
          f"pinned {float(pinned):.6f} vs legacy {float(legacy):.6f}")
    check("fixed-d0 max equals row_t's max computed independently",
          abs(float(pinned) - _row_t_max(logits, feat_full, full_npt, pair)) < 1e-6)

    # 3. the SAME uncropped d0 across three crop sizes
    rows = []
    for n_keep in (24, 36, 48):
        # CHAIN SIZES ARE 25/20/15, so an equal split overruns chain 2 at n_keep=48 and the
        # fixture died with IndexError before reaching the d0 claims. Clamp per chain.
        per = n_keep // 3
        keep = torch.cat([torch.arange(0, min(per, 25)),
                          torch.arange(25, 25 + min(per, 20)),
                          torch.arange(45, 45 + min(per, 15))])
        f_c = _feat(asym, keep)
        lg = logits[:, keep][:, :, keep]
        got = RKL.iptm_pooled(lg, f_c, cfg, pair, pool="max", n_pair_tokens=full_npt)
        want = _row_t_max(lg, f_c, full_npt, pair)
        crop_npt = int(((f_c["asym_id"] == 0) | (f_c["asym_id"] == 1)).sum())
        unpinned = _row_t_max(lg, f_c, crop_npt, pair)
        rows.append((n_keep, float(got), want, unpinned, crop_npt))
    check("every crop uses the PINNED uncropped d0, not its own",
          all(abs(g - w) < 1e-6 for _, g, w, _, _ in rows),
          "; ".join(f"n={n}: {g:.6f}" for n, g, _, _, _ in rows))
    check("pinning actually changes the number (the crop-derived d0 differs)",
          all(abs(g - u) > 1e-6 for _, g, _, u, _ in rows),
          "; ".join(f"n={n} npt {c}->{full_npt}: {g:.6f} vs {u:.6f}"
                    for n, g, _, u, c in rows))

    # 4. differentiable, and the mean path still works unchanged
    lg = logits.clone().requires_grad_(True)
    out = RKL.iptm_pooled(lg, feat_full, cfg, pair, pool="max", n_pair_tokens=full_npt)
    check("fixed-d0 max is differentiable", out.requires_grad)
    out.backward()
    check("gradient reaches the PAE logits", lg.grad is not None and float(lg.grad.abs().sum()) > 0,
          f"|grad| = {float(lg.grad.abs().sum()):.4e}" if lg.grad is not None else "no grad")
    mean_v = RKL.iptm_pooled(logits, feat_full, cfg, pair, pool="mean", n_pair_tokens=full_npt)
    check("mean path unchanged and <= the max over the same rows",
          float(mean_v) <= float(pinned) + 1e-9,
          f"mean {float(mean_v):.6f} <= max {float(pinned):.6f}")

    # 5. an unknown pooling is still rejected, and the old refusal is gone
    try:
        RKL.iptm_pooled(logits, feat_full, cfg, pair, pool="median")
        check("unknown pool rejected", False, "no error raised")
    except ValueError:
        check("unknown pool still raises ValueError", True)
    except SystemExit as e:
        check("unknown pool rejected", False, f"wrong exception: SystemExit {e}")


if __name__ == "__main__":
    main()
    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    sys.exit(1 if FAIL else 0)
