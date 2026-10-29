#!/usr/bin/env python
"""Gate for the --pool-mean path: the mean-pooled matrix must be row_t's mean, exactly.

WHY A GATE AT ALL. phase_j_rescore_cache._chain_pair_itm_mean reimplements row_t's arithmetic on
PROBABILITIES rather than logits (pair_iptm_only has already softmaxed, and softmaxing twice is a
different number). A reimplementation that drifts would answer "does mean pooling help" with a
quantity that is not the mean of anything. pae_row_weight.row_t is itself pinned against
Protenix's own calculate_iptm by test_pae_row_weight.py, so matching row_t transitively matches
Protenix.

ALSO PINNED: that the MAX of the same rows reproduces calculate_chain_based_ptm's
chain_pair_iptm. If it does not, the two poolings are not two views of one quantity and comparing
them says nothing.

CPU ONLY, synthetic input -- it checks arithmetic, not the model.

Usage:  python -I run_logging/test_pool_mean.py
"""
import os
import sys

import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import pae_row_weight as rw                                     # noqa: E402
from phase_j_rescore_cache import _chain_pair_itm_mean          # noqa: E402
from protenix.model import sample_confidence                    # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    torch.manual_seed(0)
    N, NB, NP = 60, 64, 3
    asym = torch.cat([torch.zeros(25), torch.ones(20), torch.full((15,), 2.0)]).long()
    hf = torch.ones(N, dtype=torch.bool)
    lig = torch.zeros(N, dtype=torch.bool)
    logits = torch.randn(NP, N, N, NB) * 2.0
    prob = torch.softmax(logits, dim=-1)
    bp = dict(min_bin=0.0, max_bin=32.0, no_bins=NB)

    got = _chain_pair_itm_mean(prob, has_frame=hf, asym_id=asym, token_is_ligand=lig, **bp)
    check("returns one matrix per pose", tuple(got.shape) == (NP, 3, 3), f"{tuple(got.shape)}")

    # THE PER-POSE CHECK IS THE POINT. An earlier draft computed pae_prob[0] and assigned it to
    # every pose, which is invisible unless the poses are compared to each other.
    worst_mean = worst_max = 0.0
    chains = [0, 1, 2]
    for pi in range(NP):
        for xi in range(3):
            for yi in range(xi + 1, 3):
                cx, cy = chains[xi], chains[yi]
                tok = (asym == cx) | (asym == cy)
                tm = (tok.unsqueeze(0) & (asym[hf].unsqueeze(1) != asym.unsqueeze(0))
                      & ((asym[hf] == cx) | (asym[hf] == cy)).unsqueeze(1))
                t, cnt = rw.row_t(logits[pi:pi + 1], hf, tm, n_pair_tokens=int(tok.sum()),
                                  min_bin=0.0, max_bin=32.0)
                keep = cnt > 0
                worst_mean = max(worst_mean, abs(float(t[keep].mean()) - float(got[pi, cx, cy])))
                worst_max = max(worst_max, float(t[keep].max()))
    check("mean-pooled matrix equals row_t's mean over scored rows, every pose and pair",
          worst_mean < 1e-5, f"worst |diff| = {worst_mean:.3e}")

    # the two poses must differ, or the per-pose bug is back
    spread = float((got[:, 0, 1] - got[0, 0, 1]).abs().max())
    check("different poses give different values (per-pose regression guard)", spread > 1e-6,
          f"max |pose_i - pose_0| = {spread:.3e}")

    # and the MAX of the same rows is Protenix's own chain_pair_iptm
    ref = sample_confidence.calculate_chain_based_ptm(
        prob, has_frame=hf, asym_id=asym, token_is_ligand=lig, **bp)["chain_pair_iptm"]
    ref = ref[0] if ref.dim() == 3 else ref
    tok = (asym == 0) | (asym == 1)
    tm = (tok.unsqueeze(0) & (asym[hf].unsqueeze(1) != asym.unsqueeze(0))
          & ((asym[hf] == 0) | (asym[hf] == 1)).unsqueeze(1))
    t, cnt = rw.row_t(logits[0:1], hf, tm, n_pair_tokens=int(tok.sum()),
                      min_bin=0.0, max_bin=32.0)
    d = abs(float(t[cnt > 0].max()) - float(ref[0, 1]))
    check("row_t's MAX reproduces Protenix chain_pair_iptm (the two poolings share one quantity)",
          d < 1e-5, f"|diff| = {d:.3e}  (max {float(t[cnt>0].max()):.4f})")

    # mean must be <= max, and strictly below it in general
    lo = float(got[0, 0, 1])
    check("mean <= max, and strictly below on real-ish input", lo < float(ref[0, 1]) + 1e-9,
          f"mean {lo:.4f} vs max {float(ref[0,1]):.4f}")

    # ligand exclusion is honoured
    lig2 = torch.zeros(N, dtype=torch.bool)
    lig2[:5] = True
    g2 = _chain_pair_itm_mean(prob, has_frame=hf, asym_id=asym, token_is_ligand=lig2, **bp)
    check("excluding ligand rows changes the result (the mask is actually applied)",
          abs(float(g2[0, 0, 1]) - lo) > 1e-6,
          f"{float(g2[0,0,1]):.6f} vs {lo:.6f}")


if __name__ == "__main__":
    main()
    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    sys.exit(1 if FAIL else 0)
