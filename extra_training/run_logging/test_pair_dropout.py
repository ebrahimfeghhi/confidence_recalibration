#!/usr/bin/env python
"""--pair-dropout: the rate reaches the pair representation and actually perturbs it.

WHY A FUNCTIONAL TEST AND NOT AN ATTRIBUTE CHECK. Setting p_drop proves nothing on its own: the
four pair-rep sites could have been reading a different attribute, the head could be in eval mode,
or a fused op could ignore the rate. All three failure modes leave p_drop looking correct. So this
runs the head twice on identical inputs and asserts the outputs DIFFER in train mode and MATCH in
eval mode -- which is what dropout means and the only thing that distinguishes it from a no-op.

It also pins the thing that motivated the flag: ConfidenceHead ships with dropout=0.0, overriding
Protenix's own 0.25 default for pair stacks, so the arm has been training with none.

No GPU. Run:  python -I run_logging/test_pair_dropout.py
"""
import os
import sys

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch                                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

from protenix.model.modules.confidence import ConfidenceHead                 # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


N_TOK, N_ATOM = 12, 48
DIMS = dict(n_blocks=2, c_s=384, c_z=256, c_s_inputs=449, b_pae=64, b_pde=64, b_plddt=50,
            b_resolved=2, max_atoms_per_token=24, stop_gradient=True)
feat = {"atom_to_token_idx": torch.arange(N_ATOM) % N_TOK,
        "atom_to_tokatom_idx": torch.arange(N_ATOM) % 24}


def build(rate):
    torch.manual_seed(3)
    h = ConfidenceHead(**DIMS)
    h.train()
    h.pairformer_stack.blocks_per_ckpt = None
    with torch.no_grad():
        h.linear_no_bias_pae.weight.normal_(std=0.05)
    # THE ZERO-INIT TRAP, AGAIN. Protenix zero-initialises the triangle modules' output
    # projections, so a FRESH head's tri_mul/tri_att updates are all zeros -- and dropping out
    # zeros is a no-op. A dropout test on an untrained head therefore reports "no difference" and
    # looks like a broken flag when the flag is fine. (test_lean_readouts.py hit the same thing
    # with plddt_weight.) A trained checkpoint has non-zero updates, so give the stack non-zero
    # weights or this file proves nothing in either direction.
    with torch.no_grad():
        for n, t in h.pairformer_stack.named_parameters():
            if t.dim() >= 2:
                t.normal_(std=0.02)
    if rate is not None:
        # EXACTLY WHAT phase_k_train DOES at its install site.
        for b in h.pairformer_stack.blocks:
            b.p_drop = float(rate)
            dr = getattr(b, "dropout_row", None)
            if dr is not None:
                for obj in (dr, getattr(dr, "dropout", None)):
                    for at in ("p", "r"):
                        if obj is not None and hasattr(obj, at):
                            setattr(obj, at, float(rate))
    return h


def run(h, seed):
    torch.manual_seed(seed)
    torch.manual_seed(5)
    s_trunk = torch.randn(N_TOK, 384)
    z_pair = torch.randn(N_TOK, N_TOK, 256)
    coords = torch.randn(N_TOK, 3) * 8.0
    torch.manual_seed(seed)                 # ONLY the dropout mask differs between calls
    out = h.memory_efficient_forward(
        input_feature_dict={k: v.clone() for k, v in feat.items()},
        s_trunk=s_trunk, z_pair=z_pair, pair_mask=None,
        x_pred_rep_coords=coords, inplace_safe=False, chunk_size=None)
    return out[1].detach().clone()           # pae logits


# ---- 0. the updates being dropped are NON-ZERO, or nothing below means anything -----------
import protenix.model.modules.pairformer as PF                               # noqa: E402
_seen = []
_real = PF.dropout_add_rowwise


def _spy(res, x, pr, tr):
    _seen.append(float(x.detach().abs().max()))
    return _real(res, x, pr, tr)


# ---- 1. the ship-default is zero, which is why the flag exists ----------------------------
h0 = build(None)
rates = sorted({float(b.p_drop) for b in h0.pairformer_stack.blocks})
check("ConfidenceHead ships with pair dropout 0.0", rates == [0.0], f"{rates}")

# ---- 2. at 0.0 the forward is deterministic ----------------------------------------------
a1, a2 = run(h0, 11), run(h0, 22)
check("rate 0.0: two forwards are IDENTICAL (no noise anywhere)", torch.equal(a1, a2),
      "exact" if torch.equal(a1, a2) else f"max|d| {float((a1-a2).abs().max()):.3e}")

# ---- 3. at 0.25 the forward is stochastic IN TRAIN MODE ----------------------------------
h1 = build(0.25)
PF.dropout_add_rowwise = _spy
try:
    run(h1, 11)
finally:
    PF.dropout_add_rowwise = _real
check("the tensors being dropped out are NON-ZERO (else the test is vacuous)",
      bool(_seen) and max(_seen) > 0, f"{len(_seen)} call(s), max|x| {max(_seen) if _seen else 0:.3e}")
rates = sorted({float(b.p_drop) for b in h1.pairformer_stack.blocks})
check("the rate reached every block", rates == [0.25], f"{rates}")
b1, b2 = run(h1, 11), run(h1, 22)
check("rate 0.25 train mode: two forwards DIFFER (dropout is live)", not torch.equal(b1, b2),
      f"max|d| {float((b1-b2).abs().max()):.3e}")

# ---- 4. and deterministic again in EVAL mode ---------------------------------------------
h1.eval()
c1, c2 = run(h1, 11), run(h1, 22)
check("rate 0.25 eval mode: forwards MATCH (dropout off at inference)", torch.equal(c1, c2),
      "exact" if torch.equal(c1, c2) else f"max|d| {float((c1-c2).abs().max()):.3e}")
h1.train()

# ---- 5. the 1/(1-p) rescale, checked WHERE THE INVARIANT HOLDS -------------------------
# Dropout preserves the expectation of THE TENSOR IT IS APPLIED TO. It does not preserve the mean
# of the pae logits, which sit four LayerNorms and two nonlinearities downstream -- E[f(x)] is not
# f(E[x]). An earlier version of this file asserted the latter and failed, which was the test
# being wrong and not the rescale. So assert it at the op.
_x = torch.randn(8, 16, 32)
_zero = torch.zeros_like(_x)
_draws = torch.stack([PF.dropout_add_rowwise(_zero, _x, 0.25, True) for _ in range(400)])
_err = float((_draws.mean(0) - _x).abs().mean() / _x.abs().mean())
check("dropout_add_rowwise preserves E[x] (the 1/(1-p) rescale is applied)", _err < 0.05,
      f"relative error of the 400-draw mean: {_err:.4f}")
_one = PF.dropout_add_rowwise(_zero, _x, 0.25, True)
_frac0 = float((_one == 0).float().mean())
check("a single draw really zeroes ~p of it", 0.1 < _frac0 < 0.4, f"zero fraction {_frac0:.3f}")
_rows = (_one == 0).any(dim=-1).float()
check("the mask is ROW-WISE (shared along dim -3), not per-element",
      bool(torch.equal(_rows[0], _rows[1])), "dim -3 slices share their zero pattern")

# ---- 6. the trainer's guard refuses a rate that did not take -----------------------------
src = open(os.path.join(ROOT, "phases", "phase_k_train.py")).read()
check("the trainer asserts the rate took", "--pair-dropout did not take" in src)
check("the trainer reports training mode beside the rate",
      "dropout is a NO-OP unless that is True" in src)
check("negative default leaves the head untouched",
      'ap.add_argument("--pair-dropout", type=float, default=-1.0,' in src)
check("it is a guard field on both sides", src.count('"pair_dropout": a.pair_dropout,') == 2,
      f"{src.count(chr(34)+'pair_dropout'+chr(34)+': a.pair_dropout,')} site(s)")

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("  all checks passed")
