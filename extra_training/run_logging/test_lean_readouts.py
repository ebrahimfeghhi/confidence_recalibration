#!/usr/bin/env python
"""CPU test for phases/lean_readouts.py: the pae path is UNCHANGED and the dead work is GONE.

Runs a small real ConfidenceHead twice over identical inputs -- once as Protenix ships it, once
with the lean mirror installed -- and checks three things:

  1  pae comes out BIT-IDENTICAL. Not allclose: the mirror keeps those lines verbatim, so any
     difference at all means a line was altered, and a tolerance would hide it.
  2  the dropped work is NOT PERFORMED. broadcast_token_to_atom and the pde projection are
     replaced with tripwires that raise if called, so this fails if the mirror still reaches
     them -- the "flag that changes nothing" failure that --blocks-per-ckpt had for its whole
     life (phase_k_train.py:1333).
  3  the placeholders survive what the LOSS does to them. protenix/model/loss.py:1574 calls
     every loss_fn unconditionally and the plddt lambda indexes
     pred_dict["plddt"][..., coord_mask, :] in its own body, so a scalar placeholder would
     raise there and not here. This indexes it the same way.

Plus the pin itself: install() must refuse a changed memory_efficient_forward, and must refuse
--lean-readouts without --pae-only.

No GPU. Run:  python -I run_logging/test_lean_readouts.py
"""
import os
import sys

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch                                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

from protenix.model.modules.confidence import ConfidenceHead                 # noqa: E402
import protenix.model.modules.confidence as CONF                             # noqa: E402
import lean_readouts as LR                                                   # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# A HEAD WITH THE LIVE RUN'S DIMS, only smaller in blocks and tokens. c_s/c_z/b_* are the real
# ones (read off training/runs/fnegj8/head_ep21_step2562.pt), because the placeholder shapes are
# derived from b_plddt and the pae path from c_z.
N_TOK, N_ATOM = 9, 31
torch.manual_seed(0)
head = ConfidenceHead(n_blocks=1, c_s=384, c_z=256, c_s_inputs=449, b_pae=64, b_pde=64,
                      b_plddt=50, b_resolved=2, max_atoms_per_token=24, stop_gradient=True)
head.train()
head.pairformer_stack.blocks_per_ckpt = None
# PROTENIX ZERO-INITIALISES plddt_weight AND resolved_weight (confidence.py:129-132), so a
# freshly built head returns all-zero plddt -- which would make "the lean plddt is zeros"
# trivially true and the check vacuous. The live head has trained, non-zero values; give the
# test the same, or it cannot tell the placeholder from the real thing.
with torch.no_grad():
    head.plddt_weight.normal_()
    head.resolved_weight.normal_()

feat = {
    "atom_to_token_idx": torch.arange(N_ATOM) % N_TOK,
    "atom_to_tokatom_idx": torch.arange(N_ATOM) % 24,
}
s_trunk = torch.randn(N_TOK, 384)
z_pair = torch.randn(N_TOK, N_TOK, 256)
coords = torch.randn(N_TOK, 3) * 8.0


def run():
    """One call, from inputs that are re-created each time so neither run can perturb the other."""
    torch.manual_seed(1234)
    return head.memory_efficient_forward(
        input_feature_dict={k: v.clone() for k, v in feat.items()},
        s_trunk=s_trunk.clone(), z_pair=z_pair.clone(), pair_mask=None,
        x_pred_rep_coords=coords.clone(), inplace_safe=False, chunk_size=None)


# ---- 1. the pae path is unchanged -------------------------------------------------------
orig_method = type(head).memory_efficient_forward
plddt_a, pae_a, pde_a, res_a = run()

sha = LR.install(type("M", (), {"confidence_head": head})(), pae_only=True)
check("install() verified the pin", sha == LR.PINNED_SHA, sha[:16])
check("install() replaced the method",
      head.memory_efficient_forward.__func__ is LR._lean_memory_efficient_forward)

plddt_b, pae_b, pde_b, res_b = run()

check("pae shape unchanged", tuple(pae_a.shape) == tuple(pae_b.shape),
      f"{tuple(pae_a.shape)}")
check("pae BIT-IDENTICAL", torch.equal(pae_a, pae_b),
      f"max|d| {float((pae_a - pae_b).abs().max()):.3e}" if not torch.equal(pae_a, pae_b)
      else "exact")

# ---- 2. the dead outputs really are placeholders ---------------------------------------
check("pde collapsed", pde_b.numel() == 1, f"{tuple(pde_a.shape)} -> {tuple(pde_b.shape)}")
check("resolved collapsed", res_b.numel() == 1, f"{tuple(res_a.shape)} -> {tuple(res_b.shape)}")
check("plddt keeps its N_atom axis", tuple(plddt_b.shape) == (N_ATOM, 50),
      f"{tuple(plddt_a.shape)} -> {tuple(plddt_b.shape)}")
check("plddt is no longer the gathered einsum",
      plddt_b.abs().sum().item() == 0.0 and plddt_a.abs().sum().item() != 0.0)

# ---- 3. the dropped work is not merely unused, it is NOT RUN ---------------------------
_hits = []
_real_b2a = CONF.broadcast_token_to_atom


def _tripwire_b2a(*a_, **k_):
    _hits.append("broadcast_token_to_atom")
    return _real_b2a(*a_, **k_)


CONF.broadcast_token_to_atom = _tripwire_b2a
_real_pde = head.linear_no_bias_pde.forward
head.linear_no_bias_pde.forward = lambda *a_, **k_: (_hits.append("linear_no_bias_pde")
                                                     or _real_pde(*a_, **k_))
try:
    run()
    check("lean path touches neither the atom broadcast nor the pde projection",
          _hits == [], f"called {_hits}")
finally:
    CONF.broadcast_token_to_atom = _real_b2a
    head.linear_no_bias_pde.forward = _real_pde

# the same tripwires must FIRE on the original, or they prove nothing
head.memory_efficient_forward = orig_method.__get__(head, type(head))
_hits.clear()
CONF.broadcast_token_to_atom = _tripwire_b2a
head.linear_no_bias_pde.forward = lambda *a_, **k_: (_hits.append("linear_no_bias_pde")
                                                     or _real_pde(*a_, **k_))
try:
    run()
    check("the tripwires DO fire on the original (so the test above is not vacuous)",
          sorted(set(_hits)) == ["broadcast_token_to_atom", "linear_no_bias_pde"],
          f"called {sorted(set(_hits))}")
finally:
    CONF.broadcast_token_to_atom = _real_b2a
    head.linear_no_bias_pde.forward = _real_pde

# ---- 4. the placeholder survives what loss.py:1772 does to it --------------------------
LR.install(type("M", (), {"confidence_head": head})(), pae_only=True)
plddt_c = run()[0]
coord_mask = torch.ones(N_ATOM, dtype=torch.bool)
coord_mask[3] = False
try:
    sliced = plddt_c[..., coord_mask, :]
    check("plddt placeholder indexes like loss.py:1772",
          tuple(sliced.shape) == (N_ATOM - 1, 50), f"{tuple(sliced.shape)}")
except Exception as e:                                                       # noqa: BLE001
    check("plddt placeholder indexes like loss.py:1772", False, repr(e))

# ---- 5. the guards --------------------------------------------------------------------
try:
    LR.install(type("M", (), {"confidence_head": head})(), pae_only=False)
    check("refuses --lean-readouts without --pae-only", False, "it did not refuse")
except SystemExit as e:
    check("refuses --lean-readouts without --pae-only", "--pae-only" in str(e))

_pin = LR.PINNED_SHA
LR.PINNED_SHA = "0" * 64
try:
    LR.install(type("M", (), {"confidence_head": head})(), pae_only=True)
    check("refuses a changed memory_efficient_forward", False, "it did not refuse")
except SystemExit as e:
    check("refuses a changed memory_efficient_forward",
          "PINNED_SHA" in str(e) and "re-derive" in str(e))
finally:
    LR.PINNED_SHA = _pin

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("  all checks passed")
