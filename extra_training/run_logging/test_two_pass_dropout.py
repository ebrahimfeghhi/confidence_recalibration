#!/usr/bin/env python
"""Pass A and Pass B must see the SAME dropout masks, or the coefficients describe another draw.

THE HAZARD. --rank-form rel is a two-pass step. Pass A scores every pose of the step under
no_grad to get each pose's s_i, from which the relative-regression coefficient
c_i = 2*((s_i - mean_s) - (g_i - mean_g)) is formed. Pass B recomputes s_i WITH gradients and
applies c_i. The true gradient of the loss requires Pass B's s_i to equal Pass A's s_i. With
dropout off that is automatic. With dropout ON the forward is stochastic, so unless Pass B draws
the SAME masks, c_i was computed for a different realisation than the one being differentiated --
and the step takes a gradient of nothing in particular, silently, with the loss curve still
descending.

phase_k_train guards this by saving torch.get_rng_state() and cuda.get_rng_state_all() before
Pass A and restoring both in a finally (phase_k_train.py ~4531). THAT HAS NEVER BEEN EXERCISED:
--pair-dropout is new and ConfidenceHead ships with dropout=0.0, so every arm to date had a
deterministic forward and the save/restore was a no-op.

WHAT THIS CHECKS, on a real ConfidenceHead with dropout live:
  1  the save/restore makes two passes over the same poses agree POSE FOR POSE;
  2  WITHOUT it they do NOT -- otherwise check 1 passes for the wrong reason;
  3  the coefficient a two-pass step forms equals the one a single-pass autograd step would,
     which is the property the step actually needs;
  4  the restore survives a DIFFERENT NUMBER of poses in pass A than pass B consumes.

No GPU (the cuda half of the save/restore is a no-op here, and is pinned in source instead).

Run:  python -I run_logging/test_two_pass_dropout.py
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


N_TOK, N_ATOM, N_POSE = 10, 40, 5
DIMS = dict(n_blocks=2, c_s=384, c_z=256, c_s_inputs=449, b_pae=64, b_pde=64, b_plddt=50,
            b_resolved=2, max_atoms_per_token=24, stop_gradient=True)
feat = {"atom_to_token_idx": torch.arange(N_ATOM) % N_TOK,
        "atom_to_tokatom_idx": torch.arange(N_ATOM) % 24}

torch.manual_seed(3)
head = ConfidenceHead(**DIMS)
head.train()
head.pairformer_stack.blocks_per_ckpt = None
with torch.no_grad():
    # NON-ZERO, or the triangle updates are all zeros and dropout is a no-op on them -- the trap
    # that made the first version of test_pair_dropout.py report a working flag as broken.
    for _n, _t in head.pairformer_stack.named_parameters():
        if _t.dim() >= 2:
            _t.normal_(std=0.02)
    head.linear_no_bias_pae.weight.normal_(std=0.05)
RATE = 0.25
for b in head.pairformer_stack.blocks:
    b.p_drop = RATE

torch.manual_seed(5)
S_TRUNK = torch.randn(N_TOK, 384)
Z = [torch.randn(N_TOK, N_TOK, 256) for _ in range(N_POSE)]
XYZ = [torch.randn(N_TOK, 3) * 8.0 for _ in range(N_POSE)]


def score(i):
    """One pose's scalar score, standing in for its pooled ipTM."""
    out = head.memory_efficient_forward(
        input_feature_dict={k: v.clone() for k, v in feat.items()},
        s_trunk=S_TRUNK, z_pair=Z[i], pair_mask=None,
        x_pred_rep_coords=XYZ[i], inplace_safe=False, chunk_size=None)
    return out[1].float().mean()


def pass_over(idxs, grad):
    if grad:
        return [score(i) for i in idxs]
    with torch.no_grad():
        return [float(score(i)) for i in idxs]


# ---- 1. WITH the save/restore, the two passes agree pose for pose ----------------------
torch.manual_seed(99)
rng = torch.get_rng_state()
a_scores = pass_over(range(N_POSE), grad=False)
torch.set_rng_state(rng)                                 # exactly what phase_k_train does
b_scores = [float(v) for v in pass_over(range(N_POSE), grad=True)]
d = max(abs(x - y) for x, y in zip(a_scores, b_scores))
check("pass A and pass B agree pose-for-pose WITH the rng restore", d == 0.0,
      f"max|dA-dB| {d:.3e}")

# ---- 2. and WITHOUT it they do not, so check 1 is not vacuous --------------------------
torch.manual_seed(99)
a2 = pass_over(range(N_POSE), grad=False)
b2 = [float(v) for v in pass_over(range(N_POSE), grad=True)]   # no restore
d2 = max(abs(x - y) for x, y in zip(a2, b2))
check("WITHOUT the restore they DISAGREE (so the restore is what fixes it)", d2 > 1e-6,
      f"max|dA-dB| {d2:.3e}")

# ---- 3. the two-pass coefficient equals the one-pass autograd gradient -----------------
# The loss over one interface: sum_i [(s_i - mean_s) - (g_i - mean_g)]^2.
G = torch.tensor([0.9, 0.4, 0.6, 0.2, 0.7])
Gc = G - G.mean()


def rel_loss(s_vec):
    return ((s_vec - s_vec.mean()) - Gc).pow(2).sum()


# ONE PASS: straight autograd through the whole objective.
torch.manual_seed(99)
head.zero_grad(set_to_none=True)
s_one = torch.stack(pass_over(range(N_POSE), grad=True))
rel_loss(s_one).backward()
g_one = head.linear_no_bias_pae.weight.grad.detach().clone()

# TWO PASS: coefficients from a no_grad pass, applied in a second pass -- the arm's own shape.
torch.manual_seed(99)
rng = torch.get_rng_state()
s_a = torch.tensor(pass_over(range(N_POSE), grad=False))
coef = 2.0 * ((s_a - s_a.mean()) - Gc)
# the centring makes d(s_i - mean_s)/ds_j = delta_ij - 1/P, and sum(coef) == 0 exactly, so the
# -1/P term drops out and coef_i can be applied directly to s_i. Asserted, not assumed:
check("the coefficients sum to zero, so the centring term cancels",
      abs(float(coef.sum())) < 1e-5, f"sum {float(coef.sum()):.3e}")
torch.set_rng_state(rng)
head.zero_grad(set_to_none=True)
s_b = torch.stack(pass_over(range(N_POSE), grad=True))
(s_b * coef).sum().backward()
g_two = head.linear_no_bias_pae.weight.grad.detach().clone()

num = float((g_one - g_two).abs().max())
den = float(g_one.abs().max()) + 1e-12
check("the two-pass gradient MATCHES one-pass autograd on the real objective",
      num / den < 1e-5, f"max|d| {num:.3e} on |g| {den:.3e}  (rel {num/den:.2e})")

# and it does NOT match if pass B redraws, which is the failure being guarded against
torch.manual_seed(99)
s_a3 = torch.tensor(pass_over(range(N_POSE), grad=False))
coef3 = 2.0 * ((s_a3 - s_a3.mean()) - Gc)
head.zero_grad(set_to_none=True)
s_b3 = torch.stack(pass_over(range(N_POSE), grad=True))      # no restore
(s_b3 * coef3).sum().backward()
g_bad = head.linear_no_bias_pae.weight.grad.detach().clone()
check("a pass B that REDRAWS gives a different gradient (the hazard is real)",
      float((g_one - g_bad).abs().max()) / den > 1e-4,
      f"rel {float((g_one - g_bad).abs().max())/den:.2e}")

# ---- 4. the restore survives pass A consuming a different number of poses --------------
torch.manual_seed(99)
rng = torch.get_rng_state()
pass_over(range(N_POSE), grad=False)
pass_over(range(2), grad=False)            # extra draws, as a longer pass A would make
torch.set_rng_state(rng)
b4 = [float(v) for v in pass_over(range(N_POSE), grad=True)]
check("the restore is absolute, not a rewind: extra pass-A draws do not shift pass B",
      max(abs(x - y) for x, y in zip(a_scores, b4)) == 0.0)

# ---- 5. the cuda half, pinned in source -------------------------------------------------
src = open(os.path.join(ROOT, "phases", "phase_k_train.py")).read()
check("the trainer saves BOTH cpu and cuda rng before pass A",
      "_rng_cpu = torch.get_rng_state()" in src
      and "_rng_cuda = torch.cuda.get_rng_state_all()" in src)
check("and restores both in a finally, so a raising pass A cannot leak the state",
      "finally:" in src and "torch.set_rng_state(_rng_cpu)" in src
      and "torch.cuda.set_rng_state_all(_rng_cuda)" in src)

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("  all checks passed")
