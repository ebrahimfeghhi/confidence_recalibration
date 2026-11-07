#!/usr/bin/env python
"""The lean mirror against the PRODUCTION architecture: both pose pathways installed.

WHY THIS EXISTS SEPARATELY FROM test_lean_readouts.py. That test uses a plain ConfidenceHead,
which is not what the arm runs. pose_direction.py:234 and pose_perblock.py:232 each capture
`orig_mef = head.memory_efficient_forward` and replace the attribute with a wrapper calling what
they captured, so the three patches are a CHAIN whose order is its meaning. A lean installer that
runs last silently drops both injections, and a test on a bare head cannot see that. Found in
review.

WHAT IT CHECKS, fat chain versus lean chain over identical weights and inputs:

  1  pae BIT-IDENTICAL. frame_dir_proj is RANDOMISED first, not left at its zero init -- with a
     zero projection the orientation term contributes nothing to the forward and pae would match
     even if the injection had been dropped entirely, which is precisely the bug being tested for.
  2  THE GRADIENT OF frame_dir_proj.weight is non-zero and identical. This is the check that
     actually bites: it is non-zero only if the injection is in the graph.
  3  EVERY shared parameter's gradient identical, so no other pathway lost its signal either.
  4  the per-block copies that --pose-perblock-copy adds get gradients in both.
  5  one AdamW step from identical state lands on identical weights.

No GPU. Run:  python -I run_logging/test_lean_readouts_arch.py
"""
import os
import sys

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch                                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

from protenix.model.modules.confidence import ConfidenceHead                 # noqa: E402
import lean_readouts as LR                                                   # noqa: E402
import pose_direction as PD                                                  # noqa: E402
import pose_perblock as PP                                                   # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


N_TOK, N_ATOM = 10, 40
DIMS = dict(n_blocks=2, c_s=384, c_z=256, c_s_inputs=449, b_pae=64, b_pde=64, b_plddt=50,
            b_resolved=2, max_atoms_per_token=24, stop_gradient=True)

# ONE rep atom per token, which is the contract pose_direction's index check enforces.
rep_idx = torch.arange(N_TOK) * (N_ATOM // N_TOK)
rep_mask = torch.zeros(N_ATOM, dtype=torch.bool)
rep_mask[rep_idx] = True
# frame_atom_index is per-token and indexes ATOMS: three atoms defining each token's frame.
frame_atom_index = torch.stack([rep_idx, (rep_idx + 1) % N_ATOM, (rep_idx + 2) % N_ATOM], dim=-1)
has_frame = torch.ones(N_TOK, dtype=torch.bool)

feat = {
    "atom_to_token_idx": torch.arange(N_ATOM) % N_TOK,
    "atom_to_tokatom_idx": torch.arange(N_ATOM) % 24,
    "distogram_rep_atom_mask": rep_mask,
}

torch.manual_seed(7)
s_inputs = torch.randn(N_TOK, 449)
s_trunk = torch.randn(N_TOK, 384)
z_trunk = torch.randn(N_TOK, N_TOK, 256)
# [N_sample, N_atom, 3]; one sample, matching what phase_k_train passes per pose.
coords = (torch.randn(1, N_ATOM, 3) * 8.0)

torch.manual_seed(11)
_proto = ConfidenceHead(**DIMS)
# RANDOMISED ON PURPOSE: Protenix zero-initialises the pae/pde projections and the plddt and
# resolved weights, and pose_direction zero-initialises frame_dir_proj. Every one of those zeros
# would make a comparison pass without proving anything.
with torch.no_grad():
    for t in (_proto.linear_no_bias_pae.weight, _proto.linear_no_bias_pde.weight,
              _proto.plddt_weight, _proto.resolved_weight):
        t.normal_(std=0.05)
SNAPSHOT = {k: v.detach().clone() for k, v in _proto.state_dict().items()}


def build(lean):
    """A head with the arm's chain. lean installs FIRST, as phase_k_train.py now does."""
    torch.manual_seed(11)
    head = ConfidenceHead(**DIMS)
    head.load_state_dict(SNAPSHOT)
    head.train()
    head.pairformer_stack.blocks_per_ckpt = None
    holder = type("M", (), {"confidence_head": head})()
    if lean:
        LR.install(holder, pae_only=True)
    PD.install(holder, loud=False)
    # THE ARM'S SETTING: --pose-perblock --pose-perblock-copy.
    PP.install(holder, tied=False, share_pretrained=False, copy_pretrained=True, loud=False)
    with torch.no_grad():
        # NON-ZERO, so the orientation term reaches pae and a dropped injection changes it.
        head.frame_dir_proj.weight.normal_(std=0.02)
    return head


def run(head):
    PD.set_unit(frame_atom_index, has_frame, rep_mask, pdb="test")
    torch.manual_seed(1234)
    out = head.forward(
        input_feature_dict={k: v.clone() for k, v in feat.items()},
        s_inputs=s_inputs.clone(), s_trunk=s_trunk.clone(), z_trunk=z_trunk.clone(),
        pair_mask=None, x_pred_coords=coords.clone(), use_embedding=True,
        inplace_safe=False, chunk_size=None)
    plddt, pae, pde, resolved = out
    # THE LOSS THE ARM TAKES ITS GRADIENT FROM reads pae alone (phases/rank_loss.py:238), so the
    # scalar here reads pae alone too. A scalar touching pde would differentiate a branch the arm
    # never differentiates and the comparison would be of something nobody runs.
    head.zero_grad(set_to_none=True)
    pae.float().square().mean().backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in head.named_parameters()}
    return pae.detach().clone(), (plddt, pde, resolved), grads


fat = build(lean=False)
pae_f, dead_f, g_f = run(fat)
# PD/PP keep module-level per-call caches; both are cleared by their own `finally`, and the two
# heads are run sequentially, never concurrently.
PD.clear_unit() if hasattr(PD, "clear_unit") else None

lean = build(lean=True)
check("lean chain installed under both pathways",
      getattr(lean, "_pose_direction_installed", False)
      and getattr(lean, "_pose_perblock_installed", False))
pae_l, dead_l, g_l = run(lean)

# ---- 1. the kept path ------------------------------------------------------------------
check("pae shape unchanged", tuple(pae_f.shape) == tuple(pae_l.shape), f"{tuple(pae_f.shape)}")
check("pae BIT-IDENTICAL with both pathways installed", torch.equal(pae_f, pae_l),
      "exact" if torch.equal(pae_f, pae_l)
      else f"max|d| {float((pae_f - pae_l).abs().max()):.3e}")

# ---- 2. the orientation injection is still in the graph --------------------------------
gd_f, gd_l = g_f.get("frame_dir_proj.weight"), g_l.get("frame_dir_proj.weight")
check("frame_dir_proj got a gradient in the fat chain",
      gd_f is not None and float(gd_f.abs().max()) > 0,
      "none" if gd_f is None else f"max|g| {float(gd_f.abs().max()):.3e}")
check("frame_dir_proj got a gradient in the LEAN chain",
      gd_l is not None and float(gd_l.abs().max()) > 0,
      "none" if gd_l is None else f"max|g| {float(gd_l.abs().max()):.3e}")
check("frame_dir_proj gradient IDENTICAL",
      gd_f is not None and gd_l is not None and torch.equal(gd_f, gd_l))

# ---- 3. every shared parameter -------------------------------------------------------
shared = sorted(set(g_f) & set(g_l))
check("the two chains expose the same parameter set", set(g_f) == set(g_l),
      f"fat-only {sorted(set(g_f) - set(g_l))[:3]} lean-only {sorted(set(g_l) - set(g_f))[:3]}")
diff, nograd_both, dead_names = [], [], ("plddt_weight", "resolved_weight", "pde")
for n in shared:
    a_, b_ = g_f[n], g_l[n]
    if a_ is None and b_ is None:
        nograd_both.append(n)
        continue
    if a_ is None or b_ is None or not torch.equal(a_, b_):
        # THE THREE DEAD READOUTS' OWN WEIGHTS ARE EXPECTED TO DIFFER: the fat chain reaches them
        # and the lean chain does not. They receive no gradient from the arm's loss either way --
        # the --pae-only grad probe already reports them as the dead trainable set -- so a
        # difference here is the flag working, not a regression. Anything ELSE differing is.
        (nograd_both if any(d in n for d in dead_names) else diff).append(n)
check("every shared parameter's gradient identical (outside the dead readouts)",
      not diff, f"{len(diff)} differ: {diff[:4]}")
check("the comparison was not vacuous: most parameters DID get a gradient",
      sum(1 for n in shared if g_f[n] is not None) > 0.5 * len(shared),
      f"{sum(1 for n in shared if g_f[n] is not None)}/{len(shared)} with grad")

# ---- 4. the per-block copies are live ------------------------------------------------
pb = [n for n in shared if "perblock" in n or "blk" in n or "copy" in n]
check("per-block copies exist and got gradients in both",
      bool(pb) and all(g_f[n] is not None and g_l[n] is not None for n in pb),
      f"{len(pb)} tensor(s): {pb[:3]}")

# ---- 5. the dead outputs collapsed --------------------------------------------------
check("pde collapsed under the full chain", dead_l[1].numel() == 1,
      f"{tuple(dead_f[1].shape)} -> {tuple(dead_l[1].shape)}")
check("resolved collapsed under the full chain", dead_l[2].numel() == 1)

# ---- 6. one optimiser step lands in the same place ----------------------------------
def one_step(head, grads):
    names = [n for n, _ in head.named_parameters()]
    opt = torch.optim.AdamW([p for _, p in head.named_parameters()], lr=1e-3)
    for (n, p) in head.named_parameters():
        p.grad = None if grads[n] is None else grads[n].clone()
    opt.step()
    return {n: p.detach().clone() for n, p in zip(names, (p for _, p in head.named_parameters()))}


wf, wl = one_step(fat, g_f), one_step(lean, g_l)
moved = [n for n in sorted(set(wf) & set(wl))
         if not torch.equal(wf[n], SNAPSHOT.get(n, wf[n]))]
step_diff = [n for n in sorted(set(wf) & set(wl))
             if not torch.equal(wf[n], wl[n]) and not any(d in n for d in dead_names)]
check("one AdamW step lands on identical weights (outside the dead readouts)",
      not step_diff, f"{len(step_diff)} differ: {step_diff[:4]}")
check("the step actually moved weights", bool(moved), f"{len(moved)} tensor(s) moved")

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("  all checks passed")
