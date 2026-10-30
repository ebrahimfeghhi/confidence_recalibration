#!/usr/bin/env python
"""Gate: per-block orientation is CORRECT under activation checkpointing, not merely permitted.

WHAT WAS WRONG. The orientation copies read pose_direction._CALL["dir"] from inside the block.
Under checkpointing a block's forward is RE-RUN during backward, after the outer forward's
`finally` has cleared that cache, so the recomputation saw None -- or, with several poses in
flight, a LATER pose's geometry, pairing one pose's directions with another's gradient. That was
refused rather than fixed. The fix is the one already used for the onehot/dist tensors: capture
in _prep_blocks, pass the tensor in.

WHY THE TEST IS BUILT THIS WAY.

NON-ZERO ORIENTATION WEIGHTS. frame_dir_proj is zero-initialised by construction, and every
orientation copy copies it -- so with the shipped initialisation the orientation term contributes
exactly zero, its gradient is whatever flows to a zero matrix, and a COMPLETELY BROKEN orientation
pathway would still compare equal. Every orientation projection here is given distinct non-zero
weights before the comparison, which is the only way the comparison can fail if the pathway is
wrong.

SEVERAL POSES, DIFFERENT GEOMETRY EACH. The dangerous failure is not "None during backward", which
raises loudly; it is a block recomputing with a later pose's directions, which is silent and only
observable when the poses differ. Each pose here gets its own random coordinates.

THE PRE-STACK INJECTION IS APPLIED HERE TOO, as pose_direction does for real, so frame_dir_proj
is actually called and actually carries a gradient. Without it that projection is never reached
and "its gradients match" would be vacuously true -- the gate reported it as `missing` rather
than passing, which is how this was caught.

THE CACHE IS CLEARED BETWEEN POSES, exactly as pose_direction's `finally` clears it, and the test
asserts it is None after each forward. Without that the harness would hold the cache live and the
bug it exists to catch could not occur.

GRADIENTS, NOT JUST OUTPUTS. A forward that matches proves the recomputation read something; only
the gradients prove it read the RIGHT something at every depth. All four orientation projections
are compared -- frame_dir_proj plus the three per-block copies -- along with the distance copies.

NO GPU AND NO PROTENIX MODEL. The harness is a small stand-in with the attribute surface install()
needs, driven through torch.utils.checkpoint, which is the real mechanism. It exercises the real
install(), the real _prep_blocks capture and the real _add_dir.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import torch                                                                 # noqa: E402
import torch.nn as nn                                                        # noqa: E402
from torch.utils.checkpoint import checkpoint                                # noqa: E402

import pose_direction                                                        # noqa: E402
import pose_perblock                                                         # noqa: E402
from protenix.model.modules.primitives import LinearNoBias                   # noqa: E402

N_TOK, C_Z, N_BINS, N_BLOCKS, N_POSE = 12, 8, 6, 4, 3
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


class Block(nn.Module):
    """A real parameterised block, so the graph through z is non-trivial at every depth."""

    def __init__(self, c_z, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(c_z, c_z)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(c_z, c_z, generator=g) * 0.1)
            self.lin.bias.zero_()

    def forward(self, s, z):
        return torch.tanh(self.lin(z)) + z


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(C_Z, 100 + i) for i in range(N_BLOCKS)])
        self.blocks_per_ckpt = None

    def _prep_blocks(self, pair_mask=None, **kw):
        return [b.forward for b in self.blocks]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_z, self.num_bins = C_Z, N_BINS
        self.lower_bins = torch.linspace(0.0, 10.0, N_BINS)
        self.upper_bins = torch.linspace(1.0, 11.0, N_BINS)
        self.pairformer_stack = Stack()
        self.linear_no_bias_d = LinearNoBias(in_features=N_BINS, out_features=C_Z)
        self.linear_no_bias_d_wo_onehot = LinearNoBias(in_features=1, out_features=C_Z)
        self.frame_dir_proj = LinearNoBias(in_features=3, out_features=C_Z)
        with torch.no_grad():
            g = torch.Generator().manual_seed(7)
            self.linear_no_bias_d.weight.copy_(torch.randn(C_Z, N_BINS, generator=g) * 0.05)
            self.linear_no_bias_d_wo_onehot.weight.copy_(torch.randn(C_Z, 1, generator=g) * 0.05)
            self.frame_dir_proj.weight.zero_()      # as pose_direction installs it
        self._pose_direction_installed = True

    def memory_efficient_forward(self, input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        """Stands in for confidence.py: prepare blocks, then run them under checkpoint_blocks."""
        blocks = self.pairformer_stack._prep_blocks(pair_mask=pair_mask)
        # THE PRE-STACK ORIENTATION INJECTION, which pose_direction does for real before
        # delegating here. Without it frame_dir_proj is never called, gets no gradient in EITHER
        # run, and "the gradients match" would be vacuously true for the one projection the
        # per-block copies are copied from. Caught by the gate's own "missing" report.
        _d = pose_direction._CALL.get("dir")
        z = z_pair + self.frame_dir_proj(_d).to(z_pair.dtype) if _d is not None else z_pair
        bpc = self.pairformer_stack.blocks_per_ckpt
        for fn in blocks:
            if bpc and torch.is_grad_enabled():
                z = checkpoint(fn, s_trunk, z, use_reentrant=False)
            else:
                z = fn(s_trunk, z)
        return z


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence_head = Head()


def build():
    torch.manual_seed(0)
    m = Model()
    pose_perblock.install(m, copy_pretrained=True, loud=False)
    h = m.confidence_head
    # NON-ZERO, AND DIFFERENT AT EVERY DEPTH. Zero weights would make a broken orientation
    # pathway indistinguishable from a working one.
    with torch.no_grad():
        g = torch.Generator().manual_seed(11)
        h.frame_dir_proj.weight.copy_(torch.randn(C_Z, 3, generator=g) * 0.3)
        for k, fd in enumerate(h.pose_perblock_frame_dir):
            fd.weight.copy_(torch.randn(C_Z, 3, generator=g) * (0.2 + 0.1 * k))
    return m


def run(m, bpc):
    """One forward+backward over N_POSE poses, each with its OWN geometry. Returns (out, grads)."""
    h = m.confidence_head
    h.pairformer_stack.blocks_per_ckpt = bpc
    m.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    total = None
    for pose in range(N_POSE):
        gg = torch.Generator().manual_seed(500 + pose)
        coords = torch.randn(1, N_TOK, 3, generator=gg)
        z = torch.randn(1, N_TOK, N_TOK, C_Z, generator=gg)
        s = torch.randn(1, N_TOK, C_Z, generator=gg)
        # pose_direction's contract: set the geometry BEFORE the stack, clear it after.
        pose_direction._CALL["dir"] = torch.randn(1, N_TOK, N_TOK, 3, generator=gg)
        try:
            out = h.memory_efficient_forward(
                input_feature_dict={}, s_trunk=s, z_pair=z, pair_mask=None,
                x_pred_rep_coords=coords)
        finally:
            pose_direction._CALL["dir"] = None
        # ASSERTED, NOT ASSUMED: if the harness left the cache live, the bug could not occur here
        # and this gate would pass on broken code.
        assert pose_direction._CALL["dir"] is None
        total = out.square().mean() if total is None else total + out.square().mean()
    total.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    return total.detach().clone(), grads


m = build()
h = m.confidence_head
check("install built 3 orientation copies", len(h.pose_perblock_frame_dir) == N_BLOCKS - 1,
      f"{len(h.pose_perblock_frame_dir)}")
check("all four orientation projections are NON-zero",
      float(h.frame_dir_proj.weight.abs().max()) > 0
      and all(float(f.weight.abs().max()) > 0 for f in h.pose_perblock_frame_dir))

out_off, g_off = run(m, None)
out_on, g_on = run(m, 1)

check("forward output matches with checkpointing on and off",
      torch.allclose(out_off, out_on, rtol=0, atol=1e-6),
      f"{float(out_off):.9f} vs {float(out_on):.9f}")

ORI = ["confidence_head.frame_dir_proj.weight"] + [
    f"confidence_head.pose_perblock_frame_dir.{i}.weight" for i in range(N_BLOCKS - 1)]
for n in ORI:
    a, b = g_off.get(n), g_on.get(n)
    check(f"gradient matches: {n.split('confidence_head.')[-1]}",
          a is not None and b is not None and torch.allclose(a, b, rtol=0, atol=1e-6),
          "missing" if a is None or b is None else f"max|d| {float((a-b).abs().max()):.3e}")
    # A GRADIENT OF EXACTLY ZERO WOULD MATCH TRIVIALLY, so the pathway must actually carry one.
    if a is not None:
        check(f"gradient is non-zero: {n.split('confidence_head.')[-1]}",
              float(a.abs().max()) > 0, f"max {float(a.abs().max()):.3e}")

shared = sorted(set(g_off) & set(g_on))
worst = max(((float((g_off[n] - g_on[n]).abs().max()), n) for n in shared), default=(0.0, "-"))
check(f"EVERY parameter's gradient matches ({len(shared)} tensors)", worst[0] <= 1e-6,
      f"worst {worst[0]:.3e} at {worst[1]}")
check("the two runs produced the same parameter set", set(g_off) == set(g_on))

# ---- NEGATIVE CONTROL -------------------------------------------------------------------------
# A gate that cannot fail proves nothing. This restores the OLD behaviour -- _add_dir re-reading
# pose_direction._CALL["dir"] from the global instead of taking the captured tensor -- and asserts
# the comparison above then BREAKS. If this control ever starts passing, the gate has stopped
# testing anything and the fix is no longer what makes the numbers agree.
_real_add_dir = pose_perblock._add_dir


def _old_add_dir(head, i, z, d):
    """The pre-fix code path: ignore the captured tensor, read the global."""
    dirs = getattr(head, "pose_perblock_frame_dir", None)
    if not dirs or len(dirs) == 0:
        return z
    g = pose_direction._CALL.get("dir")       # cleared by the outer `finally` before backward
    if g is None:
        raise RuntimeError("old path: direction cache was cleared")
    return z + dirs[i - 1](g).to(z.dtype)


pose_perblock._add_dir = _old_add_dir
try:
    m2 = build()
    _o_off, _g_off = run(m2, None)
    try:
        _o_on, _g_on = run(m2, 1)
        _broke = not torch.allclose(_o_off, _o_on, rtol=0, atol=1e-6) or any(
            not torch.allclose(_g_off[n], _g_on[n], rtol=0, atol=1e-6)
            for n in set(_g_off) & set(_g_on))
        _why = "numbers differ" if _broke else "NOTHING BROKE -- the gate is not testing the fix"
    except RuntimeError as e:
        _broke, _why = True, f"raised in backward: {str(e)[:60]}"
finally:
    pose_perblock._add_dir = _real_add_dir
check("NEGATIVE CONTROL: the pre-fix code path fails this same comparison", _broke, _why)

print(f"\n  {'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks pass'}")
sys.exit(1 if FAIL else 0)
