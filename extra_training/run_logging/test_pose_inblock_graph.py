#!/usr/bin/env python
"""THE TEST THAT MATTERS: the backward graph starts where --pose-inblock intends it to.

WHY THIS FILE EXISTS. Freezing pairformer blocks 0-2 and block 3's two triangle multiplications is
only a speedup if autograd BUILDS NO GRAPH through them. Autograd starts building at the first op
whose parameters or inputs require grad -- so if the pose injection moves one op earlier, to block
3's input, the `z` entering tri_mul_out requires grad, both tri-muls rejoin the backward, and the
saving evaporates with NO ERROR AND NO WARNING. A slower epoch is the only symptom, and nobody reads
epoch times against a prediction.

This asserts the boundary directly:

  * every parameter in blocks 0, 1, 2 has `.grad is None`
  * blocks.3.tri_mul_out and blocks.3.tri_mul_in have `.grad is None`
  * blocks.3.tri_att_start, tri_att_end, pair_transition and BOTH injection projections have
    non-None, NON-ZERO grads
  * AND THE NEGATIVE CONTROL: with the injection moved to block 3's input, both tri-muls' BACKWARD
    RUNS. Without that control this file proves nothing -- a test that passes because the backward
    did nothing at all is indistinguishable from one that passes because the graph is where we want
    it.

WHAT THE CONTROL HAD TO BE CHANGED TO, AND WHY IT MATTERS. The first version asserted that the
control makes the tri-muls RECEIVE GRADIENT, and it failed -- correctly. Their parameters are
FROZEN, so they never get a `.grad` wherever the pose enters. The cost of the wrong placement is
not a parameter gradient; it is that autograd builds the graph through both tri-muls and RUNS
THEIR BACKWARD to propagate dL/dz. `register_full_backward_hook` fires exactly when a module's
input gradients are computed, so it sees that and a `.grad` check cannot. Had the control been
left as written and simply "fixed" by relaxing it, the file would have been asserting the wrong
thing about the whole change.

CPU ONLY. No checkpoint, no Protenix fork, no GPU. Run:
    /home/ebrahim/Protenix/.venv/bin/python -I run_logging/test_pose_inblock_graph.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))
os.environ.setdefault("LAYERNORM_TYPE", "openfold")

import pose_inblock as PIB                                             # noqa: E402
from protenix.model.modules.pairformer import PairformerStack          # noqa: E402
from protenix.model.modules.primitives import LinearNoBias             # noqa: E402

# THE LIVE CONFIGURATION, which reproduces a block's 4,749,184 parameters exactly. c_z is 256 (NOT
# the 128 in the confidence.py docstring) and hidden_scale_up is True.
CFG = dict(n_heads=16, c_z=256, c_s=384, dropout=0.0, hidden_scale_up=True)
N_TOK, N_BINS, C_Z = 32, 39, 256
LAST = 3
FROZEN_IN_3 = ("tri_mul_out", "tri_mul_in")
TRAINED_IN_3 = ("tri_att_start", "tri_att_end", "pair_transition")
FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


class _Head(torch.nn.Module):
    """Just enough head for pose_inblock's injection: the two projections and the bin edges."""

    def __init__(self):
        super().__init__()
        self.c_z, self.num_bins = C_Z, N_BINS
        self.pose_inblock_d = LinearNoBias(in_features=N_BINS, out_features=C_Z)
        self.pose_inblock_d_wo_onehot = LinearNoBias(in_features=1, out_features=C_Z)
        with torch.no_grad():
            self.pose_inblock_d.weight.zero_()
            self.pose_inblock_d_wo_onehot.weight.zero_()


def _build(inject_at, freeze=True, freeze_tri=True):
    """A 4-block stack with blocks 0-2 frozen (and optionally block 3's tri-muls), plus the pose."""
    torch.manual_seed(0)
    stack = PairformerStack(n_blocks=4, **CFG)
    stack.train()
    head = _Head()
    if freeze:
        for i, b in enumerate(stack.blocks):
            if i != LAST:
                for p in b.parameters():
                    p.requires_grad_(False)
            else:
                for nm in (FROZEN_IN_3 if freeze_tri else ()):
                    for p in getattr(b, nm).parameters():
                        p.requires_grad_(False)
                # THE SINGLE TRACK IS FROZEN TOO, as --lean-single does, so its parameters cannot
                # be mistaken for the ones the injection is supposed to reach.
                for nm in ("attention_pair_bias", "single_transition"):
                    for p in getattr(b, nm).parameters():
                        p.requires_grad_(False)
    # THE POSE GEOMETRY, as the real wrapper stashes it.
    torch.manual_seed(1)
    d = torch.cdist(torch.randn(N_TOK, 3), torch.randn(N_TOK, 3))
    oh = torch.nn.functional.one_hot(
        torch.randint(0, N_BINS, (N_TOK, N_TOK)), N_BINS).to(torch.float32)
    return stack, head, oh, d


def _run(inject_at, freeze=True, freeze_tri=True):
    """Run blocks 0-2 then the mirrored block 3, backward, and report what the backward touched.

    THE BACKWARD HOOKS ARE THE REAL INSTRUMENT WHEN THE TRI-MULS ARE FROZEN. A frozen parameter
    never gets a `.grad` no matter where the injection goes -- but if `z` entering tri_mul_out
    requires grad, autograd still builds the graph through both tri-muls and still RUNS their
    backward to propagate dL/dz. That backward compute is the cost the placement exists to avoid,
    and a parameter-gradient check cannot see it. register_full_backward_hook fires exactly when a
    module's input gradients are computed, so it sees precisely that.
    """
    stack, head, oh, d = _build(inject_at, freeze=freeze, freeze_tri=freeze_tri)
    blk = stack.blocks[LAST]
    bwd, handles = [], []
    for nm in FROZEN_IN_3 + TRAINED_IN_3:
        handles.append(getattr(blk, nm).register_full_backward_hook(
            lambda _m, _gi, _go, _n=nm: bwd.append(_n)))
    torch.manual_seed(2)
    s = torch.randn(N_TOK, 384)
    z = torch.randn(N_TOK, N_TOK, C_Z)       # NOT requires_grad: it comes from a frozen trunk
    # BLOCKS 0-2 RUN UNINJECTED, exactly as they do live.
    for i in range(LAST):
        s, z = stack.blocks[i](s, z, None)
    ok_grad = z.requires_grad
    s, z = PIB.block_forward(head, blk, s, z, oh, d, inject_at=inject_at, pair_mask=None)
    z.sum().backward()
    for h in handles:
        h.remove()
    return stack, head, z, ok_grad, set(bwd)


def _grads(mod):
    return {n: (p.grad is not None, float(p.grad.abs().sum()) if p.grad is not None else 0.0)
            for n, p in mod.named_parameters()}


def main():
    print("1. the injection lands after tri_mul_in: the frozen ops stay out of the backward")
    stack, head, z, pre_grad, bwd = _run("after_tri_mul_in")
    ok(not pre_grad, "z entering block 3 does NOT require grad (blocks 0-2 are frozen)")
    ok(z.requires_grad, "z leaving block 3 DOES require grad (the injection is in the graph)")

    none_012 = [f"blocks.{i}.{n}" for i in range(LAST)
                for n, p in stack.blocks[i].named_parameters() if p.grad is not None]
    ok(not none_012, "every parameter in blocks 0, 1, 2 has .grad is None",
       f"{len(none_012)} did not" if none_012 else "all of them")

    got = []
    for nm in FROZEN_IN_3:
        g = _grads(getattr(stack.blocks[LAST], nm))
        got += [f"{nm}.{k}" for k, (has, _v) in g.items() if has]
    ok(not got, "blocks.3.tri_mul_out and tri_mul_in have .grad is None",
       f"{len(got)} did not: {got[:3]}" if got else "both")

    for nm in TRAINED_IN_3:
        g = _grads(getattr(stack.blocks[LAST], nm))
        n_has = sum(1 for has, _v in g.values() if has)
        n_nz = sum(1 for has, v in g.values() if has and v > 0)
        ok(n_has == len(g) and n_nz > 0, f"blocks.3.{nm} has grads, some non-zero",
           f"{n_has}/{len(g)} with grad, {n_nz} non-zero")

    gi = _grads(head)
    ok(all(has for has, _v in gi.values()), "BOTH injection projections have a grad",
       str({k: round(v, 6) for k, (_h, v) in gi.items()}))
    ok(all(v > 0 for _h, v in gi.values()), "and both are NON-ZERO",
       "zero-init weights still get a gradient, which is the point")

    # THE BACKWARD DID NOT RUN THROUGH THE FROZEN OPS AT ALL -- not merely "produced no .grad".
    ok("tri_mul_out" not in bwd and "tri_mul_in" not in bwd,
       "neither tri-mul's BACKWARD ran (full_backward_hook never fired)", str(sorted(bwd)))
    ok({"tri_att_start", "tri_att_end", "pair_transition"} <= bwd,
       "the three trained ops' backwards DID run", str(sorted(bwd)))

    print("\n2. THE FREEZE IS THE OTHER HALF: without it, placement cannot help")
    # SAID OUT LOUD BECAUSE IT IS EASY TO GET BACKWARDS. If block 3's tri-muls are TRAINABLE, their
    # own weights require grad, so the graph is built from tri_mul_out onwards NO MATTER WHERE the
    # pose enters -- the injection's placement buys exactly nothing. The placement only matters
    # once the tri-muls are frozen, and the freeze only pays once the placement is right. Both,
    # or neither.
    st_a, hd_a, _z, _pre, bwd_a = _run("after_tri_mul_in", freeze_tri=False)
    st_b, hd_b, _z2, _pre2, bwd_b = _run("block_input", freeze_tri=False)
    got_a, got_b = [], []
    for nm in FROZEN_IN_3:
        got_a += [k for k, (has, _v) in _grads(getattr(st_a.blocks[LAST], nm)).items() if has]
        got_b += [k for k, (has, _v) in _grads(getattr(st_b.blocks[LAST], nm)).items() if has]
    ok(len(got_a) > 0 and len(got_b) > 0,
       "with the tri-muls TRAINABLE they get gradient at BOTH placements",
       f"after tri_mul_in {len(got_a)}, at block input {len(got_b)}")
    ok({"tri_mul_out", "tri_mul_in"} <= bwd_a,
       "and their backward runs at both, so the freeze is load-bearing", str(sorted(bwd_a)))

    print("\n3. THE NEGATIVE CONTROL: frozen tri-muls, injection at the block INPUT")
    stack_c, head_c, z_c, pre_c, bwd_c = _run("block_input")
    ok("tri_mul_out" in bwd_c and "tri_mul_in" in bwd_c,
       "at the block input both tri-muls' backward RUNS despite frozen parameters",
       f"fired: {sorted(bwd_c)}")
    ok(bwd_c > bwd, "the control's backward strictly covers more ops than the correct placement",
       f"{len(bwd_c)} op(s) vs {len(bwd)}")
    # AND BLOCKS 0-2 STILL DO NOT, because their parameters are frozen and z entering block 3
    # carries no graph: the control shows the failure is confined to block 3, which is the point.
    ctl012 = [1 for i in range(LAST) for _n, p in stack_c.blocks[i].named_parameters()
              if p.grad is not None]
    ok(not ctl012, "and blocks 0-2 still get nothing even in the control",
       f"{len(ctl012)} did" if ctl012 else "confirmed")
    ok(all(has for has, _v in _grads(head_c).values()),
       "the control's injection still reaches its own projections")

    print("\n4. zero-init means step 0 is unchanged: the injected z equals the un-injected z")
    stack_a, head_a, oh, d = _build("after_tri_mul_in")
    blk_a = stack_a.blocks[LAST]
    torch.manual_seed(2)
    s0 = torch.randn(N_TOK, 384)
    z0 = torch.randn(N_TOK, N_TOK, C_Z)
    with torch.no_grad():
        _s_inj, z_inj = PIB.block_forward(head_a, blk_a, s0.clone(), z0.clone(), oh, d,
                                          inject_at="after_tri_mul_in", pair_mask=None)
        _s_nat, z_nat = blk_a(s0.clone(), z0.clone(), None)
    ok(torch.equal(z_inj, z_nat), "z is BIT-identical to the unmirrored block at zero-init",
       f"max|diff| = {(z_inj - z_nat).abs().max().item():.3e}")
    # AND THE MIRROR IS NOT TRIVIALLY EQUAL BECAUSE IT DID NOTHING: a non-zero injection must move
    # it, or the test above would pass for a mirror that never applies the projections at all.
    with torch.no_grad():
        head_a.pose_inblock_d.weight.fill_(0.01)
        _s2, z_on = PIB.block_forward(head_a, blk_a, s0.clone(), z0.clone(), oh, d,
                                      inject_at="after_tri_mul_in", pair_mask=None)
    ok(not torch.equal(z_on, z_nat), "a NON-zero injection does move z (the mirror applies it)",
       f"max|diff| = {(z_on - z_nat).abs().max().item():.3e}")

    print("\n5. the mirror refuses what it does not mirror")
    try:
        PIB.block_forward(head_a, blk_a, s0, z0, oh, d, inject_at="after_tri_mul_in",
                          pair_mask=None, inplace_safe=True)
        ok(False, "inplace_safe=True is refused")
    except SystemExit:
        ok(True, "inplace_safe=True is refused", "the branch is not mirrored")
    try:
        PIB.block_forward(head_a, blk_a, s0, z0, oh, d, inject_at="nowhere", pair_mask=None)
        ok(False, "an unknown inject_at is refused")
    except SystemExit:
        ok(True, "an unknown inject_at is refused")

    print("\n6. the pin matches the installed Protenix")
    from protenix.model.modules.pairformer import PairformerBlock
    try:
        PIB._pin(PairformerBlock.forward, PIB.PINNED_BLOCK_SHA, "PairformerBlock.forward")
        ok(True, "PairformerBlock.forward is at its pinned sha", PIB.PINNED_BLOCK_SHA[:16])
    except SystemExit as e:
        ok(False, "PairformerBlock.forward is at its pinned sha", str(e).splitlines()[0])

    print("\n7. the --train-only pattern reaches exactly the intended tensors")
    import re
    pat = [p for p in (r"pairformer_stack\.blocks\.3\.tri_att_start\.",
                       r"pairformer_stack\.blocks\.3\.tri_att_end\.",
                       r"pairformer_stack\.blocks\.3\.pair_transition\.",
                       r"^pose_inblock_")]
    names = [f"pairformer_stack.blocks.{i}.{n}"
             for i in range(4) for n, _p in stack.blocks[i].named_parameters()] + \
            [n for n, _p in head.named_parameters()]
    hit = [n for n in names if any(re.search(p, n) for p in pat)]
    bad_tri = [n for n in hit if "tri_mul_out" in n or "tri_mul_in" in n]
    bad_blk = [n for n in hit if ".blocks.0." in n or ".blocks.1." in n or ".blocks.2." in n]
    ok(not bad_tri, "the pattern does NOT match block 3's tri-muls", str(bad_tri[:3]))
    ok(not bad_blk, "and does not match blocks 0-2", str(bad_blk[:3]))
    ok(len([n for n in hit if n.startswith("pose_inblock_")]) == 2,
       "it matches both injection projections",
       str([n for n in hit if n.startswith("pose_inblock_")]))
    for nm in TRAINED_IN_3:
        ok(any(f".blocks.3.{nm}." in n for n in hit), f"and block 3's {nm}")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
