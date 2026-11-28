#!/usr/bin/env python
"""Gate: the nonlinear pose branch (phases/pose_mlp.py) is installed, wired at all six sites, and
INERT at step 0 -- each asserted, none assumed.

Run: python run_logging/test_pose_mlp.py      (CPU only, no GPU, no Protenix model)

WHY A SYNTHETIC HEAD. The real confidence head needs the 19.8M-parameter checkpoint and a trunk
cache; the mechanism under test is the install() calls, the geometry cache, the per-block capture
and the two pae_ln hooks, all of which are this project's code and none of which need the real
weights. The harness therefore stands in for confidence.py's memory_efficient_forward -- the two
pretrained distance terms, the stack under torch.utils.checkpoint, pae_ln, the PAE readout -- and
runs the REAL pose_direction, pose_perblock, pose_readout and pose_mlp installs on top of it. The
dimensions are the REAL ones (39 bins, c_z 128, 4 blocks), so the parameter count this prints is
the number an arm will train.

WHAT EACH CHECK EXISTS FOR, in the order they run:

BIT-IDENTITY. The whole safety story is that --pose-mlp at step 0 is the same function as the arm
without it, so anything that moves afterwards is attributable to the branch. torch.equal, not
allclose: "close" would hide a bias of 1e-8 added at six sites.

IT CHANGES SOMETHING. A flag that is inert at step 0 and inert FOREVER is the failure this project
has hit repeatedly -- parameters created and never reached. With non-zero second layers the output
must MOVE, at every site independently, which is six separate checks that each site is really
wired and really indexed to its own weights.

THE TRANSPOSE AND THE DIAGONAL. reverse[i, j] must BE forward[j, i] -- it is the cache transposed,
so this asserts the transpose is over the two token dimensions and not, say, over the last two.
Both are zero on the diagonal, which is what the head already sees from the existing terms there.

THE LENGTH. It must be the CA-CA distance, symmetric, zero on the diagonal, and NOT masked by the
degenerate-frame mask. It must also be the length of the SAME arrow the direction describes, which
is checked by multiplying them back together.

THE RAW-DISTANCE PATH IS UNTOUCHED. The branch divides the distance by 50; the pretrained
linear_no_bias_d_wo_onehot must still receive Angstroms. Checked by hooking that layer and
comparing its input against cdist, because `dd /= 50` instead of `dd / 50` would be invisible in
the output of a zero-init branch and would quietly rescale three pretrained layers.

GRADIENTS. Layer 1 can only be reached through `h`, so the branch's whole hypothesis rests on that
gradient existing. It is zero at step 0 by construction (layer 2 is zero), which is documented
rather than a defect, and non-zero once layer 2 is -- both asserted, because "no gradient" and "no
gradient yet" are the same measurement with opposite meanings.

ACTIVATION CHECKPOINTING. Sites 1-3 are inside blocks that are RE-RUN during backward. `h` is an
activation of trainable weights, not geometry, so if it were read from the module-level cache
inside the block the recomputation would see None or a LATER pose's tensor. Forward AND every
gradient are compared between blocks_per_ckpt=1 and off, over several poses with DIFFERENT geometry
each, with non-zero second layers -- and a NEGATIVE CONTROL restores the global read and asserts
the comparison then breaks, so the gate cannot pass vacuously.

THE REFUSALS. pose_mlp claims six sites. Installed without one of the modules that owns those
sites it must refuse, not run at four and report six.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import torch                                                                 # noqa: E402
import torch.nn as nn                                                        # noqa: E402
from torch.utils.checkpoint import checkpoint                                # noqa: E402

import pose_direction                                                        # noqa: E402
import pose_mlp                                                              # noqa: E402
import pose_perblock                                                         # noqa: E402
import pose_readout                                                          # noqa: E402
from protenix.model.modules.primitives import LinearNoBias                   # noqa: E402
from protenix.model.utils import one_hot                                      # noqa: E402

# THE REAL DIMENSIONS, so the parameter count below is the one an arm pays for.
N_TOK, C_Z, N_BINS, N_BLOCKS, N_POSE = 6, 128, 39, 4, 3
ATOMS_PER_TOK = 4                       # N, CA, C, CB -- tokatom 0, 1, 2, 3
N_ATOM = N_TOK * ATOMS_PER_TOK
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


class Block(nn.Module):
    def __init__(self, c_z, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(c_z, c_z)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(c_z, c_z, generator=g) * 0.05)
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
    """Stands in for confidence.py: the two pretrained distance terms, the stack, pae_ln, pae."""

    def __init__(self):
        super().__init__()
        self.c_z, self.num_bins = C_Z, N_BINS
        self.lower_bins = torch.linspace(3.25, 50.75, N_BINS)
        self.upper_bins = torch.cat([self.lower_bins[1:], torch.tensor([1e9])])
        self.pairformer_stack = Stack()
        self.linear_no_bias_d = LinearNoBias(in_features=N_BINS, out_features=C_Z)
        self.linear_no_bias_d_wo_onehot = LinearNoBias(in_features=1, out_features=C_Z)
        self.pae_ln = nn.LayerNorm(C_Z)
        self.linear_no_bias_pae = LinearNoBias(in_features=C_Z, out_features=8)

    def forward(self, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords,
                **kw):
        """Loops the samples exactly as confidence.py does, so the index check can verify it."""
        rep = input_feature_dict["distogram_rep_atom_mask"].bool()
        outs = []
        for i in range(x_pred_coords.shape[0]):
            outs.append(self.memory_efficient_forward(
                input_feature_dict=input_feature_dict, s_trunk=s_trunk, z_pair=z_trunk,
                pair_mask=pair_mask, x_pred_rep_coords=x_pred_coords[i][rep]))
        return torch.stack(outs, dim=0)

    def memory_efficient_forward(self, input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        c = x_pred_rep_coords.to(torch.float32)
        d = torch.cdist(c, c)
        z = z_pair + self.linear_no_bias_d(
            one_hot(x=d, lower_bins=self.lower_bins, upper_bins=self.upper_bins))
        z = z + self.linear_no_bias_d_wo_onehot(d.unsqueeze(-1))
        blocks = self.pairformer_stack._prep_blocks(pair_mask=pair_mask)
        bpc = self.pairformer_stack.blocks_per_ckpt
        for fn in blocks:
            if bpc and torch.is_grad_enabled():
                z = checkpoint(fn, s_trunk, z, use_reentrant=False)
            else:
                z = fn(s_trunk, z)
        # EXACTLY WHAT confidence.py DOES WITH z BEFORE THE READOUT, and it is the condition the
        # dtype branch in pose_mlp.add_site exists for: the pair tensor is upcast to float32 and
        # the whole readout runs with AUTOCAST OFF (confidence.py:321 and :330). Under AMP `h` is
        # bf16 and the pose_mlp_out weights are float32, and with autocast off torch reconciles
        # nothing. Omitting these two lines is why the first version of this gate never once
        # executed that branch -- 37 checks, all on the one dtype combination that cannot fail.
        z = z.to(torch.float32)
        with torch.amp.autocast("cpu", enabled=False):
            return self.linear_no_bias_pae(self.pae_ln(z))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence_head = Head()


def geometry(seed):
    """Coordinates and the frame definition for one synthetic unit. N, CA, C, CB per token."""
    g = torch.Generator().manual_seed(seed)
    ca = torch.randn(N_TOK, 3, generator=g) * 12.0
    full = torch.zeros(N_ATOM, 3)
    for t in range(N_TOK):
        off = t * ATOMS_PER_TOK
        full[off + 0] = ca[t] + torch.randn(3, generator=g) * 0.3 + torch.tensor([1.4, 0.0, 0.0])
        full[off + 1] = ca[t]
        full[off + 2] = ca[t] + torch.randn(3, generator=g) * 0.3 + torch.tensor([0.0, 1.5, 0.0])
        full[off + 3] = ca[t] + torch.randn(3, generator=g) * 0.4 + torch.tensor([0.0, 0.0, 1.5])
    fai = torch.tensor([[t * ATOMS_PER_TOK + k for k in (0, 1, 2)] for t in range(N_TOK)])
    has_frame = torch.ones(N_TOK, dtype=torch.bool)
    pae_mask = torch.zeros(N_ATOM, dtype=torch.bool)
    pae_mask[1::ATOMS_PER_TOK] = True                      # CA
    dis_mask = torch.zeros(N_ATOM, dtype=torch.bool)
    dis_mask[3::ATOMS_PER_TOK] = True                      # CB
    return full, fai, has_frame, pae_mask, dis_mask


def build(with_mlp=True):
    # BOTH RNGs, AND numpy IS THE ONE THAT MATTERS. Protenix's Linear initialises its weight
    # through trunc_normal_init_, which samples with scipy's truncnorm -- i.e. from NUMPY's global
    # RNG, which torch.manual_seed does not touch. So two Models built under the same torch seed
    # carry DIFFERENT linear_no_bias_d weights, and the bit-identity check below was comparing two
    # different models rather than two configurations of one: it reported max|d| 3.2 against a
    # branch whose every second layer is exactly zero. Found by this gate, on its first run.
    import numpy as np
    np.random.seed(0)
    torch.manual_seed(0)
    m = Model()
    pose_direction.install(m, loud=False)
    pose_perblock.install(m, loud=False)
    pose_readout.install(m, loud=False, sites=("pre", "post"))
    info = pose_mlp.install(m, loud=False) if with_mlp else None
    return m, info


def run(m, seeds=(500,), bpc=None, backward=False):
    """One forward per pose, each with its OWN geometry, through the PATCHED head.forward."""
    h = m.confidence_head
    h.pairformer_stack.blocks_per_ckpt = bpc
    m.zero_grad(set_to_none=True)
    total, outs = None, []
    for sd in seeds:
        full, fai, hf, pae_mask, dis_mask = geometry(sd)
        pose_direction.set_unit(fai, hf, pae_mask, pdb=f"synth{sd}")
        gg = torch.Generator().manual_seed(sd + 1)
        z = torch.randn(N_TOK, N_TOK, C_Z, generator=gg)
        s = torch.randn(N_TOK, C_Z, generator=gg)
        try:
            out = h.forward(input_feature_dict={"distogram_rep_atom_mask": dis_mask},
                            s_inputs=None, s_trunk=s, z_trunk=z, pair_mask=None,
                            x_pred_coords=full.unsqueeze(0))
        finally:
            pose_direction.clear_unit()
        outs.append(out)
        total = out.square().mean() if total is None else total + out.square().mean()
    grads = {}
    if backward:
        total.backward()
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
                 if p.grad is not None}
    return torch.cat([o.reshape(-1) for o in outs]), grads


# ---- 1. SHAPES AND THE PARAMETER COUNT --------------------------------------------------------
m, info = build()
h = m.confidence_head
print("\n  new tensors:")
for n, p in m.named_parameters():
    if "pose_mlp" in n:
        print(f"    {n.split('confidence_head.')[-1]:28} {tuple(p.shape)}")
EXPECT = N_BINS * 64 + 8 * 64 + 64 + 6 * (64 * C_Z + C_Z)
check("parameter count is n_bins*64 + 8*64 + 64 + 6*(64*c_z + c_z)",
      info["params"] == EXPECT == 52992, f"{info['params']:,} (expect {EXPECT:,})")
check("six second layers, one per site", info["sites"] == 6 and len(h.pose_mlp_out) == 6,
      f"{info['sites']}")
check("15 new tensors, all selectable by the pattern `pose_mlp`",
      len([n for n, _ in m.named_parameters() if "pose_mlp" in n]) == 15)
check("layer 1 reads the bin count from linear_no_bias_d.in_features",
      h.pose_mlp_onehot.in_features == h.linear_no_bias_d.in_features == N_BINS)
check("layer 2 weight AND bias are exactly zero at install",
      all(float(o.weight.abs().max()) == 0.0 and float(o.bias.abs().max()) == 0.0
          for o in h.pose_mlp_out))
check("layer 1 is NOT zero (default PyTorch init, so `h` is non-zero and layer 2 can move)",
      float(h.pose_mlp_onehot.weight.abs().max()) > 0
      and float(h.pose_mlp_cont.weight.abs().max()) > 0)

# ---- 2. BIT-IDENTITY AT STEP 0 ----------------------------------------------------------------
m_off, _ = build(with_mlp=False)
out_off, _ = run(m_off, seeds=(500, 501))
out_on, _ = run(m, seeds=(500, 501))
check("flag ON at fresh init is BIT-IDENTICAL to flag OFF", torch.equal(out_off, out_on),
      f"max|d| {float((out_off - out_on).abs().max()):.3e}")

# ---- 3. THE GEOMETRY INPUTS -------------------------------------------------------------------
# The caches are cleared on the way out, so they are read from inside a forward, which is also the
# only honest place to read them.
seen = {}
_real_prestack = pose_mlp.prestack


def _spy(head, z):
    import pose_direction as _pd
    import pose_perblock as _ppb
    seen["dir"] = _pd._CALL["dir"].clone()
    seen["len"] = _pd._CALL["len"].clone()
    seen["dist"] = _ppb._CALL["dist"].clone()
    return _real_prestack(head, z)


pose_mlp.prestack = _spy
try:
    run(m, seeds=(500,))
finally:
    pose_mlp.prestack = _real_prestack

dr, ln = seen["dir"], seen["len"]
rev = dr.transpose(-3, -2)
check("reverse[i, j] == forward[j, i] for every pair",
      all(torch.equal(rev[i, j], dr[j, i]) for i in range(N_TOK) for j in range(N_TOK)))
check("both are zero on the diagonal",
      float(torch.stack([dr[i, i] for i in range(N_TOK)]).abs().max()) == 0.0
      and float(torch.stack([rev[i, i] for i in range(N_TOK)]).abs().max()) == 0.0)
# THE REVERSE IS NOT THE NEGATED FORWARD, so it is a real second input and not 3 redundant numbers.
check("reverse is NOT -forward (j's frame is a different rotation, so the input is not redundant)",
      not torch.allclose(rev, -dr, atol=1e-3),
      f"max|rev+fwd| {float((rev + dr).abs().max()):.3f}")

full, fai, hf, pae_mask, dis_mask = geometry(500)
ca = full[pae_mask]
d_ca = torch.cdist(ca, ca)
check("the cached length IS the CA-CA distance (same atoms as the direction)",
      torch.allclose(ln.squeeze(-1), d_ca, atol=1e-4),
      f"max|d| {float((ln.squeeze(-1) - d_ca).abs().max()):.3e}")
check("the cached length is symmetric and zero on the diagonal",
      torch.allclose(ln, ln.transpose(-3, -2), atol=1e-5)
      and float(torch.stack([ln[i, i] for i in range(N_TOK)]).abs().max()) == 0.0)
# THE LENGTH IS THE LENGTH OF THE DIRECTION'S OWN ARROW, which is the reason it is this distance
# and not the one the one-hot is built from: direction * length is j's CA in i's frame.
check("direction * length reproduces j's CA in i's frame (one arrow, not two)",
      float((torch.linalg.vector_norm(dr * ln, dim=-1) - d_ca).abs().max()) < 1e-3,
      f"max|d| {float((torch.linalg.vector_norm(dr * ln, dim=-1) - d_ca).abs().max()):.3e}")
# THE TWO DISTANCES ARE OVER DIFFERENT ATOMS, DELIBERATELY. Asserted so that a future change which
# "fixes" one to match the other fails here and has to read the comment.
cb = full[dis_mask]
check("the one-hot distance is CB-CB and DIFFERS from the CA-CA length, by design",
      torch.allclose(seen["dist"], torch.cdist(cb, cb), atol=1e-4)
      and not torch.allclose(seen["dist"], d_ca, atol=0.2))

# ---- 4. THE PRETRAINED RAW-DISTANCE LAYER STILL SEES ANGSTROMS --------------------------------
got = {}
hk = h.linear_no_bias_d_wo_onehot.register_forward_hook(
    lambda _m, inp, _o: got.setdefault("x", inp[0].detach().clone()))
try:
    run(m, seeds=(500,))
finally:
    hk.remove()
check("linear_no_bias_d_wo_onehot still receives UNSCALED Angstroms (the /50 is on a copy)",
      torch.allclose(got["x"].squeeze(-1), torch.cdist(cb, cb), atol=1e-4),
      f"max {float(got['x'].max()):.2f} A")

# ---- 5. THE CACHES ARE RELEASED ---------------------------------------------------------------
check("pose_mlp's hidden tensor is cleared after the forward", pose_mlp._CALL["h"] is None)
check("pose_direction's length cache is cleared after the forward",
      pose_direction._CALL["len"] is None and pose_direction._CALL["dir"] is None)

# ---- 6. GRADIENT LIFE -------------------------------------------------------------------------
_, g0 = run(m, seeds=(500,), backward=True)
check("layer 2 HAS a gradient at step 0 (so the branch is not gradient-dead)",
      all(float(g0[f"confidence_head.pose_mlp_out.{k}.weight"].abs().max()) > 0
          for k in range(6)))
check("layer 1 has NO gradient at step 0, which is what zero second layers mean",
      float(g0["confidence_head.pose_mlp_onehot.weight"].abs().max()) == 0.0
      and float(g0["confidence_head.pose_mlp_cont.weight"].abs().max()) == 0.0)


def wake(model, scale=0.02):
    """Non-zero second layers, DIFFERENT at every site, as a trained arm would have."""
    with torch.no_grad():
        g = torch.Generator().manual_seed(21)
        for k, o in enumerate(model.confidence_head.pose_mlp_out):
            o.weight.copy_(torch.randn(C_Z, 64, generator=g) * (scale + 0.004 * k))
            o.bias.copy_(torch.randn(C_Z, generator=g) * scale)


wake(m)
_, g1 = run(m, seeds=(500,), backward=True)
check("layer 1 receives a gradient once layer 2 is non-zero (the two-step wake-up)",
      float(g1["confidence_head.pose_mlp_onehot.weight"].abs().max()) > 0
      and float(g1["confidence_head.pose_mlp_cont.weight"].abs().max()) > 0
      and float(g1["confidence_head.pose_mlp_cont.bias"].abs().max()) > 0)

# ---- 7. EVERY SITE IS REALLY WIRED ------------------------------------------------------------
# ONE SITE AT A TIME, because six sites sharing one hidden tensor is exactly the shape of bug where
# five are wired and the sixth silently is not -- and the per-site index is the thing that would
# be wrong.
base, _ = run(m_off, seeds=(500,))
for k in range(6):
    m_k, _ = build()
    with torch.no_grad():
        # PER-CHANNEL, NOT A CONSTANT FILL. A constant weight makes the delta the same in every
        # channel, delta = c * 1, and pae_ln subtracts the channel mean -- so site 4's injection
        # cancels EXACTLY and this check passed on 3.9e-07 of layernorm round-off while the other
        # five read 5e-02. It still distinguished wired from unwired, but it measured nothing
        # about what site 4 injects, and a more exact layernorm would have failed it spuriously.
        gk = torch.Generator().manual_seed(900 + k)
        m_k.confidence_head.pose_mlp_out[k].weight.copy_(
            torch.randn(C_Z, 64, generator=gk) * 0.02)
    out_k, _ = run(m_k, seeds=(500,))
    check(f"site {k} changes the output when only ITS second layer is non-zero",
          not torch.equal(base, out_k),
          f"max|d| {float((base - out_k).abs().max()):.3e}")

# ---- 8. ACTIVATION CHECKPOINTING --------------------------------------------------------------
m_ck, _ = build()
wake(m_ck)
POSES = (500, 501, 502)
o_off, gc_off = run(m_ck, seeds=POSES, bpc=None, backward=True)
o_on, gc_on = run(m_ck, seeds=POSES, bpc=1, backward=True)
check("forward matches with activation checkpointing on and off",
      torch.allclose(o_off, o_on, rtol=0, atol=1e-5),
      f"max|d| {float((o_off - o_on).abs().max()):.3e}")
shared = sorted(set(gc_off) & set(gc_on))
worst = max(((float((gc_off[n] - gc_on[n]).abs().max()), n) for n in shared), default=(0.0, "-"))
check(f"EVERY gradient matches under checkpointing ({len(shared)} tensors)", worst[0] <= 1e-4,
      f"worst {worst[0]:.3e} at {worst[1]}")
check("layer 1's gradient is non-zero under checkpointing, so it really reached the blocks",
      float(gc_on["confidence_head.pose_mlp_cont.weight"].abs().max()) > 0)

# NEGATIVE CONTROL. A gate that cannot fail proves nothing. This is the bug the capture in
# _prep_blocks exists to prevent: the per-block site reading the module-level cache instead of the
# tensor it was handed. Under checkpointing the recomputation then sees whatever the cache holds
# during backward -- None, or another pose's hidden tensor.
_real_add_site = pose_mlp.add_site


def _old_add_site(head, k, z, _h):
    g = pose_mlp._CALL.get("h")
    if g is None:
        raise RuntimeError("old path: the hidden cache was cleared before backward")
    return _real_add_site(head, k, z, g)


pose_mlp.add_site = _old_add_site
try:
    m_bad, _ = build()
    wake(m_bad)
    _b_off, _gb_off = run(m_bad, seeds=POSES, bpc=None, backward=True)
    try:
        _b_on, _gb_on = run(m_bad, seeds=POSES, bpc=1, backward=True)
        _broke = not torch.allclose(_b_off, _b_on, rtol=0, atol=1e-5) or any(
            not torch.allclose(_gb_off[n], _gb_on[n], rtol=0, atol=1e-4)
            for n in set(_gb_off) & set(_gb_on))
        _why = "numbers differ" if _broke else "NOTHING BROKE -- the gate is not testing the fix"
    except RuntimeError as e:
        _broke, _why = True, f"raised: {str(e)[:70]}"
finally:
    pose_mlp.add_site = _real_add_site
check("NEGATIVE CONTROL: reading the cache inside the block fails this comparison", _broke, _why)

# ---- 9. LAYER 1 IS THE SAME ON EVERY RANK ----------------------------------------------------
# THE ONE FAILURE IN THIS DESIGN THAT NOTHING DOWNSTREAM CAN SEE. The trainer builds one replica
# per rank and all-reduces HEAD GRADIENTS only -- there is no parameter broadcast (see
# phase_k_train.py:2063) -- and it never seeds torch's RNG at all. Every module this project has
# added before now was zero-init or a copy of checkpoint weights, so the question never arose;
# layer 1 is the first added parameter drawn from an RNG. Built from the global generator, three
# ranks would hold three different first layers, the all-reduce would sum gradients taken at three
# different parameter values, and the shapes, the key set, --expect-trainable and the resume guard
# would all agree. Simulated here by moving BOTH global RNGs between two builds, which is the
# cheapest stand-in for "another process, started a moment later".
torch.manual_seed(999)
import numpy as _np                                                          # noqa: E402
_np.random.seed(7)
_ = torch.randn(64)
m_r2, _ = build()
check("layer 1 is bit-identical across two independent builds (so across ranks)",
      torch.equal(h.pose_mlp_onehot.weight, m_r2.confidence_head.pose_mlp_onehot.weight)
      and torch.equal(h.pose_mlp_cont.weight, m_r2.confidence_head.pose_mlp_cont.weight)
      and torch.equal(h.pose_mlp_cont.bias, m_r2.confidence_head.pose_mlp_cont.bias))
# AND IT IS NOT IDENTICAL BY BEING ZERO, which would make the check above vacuous.
check("...and it is a real random draw, not zeros",
      float(h.pose_mlp_cont.bias.abs().max()) > 0
      and len(torch.unique(h.pose_mlp_onehot.weight)) > 100)

# ---- 10. THE AUTOCAST DTYPE PATH AT THE TWO READOUT SITES ------------------------------------
# `h` is bf16 under AMP and the pose_mlp_out weights are float32, and confidence.py runs the
# readout with autocast OFF, so the module call raises there. add_site casts the WEIGHT down
# instead. Exercised under real bf16 autocast rather than reasoned about.
m_ac, _ = build()
wake(m_ac)
dt = {}
_rp = pose_mlp.prestack


def _note(head_, z_):
    out = _rp(head_, z_)
    dt["h"] = pose_mlp._CALL["h"].dtype
    return out


pose_mlp.prestack = _note
try:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_ac, _ = run(m_ac, seeds=(500,))
    _raised = ""
except RuntimeError as e:                                                    # noqa: BLE001
    out_ac, _raised = None, str(e)[:90]
finally:
    pose_mlp.prestack = _rp
check("a bf16-autocast forward runs at all six sites without a dtype error", not _raised,
      _raised or f"h was {dt.get('h')}")
check("`h` really was bf16, so the cast branch was the one taken",
      dt.get("h") == torch.bfloat16, str(dt.get("h")))
check("the bf16 forward is finite", out_ac is not None and bool(torch.isfinite(out_ac).all()))
# NEGATIVE CONTROL for the cast: the module call this replaces does raise, so the cast is
# load-bearing rather than defensive.
_probe = torch.nn.Linear(64, 8)
with torch.amp.autocast("cpu", enabled=False):
    try:
        _probe(torch.zeros(2, 64, dtype=torch.bfloat16))
        _mismatch = ""
    except RuntimeError as e:                                                # noqa: BLE001
        _mismatch = str(e)[:60]
check("NEGATIVE CONTROL: the uncast module call raises on bf16 against a float32 weight",
      bool(_mismatch), _mismatch or "NOTHING RAISED -- the cast in add_site is not load-bearing")

# ---- 11. THE REFUSALS --------------------------------------------------------------------------
def refuses(fn, want):
    try:
        fn()
    except RuntimeError as e:
        return want in str(e)
    return False


def _no_perblock():
    torch.manual_seed(0)
    mm = Model()
    pose_direction.install(mm, loud=False)
    pose_mlp.install(mm, loud=False)


def _no_both_readout_sites():
    torch.manual_seed(0)
    mm = Model()
    pose_direction.install(mm, loud=False)
    pose_perblock.install(mm, loud=False)
    pose_readout.install(mm, loud=False, sites=("post",))
    pose_mlp.install(mm, loud=False)


def _twice():
    mm, _ = build()
    pose_mlp.install(mm, loud=False)


check("refuses without --pose-perblock and --pose-readout, naming both",
      refuses(_no_perblock, "--pose-perblock"))
check("refuses with only one readout site (5 sites, not 6)",
      refuses(_no_both_readout_sites, "--pose-readout-prenorm"))
check("refuses a second install", refuses(_twice, "ran twice"))

print(f"\n  {'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks pass'}")
sys.exit(1 if FAIL else 0)
