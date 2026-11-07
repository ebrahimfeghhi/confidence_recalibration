#!/usr/bin/env python
"""CPU test for --lean-single: the pair track is BIT-IDENTICAL and the single track was already dead.

THE TWO CLAIMS THIS HAS TO ESTABLISH, because the flag turns off real compute:

  1  z NEVER DEPENDS ON s. PairformerBlock.forward updates z through five ops that are functions
     of z alone, and only then updates s. If that is true, turning the single track off cannot
     move a single PAE logit. Asserted BIT-IDENTICALLY, not to a tolerance: this is an op-removal,
     not a reordering, so anything other than exact equality means the premise is wrong.

  2  THE SINGLE TRACK'S PARAMETERS WERE ALREADY GETTING NO GRADIENT. This is the part that makes
     dropping 10.55M parameters from AdamW a no-op rather than a capacity cut. It is checked with
     the track ON: if attention_pair_bias and single_transition have `.grad is None` after a
     backward through z even when they RUN, then they were dead before this flag existed and the
     flag only stops paying for them.

  3  AND THE FLAG MUST ACTUALLY CHANGE SOMETHING -- the failure mode --blocks-per-ckpt had for its
     whole life. The modules are asserted to be NOT CALLED, by a forward hook, rather than merely
     unused.

No GPU, no checkpoint, no Protenix fork. Run:
    /home/ebrahim/Protenix/.venv/bin/python -I run_logging/test_lean_single.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))
os.environ.setdefault("LAYERNORM_TYPE", "openfold")

import lean_readouts as LS                                              # noqa: E402
from protenix.model.modules.pairformer import (                         # noqa: E402
    PairformerBlock, PairformerStack)

# THE LIVE CONFIGURATION, not Protenix's defaults. These reproduce the head's 19,752,270 parameters
# and a block's 4,749,184 exactly; c_z is 256 (NOT the 128 in the confidence.py docstring) and
# hidden_scale_up is True, which is what makes a block 4,749,184 rather than 4,091,264.
CFG = dict(n_heads=16, c_z=256, c_s=384, dropout=0.0, hidden_scale_up=True)
N_TOK = 48
FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def _block():
    torch.manual_seed(0)
    b = PairformerBlock(**CFG)
    b.train()
    return b


def _fwd_bwd(b, seed=1):
    torch.manual_seed(seed)
    s = torch.randn(N_TOK, 384)
    z = torch.randn(N_TOK, N_TOK, 256, requires_grad=True)
    called = []
    h1 = b.attention_pair_bias.register_forward_hook(lambda *_a: called.append("apb"))
    h2 = b.single_transition.register_forward_hook(lambda *_a: called.append("st"))
    try:
        so, zo = b(s, z, None)
        zo.sum().backward()
    finally:
        h1.remove(); h2.remove()
    return so, zo.detach().clone(), called


def main():
    print("1. z is bit-identical with the single track on and off")
    full = _block()
    lean = _block()
    lean.c_s = 0
    s_f, z_f, called_f = _fwd_bwd(full)
    s_l, z_l, called_l = _fwd_bwd(lean)
    ok(torch.equal(z_f, z_l), "z is BIT-identical",
       f"max|diff| = {(z_f - z_l).abs().max().item():.3e}")
    ok(z_f.shape == z_l.shape == (N_TOK, N_TOK, 256), "z keeps its shape", str(tuple(z_f.shape)))

    print("\n2. the single track was ALREADY dead -- it gets no gradient even when it runs")
    dead_names = [n for n, _p in full.named_parameters()
                  if n.startswith(("attention_pair_bias.", "single_transition."))]
    ok(len(dead_names) > 0, "the single track has parameters at all", f"{len(dead_names)} tensor(s)")
    got_grad_on = [n for n, p in full.named_parameters()
                   if n in set(dead_names) and p.grad is not None]
    ok(not got_grad_on, "with the track ON, none of them receives a gradient from z",
       f"{len(got_grad_on)} did" if got_grad_on else f"0 of {len(dead_names)}")
    # AND THE PAIR TRACK DOES, so the test is not passing because the backward did nothing.
    live_grad = [n for n, p in full.named_parameters()
                 if n not in set(dead_names) and p.grad is not None]
    nonzero = [n for n, p in full.named_parameters()
               if n not in set(dead_names) and p.grad is not None and p.grad.abs().sum() > 0]
    ok(len(live_grad) > 20, "the pair track DOES receive gradient (the backward was real)",
       f"{len(live_grad)} tensor(s) with a grad")
    # NON-ZERO IS A WEAKER CLAIM THAN IT LOOKS and only 5 of 41 satisfy it, because several of
    # Protenix's output projections are zero-initialised -- which makes the gradient of everything
    # feeding them exactly zero at init. That is init, not a dead path, so the real assertion is
    # "has a .grad at all"; this one only has to be non-empty.
    ok(len(nonzero) > 0, "and at least one of them is non-zero at init",
       f"{len(nonzero)} of {len(live_grad)}")

    print("\n3. the flag changes something: the modules are NOT CALLED")
    ok(called_f == ["apb", "st"], "with the track on, both modules are called", str(called_f))
    ok(called_l == [], "with c_s=0, NEITHER module is called", str(called_l))
    ok(s_l is not None and torch.equal(s_l, s_f * 0 + s_l),
       "s passes through rather than becoming None")

    print("\n4. s is returned UNCHANGED by a c_s=0 block, so the mirror's .to(float32) is safe")
    torch.manual_seed(1)
    s_in = torch.randn(N_TOK, 384)
    z_in = torch.randn(N_TOK, N_TOK, 256)
    s_out, _ = lean(s_in, z_in, None)
    ok(s_out is s_in, "the same tensor object comes back out")

    print("\n5. the stack passes s through too, so one attribute per block is enough")
    torch.manual_seed(0)
    st = PairformerStack(n_blocks=2, n_heads=16, c_z=256, c_s=384, dropout=0.0,
                         hidden_scale_up=True)
    st.train()
    for b in st.blocks:
        b.c_s = 0
    s_o, z_o = st(s_in, z_in, None)
    ok(s_o is s_in, "PairformerStack returns the input s untouched")
    ok(z_o.shape == z_in.shape, "and a z of the right shape", str(tuple(z_o.shape)))

    print("\n6. the pins match the installed Protenix")
    for fn, want, where in ((PairformerBlock.forward, LS.PINNED_BLOCK_SHA, "PairformerBlock.forward"),
                            (PairformerStack.forward, LS.PINNED_STACK_SHA, "PairformerStack.forward")):
        try:
            LS._pin(fn, want, where)
            ok(True, f"{where} is at its pinned sha", want[:16])
        except SystemExit as e:
            ok(False, f"{where} is at its pinned sha", str(e).splitlines()[0])

    print("\n7. the dead-parameter list is the one the measurement claims")
    # A STAND-IN HEAD, so this does not need a checkpoint. Only the names matter.
    class _H(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pairformer_stack = PairformerStack(n_blocks=4, n_heads=16, c_z=256, c_s=384,
                                                    dropout=0.0, hidden_scale_up=True)
            self.plddt_weight = torch.nn.Parameter(torch.zeros(24, 384, 50))
            self.resolved_weight = torch.nn.Parameter(torch.zeros(24, 384, 2))
            self.linear_no_bias_pae = torch.nn.Linear(256, 64, bias=False)
            self.linear_no_bias_pde = torch.nn.Linear(256, 64, bias=False)
            self.pae_ln = torch.nn.LayerNorm(256)
            self.pde_ln = torch.nn.LayerNorm(256)
            self.plddt_ln = torch.nn.LayerNorm(384)
            self.resolved_ln = torch.nn.LayerNorm(384)
            self.input_strunk_ln = torch.nn.LayerNorm(384)
            self.linear_no_bias_s1 = torch.nn.Linear(449, 256, bias=False)
            self.linear_no_bias_s2 = torch.nn.Linear(449, 256, bias=False)
            self.linear_no_bias_d = torch.nn.Linear(39, 256, bias=False)
            self.linear_no_bias_d_wo_onehot = torch.nn.Linear(1, 256, bias=False)
    h = _H()
    tot = sum(p.numel() for p in h.parameters())
    dead = LS.dead_head_params(h, single_dropped=True)
    n_dead = sum(p.numel() for n, p in h.named_parameters() if n in set(dead))
    # THE STAND-IN IS 78 PARAMETERS SHORT OF THE REAL HEAD (lower_bins + upper_bins, 39 each,
    # which are requires_grad=False buffers-as-Parameters and are in neither set), so 19,752,192
    # here against 19,752,270 live. The dead count is identical either way.
    ok(n_dead == 10_551_552, "the dead set is 10,551,552 parameters", f"{n_dead:,}")
    ok(len(dead) == 75, "across 75 tensors", f"{len(dead)}")
    # NOTHING LIVE MAY BE IN IT. These four are the pae path and must survive.
    live_must_keep = ["linear_no_bias_pae.weight", "pae_ln.weight", "linear_no_bias_s1.weight",
                      "linear_no_bias_d.weight"]
    bad = [n for n in live_must_keep if n in set(dead)]
    ok(not bad, "no live pae-path tensor is in the dead set", str(bad))
    # AND THE PAIR TRACK MUST SURVIVE.
    pair_in_dead = [n for n in dead if any(
        k in n for k in ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
                         "pair_transition"))]
    ok(not pair_in_dead, "no pair-track tensor is in the dead set",
       f"{len(pair_in_dead)} were")
    print(f"      (head stand-in holds {tot:,} parameters; dead is "
          f"{100.0 * n_dead / tot:.2f}%)")
    # WITHOUT single_dropped THE SINGLE TRACK STAYS IN, so the function cannot silently change
    # pre-existing behaviour.
    dead_ro = LS.dead_head_params(h, single_dropped=False)
    ok(all("attention_pair_bias" not in n and "single_transition" not in n for n in dead_ro),
       "single_dropped=False leaves the single track in the optimiser",
       f"{len(dead_ro)} tensor(s) dead")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
