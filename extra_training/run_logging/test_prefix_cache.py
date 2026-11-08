#!/usr/bin/env python
"""The frozen-prefix cache on a REAL confidence head, through install(), with blocks_per_ckpt=1.

WHAT THIS HAS TO PROVE, in the order the claims are made:

  1  THE CACHE CHANGES SOMETHING. Call counts on tri_mul_out, tri_mul_in and blocks 0-2 are ZERO on
     a hit and non-zero on a miss. Without this every equality check below is satisfied by a cache
     that never engages, which is the --blocks-per-ckpt failure mode: a flag that prints a line and
     changes nothing. This is checked FIRST and nothing after it is believed without it.
  2  IT IS THE SAME FUNCTION. Cached and uncached PAE logits, and the gradients to every one of the
     28 trainable tensors, in fp32 and then separately under bf16 autocast.
  3  NOTHING FROZEN MOVES. No prefix tensor receives a gradient on either path.
  4  A MISS IS A RECOMPUTE, NOT A WRONG ANSWER. A cleared cache, a changed token count and an
     over-cap pose all fall back and all give the same logits.
  5  THE CAP BOUNDS BYTES, not just entries, and the reservation is released on every path.
  6  THE PRECONDITIONS REFUSE. No --pose-inblock, no --lean-single, a live dropout rate, a
     trainable prefix tensor, the wrong injection point, and a frozen tensor that moves mid-run.

THE THREE TRAPS THIS FILE INHERITS FROM run_logging/test_pose_inblock_integration.py, every one of
which makes a cache bug invisible:

  (a) a fresh ConfidenceHead zero-initialises linear_no_bias_pae, so the PAE logits are identically
      zero whatever the stack does and every "same logits" assertion is 0 == 0;
  (b) every PairformerBlock residual branch ends in a zero-init projection, so a FRESH block is an
      EXACT identity on z and a fresh four-block stack is an identity too -- meaning the value at
      the cut equals the value at the stack's input and a cache that returned the wrong tensor
      would still be right. 108 projections per head are filled to make the stack a function;
  (c) ConfidenceHead construction is NOT deterministic under torch.manual_seed (84 tensors differ
      between two constructions at the same seed), so two heads are made equal with
      load_state_dict and never by re-seeding.

AND ONE OF ITS OWN: the prefix must not be an identity either. Section 2 measures that z at the cut
actually differs from z entering the stack, because on an identity prefix the cache would be
returning the input and nothing would notice.

CPU ONLY. ~19.8M parameters at the live configuration and 16 tokens of geometry.
"""
import os
import sys

import torch
# EXPLICITLY, because protenix.model.utils reaches for torch.utils.checkpoint.checkpoint without
# importing the submodule. The trainer gets it transitively; a -I test process does not, and the
# failure is an AttributeError the first time blocks_per_ckpt is honoured -- i.e. only on the
# production path this test exists to exercise.
import torch.utils.checkpoint  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")

import lean_readouts as LS                                               # noqa: E402
import pose_inblock                                                      # noqa: E402
import prefix_cache as PFC                                               # noqa: E402
from protenix.model.modules.confidence import ConfidenceHead             # noqa: E402

FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


# THE LIVE CONFIGURATION, not the defaults in confidence.py's docstring.
CFG = dict(n_blocks=4, c_s=384, c_z=256, c_s_inputs=449, b_pae=64, b_pde=64, b_plddt=50,
           b_resolved=2, max_atoms_per_token=24, pairformer_dropout=0.0, hidden_scale_up=True)
N_TOK = 16
BLK = 3

# THE LAUNCHER'S --train-only FOR THE DEFAULT ARM, as name prefixes rather than the shell regex, so
# this file does not have to track the quoting. INBLOCK_READOUT=1 is the default, which is why
# linear_no_bias_pae and pae_ln are here: 28 tensors, 1,484,800 parameters.
TRAINABLE = ("pairformer_stack.blocks.3.tri_att_start.",
             "pairformer_stack.blocks.3.tri_att_end.",
             "pairformer_stack.blocks.3.pair_transition.",
             "pose_inblock_", "linear_no_bias_pae.", "pae_ln.")
N_TRAINABLE, N_TRAINABLE_P = 28, 1_484_800


class _Model(torch.nn.Module):
    """install() reaches for model.confidence_head, so it needs a holder. See traps (a) and (b)."""

    def __init__(self, blocks_per_ckpt=1):
        super().__init__()
        torch.manual_seed(0)
        self.confidence_head = ConfidenceHead(blocks_per_ckpt=blocks_per_ckpt, **CFG)
        h = self.confidence_head
        with torch.no_grad():
            w = h.linear_no_bias_pae.weight
            assert float(w.abs().max()) == 0.0, (
                "linear_no_bias_pae was expected to arrive zero-initialised; confidence.py has "
                "changed and this file's vacuity guards need rereading")
            w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
            n_filled = 0
            for i, (name, p) in enumerate(h.pairformer_stack.named_parameters()):
                if float(p.abs().max()) == 0.0:
                    p.copy_(torch.randn(p.shape,
                                        generator=torch.Generator().manual_seed(1000 + i)) * 0.05)
                    n_filled += 1
            assert n_filled >= 40, (
                f"expected the fresh stack to arrive with many zero-init projections, filled "
                f"{n_filled}; pairformer.py has changed")
        self.n_filled = n_filled


def _freeze(head):
    for n, p in head.named_parameters():
        p.requires_grad_(any(n.startswith(t) for t in TRAINABLE))
    return ([n for n, p in head.named_parameters() if p.requires_grad],
            [n for n, p in head.named_parameters() if not p.requires_grad])


def _built(blocks_per_ckpt=1, inject_at="after_tri_mul_in", lean=True, freeze=True):
    """A head in the TRAINER'S INSTALL ORDER: lean_readouts, the single drop, then the pose."""
    m = _Model(blocks_per_ckpt=blocks_per_ckpt)
    LS.install(m, pae_only=True)
    if lean:
        LS.drop_single_pathway(m)
    pose_inblock.install(m, block_index=BLK, pae_inject=True, inject_at=inject_at, loud=False)
    h = m.confidence_head
    # THE INJECTIONS, MADE NON-ZERO. They ship exactly zero so that step 0 is bit-identical to the
    # un-injected head; a zero injection here would decouple the suffix from the cut point.
    with torch.no_grad():
        for i, t in enumerate(("pose_inblock_d", "pose_inblock_d_wo_onehot",
                               "pose_inblock_pae_d", "pose_inblock_pae_d_wo_onehot")):
            w = getattr(h, t).weight
            w.copy_(torch.randn(w.shape,
                                generator=torch.Generator().manual_seed(500 + i)) * 0.2)
    if freeze:
        _freeze(h)
    return m


def _inputs(seed=7, n=N_TOK):
    """One sample's worth of geometry and trunk tensors, in the shapes the head's own forward takes.

    run_confidence_head is not reachable without the full Protenix model, so the test drives
    ConfidenceHead.memory_efficient_forward directly -- which is exactly what the trainer's
    forward reduces to, since the patched attribute IS the call the model makes.
    """
    g = torch.Generator().manual_seed(seed)
    f = lambda *s: torch.randn(*s, generator=g)                          # noqa: E731
    return {
        "input_feature_dict": {"atom_to_token_idx": torch.arange(n)},
        "s_trunk": f(n, CFG["c_s"]),
        "z_pair": f(n, n, CFG["c_z"]),
        "pair_mask": None,
        # SPREAD OVER THE BIN RANGE (3.25 to 52 A), so one_hot does not put every pair in one bin.
        "x_pred_rep_coords": f(n, 3) * 12.0,
    }


def _pae(model, inp):
    """The PAE logits only. lean_readouts returns them second."""
    return model.confidence_head.memory_efficient_forward(**inp)[1]


class _Counter:
    """Forward-hook call counts on the modules the cut is supposed to skip."""

    def __init__(self, head, block_index=BLK):
        self.n = {}
        self.h = []
        tgt = {"tri_mul_out": head.pairformer_stack.blocks[block_index].tri_mul_out,
               "tri_mul_in": head.pairformer_stack.blocks[block_index].tri_mul_in,
               "tri_att_start": head.pairformer_stack.blocks[block_index].tri_att_start,
               "native_d": head.linear_no_bias_d}
        for i in range(block_index):
            tgt[f"block{i}.tri_mul_out"] = head.pairformer_stack.blocks[i].tri_mul_out
        for k, mod in tgt.items():
            self.n[k] = 0
            self.h.append(mod.register_forward_hook(self._mk(k)))

    def _mk(self, k):
        def hook(*_a):
            self.n[k] += 1
        return hook

    def zero(self):
        for k in self.n:
            self.n[k] = 0

    def close(self):
        for h in self.h:
            h.remove()


def _enable(m, cap_gb, loud=False):
    PFC.uninstall()
    return PFC.install(m, cap_gb, lean_single=True, device=None, loud=loud)


def _two_pass(m, inp, key=("1abc", "free_500000", 0), cached=True, grad=True, autocast=None):
    """Pass A then pass B over one pose, exactly as phase_k_train sequences them.

    Returns (pae_logits_from_pass_B, {param name -> grad}).
    """
    h = m.confidence_head
    for p in h.parameters():
        p.grad = None

    def _fwd():
        if autocast is None:
            return _pae(m, inp)
        with torch.autocast(device_type="cpu", dtype=autocast):
            return _pae(m, inp)

    # PASS A: a no_grad forward, which is where the cache fills.
    if cached:
        PFC.set_pose(key, fill=True, n_tok=N_TOK)
    try:
        with torch.no_grad():
            _fwd()
    finally:
        if cached:
            PFC.clear_pose()
    # PASS B: the same pose, rebuilt with autograd, which is where the cache hits.
    if cached:
        PFC.set_pose(key, fill=False, n_tok=N_TOK)
    try:
        if grad:
            out = _fwd()
        else:
            with torch.no_grad():
                out = _fwd()
    finally:
        if cached:
            PFC.clear_pose()
    g = {}
    if grad:
        out.float().pow(2).mean().backward()
        g = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in h.named_parameters()}
    return out.detach().clone(), g


def _cmp(a, b):
    """(max abs diff, max rel diff) between two tensors."""
    d = (a.float() - b.float()).abs()
    den = b.float().abs().clamp_min(1e-12)
    return float(d.max()), float((d / den).max())


def _refuses(fn, must_contain):
    try:
        fn()
    except SystemExit as e:
        return must_contain.lower() in str(e).lower(), str(e).splitlines()[0][:90]
    return False, "did not raise SystemExit"


def main():
    print(__doc__.strip().splitlines()[0])
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    inp = _inputs()

    # ================================================================================= 1
    print("\n1. install() on a REAL head at the live freeze, and what it says it cut")
    m = _built()
    live, dead = _freeze(m.confidence_head)
    ok(len(live) == N_TRAINABLE, f"{N_TRAINABLE} tensors are trainable", f"{len(live)}")
    n_live = sum(p.numel() for p in m.confidence_head.parameters() if p.requires_grad)
    ok(n_live == N_TRAINABLE_P, f"and they hold {N_TRAINABLE_P:,} parameters", f"{n_live:,}")
    info = _enable(m, 1.0, loud=True)
    ok(info is not None and info["block_index"] == BLK,
       f"the cut is INSIDE block {BLK}", str(info and info["block_index"]))
    # THE PREFIX SET IS DERIVED FROM THE MODULE TREE, so this is a check on the derivation and not
    # on a hand-written list. 3 whole blocks + 2 tri-muls + the native injection + s1/s2 + the bins.
    ok(info["prefix_tensors"] > 100,
       "the prefix is the blocks before the cut plus the cut block's two tri-muls",
       f"{info['prefix_tensors']} tensor(s), {info['prefix_params']:,} parameter(s)")
    pfx = PFC._prefix_param_names(m.confidence_head, BLK)
    bad = [n for n in pfx if n.startswith(f"pairformer_stack.blocks.{BLK}.")
           and not any(f".{s}." in n for s in PFC.PREFIX_CUT_BLOCK_SUBS)]
    ok(not bad, "no tensor downstream of the cut is counted as prefix", str(bad[:3]))
    ok(all(not dict(m.confidence_head.named_parameters())[n].requires_grad for n in pfx),
       "every prefix tensor reads requires_grad=False at runtime")
    ok(not (set(pfx) & set(live)), "the prefix set and the trainable set are DISJOINT")

    # ================================================================================= 2
    print("\n2. the vacuity guards: the readout, the stack AND the prefix are real functions")
    ok(float(m.confidence_head.linear_no_bias_pae.weight.abs().max()) > 0,
       "the PAE readout is non-zero, so the logits can move")
    ok(m.n_filled >= 40, "the stack was made a function, not left an identity",
       f"{m.n_filled} zero-init projection(s) filled")
    ok(all(float(getattr(m.confidence_head, t).weight.abs().max()) > 0
           for t in ("pose_inblock_d", "pose_inblock_d_wo_onehot",
                     "pose_inblock_pae_d", "pose_inblock_pae_d_wo_onehot")),
       "all four injections are non-zero, so the suffix depends on the cut point")
    # THE PREFIX IS NOT AN IDENTITY. Captured through the cache's own fill path: the value it
    # stores IS the value at the cut, so comparing it against z entering the head is the direct
    # measurement. On an identity prefix the cache would be handing back its own input.
    PFC.step_reset()
    PFC.set_pose(("probe", "s", 0), fill=True, n_tok=N_TOK)
    with torch.no_grad():
        _pae(m, inp)
    PFC.clear_pose()
    cut_z = list(PFC._S["cache"].values())[0]
    d_cut, _ = _cmp(cut_z, inp["z_pair"])
    ok(d_cut > 1e-3, "z at the cut DIFFERS from z entering the head -- the prefix is a function",
       f"max|diff| = {d_cut:.3e}")
    ok(tuple(cut_z.shape) == (N_TOK, N_TOK, CFG["c_z"]),
       "and the cached tensor is the pair representation", str(tuple(cut_z.shape)))
    ok(cut_z.device.type == "cpu", "held on the HOST, not the compute device",
       str(cut_z.device))

    # ================================================================================= 3
    print("\n3. THE CACHE CHANGES SOMETHING: what runs on a miss and what does not on a hit")
    cnt = _Counter(m.confidence_head)
    PFC.step_reset()
    # COUNTERS ZEROED, because section 2's probe already filled and missed once on this cache and
    # a cumulative counter would make the "one fill, one hit" claim below read 2/2.
    PFC.reset_stats()
    # PASS A -- a miss by construction, since the step's cache was just dropped.
    PFC.set_pose(("1abc", "free_500000", 0), fill=True, n_tok=N_TOK)
    with torch.no_grad():
        _pae(m, inp)
    PFC.clear_pose()
    miss_n = dict(cnt.n)
    ok(miss_n["tri_mul_out"] == 1 and miss_n["tri_mul_in"] == 1,
       "on the FILL pass the cut block's tri-muls DO run", str(miss_n))
    ok(all(miss_n[f"block{i}.tri_mul_out"] == 1 for i in range(BLK)),
       "and so does every block before the cut")
    cnt.zero()
    # PASS B -- a hit.
    PFC.set_pose(("1abc", "free_500000", 0), fill=False, n_tok=N_TOK)
    out_hit = _pae(m, inp)
    PFC.clear_pose()
    hit_n = dict(cnt.n)
    ok(hit_n["tri_mul_out"] == 0 and hit_n["tri_mul_in"] == 0,
       "on the HIT pass NEITHER tri-mul of the cut block runs", str(hit_n))
    ok(all(hit_n[f"block{i}.tri_mul_out"] == 0 for i in range(BLK)),
       f"and blocks 0..{BLK - 1} are not entered at all")
    ok(hit_n["tri_att_start"] == 1, "but the trainable suffix DOES run",
       f"tri_att_start x{hit_n['tri_att_start']}")
    # THE HONEST CAVEAT, ASSERTED SO IT CANNOT QUIETLY CHANGE. The native pre-stack injection is
    # upstream of the stack and is NOT skipped; the module docstring says so and this is the check.
    ok(hit_n["native_d"] == 1,
       "the pre-stack native injection still runs on a hit (documented, ~2% of the prefix)",
       f"linear_no_bias_d x{hit_n['native_d']}")
    st = PFC.stats()
    ok(st["hit"] == 1 and st["put"] == 1, "one fill, one hit", str({k: st[k] for k in
                                                                   ("hit", "miss", "put", "full")}))
    cnt.close()

    # ================================================================================= 4
    print("\n4. fp32: cached == uncached, logits AND gradients to all 28 trainable tensors")
    PFC.step_reset()
    out_c, g_c = _two_pass(m, inp, cached=True)
    PFC.uninstall()
    out_u, g_u = _two_pass(m, inp, cached=False)
    da, dr = _cmp(out_c, out_u)
    ok(da == 0.0, "PAE logits are BIT-IDENTICAL with and without the cache",
       f"max|abs| = {da:.3e}, max|rel| = {dr:.3e}")
    ok(float(out_u.abs().max()) > 0, "and the logits are not identically zero",
       f"max|logit| = {float(out_u.abs().max()):.3e}")
    got = [n for n in g_c if g_c[n] is not None]
    ok(len(got) == N_TRAINABLE, f"exactly {N_TRAINABLE} tensors received a gradient",
       f"{len(got)}")
    worst, worst_n = 0.0, ""
    worst_r = 0.0
    for n in got:
        a_, r_ = _cmp(g_c[n], g_u[n])
        if a_ > worst:
            worst, worst_n = a_, n
        worst_r = max(worst_r, r_)
    ok(worst == 0.0, "every trainable gradient is BIT-IDENTICAL",
       f"max|abs| = {worst:.3e} ({worst_n or 'none'}), max|rel| = {worst_r:.3e}")
    ok(all(float(g_c[n].abs().max()) > 0 for n in got),
       "and every one of them is NON-ZERO, so the comparison is not 0 == 0")
    ok(all(g_c[n] is None for n in g_c if n not in got) and
       not (set(got) & set(PFC._prefix_param_names(m.confidence_head, BLK))),
       "NOT ONE frozen/prefix tensor received a gradient on either path")

    # ================================================================================= 5
    print("\n5. bf16 autocast, reported SEPARATELY from fp32")
    _enable(m, 1.0)
    out_cb, g_cb = _two_pass(m, inp, cached=True, autocast=torch.bfloat16)
    PFC.uninstall()
    out_ub, g_ub = _two_pass(m, inp, cached=False, autocast=torch.bfloat16)
    da_b, dr_b = _cmp(out_cb, out_ub)
    ok(da_b == 0.0, "bf16: PAE logits are BIT-IDENTICAL with and without the cache",
       f"max|abs| = {da_b:.3e}, max|rel| = {dr_b:.3e}")
    wb, wb_n, wb_r = 0.0, "", 0.0
    for n in (k for k in g_cb if g_cb[k] is not None):
        a_, r_ = _cmp(g_cb[n], g_ub[n])
        if a_ > wb:
            wb, wb_n = a_, n
        wb_r = max(wb_r, r_)
    ok(wb == 0.0, "bf16: every trainable gradient is BIT-IDENTICAL",
       f"max|abs| = {wb:.3e} ({wb_n or 'none'}), max|rel| = {wb_r:.3e}")
    # AND THE TWO PRECISIONS REALLY ARE DIFFERENT COMPUTATIONS, or the bf16 row above is the fp32
    # row again under another name.
    d_prec, _ = _cmp(out_ub, out_u)
    ok(d_prec > 1e-4, "and bf16 is a DIFFERENT computation from fp32, so this row is its own test",
       f"max|diff| vs fp32 = {d_prec:.3e}")

    # ================================================================================= 6
    print("\n6. a MISS is a recompute, never a wrong answer")
    _enable(m, 1.0)
    # (a) the cache dropped between the passes -- which is what a step boundary does.
    PFC.set_pose(("1abc", "free_500000", 0), fill=True, n_tok=N_TOK)
    with torch.no_grad():
        _pae(m, inp)
    PFC.clear_pose()
    PFC.step_reset()
    PFC.reset_stats()
    PFC.set_pose(("1abc", "free_500000", 0), fill=False, n_tok=N_TOK)
    with torch.no_grad():
        out_m = _pae(m, inp)
    PFC.clear_pose()
    ok(PFC.stats()["miss"] == 1 and PFC.stats()["hit"] == 0,
       "a cache dropped at the step boundary is a MISS", str(PFC.stats()["miss"]))
    ok(_cmp(out_m, out_u)[0] == 0.0, "and the recomputed answer is the uncached answer",
       f"max|abs| = {_cmp(out_m, out_u)[0]:.3e}")
    # (b) a key that is stale in the way cropping would make it stale.
    PFC.step_reset()
    PFC.set_pose(("1abc", "free_500000", 0), fill=True, n_tok=N_TOK)
    with torch.no_grad():
        _pae(m, inp)
    PFC.clear_pose()
    PFC.reset_stats()
    PFC.set_pose(("1abc", "free_500000", 0), fill=False, n_tok=N_TOK + 1)
    with torch.no_grad():
        out_k = _pae(m, inp)
    PFC.clear_pose()
    ok(PFC.stats()["miss"] == 1, "a DIFFERENT token count on the same pose is a MISS")
    ok(_cmp(out_k, out_u)[0] == 0.0, "and still gives the uncached answer")
    # (c) a different pose of the same unit does not read another pose's prefix.
    PFC.step_reset()
    inp2 = _inputs(seed=99)
    PFC.set_pose(("1abc", "free_500000", 0), fill=True, n_tok=N_TOK)
    with torch.no_grad():
        _pae(m, inp)
    PFC.clear_pose()
    PFC.reset_stats()
    PFC.set_pose(("1abc", "free_500000", 1), fill=False, n_tok=N_TOK)
    with torch.no_grad():
        out_p = _pae(m, inp2)
    PFC.clear_pose()
    PFC.uninstall()
    with torch.no_grad():
        out_p_ref = _pae(m, inp2)
    ok(PFC.stats()["miss"] == 1, "pose 1 does not hit on pose 0's entry -- the cut is PER POSE")
    ok(_cmp(out_p, out_p_ref)[0] == 0.0, "and pose 1's answer is its own")
    ok(_cmp(out_p_ref, out_u)[0] > 1e-3,
       "(and the two poses really do give different logits, so the check is not vacuous)",
       f"max|diff| = {_cmp(out_p_ref, out_u)[0]:.3e}")
    # (d) NO POSE KEY AT ALL -- the evaluator and the held-out monitor forward without one.
    _enable(m, 1.0)
    PFC.reset_stats()
    with torch.no_grad():
        out_nk = _pae(m, inp)
    ok(PFC.stats()["nokey"] >= 1 and PFC.stats()["hit"] == 0,
       "a forward with no pose key uses the uncached path", str(PFC.stats()["nokey"]))
    ok(_cmp(out_nk, out_u)[0] == 0.0, "and gives the uncached answer")

    # ================================================================================= 7
    print("\n7. the byte cap: an over-cap pose falls back instead of failing")
    nb = cut_z.numel() * cut_z.element_size()
    # A CAP ONE BYTE SHORT OF ONE ENTRY. Not zero -- zero turns the cache OFF, which would test
    # the off switch and not the cap.
    _enable(m, (nb - 1) / (1 << 30))
    PFC.reset_stats()
    out_cap, g_cap = _two_pass(m, inp, cached=True)
    s_cap = PFC.stats()
    ok(s_cap["full"] >= 1 and s_cap["put"] == 0,
       "a pose one byte over the cap is REFUSED, not admitted",
       f"full={s_cap['full']} put={s_cap['put']} bytes={s_cap['bytes']}")
    ok(s_cap["bytes"] == 0, "so the cache holds ZERO bytes", f"{s_cap['bytes']}")
    ok(s_cap["resv"] == 0, "and no reservation leaked", f"{s_cap['resv']}")
    ok(_cmp(out_cap, out_u)[0] == 0.0, "and the refused pose gives the uncached logits")
    wc = max((_cmp(g_cap[n], g_u[n])[0] for n in g_cap if g_cap[n] is not None), default=-1.0)
    ok(wc == 0.0, "and the uncached gradients", f"max|abs| = {wc:.3e}")
    # EXACTLY ONE ENTRY FITS, and the second is refused -- so the cap is a bound and not a label.
    _enable(m, nb / (1 << 30))
    PFC.reset_stats()
    for j in range(3):
        PFC.set_pose(("1abc", "free_500000", j), fill=True, n_tok=N_TOK)
        with torch.no_grad():
            _pae(m, _inputs(seed=7 + j))
        PFC.clear_pose()
    s2 = PFC.stats()
    ok(s2["put"] == 1 and s2["full"] == 2, "a cap of exactly one entry admits one and refuses two",
       f"put={s2['put']} full={s2['full']}")
    ok(s2["bytes"] == nb and s2["bytes"] <= s2["cap"],
       "and the admitted bytes equal one entry and never exceed the cap",
       f"{s2['bytes']} <= {s2['cap']}")
    PFC.step_reset()
    ok(PFC.stats()["bytes"] == 0 and PFC.stats()["entries"] == 0,
       "the step boundary returns the byte count to zero")

    # ================================================================================= 8
    print("\n8. blocks_per_ckpt=1: the cut block's forward is RE-RUN inside backward")
    # THE CASE WITH TEETH, and the same one test_pose_inblock_integration.py exists for. On a MISS
    # in a grad-enabled pass the block runs twice, so the fill callback is invoked twice and must
    # not double-charge the cap or pay a second copy.
    ok(m.confidence_head.pairformer_stack.blocks_per_ckpt == 1,
       "the stack is at blocks_per_ckpt=1 (the production path)")
    _enable(m, 1.0)
    PFC.reset_stats()
    PFC.set_pose(("1abc", "free_500000", 0), fill=True, n_tok=N_TOK)
    out_r = _pae(m, inp)
    out_r.float().pow(2).mean().backward()
    PFC.clear_pose()
    s8 = PFC.stats()
    ok(s8["put"] == 1, "the fill is IDEMPOTENT across the checkpoint recomputation",
       f"put={s8['put']} bytes={s8['bytes']} (one entry = {nb})")
    ok(s8["bytes"] == nb, "and the cap was charged exactly once", f"{s8['bytes']}")
    ok(s8["resv"] == 0, "with no reservation left outstanding", f"{s8['resv']}")
    # AND THE HIT PATH SURVIVES A BACKWARD THAT RE-ENTERS THE BLOCK. The cached tensor is held by
    # the _prep_blocks closure for exactly this reason; if it were read from a dict that
    # clear_pose() emptied, this is where it would raise.
    for p in m.confidence_head.parameters():
        p.grad = None
    PFC.set_pose(("1abc", "free_500000", 0), fill=False, n_tok=N_TOK)
    out_h = _pae(m, inp)
    out_h.float().pow(2).mean().backward()
    PFC.clear_pose()
    ok(_cmp(out_h, out_u)[0] == 0.0, "a HIT whose block is recomputed in backward is still exact")
    g_h = {n: p.grad for n, p in m.confidence_head.named_parameters() if p.grad is not None}
    w8 = max((_cmp(g_h[n], g_u[n])[0] for n in g_h), default=-1.0)
    ok(len(g_h) == N_TRAINABLE and w8 == 0.0,
       "and its gradients match the uncached run exactly",
       f"{len(g_h)} tensor(s), max|abs| = {w8:.3e}")

    # ================================================================================= 9
    print("\n9. the preconditions REFUSE rather than degrading")
    PFC.uninstall()
    # (a) no --pose-inblock.
    m_np = _Model()
    LS.install(m_np, pae_only=True)
    LS.drop_single_pathway(m_np)
    _freeze(m_np.confidence_head)
    r, msg = _refuses(lambda: PFC.install(m_np, 1.0, lean_single=True, loud=False),
                      "without --pose-inblock")
    ok(r, "refuses without --pose-inblock", msg)
    # (b) no --lean-single: the flag says no.
    m_ls = _built(lean=True)
    r, msg = _refuses(lambda: PFC.install(m_ls, 1.0, lean_single=False, loud=False),
                      "requires --lean-single")
    ok(r, "refuses when --lean-single was not passed", msg)
    # (c) no --lean-single: the BLOCKS say no, whatever the flag says. This is the check that
    #     matters, because the flag is an argument and c_s is what the forward branches on.
    m_cs = _built(lean=False)
    r, msg = _refuses(lambda: PFC.install(m_cs, 1.0, lean_single=True, loud=False),
                      "requires --lean-single")
    ok(r, "refuses when a block still has c_s > 0 even though the flag claims --lean-single", msg)
    # (d) a live dropout rate in the prefix, set the way --pair-dropout sets it.
    m_dr = _built()
    m_dr.confidence_head.pairformer_stack.blocks[1].p_drop = 0.1
    r, msg = _refuses(lambda: PFC.install(m_dr, 1.0, lean_single=True, loud=False),
                      "deterministic")
    ok(r, "refuses a non-zero dropout rate on a prefix block", msg)
    # (e) a prefix tensor that is trainable -- i.e. a --train-only wider than this arm's.
    m_tr = _built()
    m_tr.confidence_head.pairformer_stack.blocks[BLK].tri_mul_in.linear_z.weight.requires_grad_(
        True)
    r, msg = _refuses(lambda: PFC.install(m_tr, 1.0, lean_single=True, loud=False),
                      "require grad")
    ok(r, "refuses when a prefix tensor requires grad (read from the tensor, not the flag)", msg)
    # (f) the injection at the block INPUT, where --pose-perblock puts it: no frozen prefix exists
    #     inside the block at all.
    m_bi = _built(inject_at="block_input")
    r, msg = _refuses(lambda: PFC.install(m_bi, 1.0, lean_single=True, loud=False),
                      "after tri_mul_in")
    ok(r, "refuses an injection at the block input", msg)
    # (g) A FROZEN TENSOR THAT MOVES MID-RUN. The step boundary re-checks both the freeze and the
    #     values, because that is the only moment an optimiser step could have touched them.
    m_mv = _built()
    _enable(m_mv, 1.0)
    with torch.no_grad():
        m_mv.confidence_head.pairformer_stack.blocks[0].tri_mul_in.linear_z.weight.add_(1e-3)
    r, msg = _refuses(PFC.step_reset, "frozen prefix changed value")
    ok(r, "the step boundary hard-stops if a frozen prefix tensor changed value", msg)
    m_mv2 = _built()
    _enable(m_mv2, 1.0)
    m_mv2.confidence_head.linear_no_bias_d.weight.requires_grad_(True)
    r, msg = _refuses(PFC.step_reset, "require grad")
    ok(r, "and if one of them became trainable", msg)
    # (h) 0 IS OFF, and off means the planner is not installed at all.
    PFC.uninstall()
    m_off = _built()
    ok(PFC.install(m_off, 0.0, lean_single=True, loud=False) is None,
       "--prefix-cache-gb 0 returns None and installs nothing")
    ok(pose_inblock.CUT["plan"] is None, "and leaves pose_inblock's cut hook unset")

    PFC.uninstall()
    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
