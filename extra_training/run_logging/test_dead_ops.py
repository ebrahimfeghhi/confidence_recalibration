#!/usr/bin/env python
"""Dead-operation elimination on a REAL confidence head holding the REAL SHIPPED WEIGHTS.

WHAT THIS HAS TO PROVE, in the order the claims are made:

  1  THE MAP IS RIGHT, AND IT IS RIGHT IN BOTH DIRECTIONS. With the shipped head loaded, exactly
     the nine expected O(N^3) operations are detected dead -- and the seven expected ones are NOT.
     Every live control is named explicitly and its measured output is required to be non-zero.
  2  THE SKIP CHANGES SOMETHING. Forward-hook call counts inside each stubbed module are ZERO and
     inside each kept module are non-zero. Without this every equality below is satisfied by a flag
     that prints a line and does nothing, which is the --blocks-per-ckpt failure mode.
  3  THE TREE IS UNTOUCHED. The head's state-dict key set, its named_parameters() set and the
     identity of every parameter object survive the swap -- which is what phase_k_eval's
     set-equality guard, --train-only's name matching and prefix_cache's prefix derivation rest on.
  4  IT IS THE SAME FUNCTION. PAE logits and the gradient to every trainable tensor, bit-identical,
     in fp32 and then SEPARATELY under bf16 autocast.
  5  IT REFUSES RATHER THAN DEGRADING: a dead module in the trainable set (which is this arm's
     actual --train-only), a hooked module, and a stubbed weight that moves or becomes trainable
     mid-run.
  6  IT COMPOSES with phases/prefix_cache.py: stubs plus cache still give bit-identical logits and
     gradients.

THE TRAP, WHICH ALREADY PRODUCED A WRONG ANSWER ONCE. A FRESHLY CONSTRUCTED ConfidenceHead has
every residual branch ending in a ZERO-INIT projection, so a fresh PairformerBlock is an exact
identity on z and EVERY candidate module returns exactly 0.0. A fresh head therefore looks
identically like a fully-dead head and any "is the output zero?" test passes vacuously on all
sixteen cubic ops. Section 2 reproduces that explicitly -- it asserts that the probe ALONE calls
20 of 20 candidates dead on a fresh head while the full rule (weights below the floor AND the
output measured zero) calls 0 of 20 -- so the shipped map in section 1 cannot be an artifact of an
unloaded head. The other half of the same trap is a load that silently does nothing: the shipped
state dict is loaded with its key count asserted non-zero and both missing_keys and unexpected_keys
asserted empty.

AND THE SHIPPED WEIGHTS DISPOSE OF THE OTHER TWO TRAPS run_logging/test_prefix_cache.py has to work
around. It fills 108 zero-init projections by hand to make the stack a function and randomises
linear_no_bias_pae so the logits can move. Here neither is needed or wanted: the shipped readout is
0.98 and the shipped stack emits up to 1.29e+02, both asserted below. Only the four pose_inblock
projections are filled, because those really do ship exactly zero (by design -- step 0 is meant to
be bit-identical to the un-injected head) and a zero injection would decouple the suffix from the
cut point that section 6 is about.

CPU ONLY. One ~19.8M-parameter head, 252 shipped tensors, 16 tokens of geometry.
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

import dead_ops as DOPS                                                  # noqa: E402
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

# THE REAL SHIPPED HEAD, saved from /home/ebrahim/checkpoint/protenix-v2.pt by the in-block smoke
# run. 252 tensors, keys prefixed "confidence_head.", and it already carries pose_inblock's four
# projections at their zero init.
SHIPPED = ("/home/ebrahim/confidence_recalibration/extra_training/training/runs/"
           "inblock_smoke_20261107_181533_g1/head_init.pt")
N_SHIPPED = 252

# ---- THE EXPECTED MAP, WRITTEN OUT SO IT CANNOT DRIFT -------------------------------------------
# Nine of the sixteen O(N^3) pair operations, plus block 1's pair_transition (which is O(N^2)).
EXPECT_DEAD_CUBIC = [
    "pairformer_stack.blocks.0.tri_mul_out", "pairformer_stack.blocks.0.tri_mul_in",
    "pairformer_stack.blocks.0.tri_att_start", "pairformer_stack.blocks.0.tri_att_end",
    "pairformer_stack.blocks.1.tri_att_start", "pairformer_stack.blocks.1.tri_att_end",
    "pairformer_stack.blocks.2.tri_att_start", "pairformer_stack.blocks.2.tri_att_end",
    "pairformer_stack.blocks.3.tri_att_end",
]
EXPECT_ALIVE_CUBIC = [
    "pairformer_stack.blocks.1.tri_mul_out", "pairformer_stack.blocks.1.tri_mul_in",
    "pairformer_stack.blocks.2.tri_mul_out", "pairformer_stack.blocks.2.tri_mul_in",
    "pairformer_stack.blocks.3.tri_mul_out", "pairformer_stack.blocks.3.tri_mul_in",
    "pairformer_stack.blocks.3.tri_att_start",
]
EXPECT_DEAD_OTHER = ["pairformer_stack.blocks.1.pair_transition"]
EXPECT_ALIVE_OTHER = ["pairformer_stack.blocks.0.pair_transition",
                      "pairformer_stack.blocks.2.pair_transition",
                      "pairformer_stack.blocks.3.pair_transition"]
EXPECT_DEAD = EXPECT_DEAD_CUBIC + EXPECT_DEAD_OTHER
EXPECT_ALIVE = EXPECT_ALIVE_CUBIC + EXPECT_ALIVE_OTHER

# THE LAUNCHER'S --train-only FOR THE DEFAULT ARM, as name prefixes rather than the shell regex.
# 28 tensors, 1,484,800 parameters -- AND IT INCLUDES BLOCK 3's tri_att_end, WHICH IS DEAD, so
# installing against this set is a hard stop. That is the point of section 5(a).
TRAINABLE_ARM = ("pairformer_stack.blocks.3.tri_att_start.",
                 "pairformer_stack.blocks.3.tri_att_end.",
                 "pairformer_stack.blocks.3.pair_transition.",
                 "pose_inblock_", "linear_no_bias_pae.", "pae_ln.")
N_ARM, N_ARM_P = 28, 1_484_800
# THE SET THE REFUSAL ASKS THE CALLER FOR: the same thing with the dead tri_att_end dropped.
TRAINABLE_FIXED = tuple(t for t in TRAINABLE_ARM
                        if t != "pairformer_stack.blocks.3.tri_att_end.")
N_FIXED, N_FIXED_P = 20, 1_154_560

# ONE CHILD PER CANDIDATE, for the call counts. A hook on the stub itself would still fire (the
# husk's __call__ runs), so the measurement has to be INSIDE the module -- and a hook on the
# original module object is refused by install() for exactly the reason it would be misleading.
# THE KERNELS THIS TEST PROBES THROUGH. "torch" on both, because cuequivariance's fused Triton
# kernels need a GPU and this file is CPU-only. THE PRODUCTION RUN PASSES "cuequivariance" FOR
# BOTH (configs_base.py:129-130) and dead_ops.install() now takes them from `configs`, so the
# verdict certified HERE is about the reference implementation and the verdict certified ON A CARD
# will be about the fused kernels. That is the right division -- the measurement happens at every
# install, so the first GPU run does its own -- but it means THIS FILE DOES NOT CERTIFY THE GPU
# PATH, and saying so is the point of this comment.
KERNELS = dict(triangle_multiplicative="torch", triangle_attention="torch")

PROBE_CHILD = {"tri_mul_out": "layer_norm_in", "tri_mul_in": "layer_norm_in",
               "tri_att_start": "layer_norm", "tri_att_end": "layer_norm",
               "pair_transition": "layernorm1"}


class _Model(torch.nn.Module):
    """install() reaches for model.confidence_head, so it needs a holder.

    NOTHING IS RANDOMISED HERE. Unlike test_prefix_cache.py's holder, this head is going to be
    loaded with the real shipped weights, and filling its zero-init projections first would be
    overwritten anyway -- worse, it would hide a load that silently did nothing.
    """

    def __init__(self, blocks_per_ckpt=1):
        super().__init__()
        torch.manual_seed(0)
        self.confidence_head = ConfidenceHead(blocks_per_ckpt=blocks_per_ckpt, **CFG)


def _shipped_state():
    """The shipped head's tensors, with the "confidence_head." prefix stripped."""
    raw = torch.load(SHIPPED, map_location="cpu")
    pre = "confidence_head."
    sd = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
    return raw, sd


def _built(trainable=TRAINABLE_FIXED, blocks_per_ckpt=1, load=True, fill_pose=True,
           inject_at="after_tri_mul_in"):
    """A head in the TRAINER'S INSTALL ORDER, holding the shipped weights, at the given freeze.

    lean_readouts REPLACES memory_efficient_forward rather than wrapping it, so it must come first;
    pose_inblock.install() hard-stops otherwise. The shipped state dict is loaded AFTER the install
    because it carries pose_inblock's four projections.
    """
    m = _Model(blocks_per_ckpt=blocks_per_ckpt)
    LS.install(m, pae_only=True)
    LS.drop_single_pathway(m)
    pose_inblock.install(m, block_index=BLK, pae_inject=True, inject_at=inject_at, loud=False)
    h = m.confidence_head
    rep = None
    if load:
        _raw, sd = _shipped_state()
        rep = h.load_state_dict(sd, strict=False)
    if fill_pose:
        # THE INJECTIONS, MADE NON-ZERO. They SHIP exactly zero so that step 0 is bit-identical to
        # the un-injected head; left at zero the suffix would not depend on the cut point and
        # section 6 would be testing nothing.
        with torch.no_grad():
            for i, t in enumerate(("pose_inblock_d", "pose_inblock_d_wo_onehot",
                                   "pose_inblock_pae_d", "pose_inblock_pae_d_wo_onehot")):
                w = getattr(h, t).weight
                w.copy_(torch.randn(w.shape,
                                    generator=torch.Generator().manual_seed(500 + i)) * 0.2)
    for n, p in h.named_parameters():
        p.requires_grad_(any(n.startswith(t) for t in trainable))
    return m, rep


def _inputs(seed=7, n=N_TOK):
    """One sample's worth of geometry and trunk tensors, in the shapes the head's forward takes."""
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


def _fwd_bwd(m, inp, autocast=None, grad=True):
    """One forward (and backward), returning (pae logits, {name -> grad})."""
    h = m.confidence_head
    for p in h.parameters():
        p.grad = None
    if autocast is None:
        out = _pae(m, inp)
    else:
        with torch.autocast(device_type="cpu", dtype=autocast):
            out = _pae(m, inp)
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


def _worst(ga, gb):
    """(max abs diff, name, max rel diff, how many compared) over two gradient dicts."""
    w, wn, wr, n = 0.0, "", 0.0, 0
    for k in ga:
        if ga[k] is None or gb.get(k) is None:
            continue
        n += 1
        a_, r_ = _cmp(ga[k], gb[k])
        if a_ > w:
            w, wn = a_, k
        wr = max(wr, r_)
    return w, wn, wr, n


class _Counter:
    """Forward-hook call counts on ONE CHILD of every candidate module."""

    def __init__(self, head):
        self.n, self.h = {}, []
        for bi, blk in enumerate(head.pairformer_stack.blocks):
            for sub, child in PROBE_CHILD.items():
                mod = getattr(getattr(blk, sub), child)
                key = f"pairformer_stack.blocks.{bi}.{sub}"
                self.n[key] = 0
                self.h.append(mod.register_forward_hook(self._mk(key)))

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


def _refuses(fn, must_contain):
    try:
        fn()
    except SystemExit as e:
        return must_contain.lower() in str(e).lower(), str(e).splitlines()[0][:100]
    return False, "did not raise SystemExit"


def main():
    print(__doc__.strip().splitlines()[0])
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    DOPS.uninstall()
    PFC.uninstall()
    inp = _inputs()

    # ================================================================================= 1
    print("\n1. the SHIPPED head really loaded, and the detected map is exactly the expected one")
    raw, sd = _shipped_state()
    ok(len(raw) == N_SHIPPED and len(sd) == N_SHIPPED,
       f"the shipped head holds {N_SHIPPED} tensors and all of them carry the expected prefix",
       f"{len(raw)} raw / {len(sd)} extracted")
    m, rep = _built()
    # A LOAD THAT SILENTLY DID NOTHING IS THE FIRST VERSION OF THIS MEASUREMENT. A wrong key
    # prefix with strict=False reports every module dead, including the live ones.
    ok(not rep.unexpected_keys, "load_state_dict returned NO unexpected keys",
       str(rep.unexpected_keys[:3]))
    ok(not rep.missing_keys, "and no missing keys", str(rep.missing_keys[:3]))
    h = m.confidence_head
    ok(float(h.linear_no_bias_pae.weight.abs().max()) > 0.5,
       "the shipped PAE readout is non-zero, so the logits can move",
       f"max|w| = {float(h.linear_no_bias_pae.weight.abs().max()):.4f}")

    rows = DOPS.scan(h, autocast_dtype=torch.bfloat16, **KERNELS)
    by = {r["name"]: r for r in rows}
    ok(len(rows) == 20, "20 candidate modules were measured (4 blocks x 5)", f"{len(rows)}")
    got_dead = sorted(r["name"] for r in rows if r["dead"])
    ok(got_dead == sorted(EXPECT_DEAD),
       f"exactly the {len(EXPECT_DEAD)} expected modules are dead "
       f"({len(EXPECT_DEAD_CUBIC)} of the 16 O(N^3) ops + block 1's pair_transition)",
       f"{len(got_dead)} detected; unexpected={sorted(set(got_dead) - set(EXPECT_DEAD))}, "
       f"missed={sorted(set(EXPECT_DEAD) - set(got_dead))}")
    n_cubic_dead = sum(1 for r in rows if r["dead"] and r["cubic"])
    ok(n_cubic_dead == 9, "nine of the sixteen O(N^3) pair operations", f"{n_cubic_dead}")
    # THE LIVE CONTROLS, NAMED, AND REQUIRED TO BE ALIVE. Without this half the map is satisfied by
    # any rule that says "everything is dead".
    not_dead = [n for n in EXPECT_ALIVE if by[n]["dead"]]
    ok(not not_dead, f"and the {len(EXPECT_ALIVE)} expected LIVE modules are NOT dead",
       str(not_dead))
    ok(all(by[n]["max_w"] > 1e-3 for n in EXPECT_ALIVE),
       "every live control holds real weights", f"min max|w| = "
       f"{min(by[n]['max_w'] for n in EXPECT_ALIVE):.4f}")
    # THE CLAIM "every live control's measured output is required to be non-zero" WAS NOT
    # IMPLEMENTED, and review caught it: scan() skips the probe entirely when the weights are
    # live (there is no point running a module the AND has already decided about), so all ten
    # live candidates came back with zero probe regimes. The map was still proven in both
    # directions -- by weights above and by the hook counts in section 3 -- but the
    # strongest-sounding half of the claim was absent. So it is forced here.
    live_out = {}
    for n in EXPECT_ALIVE:
        bi, sub = int(n.split(".")[2]), n.split(".")[3]
        mod = getattr(h.pairformer_stack.blocks[bi], sub)
        _z, det = DOPS._probe_zero(mod, sub, CFG["c_z"], **KERNELS)
        live_out[n] = max(d[1] for d in det)
    ok(all(v > 0 for v in live_out.values()),
       f"and all {len(EXPECT_ALIVE)} live controls were RUN and measured NON-ZERO output",
       f"min max|out| = {min(live_out.values()):.4e}, max = {max(live_out.values()):.4e}")
    ok(all(not _z for _z in [DOPS._probe_zero(
        getattr(h.pairformer_stack.blocks[int(n.split('.')[2])], n.split('.')[3]),
        n.split('.')[3], CFG["c_z"], **KERNELS)[0] for n in EXPECT_ALIVE]),
       "so the zero-probe itself says NO for every one of them, not just the weight test")
    ok(all(by[n]["max_w"] < DOPS.FLOOR for n in EXPECT_DEAD),
       f"every dead module's weights are below the {DOPS.FLOOR:g} floor",
       f"max = {max(by[n]['max_w'] for n in EXPECT_DEAD):.6e}")
    print("      measured max|w| per candidate:")
    for r in rows:
        tag = "DEAD " if r["dead"] else "alive"
        pz = [d[1] for d in r["probe"]]
        print(f"        {tag} {r['name']:<46} max|w| = {r['max_w']:.6e}"
              + (f"  probe max|out| = {max(pz):.6e} over {len(pz)} regime(s)" if pz else ""))

    # ================================================================================= 2
    print("\n2. THE TRAP: a FRESH head looks exactly like a fully-dead one to the probe alone")
    m_fresh, _ = _built(load=False, fill_pose=False)
    rows_f = DOPS.scan(m_fresh.confidence_head, autocast_dtype=torch.bfloat16, **KERNELS)
    # A FRESH PairformerBlock is an EXACT IDENTITY on z: every residual branch ends in a zero-init
    # projection. So "did it return zero?" is True for all 20 candidates. That is the measurement
    # that reported all 16 cubic ops dead the first time, including the live ones.
    probe_only = 0
    for bi, blk in enumerate(m_fresh.confidence_head.pairformer_stack.blocks):
        for sub in DOPS.CANDIDATES:
            # THE SAME REGIMES rows_f was scanned under, including autocast, or the "20 of 20"
            # and the "0 of 20" would be measured under two different rules.
            z, _d = DOPS._probe_zero(getattr(blk, sub), sub, CFG["c_z"],
                                     autocast_dtype=torch.bfloat16, **KERNELS)
            probe_only += int(z)
    ok(probe_only == 20,
       "the OUTPUT-IS-ZERO probe alone calls 20 of 20 candidates dead on a fresh head",
       f"{probe_only}/20")
    ok(sum(1 for r in rows_f if r["dead"]) == 0,
       "but the full rule (below the floor AND measured zero) calls 0 of 20 dead",
       f"{sum(1 for r in rows_f if r['dead'])}/20")
    ok(all(not r["below_floor"] for r in rows_f),
       "because a fresh module's layer norm ships at 1.0, nowhere near the floor",
       f"min max|w| = {min(r['max_w'] for r in rows_f):.4f}")
    # AND THE CONVERSE HALF: on the shipped head the threshold alone would be right here, but it is
    # the AND that makes the verdict safe against both failures. Stated as a measurement, not a
    # claim: every shipped module below the floor also measured zero.
    ok(all(r["probe_zero"] for r in rows if r["below_floor"]),
       "on the SHIPPED head every below-floor module also measured exactly 0.0 when RUN",
       f"{sum(1 for r in rows if r['below_floor'])} below floor")
    del m_fresh, rows_f

    # ================================================================================= 3
    print("\n3. THE SKIP CHANGES SOMETHING: what runs with the stubs in and what does not")
    info = DOPS.install(m, autocast_dtype=torch.bfloat16, loud=True, **KERNELS)
    ok(info is not None and info["n_stubbed"] == len(EXPECT_DEAD),
       f"install() stubbed {len(EXPECT_DEAD)} module(s)", str(info and info["n_stubbed"]))
    ok(sorted(info["stubbed"]) == sorted(EXPECT_DEAD), "and they are the expected ones")
    cnt = _Counter(h)
    DOPS.reset_stats()
    with torch.no_grad():
        _pae(m, inp)
    n_in = dict(cnt.n)
    dead_run = {k: v for k, v in n_in.items() if k in set(EXPECT_DEAD) and v > 0}
    ok(not dead_run, "NOT ONE stubbed module executed -- zero forward calls inside all 10",
       str(dead_run))
    alive_idle = {k: v for k, v in n_in.items() if k in set(EXPECT_ALIVE) and v == 0}
    ok(not alive_idle, "and every KEPT module did execute, so the counter is not broken",
       f"counts {sorted({v for k, v in n_in.items() if k in set(EXPECT_ALIVE)})}")
    st = DOPS.stats()
    # ONE FEWER CALL THAN THERE ARE STUBS, deliberately. Block 3's tri_att_end is bracketed by two
    # transposes that exist only to flip z onto the axis it attends over; once its branch is a
    # no-op, pose_inblock.block_forward elides the WHOLE span -- both `.contiguous()` copies and
    # the module -- so that stub is never reached at all. A stub that answers is a stub whose two
    # 2.50 GiB pair copies (at 2,289 tokens) were still paid for.
    _elided = sum(1 for n in EXPECT_DEAD if n == "pairformer_stack.blocks.3.tri_att_end")
    ok(st["calls"] == len(EXPECT_DEAD) - _elided,
       f"the stubs answered {len(EXPECT_DEAD) - _elided} call(s): one per skipped module, LESS "
       f"the {_elided} whose entire branch is elided",
       f"{st['calls']}")
    cnt.close()

    # ================================================================================= 4
    print("\n4. THE TREE IS UNTOUCHED: keys, names and parameter identity survive the swap")
    DOPS.uninstall()
    keys_u = set(h.state_dict().keys())
    names_u = [n for n, _ in h.named_parameters()]
    ids_u = {n: id(p) for n, p in h.named_parameters()}
    mods_u = [n for n, _ in h.named_modules()]
    pfx_u = PFC._prefix_param_names(h, BLK)
    DOPS.install(m, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    ok(set(h.state_dict().keys()) == keys_u,
       "the head's state-dict key set is identical (phase_k_eval guards on exactly this)",
       f"{len(keys_u)} keys")
    ok([n for n, _ in h.named_parameters()] == names_u,
       "named_parameters() is identical, in the same order (--train-only matches by name)",
       f"{len(names_u)} tensors")
    ok({n: id(p) for n, p in h.named_parameters()} == ids_u,
       "and every parameter is the SAME OBJECT, so requires_grad and the optimiser still reach it")
    ok([n for n, _ in h.named_modules()] == mods_u,
       "named_modules() is identical too (prefix_cache scans it for live dropout)",
       f"{len(mods_u)} modules")
    ok(all(isinstance(getattr(h.pairformer_stack.blocks[int(n.split('.')[2])],
                              n.split('.')[3]), DOPS._DeadStub) for n in EXPECT_DEAD),
       "each dead slot now holds a husk, not the original module")
    ok(set(PFC._prefix_param_names(h, BLK)) == set(pfx_u) and len(pfx_u) > 100,
       "prefix_cache derives the IDENTICAL frozen-prefix set from the stubbed tree",
       f"{len(pfx_u)} tensor(s), symmetric difference "
       f"{len(set(PFC._prefix_param_names(h, BLK)) ^ set(pfx_u))}")

    # ================================================================================= 5
    print("\n5. fp32: logits AND gradients to all 20 trainable tensors are BIT-IDENTICAL")
    live = [n for n, p in h.named_parameters() if p.requires_grad]
    n_live_p = sum(p.numel() for p in h.parameters() if p.requires_grad)
    ok(len(live) == N_FIXED and n_live_p == N_FIXED_P,
       f"the freeze is the arm's --train-only with the dead tri_att_end dropped: {N_FIXED} "
       f"tensors / {N_FIXED_P:,} parameters", f"{len(live)} / {n_live_p:,}")
    out_s, g_s = _fwd_bwd(m, inp)
    DOPS.uninstall()
    out_u, g_u = _fwd_bwd(m, inp)
    da, dr = _cmp(out_s, out_u)
    ok(da == 0.0, "PAE logits are BIT-IDENTICAL with and without the skip",
       f"max|abs| = {da:.3e}, max|rel| = {dr:.3e}")
    ok(float(out_u.abs().max()) > 0, "and the logits are not identically zero",
       f"max|logit| = {float(out_u.abs().max()):.4e}")
    w, wn, wr, nc = _worst(g_s, g_u)
    ok(nc == N_FIXED, f"exactly {N_FIXED} tensors received a gradient on both paths", f"{nc}")
    ok(w == 0.0, "every trainable gradient is BIT-IDENTICAL",
       f"max|abs| = {w:.3e} ({wn or 'none'}), max|rel| = {wr:.3e}")
    ok(all(float(g_s[n].abs().max()) > 0 for n in g_s if g_s[n] is not None),
       "and every one of them is NON-ZERO, so the comparison is not 0 == 0")
    # THE ONE THAT COULD HAVE GONE EITHER WAY. Block 3's tri_att_end sits AFTER the injection, so
    # the z entering it DOES require grad; skipping it replaces (I + J) with I on the backward
    # path, which is only bit-identical if the module's Jacobian underflows to exactly zero too.
    # The forward being zero does not imply it. Measured, not assumed.
    ok(by["pairformer_stack.blocks.3.tri_att_end"]["dead"]
       and _cmp(g_s["pose_inblock_d.weight"], g_u["pose_inblock_d.weight"])[0] == 0.0,
       "including the gradient THROUGH block 3's stubbed tri_att_end, which is downstream of the "
       "injection -- its Jacobian underflows to exactly zero as well",
       f"max|abs| on pose_inblock_d = "
       f"{_cmp(g_s['pose_inblock_d.weight'], g_u['pose_inblock_d.weight'])[0]:.3e}")
    frozen_got = [n for n, p in h.named_parameters() if p.grad is not None and not p.requires_grad]
    ok(not frozen_got, "NOT ONE frozen tensor received a gradient on either path",
       f"{len(frozen_got)}")

    # ================================================================================= 6
    print("\n6. bf16 autocast, reported SEPARATELY from fp32")
    DOPS.install(m, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    out_sb, g_sb = _fwd_bwd(m, inp, autocast=torch.bfloat16)
    DOPS.uninstall()
    out_ub, g_ub = _fwd_bwd(m, inp, autocast=torch.bfloat16)
    dab, drb = _cmp(out_sb, out_ub)
    ok(dab == 0.0, "bf16: PAE logits are BIT-IDENTICAL with and without the skip",
       f"max|abs| = {dab:.3e}, max|rel| = {drb:.3e}")
    wb, wbn, wbr, ncb = _worst(g_sb, g_ub)
    ok(wb == 0.0 and ncb == N_FIXED, "bf16: every trainable gradient is BIT-IDENTICAL",
       f"{ncb} tensor(s), max|abs| = {wb:.3e} ({wbn or 'none'}), max|rel| = {wbr:.3e}")
    # AND THE TWO PRECISIONS REALLY ARE DIFFERENT COMPUTATIONS, or the bf16 row above is the fp32
    # row again under another name.
    dprec, _ = _cmp(out_ub, out_u)
    ok(dprec > 1e-4, "and bf16 is a DIFFERENT computation from fp32, so this row is its own test",
       f"max|diff| vs fp32 = {dprec:.3e}")

    # ================================================================================= 7
    print("\n7. it COMPOSES with the frozen-prefix cache")
    DOPS.install(m, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    PFC.uninstall()
    pinfo = PFC.install(m, 1.0, lean_single=True, device=None, loud=False)
    ok(pinfo is not None and pinfo["block_index"] == BLK,
       f"the cache installs on the stubbed tree and cuts inside block {BLK}",
       str(pinfo and pinfo["block_index"]))

    def _two_pass(key=("1abc", "free_500000", 0), cached=True):
        """Pass A then pass B over one pose, exactly as phase_k_train sequences them."""
        for p in h.parameters():
            p.grad = None
        if cached:
            PFC.set_pose(key, fill=True, n_tok=N_TOK)
        try:
            with torch.no_grad():
                _pae(m, inp)
        finally:
            if cached:
                PFC.clear_pose()
        if cached:
            PFC.set_pose(key, fill=False, n_tok=N_TOK)
        try:
            out = _pae(m, inp)
        finally:
            if cached:
                PFC.clear_pose()
        out.float().pow(2).mean().backward()
        return (out.detach().clone(),
                {n: (None if p.grad is None else p.grad.detach().clone())
                 for n, p in h.named_parameters()})

    PFC.step_reset()
    PFC.reset_stats()
    DOPS.reset_stats()
    cnt = _Counter(h)
    out_both, g_both = _two_pass(cached=True)
    ps = PFC.stats()
    n_both = dict(cnt.n)
    cnt.close()
    ok(ps["hit"] >= 1 and ps["put"] >= 1, "the cache filled in pass A and hit in pass B",
       f"hit={ps['hit']} miss={ps['miss']} put={ps['put']}")
    ok(not {k: v for k, v in n_both.items() if k in set(EXPECT_DEAD) and v > 0},
       "and still not one stubbed module executed, in either pass")
    ok(DOPS.stats()["calls"] >= len(EXPECT_DEAD) - _elided,
       "the stubs fired, bar the elided branch(es)", f"{DOPS.stats()['calls']} call(s)")
    # ON A HIT the blocks before the cut are identities, so their stubs are not even reached --
    # which is the honest reason the two optimisations are complementary rather than redundant.
    ok(n_both["pairformer_stack.blocks.0.pair_transition"] == 1,
       "a kept module BEFORE the cut ran once (pass A only; pass B skipped the whole prefix)",
       f"x{n_both['pairformer_stack.blocks.0.pair_transition']}")
    PFC.uninstall()
    DOPS.uninstall()
    out_ref, g_ref = _two_pass(cached=False)
    dc, _ = _cmp(out_both, out_ref)
    ok(dc == 0.0, "stubs + cache give BIT-IDENTICAL PAE logits to a run with neither",
       f"max|abs| = {dc:.3e}")
    wc, wcn, wcr, ncc = _worst(g_both, g_ref)
    ok(wc == 0.0 and ncc == N_FIXED, "and bit-identical gradients to all of them",
       f"{ncc} tensor(s), max|abs| = {wc:.3e} ({wcn or 'none'}), max|rel| = {wcr:.3e}")

    # ================================================================================= 8
    print("\n8. the preconditions REFUSE rather than degrading")
    DOPS.uninstall()
    PFC.uninstall()
    # (a) THE ONE THAT FIRES ON TODAY'S ARM. --train-only includes block 3's tri_att_end, which is
    #     dead; a stubbed module gets NO gradient (its .grad stays None rather than becoming an
    #     explicit zero), so the optimiser and the collective would see a different set than a run
    #     without the flag. The message must name the module and say to drop it from --train-only.
    m_arm, _ = _built(trainable=TRAINABLE_ARM)
    n_arm = sum(1 for _n, p in m_arm.confidence_head.named_parameters() if p.requires_grad)
    n_arm_p = sum(p.numel() for p in m_arm.confidence_head.parameters() if p.requires_grad)
    ok(n_arm == N_ARM and n_arm_p == N_ARM_P,
       f"the arm's --train-only really is {N_ARM} tensors / {N_ARM_P:,} parameters",
       f"{n_arm} / {n_arm_p:,}")
    r, msg = _refuses(lambda: DOPS.install(m_arm, autocast_dtype=torch.bfloat16, loud=False, **KERNELS),
                      "drop")
    ok(r, "refuses when a dead module is in the TRAINABLE set", msg)
    full = ""
    try:
        DOPS.install(m_arm, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    except SystemExit as e:
        full = str(e)
    ok("blocks.3.tri_att_end" in full and "--train-only" in full,
       "and the message NAMES the module and says to drop it from --train-only")
    ok(not DOPS.stats()["on"] and not isinstance(
        m_arm.confidence_head.pairformer_stack.blocks[0].tri_mul_out, DOPS._DeadStub),
       "and nothing was stubbed on the way to refusing")
    # (b) A HOOK ALREADY ON A DEAD MODULE. The husk does not share hook dicts, so stubbing would
    #     stop it firing -- silently.
    m_hk, _ = _built()
    _hh = m_hk.confidence_head.pairformer_stack.blocks[0].tri_mul_out.register_forward_hook(
        lambda *_a: None)
    r, msg = _refuses(lambda: DOPS.install(m_hk, autocast_dtype=torch.bfloat16, loud=False, **KERNELS),
                      "hook")
    ok(r, "refuses to stub a module that already carries a hook", msg)
    _hh.remove()
    # (b2) A HOOK ON A *CHILD* of a dead module. The husk shares `_modules`, so the child stays
    #      reachable and would silently stop firing -- the same failure one level down. The first
    #      version of the refusal counted hooks only on the candidate itself and would have
    #      allowed this; caught in review.
    m_hk2, _ = _built()
    _hh2 = m_hk2.confidence_head.pairformer_stack.blocks[0].tri_mul_out.layer_norm_in \
        .register_forward_hook(lambda *_a: None)
    r, msg = _refuses(lambda: DOPS.install(m_hk2, autocast_dtype=torch.bfloat16, loud=False,
                                           **KERNELS), "hook")
    ok(r, "refuses a hook on a CHILD of a dead module too, not just on the module", msg)
    _hh2.remove()
    # (b3) THE ROLLBACK IS REAL. The three post-swap guards say "Rolled back; do not launch", and
    #      they were SILENT NO-OPS: uninstall() finds the modules to restore through _S["head"],
    #      which was only assigned after those guards. Forced here by making the husk stop sharing
    #      `_modules`, which is exactly what the key-set guard exists to catch. NOTE it has to be
    #      `_modules` and not `_parameters`: every one of these modules holds its tensors in CHILD
    #      modules (layer_norm_in.weight, mha.linear_q.weight, ...) and its own `_parameters` is
    #      already empty, so emptying that would have changed nothing and the test would have
    #      "passed" while exercising neither the guard nor the rollback.
    m_rb, _ = _built()
    keys_rb = set(m_rb.confidence_head.state_dict().keys())

    class _BadStub(DOPS._DeadStub):
        def __init__(self, orig, where):
            super().__init__(orig, where)
            object.__setattr__(self, "_modules", type(orig._modules)())

    _real = DOPS._DeadStub
    DOPS._DeadStub = _BadStub
    try:
        r, msg = _refuses(lambda: DOPS.install(m_rb, autocast_dtype=torch.bfloat16, loud=False,
                                               **KERNELS), "rolled back")
    finally:
        DOPS._DeadStub = _real
    ok(r, "a husk that drops parameters is caught by the key-set guard", msg)
    ok(set(m_rb.confidence_head.state_dict().keys()) == keys_rb,
       "and the rollback really restored the head -- all 252 keys are back",
       f"{len(m_rb.confidence_head.state_dict())} of {len(keys_rb)}")
    ok(not any(isinstance(getattr(m_rb.confidence_head.pairformer_stack.blocks[
        int(n.split('.')[2])], n.split('.')[3]), (_real, _BadStub)) for n in EXPECT_DEAD),
       "with no stub of either kind left in the tree")
    DOPS.uninstall()
    # (c) A STUBBED WEIGHT THAT MOVES MID-RUN, caught at the step boundary -- the one moment an
    #     optimiser step could have touched it.
    m_mv, _ = _built()
    DOPS.install(m_mv, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    with torch.no_grad():
        m_mv.confidence_head.pairformer_stack.blocks[0].tri_mul_out.linear_z.weight.add_(1e-3)
    r, msg = _refuses(DOPS.step_check, "changed value")
    ok(r, "the step boundary hard-stops if a stubbed module's weights changed value", msg)
    DOPS.uninstall()
    # (d) AND IF ONE OF THEM BECAME TRAINABLE while the stub was still installed.
    m_tr, _ = _built()
    DOPS.install(m_tr, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    m_tr.confidence_head.pairformer_stack.blocks[2].tri_att_start.layer_norm.weight.requires_grad_(
        True)
    r, msg = _refuses(DOPS.step_check, "require grad")
    ok(r, "and if a stubbed parameter became trainable", msg)
    DOPS.uninstall()
    # (e) A CLEAN STEP BOUNDARY DOES NOT RAISE, or (c) and (d) prove only that it always raises.
    m_cl, _ = _built()
    DOPS.install(m_cl, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    try:
        for _ in range(3):
            DOPS.step_check()
        clean = True
    except SystemExit as e:
        clean, _m = False, str(e)
    ok(clean, "three clean step boundaries in a row raise nothing")
    # (f) THE IN-PLACE BRANCH IS REFUSED BY THE STUB ITSELF. There a tri-mul returns the UPDATED z
    #     rather than the update, so zeros would ERASE the pair representation, not leave it alone.
    stub = m_cl.confidence_head.pairformer_stack.blocks[0].tri_mul_out
    r, msg = _refuses(lambda: stub(torch.zeros(4, 4, CFG["c_z"]), mask=None, inplace_safe=True),
                      "inplace")
    ok(r, "a stub called on the inplace_safe branch hard-stops instead of erasing z", msg)
    # (g) INSTALLING TWICE, and scanning a tree that already holds stubs (which would read the
    #     stub's own zeros as evidence the module is dead -- circular).
    r, msg = _refuses(lambda: DOPS.install(m_cl, loud=False, **KERNELS), "already installed")
    ok(r, "refuses a second install", msg)
    r, msg = _refuses(lambda: DOPS.scan(m_cl.confidence_head, **KERNELS), "already installed")
    ok(r, "and refuses to SCAN a tree that already holds stubs", msg)

    # ================================================================================= 9
    print("\n9. uninstall() puts the originals back")
    orig_t = {n: type(getattr(m_cl.confidence_head.pairformer_stack.blocks[int(n.split('.')[2])],
                              n.split('.')[3])).__name__ for n in EXPECT_DEAD}
    ok(all(v == "_DeadStub" for v in orig_t.values()), "they are husks before uninstall()")
    DOPS.uninstall()
    after_t = {n: type(getattr(m_cl.confidence_head.pairformer_stack.blocks[int(n.split('.')[2])],
                               n.split('.')[3])).__name__ for n in EXPECT_DEAD}
    ok(all(v != "_DeadStub" for v in after_t.values()),
       "and real modules after it", str(sorted(set(after_t.values()))))
    ok(not DOPS.stats()["on"] and DOPS.stats()["stubbed"] == 0,
       "and the module reports itself off")
    cnt = _Counter(m_cl.confidence_head)
    with torch.no_grad():
        _pae(m_cl, inp)
    ok(all(v == 1 for v in cnt.n.values()),
       "every candidate -- dead and alive -- runs again after uninstall()",
       f"counts {sorted(set(cnt.n.values()))}")
    cnt.close()

    # MOVED TO THE END ON PURPOSE: section 8 deliberately leaves m_cl STUBBED so that
    # section 9 can watch uninstall() put the originals back, and install() refuses a
    # second install while that is live. This needs a clean slate, so it runs after it.
    # ================================================================================ 10
    print("\n10. a NON-ZERO --pair-dropout is still bit-identical, which is a named claim")
    # dropout_add_rowwise derives its mask shape from x.shape and draws the same number of values
    # whether x is the module's zeros or the stub's, and 0 * mask / (1-p) is still 0. The claim is
    # in dead_ops.py's docstring and had no coverage: every other section runs at p_drop = 0.
    m_dp, _ = _built()
    for _b in m_dp.confidence_head.pairformer_stack.blocks:
        _b.p_drop = 0.1
    m_dp.confidence_head.train()
    _sd_dp = {k: v.detach().clone() for k, v in m_dp.confidence_head.state_dict().items()}
    torch.manual_seed(4242)
    out_dp_u, g_dp_u = _fwd_bwd(m_dp, inp)
    DOPS.install(m_dp, autocast_dtype=torch.bfloat16, loud=False, **KERNELS)
    torch.manual_seed(4242)
    out_dp_s, g_dp_s = _fwd_bwd(m_dp, inp)
    DOPS.uninstall()
    ok(float((out_dp_u - out_dp_s).abs().max()) == 0.0,
       "at --pair-dropout 0.1 in train mode the logits are still BIT-IDENTICAL",
       f"max|abs| = {float((out_dp_u - out_dp_s).abs().max()):.3e}")
    w_dp, w_dpn, _wr, n_dp = _worst(g_dp_s, g_dp_u)
    ok(w_dp == 0.0 and n_dp == N_FIXED, "and so are the gradients, so the RNG stream is unchanged",
       f"{n_dp} tensor(s), max|abs| = {w_dp:.3e} ({w_dpn or 'none'})")
    # AND THE DROPOUT WAS ACTUALLY ACTIVE, or this section is the p_drop = 0 case again.
    torch.manual_seed(9999)
    out_dp_other, _ = _fwd_bwd(m_dp, inp, grad=False)
    ok(float((out_dp_other - out_dp_u).abs().max()) > 1e-6,
       "(and the dropout really was live -- a different seed gives different logits)",
       f"max|diff| = {float((out_dp_other - out_dp_u).abs().max()):.3e}")
    ok(all(torch.equal(_sd_dp[k], v) for k, v in m_dp.confidence_head.state_dict().items()),
       "and nothing in the head moved while we were doing that")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
