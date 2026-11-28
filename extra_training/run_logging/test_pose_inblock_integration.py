#!/usr/bin/env python
"""The in-block injection on a REAL confidence head, through install(), with blocks_per_ckpt=1.

WHY THIS EXISTS SEPARATELY FROM test_pose_inblock_graph.py. That test calls
pose_inblock.block_forward directly on a stand-in head. It establishes the autograd boundary for
a standalone block, which is the thing most likely to be got wrong -- but it never runs
install(), its memory_efficient_forward wrapper, its _prep_blocks replacement, the pae_ln hook, or
the production activation-checkpointing path. Review's objection was exactly that: a unit test of
the mechanism is not evidence that the installed trainer respects it when a block's forward is
RE-RUN during backward. So this one builds the real ConfidenceHead at the live configuration, puts
the real freeze on it, and drives it end to end.

The checkpoint-recomputation case is the one with teeth. With blocks_per_ckpt=1 the block's forward
runs a second time inside backward, by which point the wrapper's `finally` has already cleared the
stashed geometry -- the _NoPose-on-backward bug that pose_perblock actually shipped. The closure
holds the geometry in a local for that reason, and section 5 is what proves it.

CPU ONLY. The head is ~19.8M parameters at the live configuration (c_z 256, c_s 384, n_blocks 4),
which is a couple of seconds and ~250 MB to instantiate, and 16 tokens of geometry.
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

import pose_inblock                                                      # noqa: E402
import lean_readouts as LS                                               # noqa: E402
from protenix.model.modules.confidence import ConfidenceHead             # noqa: E402

FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


# THE LIVE CONFIGURATION, not the defaults in confidence.py's docstring -- c_z is 256 there and
# 128 in the docstring, and the whole point of this test is the real head.
CFG = dict(n_blocks=4, c_s=384, c_z=256, c_s_inputs=449, b_pae=64, b_pde=64, b_plddt=50,
           b_resolved=2, max_atoms_per_token=24, pairformer_dropout=0.0, hidden_scale_up=True)

N_TOK = 16
# THE FREEZE THE LAUNCHER APPLIES, as a name test rather than a regex, so this file does not have
# to stay in step with the shell quoting. Trainable: block 3's two triangle ATTENTIONS, its pair
# transition, and the injections. Everything else -- blocks 0-2 entirely, and block 3's two
# triangle MULTIPLICATIONS -- is frozen.
# INCLUDING THE READOUT, because INBLOCK_READOUT defaults to 1. This tuple omitted
# linear_no_bias_pae and pae_ln, which was correct when the default was 0 and became wrong the
# moment it was flipped -- so this file was asserting the ABLATION's 25 tensors / 1,467,904
# parameters while the arm it is supposed to describe trains 28 / 1,484,800. Caught by a second
# reader measuring the launcher's own pattern against a real head rather than reading this tuple.
TRAINABLE = ("pairformer_stack.blocks.3.tri_att_start.",
             "pairformer_stack.blocks.3.tri_att_end.",
             "pairformer_stack.blocks.3.pair_transition.",
             "pose_inblock_",
             "linear_no_bias_pae.",
             "pae_ln.")
FROZEN_MUST = ("pairformer_stack.blocks.0.", "pairformer_stack.blocks.1.",
               "pairformer_stack.blocks.2.", "pairformer_stack.blocks.3.tri_mul_out.",
               "pairformer_stack.blocks.3.tri_mul_in.")


class _Model(torch.nn.Module):
    """install() reaches for model.confidence_head, so it needs a holder.

    TWO THINGS ARE DONE TO THE FRESH HEAD, and without either one this whole file passes for the
    wrong reason.

    FIRST, THE PAE READOUT IS GIVEN A NON-ZERO WEIGHT. confidence.py:127-129 zero-initialises
    linear_no_bias_pae.weight, so a freshly constructed head emits PAE logits that are identically
    zero no matter what the stack does, and every check of the form "X changes / does not change
    the logits" reduces to 0 == 0.

    SECOND, EVERY ZERO-INIT TENSOR IN THE PAIRFORMER STACK IS FILLED. Each of a block's five
    residual branches ends in a zero-initialised projection -- tri_mul_out/tri_mul_in's linear_z
    and linear_g, tri_att_start/tri_att_end's mha.linear_o and mha.linear_g, pair_transition's
    linear_no_bias -- so a FRESH PairformerBlock is an EXACT IDENTITY on z: measured
    torch.equal(z_in, z_out) is True. A fresh four-block stack is therefore an identity too, which
    means an injected delta reaches the readout unchanged from ANY depth, and "the same weight at
    block 2 gives different logits than at block 3" is false on such a head -- bitwise, for all
    four depths. That is not a bug in the injection; it is a stack that is not yet a function.
    Filling the zero-init projections makes it one.

    Everything is filled from dedicated generators, so two instances agree -- but see _loaded():
    cross-instance equality is NOT assumed anywhere, because ConfidenceHead construction turns out
    not to be deterministic under torch.manual_seed (84 tensors differ between two consecutive
    constructions at the same seed).
    """

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
            # THE STACK, MADE INTO A FUNCTION. Only pairformer_stack tensors are touched: the
            # head's lower_bins / upper_bins are requires_grad=False Parameters holding the
            # distance bin EDGES, and randomising those would silently move the one-hot bins.
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


def _built(block_index=3, blocks_per_ckpt=1, pae_inject=True):
    """A head in the TRAINER'S INSTALL ORDER: lean_readouts, then the single drop, then the pose.

    pose_inblock.install() hard-stops if lean_readouts arrives second, because lean_readouts
    REPLACES memory_efficient_forward rather than wrapping it and would throw the injection away.
    """
    m = _Model(blocks_per_ckpt=blocks_per_ckpt)
    LS.install(m, pae_only=True)
    LS.drop_single_pathway(m)
    info = pose_inblock.install(m, block_index=block_index, pae_inject=pae_inject, loud=False)
    return m, info


def _loaded(src, block_index=3, blocks_per_ckpt=None, pae_inject=True):
    """A second head, installed at `block_index`, made VALUE-IDENTICAL to `src`.

    NOT built-from-the-same-seed. ConfidenceHead construction is not deterministic under
    torch.manual_seed: two consecutive constructions at seed 0 differ in 84 tensors. Every
    cross-instance comparison in this file therefore goes through load_state_dict, so the only
    difference between the two heads is the one the check is about.
    """
    m, _ = _built(block_index=block_index, blocks_per_ckpt=blocks_per_ckpt, pae_inject=pae_inject)
    with torch.no_grad():
        m.confidence_head.load_state_dict(src.confidence_head.state_dict())
    return m


def _inputs(seed=7, n=N_TOK):
    """One sample's worth of geometry and trunk tensors. Deterministic."""
    g = torch.Generator().manual_seed(seed)
    f = lambda *s: torch.randn(*s, generator=g)                          # noqa: E731
    return {
        "input_feature_dict": {"atom_to_token_idx": torch.arange(n)},
        "s_trunk": f(n, CFG["c_s"]),
        "z_pair": f(n, n, CFG["c_z"]),
        "pair_mask": torch.ones(n, n),
        # SPREAD OVER THE BIN RANGE (3.25 to 52 A), so one_hot does not put every pair in one bin
        # and the one-hot term is not a constant.
        "x_pred_rep_coords": f(n, 3) * 12.0,
    }


def _freeze(head):
    for n, p in head.named_parameters():
        p.requires_grad_(any(n.startswith(t) or t in n for t in TRAINABLE))
    return ([n for n, p in head.named_parameters() if p.requires_grad],
            [n for n, p in head.named_parameters() if not p.requires_grad])


def _pae(model, inp):
    """The PAE logits only. lean_readouts returns them second."""
    return model.confidence_head.memory_efficient_forward(**inp)[1]


def main():
    print(__doc__.strip().splitlines()[0])

    print("\n1. install() on a REAL head, in the trainer's order (lean_readouts first)")
    m = _Model()
    LS.install(m, pae_only=True)
    LS.drop_single_pathway(m)
    before = {n for n, _ in m.confidence_head.named_parameters()}
    info = pose_inblock.install(m, block_index=3, pae_inject=True, loud=False)
    after = {n for n, _ in m.confidence_head.named_parameters()}
    added = sorted(after - before)
    ok(len(added) == 4, "install() added exactly 4 tensors", str(len(added)))
    ok(all("pose_inblock" in n for n in added), "all four are pose_inblock tensors", str(added))
    ok(all(float(dict(m.confidence_head.named_parameters())[n].abs().max()) == 0.0
           for n in added), "all four are EXACTLY zero at init")
    ok(info["block_index"] == 3 and info["pae_inject"] is True,
       "install() reports block 3 with the readout injection", str(info["block_index"]))
    ok(m.confidence_head.pairformer_stack.blocks_per_ckpt == 1,
       "the stack is still at blocks_per_ckpt=1 (the production path)")

    print("\n2. ZERO-INIT IS A NO-OP: the installed head's PAE logits match an un-installed one")
    # THE VACUITY GUARD FIRST. If the readout were still at its zero init, every comparison below
    # would be 0 == 0 and would pass while proving nothing.
    ok(float(m.confidence_head.linear_no_bias_pae.weight.abs().max()) > 0,
       "the PAE readout is non-zero, so the logits can actually move")
    m_ref = _Model()
    LS.install(m_ref, pae_only=True)
    LS.drop_single_pathway(m_ref)
    # LOADED, NOT RE-SEEDED. m_ref has no injection installed, so it takes m's state minus the
    # four injection tensors; load_state_dict is what makes the two heads equal, because the seed
    # does not (84 tensors differ between two constructions at seed 0).
    m_ref.confidence_head.load_state_dict(
        {k: v for k, v in m.confidence_head.state_dict().items() if "pose_inblock" not in k})
    ok(m.n_filled == m_ref.n_filled and m.n_filled >= 40,
       "the stack was made a real function, not left as an identity",
       f"{m.n_filled} zero-init projection(s) filled per head")
    inp = _inputs()
    with torch.no_grad():
        p_inj = _pae(m, inp)
        p_ref = _pae(m_ref, inp)
    same = torch.equal(p_inj, p_ref)
    ok(same, "PAE logits are BIT-IDENTICAL with the zero-init injection installed",
       f"max|diff| = {float((p_inj - p_ref).abs().max()):.3e}")
    ok(p_inj.shape == (N_TOK, N_TOK, CFG["b_pae"]), "PAE logits have the expected shape",
       str(tuple(p_inj.shape)))
    ok(float(p_inj.abs().max()) > 0, "and the baseline logits are not identically zero",
       f"max|logit| = {float(p_inj.abs().max()):.3e}")

    print("\n3. the freeze: which tensors carry gradient at all")
    live, dead = _freeze(m.confidence_head)
    # 28 TENSORS AND 1,484,800 PARAMETERS at the INBLOCK_READOUT=1 default, so
    # these two numbers are the arm's advertised capacity and a change to either is a change to
    # the experiment. The head totals 19,772,750 here: the live 19,752,270 plus the injections'
    # 20,480 (2 x 39 x 256 for the one-hot terms, 2 x 1 x 256 for the raw-distance terms).
    ok(len(live) == 28, "28 tensors are trainable (the INBLOCK_READOUT=1 default)",
       f"{len(live)}")
    ok(all(any(n.startswith(t) or t in n for t in TRAINABLE) for n in live),
       "every trainable tensor is block 3's attentions, its transition, or an injection")
    bad = [n for n in live if any(n.startswith(f) for f in FROZEN_MUST)]
    ok(not bad, "no block 0-2 tensor and no block 3 tri-mul tensor is trainable", str(bad[:3]))
    n_live = sum(p.numel() for n, p in m.confidence_head.named_parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in m.confidence_head.parameters())
    ok(n_live == 1_484_800, "and they hold 1,484,800 parameters", f"{n_live:,}")
    ok(n_tot == 19_772_750, "the head totals 19,772,750 (19,752,270 live + 20,480 injected)",
       f"{n_tot:,}")
    print(f"      ({n_live:,} trainable of {n_tot:,} -- {100.0 * n_live / n_tot:.2f}%)")

    print("\n4. THE BOUNDARY: what requires grad as z moves through block 3")
    # FORWARD HOOKS RECORDING requires_grad, which is the deterministic probe. A tensor's
    # requires_grad IS whether autograd will build a graph from the ops that consume it, so
    # "tri_mul_in's input does not require grad" and "tri_mul_in is not in the backward" are the
    # same statement. Recorded on the LAST call so checkpoint recomputation does not hide it.
    seen = {}
    blk = m.confidence_head.pairformer_stack.blocks[3]

    def probe(tag):
        def hook(_mod, inp, out):
            t = inp[0] if isinstance(inp, tuple) else inp
            seen[tag] = (bool(t.requires_grad),
                         bool(out.requires_grad) if torch.is_tensor(out) else None)
        return hook

    hs = [blk.tri_mul_out.register_forward_hook(probe("tri_mul_out")),
          blk.tri_mul_in.register_forward_hook(probe("tri_mul_in")),
          blk.tri_att_start.register_forward_hook(probe("tri_att_start"))]
    out = _pae(m, inp)
    loss = out.float().pow(2).mean()
    loss.backward()
    for h in hs:
        h.remove()
    ok(seen.get("tri_mul_out", (None,))[0] is False,
       "tri_mul_out's INPUT does not require grad -- it is outside the graph",
       str(seen.get("tri_mul_out")))
    ok(seen.get("tri_mul_in", (None,))[0] is False,
       "tri_mul_in's INPUT does not require grad -- it is outside the graph",
       str(seen.get("tri_mul_in")))
    ok(seen.get("tri_att_start", (None,))[0] is True,
       "tri_att_start's INPUT DOES require grad -- the injection landed before it",
       str(seen.get("tri_att_start")))

    print("\n5. after backward, through the checkpointed path")
    gp = dict(m.confidence_head.named_parameters())
    no_grad_names = [n for n in dead if gp[n].grad is not None]
    ok(not no_grad_names, "not one frozen tensor received a gradient",
       f"{len(no_grad_names)} did")
    got = [n for n in live if gp[n].grad is not None and torch.isfinite(gp[n].grad).all()]
    ok(len(got) == len(live), "every trainable tensor received a FINITE gradient",
       f"{len(got)} of {len(live)}")
    inj = [n for n in live if "pose_inblock" in n]
    nz = [n for n in inj if float(gp[n].grad.abs().max()) > 0]
    ok(len(nz) == 4, "all FOUR injection tensors received a NON-ZERO gradient",
       f"{len(nz)} of 4: {[n.split('.')[0] for n in nz]}")
    # BOTH SITES, NAMED SEPARATELY. "four non-zero" would pass if one site got two tensors and
    # the other got two zeros, which is the failure where one injection is silently inert.
    ok(any("pose_inblock_d" == n.split(".")[0] for n in nz)
       and any("pose_inblock_pae_d" == n.split(".")[0] for n in nz),
       "BOTH the in-block site and the readout site have live gradient")

    print("\n6. the injections are not inert: each term MOVES the logits, one at a time")
    # ONE HEAD, ONE TENSOR AT A TIME. The earlier version built a second head per term and
    # compared against the first head's logits -- which measured the 84-tensor difference between
    # two constructions, not the injection. Toggling a single tensor on a single head is the only
    # comparison here that isolates the thing being claimed.
    m6 = _loaded(m, block_index=3, blocks_per_ckpt=None)
    h6 = m6.confidence_head
    with torch.no_grad():
        for t in ("pose_inblock_d", "pose_inblock_d_wo_onehot", "pose_inblock_pae_d",
                  "pose_inblock_pae_d_wo_onehot"):
            getattr(h6, t).weight.zero_()
        base6 = _pae(m6, inp).clone()
    for t, label in (("pose_inblock_d", "in-block one-hot"),
                     ("pose_inblock_d_wo_onehot", "in-block raw-distance"),
                     ("pose_inblock_pae_d", "readout one-hot"),
                     ("pose_inblock_pae_d_wo_onehot", "readout raw-distance")):
        w = getattr(h6, t).weight
        with torch.no_grad():
            w.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(3))
            moved = _pae(m6, inp)
            d = float((moved - base6).abs().max())
            w.zero_()
        ok(d > 1e-6, f"the {label} term changes the PAE logits", f"max|diff| = {d:.3e}")

    print("\n7. CHECKPOINT ROUND TRIP: what the trainer saves, an evaluator can load")
    # THE EVALUATOR'S GUARD IS SET EQUALITY over the head's parameters against the checkpoint's
    # keys, so this reproduces it exactly rather than approximating it. phase_k_eval installs the
    # pathway from --pose-inblock / --pose-inblock-pae / --pose-inblock-block before loading; a
    # model built WITHOUT that install is the failure this round trip is here to catch.
    sd = {n: p.detach().clone() for n, p in m.confidence_head.named_parameters()
          if p.requires_grad}
    m_eval = _Model(blocks_per_ckpt=None)
    # THE TRAINER'S ORDER, which pose_inblock enforces with a hard stop: lean_readouts REPLACES
    # memory_efficient_forward, so installing it second would throw the injection wrapper away.
    LS.install(m_eval, pae_only=True)
    LS.drop_single_pathway(m_eval)
    pose_inblock.install(m_eval, block_index=3, pae_inject=True, loud=False)
    eval_head = {n for n, p in m_eval.confidence_head.named_parameters()}
    extra = sorted(set(sd) - eval_head)
    ok(not extra, "no saved tensor is absent from an identically-installed eval head", str(extra))
    m_noinstall = _Model(blocks_per_ckpt=None)
    missing = sorted(set(sd) - {n for n, _ in m_noinstall.confidence_head.named_parameters()})
    ok(len(missing) == 4,
       "WITHOUT the install, the eval head is missing exactly the 4 injection tensors",
       str(len(missing)))
    # AND THE NUMBERS AGREE. Key-set equality alone would pass if the install put the injection at
    # a different depth, which is why --pose-inblock-block is forwarded.
    with torch.no_grad():
        m_eval.confidence_head.load_state_dict(m.confidence_head.state_dict())
        p_eval = _pae(m_eval, inp)
    with torch.no_grad():
        p_train = _pae(m, inp)
    d = float((p_eval - p_train).abs().max())
    ok(d == 0.0, "an identically-installed eval head reproduces the PAE logits EXACTLY",
       f"max|diff| = {d:.3e}")
    # THE WRONG DEPTH IS A DIFFERENT OPERATOR, and nothing in the key set can tell.
    #
    # BOTH HEADS ARE MADE VALUE-IDENTICAL FROM m, so install depth is the only difference. This
    # check reads 0.000e+00 on a head whose stack is still at its zero init, because a fresh
    # PairformerBlock is an exact identity on z and an injected delta then reaches the readout
    # unchanged from any depth -- see _Model's docstring. It is only a real check because
    # _Model fills those projections.
    m_b3 = _loaded(m, block_index=3)
    m_b2 = _loaded(m, block_index=2)
    ok({n for n, _ in m_b2.confidence_head.named_parameters()}
       == {n for n, _ in m_b3.confidence_head.named_parameters()},
       "installing at block 2 gives the SAME key set as block 3 -- the guard cannot see depth")
    W = torch.randn(CFG["c_z"], 39, generator=torch.Generator().manual_seed(5)) * 0.5
    with torch.no_grad():
        for mm in (m_b3, m_b2):
            mm.confidence_head.pose_inblock_d.weight.copy_(W)
        sd3 = m_b3.confidence_head.state_dict()
        sd2 = m_b2.confidence_head.state_dict()
        ok(all(torch.equal(sd3[k], sd2[k]) for k in sd3),
           "the two heads are identical in every stored value")
        d_depth = float((_pae(m_b3, inp) - _pae(m_b2, inp)).abs().max())
    ok(d_depth > 1e-6,
       "and the same weight at block 2 gives DIFFERENT logits, so the depth must be forwarded",
       f"max|diff| = {d_depth:.3e}")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
