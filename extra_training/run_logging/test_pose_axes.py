#!/usr/bin/env python
"""phases/pose_axes.py and the compositional pose-injection surface. CPU only, no CUDA.

WHAT IS UNDER TEST, and why each part is here rather than taken on trust:

  1. THE DESUGARING TABLE. Every legacy flag must still mean exactly the axes it used to mean --
     plain --pose-perblock is onehot at zero init untied, --pose-perblock-copy is onehot+dist at
     the pretrained init, --pose-perblock-share is onehot+dist on block 0's own tensors, and
     `orient` joins all of them whenever --frame-direction is on. Reproducing an earlier arm from
     the command line it was launched with depends on this and nothing else.

  2. THE CHECKPOINT KEY SET IS UNCHANGED for every legacy point. The tensor names are a function
     of the TERMS, chosen so that a onehot-only arm still carries pose_perblock_proj and a
     both-terms arm still carries pose_perblock_linear_no_bias_d(_wo_onehot). A name that moved
     would make every existing checkpoint unloadable, and the load guard's message would be about
     tensors rather than about the change that caused it.

  3. THE NEW POINT: blocks carrying onehot AND the raw distance at ZERO init -- the configuration
     the old surface could not express at all, because the forward keyed the both-terms path off
     an init-named flag. Asserted here on three counts: the right tensor names and counts, a
     forward that is EXACTLY equal to the un-injected one (probed, not assumed), and -- the part
     a smoke test would miss -- that it is a DIFFERENT FUNCTION from the one-hot-only arm once the
     weights are non-zero. A feature whose two settings compute the same thing is worse than no
     feature, because the run would be mislabelled.

  4. THE REFUSALS. Every illegal combination must refuse by naming what to pass instead, not
     install something defensible-looking.

  5. THE ARGV. One builder feeds every forwarding path, so a child eval cannot be built from a
     different architecture than its parent -- the failure of commit 7d9d4d10. Checked both ways:
     a legacy args.json still yields the flags that rebuild its architecture, and the flags it
     yields are ones phase_k_eval really declares.

NOT TESTED HERE, AND IT NEEDS A GPU: nothing. Every property above is shape and value algebra on
small tensors. phases/phase_k_perblock_check.py, which exercises the same module against the real
ConfidenceHead, CANNOT run on CPU -- Protenix's fused LayerNorm raises "input must be a CUDA
tensor" -- so the real-head equivalents of 2 and 3 are GPU-only and are not run by this file.

Usage:  python -I run_logging/test_pose_axes.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import torch                                                                 # noqa: E402
import torch.nn as nn                                                        # noqa: E402

import pose_axes as PA                                                       # noqa: E402
import pose_direction                                                        # noqa: E402
import pose_perblock                                                         # noqa: E402
import pose_readout                                                          # noqa: E402
from protenix.model.modules.primitives import LinearNoBias                   # noqa: E402
from protenix.model.utils import one_hot                                     # noqa: E402

N_TOK, C_Z, N_BINS, N_BLOCKS = 9, 8, 6, 4
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ----------------------------------------------------------------------------------------------
# THE HARNESS: the surface the pose modules touch, and nothing else. Modelled on
# test_perblock_ckpt_equiv.py's, which is the proven stand-in for confidence.py on CPU.
# ----------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(C_Z, C_Z)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(C_Z, C_Z, generator=g) * 0.1)
            self.lin.bias.zero_()

    def forward(self, s, z):
        return torch.tanh(self.lin(z)) + z


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(100 + i) for i in range(N_BLOCKS)])
        self.blocks_per_ckpt = None

    def _prep_blocks(self, pair_mask=None, **kw):
        return [b.forward for b in self.blocks]


class Head(nn.Module):
    def __init__(self, with_direction=False):
        super().__init__()
        self.c_z, self.num_bins = C_Z, N_BINS
        _edges = torch.linspace(1.0, 11.0, N_BINS + 1)
        self.lower_bins, self.upper_bins = _edges[:-1].clone(), _edges[1:].clone()
        self.pairformer_stack = Stack()
        self.linear_no_bias_d = LinearNoBias(in_features=N_BINS, out_features=C_Z)
        self.linear_no_bias_d_wo_onehot = LinearNoBias(in_features=1, out_features=C_Z)
        self.pae_ln = nn.LayerNorm(C_Z)
        self.linear_no_bias_pae = LinearNoBias(in_features=C_Z, out_features=N_BINS)
        with torch.no_grad():
            g = torch.Generator().manual_seed(7)
            self.linear_no_bias_d.weight.copy_(torch.randn(C_Z, N_BINS, generator=g) * 0.05)
            self.linear_no_bias_d_wo_onehot.weight.copy_(torch.randn(C_Z, 1, generator=g) * 0.05)
        if with_direction:
            self.frame_dir_proj = LinearNoBias(in_features=3, out_features=C_Z)
            with torch.no_grad():
                self.frame_dir_proj.weight.zero_()   # as pose_direction installs it
            self._pose_direction_installed = True

    def memory_efficient_forward(self, input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        blocks = self.pairformer_stack._prep_blocks(pair_mask=pair_mask)
        z = z_pair
        _d = pose_direction._CALL.get("dir")
        if getattr(self, "_pose_direction_installed", False) and _d is not None:
            z = z + self.frame_dir_proj(_d).to(z.dtype)
        for fn in blocks:
            z = fn(s_trunk, z)
        return self.readout(z)

    def readout(self, z):
        return self.linear_no_bias_pae(self.pae_ln(z))


class Model(nn.Module):
    def __init__(self, with_direction=False):
        super().__init__()
        self.confidence_head = Head(with_direction)


def inputs():
    g = torch.Generator().manual_seed(3)
    s = torch.randn(N_TOK, C_Z, generator=g)
    z = torch.randn(N_TOK, N_TOK, C_Z, generator=g)
    x = torch.randn(N_TOK, 3, generator=g) * 4.0
    return s, z, x


def run(model, with_dir=False):
    """One forward of the stand-in head, with the geometry published as the real path does."""
    s, z, x = inputs()
    head = model.confidence_head
    if with_dir:
        g = torch.Generator().manual_seed(5)
        pose_direction._CALL["dir"] = torch.nn.functional.normalize(
            torch.randn(N_TOK, N_TOK, 3, generator=g), dim=-1)
    try:
        return head.memory_efficient_forward({}, s, z, None, x)
    finally:
        pose_direction._CALL["dir"] = None


def param_names(model, prefix="pose_"):
    return sorted(n for n, _ in model.confidence_head.named_parameters() if n.startswith(prefix))


def main():
    print(__doc__.strip().splitlines()[0])

    # ------------------------------------------------------------------------------------------
    print("\n1  THE DESUGARING TABLE: every legacy flag means exactly what it meant")
    # (legacy flags) -> (block axes, readout axes). Written out in full rather than computed, so
    # a change to the desugaring has to be made here too, deliberately.
    TABLE = [
        ({"pose_perblock": True},
         ("onehot", "zero", "untied"), None),
        ({"pose_perblock": True, "pose_perblock_tied": True},
         ("onehot", "zero", "tied"), None),
        ({"pose_perblock": True, "pose_perblock_copy": True},
         ("onehot,dist", "pretrained", "untied"), None),
        ({"pose_perblock": True, "pose_perblock_share": True},
         ("onehot,dist", "zero", "shared"), None),
        ({"pose_perblock": True, "frame_direction": True},
         ("onehot,orient", "zero", "untied"), None),
        ({"pose_perblock": True, "pose_perblock_copy": True, "frame_direction": True},
         ("onehot,dist,orient", "pretrained", "untied"), None),
        ({"pose_perblock": True, "pose_readout": True},
         ("onehot", "zero", "untied"), ("post", "onehot,dist", "zero")),
        ({"pose_perblock": True, "pose_readout": True, "pose_readout_prenorm": True},
         ("onehot", "zero", "untied"), ("pre", "onehot,dist", "zero")),
        ({"pose_perblock": True, "pose_readout": True, "pose_readout_prenorm": True,
          "pose_readout_post_zero": True},
         ("onehot", "zero", "untied"), ("pre,post", "onehot,dist", "zero")),
        ({"pose_perblock": True, "pose_readout": True, "pose_readout_copy_distance": True},
         ("onehot", "zero", "untied"), ("post", "onehot,dist", "pretrained")),
        ({"pose_perblock": True, "pose_readout": True, "frame_direction": True},
         ("onehot,orient", "zero", "untied"), ("post", "onehot,dist,orient", "zero")),
        # AND THE IN-BLOCK PATHWAY STILL PRODUCES THE READOUT's GEOMETRY, so the readout alone is
        # legal with it and the block axes stay off.
        ({"pose_inblock": True, "pose_readout": True},
         None, ("post", "onehot,dist", "zero")),
    ]
    for legacy, want_blk, want_rd in TABLE:
        rec = PA.record(*PA.resolve(legacy))
        got_blk = ((rec["pose_block_terms"], rec["pose_block_init"], rec["pose_block_weights"])
                   if rec["pose_block_terms"] else None)
        got_rd = ((rec["pose_readout_sites"], rec["pose_readout_terms"], rec["pose_readout_init"])
                  if rec["pose_readout_sites"] else None)
        name = " ".join("--" + k.replace("_", "-") for k in sorted(legacy))
        check(f"{name}", got_blk == want_blk and got_rd == want_rd,
              f"blocks {got_blk} readout {got_rd}")

    # AND THE LEGACY KEYS COME BACK OUT, so args.json keeps describing the arm the old way too.
    rec = PA.record(*PA.resolve({"pose_perblock": True, "pose_perblock_copy": True}))
    check("a copy arm still records pose_perblock_copy true",
          rec["pose_perblock"] and rec["pose_perblock_copy"]
          and not rec["pose_perblock_share"] and not rec["pose_perblock_tied"])
    # THE ONE PLACE THE LEGACY KEYS ARE WEAKER THAN THE AXES, said out loud: the new point records
    # pose_perblock true and copy FALSE, which is true and is not enough to rebuild it. That is
    # why the axis keys are in args.json and why argv() reads them.
    rec2 = PA.record(*PA.resolve({"pose_block_terms": "onehot,dist"}))
    check("the new onehot+dist-at-zero point records copy FALSE (legacy cannot express it)",
          rec2["pose_perblock"] and not rec2["pose_perblock_copy"]
          and rec2["pose_block_terms"] == "onehot,dist" and rec2["pose_block_init"] == "zero")
    check("and the two are distinguishable only by the axis keys",
          rec["pose_block_terms"] == rec2["pose_block_terms"]
          and rec["pose_block_init"] != rec2["pose_block_init"])

    # ------------------------------------------------------------------------------------------
    print("\n2  THE CHECKPOINT KEY SET IS UNCHANGED FOR EVERY LEGACY POINT")
    # The names are hardcoded, because the claim IS the exact historical key set: computing them
    # from the module would make this test agree with whatever the code does.
    LEGACY_NAMES = {
        ("onehot", "zero", "untied"): ["pose_perblock_proj.%d.weight" % i for i in range(3)],
        ("onehot", "zero", "tied"): ["pose_perblock_proj.0.weight"],
        ("onehot,dist", "pretrained", "untied"):
            ["pose_perblock_linear_no_bias_d.%d.weight" % i for i in range(3)]
            + ["pose_perblock_linear_no_bias_d_wo_onehot.%d.weight" % i for i in range(3)],
        ("onehot,dist", "zero", "shared"): [],
    }
    for (terms, init, weights), want in LEGACY_NAMES.items():
        m = Model()
        pose_perblock.install(m, terms=tuple(terms.split(",")), init=init, weights=weights,
                              loud=False)
        got = param_names(m)
        check(f"terms {terms} / {init} / {weights} -> the historical tensor names",
              got == sorted(want), f"{got}")

    m = Model(with_direction=True)
    pose_perblock.install(m, terms=("onehot", "dist", "orient"), init="pretrained", loud=False)
    check("the copy+orientation arm's names are unchanged too",
          param_names(m) == sorted(
              ["pose_perblock_linear_no_bias_d.%d.weight" % i for i in range(3)]
              + ["pose_perblock_linear_no_bias_d_wo_onehot.%d.weight" % i for i in range(3)]
              + ["pose_perblock_frame_dir.%d.weight" % i for i in range(3)]),
          f"{param_names(m)}")
    check("and the orientation copies are copies of frame_dir_proj, not independently zeroed",
          all(torch.equal(c.weight, m.confidence_head.frame_dir_proj.weight)
              for c in m.confidence_head.pose_perblock_frame_dir))

    # ------------------------------------------------------------------------------------------
    print("\n3  THE NEW POINT: blocks carry onehot AND the raw distance, at ZERO init")
    # ONE INSTANCE, MEASURED BEFORE AND AFTER INSTALL. Two separately constructed Models do NOT
    # carry the same weights even under one torch seed: LinearNoBias initialises through
    # trunc_normal_init_, which samples from NUMPY's global RNG. Comparing two instances was
    # comparing two different heads and reported max|d| 4.7 against an injection that is exactly
    # zero -- the same trap test_pose_mlp.py and phase_k_perblock_check.py both record.
    m = Model()
    want = run(m).clone()
    info = pose_perblock.install(m, terms=("onehot", "dist"), init="zero", loud=False)
    check("tensor names and counts",
          param_names(m) == sorted(
              ["pose_perblock_linear_no_bias_d.%d.weight" % i for i in range(3)]
              + ["pose_perblock_linear_no_bias_d_wo_onehot.%d.weight" % i for i in range(3)]),
          f"{len(param_names(m))} tensor(s): {param_names(m)}")
    check("the reported parameter count is the sum of both terms at every depth",
          info["params"] == 3 * (N_BINS * C_Z + 1 * C_Z)
          and info["n_tensors"] == 6 and info["terms"] == ("onehot", "dist"),
          f"{info['params']} params, {info['n_tensors']} tensors, terms {info['terms']}")
    # EXACTLY ZERO, PROBED THROUGH THE MODULES, as install() itself does -- and then asserted on
    # the FORWARD, which is the property the zero init exists for.
    worst = max(float(p.abs().max()) for n, p in m.confidence_head.named_parameters()
                if n.startswith("pose_"))
    check("every new tensor is EXACTLY zero", worst == 0.0, f"max|w| {worst:.3e}")
    got = run(m)
    check("the forward is BIT-identical to the un-injected head",
          bool(torch.equal(want, got)), f"max|d| {float((got - want).abs().max()):.3e}")

    # THE FEATURE CHANGES SOMETHING. Same weights in the one-hot arm, non-zero: the both-terms
    # arm must differ, and it must differ BY THE DISTANCE TERM rather than by anything else, so
    # the one-hot matrices are set identically in both.
    m_one, m_both = Model(), Model()
    m_both.load_state_dict(m_one.state_dict())
    base_one = run(m_one).clone()
    pose_perblock.install(m_one, terms=("onehot",), loud=False)
    pose_perblock.install(m_both, terms=("onehot", "dist"), loud=False)
    with torch.no_grad():
        g = torch.Generator().manual_seed(21)
        for i in range(3):
            w = torch.randn(C_Z, N_BINS, generator=g) * 0.1
            m_one.confidence_head.pose_perblock_proj[i].weight.copy_(w)
            m_both.confidence_head.pose_perblock_linear_no_bias_d[i].weight.copy_(w)
            m_both.confidence_head.pose_perblock_linear_no_bias_d_wo_onehot[i].weight.copy_(
                torch.randn(C_Z, 1, generator=g) * 0.1)
    one, both = run(m_one), run(m_both)
    check("with non-zero weights the both-terms arm is a DIFFERENT FUNCTION from onehot-only",
          not torch.equal(one, both), f"max|d| {float((both - one).abs().max()):.3e}")
    check("and the difference is not vacuous: the onehot arm itself moved off the baseline",
          not torch.equal(one, base_one), f"max|d| {float((one - base_one).abs().max()):.3e}")
    # AND THE RAW-DISTANCE TERM IS GRADIENT-LIVE AT ZERO INIT, which is what makes the new point
    # trainable at all: at W = 0 the output is zero but dL/dW = g (x) x^T with x the distance.
    m_z = Model()
    pose_perblock.install(m_z, terms=("onehot", "dist"), loud=False)
    run(m_z).square().mean().backward()
    gmin = min(float(p.grad.abs().max()) for n, p in m_z.confidence_head.named_parameters()
               if n.startswith("pose_") and p.grad is not None)
    ngrad = sum(1 for n, p in m_z.confidence_head.named_parameters()
                if n.startswith("pose_") and p.grad is not None)
    check("every new tensor gets a NON-ZERO gradient at zero init (not gradient-dead)",
          ngrad == 6 and gmin > 0, f"{ngrad} with grad, smallest max|g| {gmin:.3e}")

    # THE SAME POINT WITH ORIENTATION AND BOTH READOUT SITES -- the arm this refactor is for.
    m_all = Model(with_direction=True)
    want_all = run(m_all, with_dir=True).clone()      # the SAME instance, before install
    pose_perblock.install(m_all, terms=("onehot", "dist", "orient"), init="zero", loud=False)
    pose_readout.install(m_all, sites=("pre", "post"), terms=("onehot", "dist", "orient"),
                         init="zero", loud=False)
    names_all = param_names(m_all)
    check("the five-site / three-term / zero-init arm: 15 new tensors",
          len(names_all) == 15, f"{len(names_all)}: {names_all}")
    got_all = run(m_all, with_dir=True)
    check("and it is BIT-identical to the same head without any of it",
          bool(torch.equal(want_all, got_all)),
          f"max|d| {float((got_all - want_all).abs().max()):.3e}")
    # AND EVERY ONE OF THE 15 IS GRADIENT-LIVE, which is what makes the arm trainable from zero.
    got_all.square().mean().backward()
    _live = [n for n, pp in m_all.confidence_head.named_parameters()
             if n.startswith("pose_") and pp.grad is not None and float(pp.grad.abs().max()) > 0]
    check("all 15 receive a non-zero gradient at zero init", len(_live) == 15,
          f"{len(_live)} of 15 live")

    # A BUNDLE CANNOT CROSS A CHANGE OF TERMS, which is the property the resume guard's new
    # fields exist to NAME rather than to discover as an absent tensor. Asserted at the tensor
    # level here: the onehot-only arm's state dict is not loadable into the both-terms arm, in
    # either direction. (The trainer's own refusal -- the missing-trainable-tensor check in the
    # resume block -- cannot run on CPU: it needs the real ConfidenceHead, whose fused LayerNorm
    # raises "input must be a CUDA tensor".)
    sd_one = {k: v for k, v in m_one.state_dict().items() if "pose_" in k}
    sd_both = {k: v for k, v in m_both.state_dict().items() if "pose_" in k}
    check("the two arms' key sets are disjoint, so a bundle cannot silently cross them",
          set(sd_one) != set(sd_both) and not (set(sd_one) & set(sd_both)),
          f"{sorted(k.split('.')[-3] for k in sd_one)[:1]} vs "
          f"{sorted(set(k.split('.')[-3] for k in sd_both))}")
    try:
        m_both.load_state_dict(m_one.state_dict(), strict=True)
        check("and a strict load across the change is refused", False, "it loaded")
    except RuntimeError as e:
        check("and a strict load across the change is refused",
              "pose_perblock" in str(e), str(e)[:80])

    # ------------------------------------------------------------------------------------------
    print("\n4  THE READOUT AXES")
    m = Model()
    m.confidence_head._pose_perblock_installed = True
    pose_readout.install(m, sites=("post",), terms=("onehot",), loud=False)
    check("a one-term readout site builds ONE tensor, not a zero one it never trains",
          param_names(m) == ["pose_readout_d.weight"], f"{param_names(m)}")
    check("and the hook adds only that term",
          not hasattr(m.confidence_head, "pose_readout_d_wo_onehot"))
    m2 = Model()
    m2.confidence_head._pose_perblock_installed = True
    pose_readout.install(m2, sites=("pre", "post"), terms=("onehot", "dist"), loud=False)
    check("two sites keep the historical prefixes: pre is unprefixed, post is post_",
          param_names(m2) == sorted(["pose_readout_d.weight", "pose_readout_d_wo_onehot.weight",
                                     "pose_readout_post_d.weight",
                                     "pose_readout_post_d_wo_onehot.weight"]),
          f"{param_names(m2)}")
    check("and the legacy attributes still describe the sites",
          m2.confidence_head._pose_readout_prenorm is True
          and m2.confidence_head._pose_readout_post_zero is True)

    # ------------------------------------------------------------------------------------------
    print("\n5  THE REFUSALS, each naming what to pass instead")
    REFUSE = [
        ("both surfaces at once", {"pose_perblock": True, "pose_block_terms": "onehot"},
         "--pose-block-terms"),
        ("orientation without --frame-direction", {"pose_block_terms": "onehot,orient"},
         "--frame-direction"),
        ("an unknown term", {"pose_block_terms": "onehot,distance"}, "legal terms"),
        ("an empty term list", {"pose_block_terms": ",,"}, "is empty"),
        ("a repeated term", {"pose_block_terms": "onehot,onehot"}, "more than once"),
        ("shared with one distance term",
         {"pose_block_terms": "onehot", "pose_block_weights": "shared"}, "BOTH terms"),
        ("shared with a pretrained init",
         {"pose_block_terms": "onehot,dist", "pose_block_weights": "shared",
          "pose_block_init": "pretrained"}, "contradiction"),
        ("an init with no terms", {"pose_block_init": "pretrained"}, "not WHETHER"),
        ("a weights mode with no terms", {"pose_block_weights": "tied"}, "not WHETHER"),
        ("a legacy mode flag without --pose-perblock", {"pose_perblock_copy": True},
         "without --pose-perblock"),
        ("--pose-readout-post-zero without --pose-readout-prenorm",
         {"pose_perblock": True, "pose_readout": True, "pose_readout_post_zero": True},
         "SECOND site"),
        ("readout sites with no terms",
         {"pose_block_terms": "onehot", "pose_readout_sites": "pre"}, "--pose-readout-terms"),
        ("readout terms with no sites",
         {"pose_block_terms": "onehot", "pose_readout_terms": "onehot"}, "--pose-readout-sites"),
        ("a readout with no geometry producer",
         {"pose_readout_sites": "pre", "pose_readout_terms": "onehot"}, "silent no-op"),
        ("an unknown readout site",
         {"pose_block_terms": "onehot", "pose_readout_sites": "middle",
          "pose_readout_terms": "onehot"}, "unknown site"),
        ("a pretrained readout with no distance term",
         {"pose_block_terms": "onehot", "frame_direction": True,
          "pose_readout_sites": "post", "pose_readout_terms": "orient",
          "pose_readout_init": "pretrained"}, "no pretrained orientation matrix"),
    ]
    for name, d, want_text in REFUSE:
        try:
            PA.resolve(d)
            check(f"refuses {name}", False, "it resolved")
        except PA.PoseConfigError as e:
            check(f"refuses {name}", want_text in str(e), str(e)[:90])

    # AND install() REFUSES THE SAME THINGS, because phase_k_eval, the checkers and these tests
    # call it directly rather than through the CLI.
    try:
        pose_perblock.install(Model(), terms=("onehot", "orient"), loud=False)
        check("install() refuses orientation with no geometry", False, "it installed")
    except PA.PoseConfigError as e:
        check("install() refuses orientation with no geometry", "--frame-direction" in str(e))
    try:
        pose_perblock.install(Model(), terms=("onehot",), weights="shared", loud=False)
        check("install() refuses a one-term shared arm", False, "it installed")
    except PA.PoseConfigError as e:
        check("install() refuses a one-term shared arm", "BOTH terms" in str(e))

    # ------------------------------------------------------------------------------------------
    print("\n6  THE ARGV: one builder, and it rebuilds what the run trained")
    import pathway_flags
    defined = PA.eval_long_opts()
    check("phase_k_eval declares every axis flag the eval target emits",
          defined is not None
          and {"--pose-block-terms", "--pose-block-weights", "--pose-readout-sites",
               "--pose-readout-terms"} <= defined)
    check("and pathway_flags reads the same vocabulary", pathway_flags.eval_flags() == defined)
    for legacy, want_blk, want_rd in TABLE:
        rec = dict(legacy, **PA.record(*PA.resolve(legacy)))
        flags, notes = PA.argv(rec, target="eval", defined=defined)
        check(f"eval argv has no wiring note for {sorted(legacy)}", not notes, f"{notes}")
        # ROUND TRIP: the emitted flags, parsed back, must resolve to the SAME axes.
        back = {}
        for i in range(0, len(flags), 2):
            back[flags[i][2:].replace("-", "_")] = flags[i + 1]
        back["frame_direction"] = legacy.get("frame_direction", False)
        back["pose_inblock"] = legacy.get("pose_inblock", False)
        blk2, rd2 = PA.resolve(back)
        blk1, rd1 = PA.resolve(legacy)
        same = ((blk1 is None) == (blk2 is None)
                and (blk1 is None or (blk1["terms"] == blk2["terms"]
                                      and blk1["weights"] == blk2["weights"]))
                and (rd1 is None) == (rd2 is None)
                and (rd1 is None or (rd1["sites"] == rd2["sites"]
                                     and rd1["terms"] == rd2["terms"])))
        check(f"and it round-trips to the same architecture: {' '.join(flags) or '(none)'}", same)
    # THE INITS GO TO THE TRAINER's CHILD AND NOT TO THE EVALUATOR, which is the one asymmetry.
    rec = PA.record(*PA.resolve({"pose_perblock": True, "pose_perblock_copy": True}))
    tr = PA.argv(rec, target="train")[0]
    ev = PA.argv(rec, target="eval")[0]
    check("the --loss-only child gets the init; the evaluator does not",
          "--pose-block-init" in tr and "--pose-block-init" not in ev, f"train {tr} / eval {ev}")
    # AN OLD args.json WITH NO AXIS KEYS AT ALL still yields the right flags.
    old = {"pose_perblock": True, "pose_perblock_copy": True, "frame_direction": True,
           "pose_readout": True, "pose_readout_prenorm": True}
    check("an args.json predating the axes is backfilled from its legacy keys",
          PA.argv(old, target="eval")[0]
          == ["--pose-block-terms", "onehot,dist,orient",
              "--pose-readout-sites", "pre", "--pose-readout-terms", "onehot,dist,orient"],
          f"{PA.argv(old, target='eval')[0]}")
    check("and a record with NO pose flags emits nothing", PA.argv({}, target="eval")[0] == [])

    # ------------------------------------------------------------------------------------------
    print("\n7  A LEGACY RESUME MUST NOT READ AS DRIFT (ground truth: the bundles on disk)")
    # The resume guard now compares four axis fields no existing bundle records. Absence there is
    # evidence of the LEGACY point in the space, so the guard backfills them from the bundle's own
    # legacy keys -- and if that backfill were wrong, every pose arm's resume would report drift on
    # a setting that did not change, in the loudest severity group. Checked against real bundles
    # rather than hand-built dicts, because the bundles are what a resume will actually read.
    import glob
    import json
    GUARDED = ("pose_block_terms", "pose_block_weights", "pose_readout_sites",
               "pose_readout_terms")
    src = open(os.path.join(ROOT, "phases", "phase_k_train.py")).read()
    for k in GUARDED:
        want = f'"{k}": a.{k},'
        check(f"{k} is COMPARED and SAVED by the resume guard", src.count(want) == 2,
              f"{src.count(want)} occurrence(s) of {want!r} (want 2: compared and saved)")
    # AND ITS MIGRATION DEFAULT IS THE NO-INJECTION VALUE, so a bundle from a run with no pose
    # flags at all -- which backfill() correctly declines to invent axes for -- compares equal.
    for k in GUARDED:
        needle = f'"{k}": "{PA.EMPTY[k]}"'
        check(f"{k}'s migration default is the no-injection value", needle in src,
              f"looked for {needle} in _PRE_CROP")
    bundles = sorted(glob.glob(os.path.join(ROOT, "training/runs/*/resume_ep*.pt")))[-8:]
    check("there are real bundles to check", len(bundles) > 0, f"{len(bundles)}")
    import torch as _t
    drifted = []
    for b in bundles:
        g = (_t.load(b, map_location="cpu", weights_only=False).get("guard") or {})
        try:
            aj = json.load(open(os.path.join(os.path.dirname(b), "args.json")))
        except (OSError, ValueError):
            aj = {}
        if not aj:
            continue
        # THE GUARD's OWN CHAIN: bundle, then args.json, then the migration default.
        g = dict(g, **PA.backfill(g))
        aj_bf = dict(aj, **PA.backfill(aj))
        mine = PA.record(*PA.resolve(aj))
        for k in GUARDED:
            was = g.get(k, aj_bf.get(k, PA.EMPTY[k]))
            if was != mine[k]:
                drifted.append((os.path.relpath(b, ROOT), k, was, mine[k]))
    check("no real bundle reads as drift against its own run's flags", not drifted,
          f"{len(bundles)} bundle(s) checked" if not drifted else f"{drifted[:3]}")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
