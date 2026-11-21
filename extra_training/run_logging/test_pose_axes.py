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

  6. THE FOUR DEFECTS AN EXTERNAL REVIEW OF 5caf2961 FOUND, each as the thing that would have
     caught it: that the resume guard now SEES a zero-vs-pretrained init difference (it is the
     whole architecture for a projection outside --train-only, since a bundle holds the trainable
     tensors alone); that a legacy post-LN readout resumes without false drift, which needs the
     bundle's legacy fields MERGED with its args.json before the axes are derived, because the
     bundle records pose_readout_prenorm but not pose_readout; that both child-argv builders --
     run_logging/valpae_flags.py and run_logging/eval_fawgtu_bundles.py's --loss-only stage --
     emit a command the child resolver accepts and rebuilds the same architecture from; and that
     phases/phase_k_perblock_check.py's fake head carries every attribute install() reads.

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
    # THE INITS ARE IN THIS TUPLE NOW. They were left out of the guard on the argument that a
    # resume overwrites every trained tensor, which holds only for a tensor the bundle CARRIES --
    # and a bundle carries the trainable ones alone. Section 8 is the case that proves it.
    GUARDED = ("pose_block_terms", "pose_block_init", "pose_block_weights",
               "pose_readout_sites", "pose_readout_terms", "pose_readout_init")
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
        # THE GUARD's OWN CHAIN: bundle, then args.json, then the migration default -- and the
        # backfill is fed BOTH records, bundle first, exactly as the guard now does it. Backfilling
        # the bundle alone was finding 2: its guard record carries pose_readout_prenorm and
        # pose_readout_post_zero but not pose_readout itself, so a legacy POST-LN arm derived no
        # readout site at all.
        g = dict(g, **PA.backfill(dict(aj, **g)))
        aj_bf = dict(aj, **PA.backfill(aj))
        mine = PA.record(*PA.resolve(aj))
        for k in GUARDED:
            was = g.get(k, aj_bf.get(k, PA.EMPTY[k]))
            if was != mine[k]:
                drifted.append((os.path.relpath(b, ROOT), k, was, mine[k]))
    check("no real bundle reads as drift against its own run's flags", not drifted,
          f"{len(bundles)} bundle(s) checked" if not drifted else f"{drifted[:3]}")

    # ------------------------------------------------------------------------------------------
    print("\n8  THE FOUR DEFECTS FOUND IN REVIEW OF 5caf2961")
    import ast
    import json
    import re
    import shlex
    import subprocess
    import tempfile

    TRAIN_SRC = os.path.join(ROOT, "phases", "phase_k_train.py")
    tsrc = open(TRAIN_SRC).read()

    def guard_dicts():
        """(compared, saved) key sets, read out of the trainer the way test_guard_fields.py does."""
        big = []
        for n in ast.walk(ast.parse(tsrc)):
            if isinstance(n, ast.Dict):
                ks = {k.value for k in n.keys
                      if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if len(ks) >= 10:
                    big.append(ks)
        saved = next(ks for ks in big if "drift_history" in ks)
        compared = next(ks for ks in big if "lr_decay_shape" in ks and "drift_history" not in ks)
        return compared, saved

    COMPARED, SAVED = guard_dicts()

    # 8a  AN INIT CHANGE IS VISIBLE TO THE RESUME GUARD.
    #
    # THE DEFECT: the guard compared the terms and the weights mode and deliberately not the
    # inits, on the reasoning that a resume overwrites every trained tensor's value. A bundle
    # saves the TRAINABLE tensors only. So for `--pose-block-terms onehot --train-only pae_ln`,
    # moving --pose-block-init from zero to pretrained left every compared field identical
    # (pose_perblock_copy is False on both sides: legacy's copy mode meant BOTH distance terms),
    # left the trainable tensor NAMES identical, and rebuilt the FROZEN projection with different
    # values -- which, being frozen, are the values the whole run computes with.
    A = {"pose_block_terms": "onehot", "pose_block_init": "zero"}
    B = {"pose_block_terms": "onehot", "pose_block_init": "pretrained"}
    ra, rb = PA.record(*PA.resolve(A)), PA.record(*PA.resolve(B))
    check("zero vs pretrained differ in NO field the legacy surface records",
          not [k for k in ra if ra[k] != rb[k] and k in PA.LEGACY_KEYS],
          "which is why the guard needs the axis key")
    seen = sorted(k for k in ra if ra[k] != rb[k] and k in COMPARED)
    check("the resume guard COMPARES the field that differs", seen == ["pose_block_init"],
          f"compared fields that differ: {seen}")
    for k in ("pose_block_init", "pose_readout_init"):
        check(f"{k} is compared AND saved AND defaulted",
              k in COMPARED and k in SAVED and f'"{k}": "{PA.EMPTY[k]}"' in tsrc,
              f"compared={k in COMPARED} saved={k in SAVED}")
    check("and both inits are in the loudest drift group, like the axes they belong to",
          re.search(r'"pose_block_init",\s*"pose_readout_init"', tsrc) is not None)
    # THE ARCHITECTURE REALLY DOES DIFFER, so the field is not a formality: same tensor names,
    # different tensors, and --train-only pae_ln carries neither of them in the bundle.
    mz, mp = Model(), Model()
    mp.load_state_dict(mz.state_dict())
    pose_perblock.install(mz, terms=("onehot",), init="zero", loud=False)
    pose_perblock.install(mp, terms=("onehot",), init="pretrained", loud=False)
    check("the two inits build the SAME tensor names", param_names(mz) == param_names(mp),
          f"{param_names(mz)}")
    zw = [p for n, p in mz.confidence_head.named_parameters() if n.startswith("pose_")]
    pw = [p for n, p in mp.confidence_head.named_parameters() if n.startswith("pose_")]
    check("and DIFFERENT values, so a resume across the change is a different head",
          all(float(z.abs().max()) == 0.0 for z in zw)
          and all(torch.equal(p, mp.confidence_head.linear_no_bias_d.weight) for p in pw),
          f"max|zero| {max(float(z.abs().max()) for z in zw):.3e}, "
          f"pretrained copies linear_no_bias_d")
    kept = [n for n, _ in mz.confidence_head.named_parameters() if re.search("pae_ln", n)]
    check("--train-only pae_ln leaves the projection OUT of the bundle, so the init is final",
          kept and not any(n.startswith("pose_") for n in kept), f"bundle would hold {kept}")
    # AND THE FORWARD DIFFERS, which is the whole point: the frozen tensor is in the graph.
    check("the two heads are DIFFERENT FUNCTIONS at step 0",
          not torch.equal(run(mz), run(mp)),
          f"max|d| {float((run(mp) - run(mz)).abs().max()):.3e}")

    # 8b  A LEGACY POST-LN READOUT RESUMES WITHOUT DRIFT.
    #
    # THE DEFECT: the bundle's guard record carries pose_readout_prenorm and
    # pose_readout_post_zero but NOT pose_readout, so for --pose-perblock --pose-readout (the
    # plain post-LN arm) every readout key the bundle has is False. Backfilling the bundle ALONE
    # therefore derived "no readout at all" and that inferred empty value won the comparison,
    # reporting pose_readout_sites and pose_readout_terms as drift on an ordinary resume --
    # refused outright under --resume-strict.
    LAUNCH = {"pose_perblock": True, "pose_readout": True}
    mine = PA.record(*PA.resolve(LAUNCH))
    aj = {k: dict(LAUNCH, **mine)[k] for k in PA.LEGACY_KEYS}        # a pre-axes args.json
    # exactly the legacy keys the saved guard dict carries, and no others
    gkeys = [k for k in PA.LEGACY_KEYS if k in SAVED]
    bundle_g = {k: dict(LAUNCH, **mine)[k] for k in gkeys}
    check("the bundle's guard record really does omit pose_readout",
          "pose_readout" not in bundle_g and "pose_readout_prenorm" in bundle_g,
          f"it carries {sorted(bundle_g)}")
    check("backfilling the bundle ALONE still derives no readout site (the defect's cause)",
          dict(bundle_g, **PA.backfill(bundle_g)).get("pose_readout_sites") == "",
          "unchanged on purpose: the bundle cannot know, which is why the merge exists")
    merged = dict(bundle_g, **PA.backfill(dict(aj, **bundle_g)))
    drift = [(k, merged.get(k, PA.EMPTY[k]), mine[k])
             for k in GUARDED if merged.get(k, PA.EMPTY[k]) != mine[k]]
    check("merged with the run's own args.json it derives the post-LN site and NO drift",
          merged.get("pose_readout_sites") == "post" and not drift, f"{drift}")
    check("and the trainer's guard feeds backfill both records, bundle first",
          "_PAX.backfill(dict(_fallback or {}, **_g))" in tsrc)
    # THE MERGE MUST NOT INVENT ANYTHING for a bundle that records the axes itself: a newer
    # args.json beside an older bundle has to leave the per-field fallback to do the comparing.
    newer = dict(bundle_g, pose_block_terms="onehot", pose_block_weights="untied")
    check("a bundle that already carries axis keys is left alone by the merge",
          PA.backfill(dict(aj, **newer)) == {})

    # 8c  BOTH CHILD-ARGV BUILDERS PRODUCE A COMMAND THE CHILD RESOLVER ACCEPTS.
    def as_namespace(toks):
        """The child's namespace, as argparse would see this flag list."""
        d, i = {}, 0
        while i < len(toks):
            t = toks[i]
            if not t.startswith("--"):
                i += 1
                continue
            k = t[2:].replace("-", "_")
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                d[k], i = toks[i + 1], i + 2
            else:
                d[k], i = True, i + 1
        return d

    # valpae_flags.py: run it for real, on a legacy in-block + readout args.json.
    CASES = {
        # THE REFUSAL CASE: the readout site reads the geometry --pose-inblock publishes, so the
        # flags without it are not merely incomplete, they are unresolvable.
        "legacy pose_inblock + pose_readout": {
            "pose_inblock": True, "pose_inblock_pae": True, "pose_inblock_block": 2,
            "pose_readout": True, "pae_only": True, "nsample": 5, "alpha_pae": 1.0,
            "train_only": "pae_ln,linear_no_bias_pae"},
        # THE SILENT CASE: a frozen pretrained copy arm, where a missing init flag builds the
        # same tensor names with different values.
        "legacy copy arm, projections frozen": {
            "pose_perblock": True, "pose_perblock_copy": True, "frame_direction": True,
            "pose_readout": True, "pose_readout_copy_distance": True,
            "nsample": 5, "alpha_pae": 1.0, "train_only": "pae_ln"},
    }
    with tempfile.TemporaryDirectory() as td:
        for _i, (label, args_json) in enumerate(CASES.items()):
            rid = f"case{_i}"          # not hash(label): PYTHONHASHSEED makes that per-process
            d = os.path.join(td, "training", "runs", rid)
            os.makedirs(d)
            json.dump(args_json, open(os.path.join(d, "args.json"), "w"))
            rc = subprocess.run([sys.executable, "-I",
                                 os.path.join(ROOT, "run_logging", "valpae_flags.py"), rid],
                                cwd=td, capture_output=True, text=True)
            out = os.path.join(d, "valpae_flags.txt")
            toks = shlex.split(open(out).read()) if rc.returncode == 0 else []
            ns = as_namespace(toks)
            ns["frame_direction"] = bool(args_json.get("frame_direction"))
            want = PA.record(*PA.resolve(args_json))
            try:
                got = PA.record(*PA.resolve(ns))
                ok = all(got[k] == want[k] for k in PA.AXIS_KEYS)
                why = f"{[ (k, got[k], want[k]) for k in PA.AXIS_KEYS if got[k] != want[k] ]}"
            except PA.PoseConfigError as e:
                ok, why = False, "CHILD REFUSES: " + str(e)[:70]
            check(f"valpae_flags: the --loss-only child accepts and rebuilds {label}",
                  rc.returncode == 0 and ok, why)
            check(f"valpae_flags: and it forwards --pose-inblock for {label}",
                  bool(args_json.get("pose_inblock")) == ("--pose-inblock" in toks),
                  f"{' '.join(toks)}")

    # eval_fawgtu_bundles.py: the valpae stage is a TRAINER child, so it must not reuse the
    # evaluator's flags -- the eval target omits the inits because every head tensor comes from
    # the checkpoint, which is false for a bundle that holds the trainable tensors only.
    FSRC = open(os.path.join(ROOT, "run_logging", "eval_fawgtu_bundles.py")).read()
    ftree = ast.parse(FSRC)
    # OVER THE PARSED NAMES, not the text: a comment in stage_valpae that says "POSE_TRAIN, not
    # POSE" is not a reference to POSE, and a grep-shaped assertion failed on exactly that.
    bodies = {n.name: {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
              for n in ast.walk(ftree) if isinstance(n, ast.FunctionDef)}
    check("eval_fawgtu_bundles builds a train-target list as well as an eval one",
          'target="train"' in FSRC and 'target="eval"' in FSRC)
    for fn, want in (("stage_valpae", "POSE_TRAIN"), ("stage_train", "POSE"),
                     ("stage_val", "POSE")):
        names = {n for n in bodies.get(fn, set()) if n in ("POSE", "POSE_TRAIN")}
        check(f"{fn} passes {want} to its child", names == {want},
              f"it references {sorted(names) or 'nothing'}")
    # AND THE FLAGS IT WOULD BUILD REALLY DO REBUILD A PRETRAINED ARM, checked against a real
    # args.json on disk rather than a hand-made one.
    real = os.path.join(ROOT, "training", "runs", "fobq26", "args.json")
    if os.path.exists(real):
        RA = json.load(open(real))
        tr, tnotes = PA.argv(RA, target="train",
                             defined=PA.eval_long_opts(TRAIN_SRC))
        ev = PA.argv(RA, target="eval", defined=PA.eval_long_opts())[0]
        ns = as_namespace(tr)
        ns["frame_direction"] = bool(RA.get("frame_direction"))
        got = PA.record(*PA.resolve(ns))
        want = dict(RA, **PA.backfill(RA))
        check("the train-target argv rebuilds fobq26's pretrained inits; the eval one omits them",
              not tnotes
              and got["pose_block_init"] == want["pose_block_init"] == "pretrained"
              and got["pose_readout_init"] == want["pose_readout_init"] == "pretrained"
              and "--pose-block-init" not in ev,
              f"notes {tnotes}; train {' '.join(tr)}")

    # AND EVERY TRAINER-CHILD BUILDER FORWARDS THE IN-BLOCK PATHWAY, which is not one of the axes
    # and so is not pose_axes.argv's to emit. Three files build a phase_k_train --loss-only argv
    # from an args.json or a bundle guard; two of them had never forwarded it, and the third --
    # recover_val_pae.py -- was found by asking this question of all three rather than of the two
    # the review named.
    for f in ("valpae_flags.py", "eval_fawgtu_bundles.py", "recover_val_pae.py"):
        fs = open(os.path.join(ROOT, "run_logging", f)).read()
        check(f"{f} builds a train-target argv AND forwards --pose-inblock",
              'target="train"' in fs and '"--pose-inblock"' in fs
              and '"--pose-inblock-block"' in fs and '"--pose-inblock-pae"' in fs,
              "a child without it refuses the bundle, or refuses the command outright when a "
              "legacy readout site rides on it")

    # 8d  THE PER-BLOCK CHECKER's FIXTURE SATISFIES WHAT THE INSTALLER READS.
    #
    # THE DEFECT: install() builds its term table over BOTH distance sources before it looks at
    # which terms are enabled (pose_perblock.py:208-211), so it reads
    # linear_no_bias_d_wo_onehot.weight even for terms=("onehot",) -- and the fake head defined
    # only linear_no_bias_d. The first install raised AttributeError, after the CUDA-dependent
    # baseline forward. THE FIXTURE was the thing that was wrong: every real head with
    # linear_no_bias_d has the wo_onehot layer beside it (confidence.py:286-305), shared mode
    # re-applies exactly that pair and the pretrained init copies from it, so a head without it
    # cannot represent two of the three modes the checker exists to check.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_pbc", os.path.join(ROOT, "phases", "phase_k_perblock_check.py"))
    _pbc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pbc)
    fm = _pbc.FakeModel(n_blocks=4, c_z=16, c_s=8, num_bins=39)
    # EVERY ATTRIBUTE install() READS, not just the one that broke: a list read out of the
    # installer's own source, so the next attribute added there fails HERE rather than in a
    # checker run that needs a GPU to reach the first install.
    import inspect
    reads = set(re.findall(r"head\.([A-Za-z_][A-Za-z_0-9]*)",
                           inspect.getsource(pose_perblock.install)))
    reads -= {n for n in reads if n.startswith("_pose")} | {"add_module"}
    # frame_dir_proj is read ONLY when `orient` is in the terms, which pose_axes refuses without
    # --frame-direction; the checker never asks for it and the fake head has no pose_direction.
    reads -= {"frame_dir_proj"}
    missing = sorted(r for r in reads if not hasattr(fm.confidence_head, r))
    check("the checker's fake head has every attribute install() reads", not missing,
          f"missing {missing}" if missing else f"{len(reads)} attribute(s): {sorted(reads)}")
    try:
        pose_perblock.install(fm, terms=("onehot",), loud=False)
        check("so install(terms=('onehot',)) on the fake head does not raise", True)
    except Exception as e:
        check("so install(terms=('onehot',)) on the fake head does not raise", False,
              f"{type(e).__name__}: {e}")
    # AND THE FIXTURE's OWN FORWARD IS BLOCK 0's COMPLETE INJECTION, which is what "shared" and
    # "pretrained" are defined against.
    check("and its pre-stack forward adds both of block 0's terms",
          "linear_no_bias_d_wo_onehot(d.unsqueeze(-1))" in
          inspect.getsource(_pbc.FakeHead.memory_efficient_forward))

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
