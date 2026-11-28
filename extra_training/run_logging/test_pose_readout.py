"""phases/pose_readout.py -- the fifth injection site, the one that works with --pose-perblock.

FIVE THINGS, each of which has a way of being wrong that a smoke test would miss:

  1. it REFUSES with no geometry producer, rather than installing a hook that never fires. That
     silent no-op is exactly what --pose-inblock-pae is refused for (phase_k_train.py:1896).
  2. zero-init leaves the readout BIT-IDENTICAL, not merely close.
  3. the arms are NOT gradient-dead at zero init. This is the claim worth testing: a zero-init
     multi-layer branch never learns, and the request explicitly warned about it. These are single
     LinearNoBias layers, so dL/dW = g (x) x^T is non-zero -- asserted here, not assumed.
  4. copy_distance ACTUALLY CHANGES step 0. A flag whose two settings behave identically is worse
     than no flag, because the run would be mislabelled.
  5. the shape guard fires instead of broadcasting a mismatched pose into the readout.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))

from protenix.model.modules.primitives import LinearNoBias  # noqa: E402
from protenix.model.utils import one_hot                    # noqa: E402

import pose_perblock                                        # noqa: E402
import pose_readout                                         # noqa: E402

N, C_Z, NUM_BINS = 7, 16, 39
# BIN EDGES, NOT SCALARS. protenix one_hot takes [bins]-shaped bounds and broadcasts x[..., None]
# against them; passing floats yields a 1-wide "one-hot" and the matmul fails 39 columns short.
_EDGES = torch.linspace(3.25, 21.75, NUM_BINS + 1)
LOWER, UPPER = _EDGES[:-1].clone(), _EDGES[1:].clone()


class _Head(torch.nn.Module):
    """The surface pose_readout touches, and nothing else."""
    def __init__(self):
        super().__init__()
        self.c_z, self.num_bins = C_Z, NUM_BINS
        self.lower_bins, self.upper_bins = LOWER, UPPER
        self.pae_ln = torch.nn.LayerNorm(C_Z)
        self.linear_no_bias_d = LinearNoBias(in_features=NUM_BINS, out_features=C_Z)
        self.linear_no_bias_d_wo_onehot = LinearNoBias(in_features=1, out_features=C_Z)
        self.linear_no_bias_pae = LinearNoBias(in_features=C_Z, out_features=NUM_BINS)

    def readout(self, z):
        return self.linear_no_bias_pae(self.pae_ln(z))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence_head = _Head()


def _geometry(n=N):
    g = torch.Generator().manual_seed(11)
    c = torch.randn(n, 3, generator=g)
    d = torch.cdist(c, c)
    oh = one_hot(x=d, lower_bins=LOWER, upper_bins=UPPER)
    dr = torch.nn.functional.normalize(torch.randn(n, n, 3, generator=g), dim=-1)
    return oh, d, dr


def _publish(oh, d, dr=None):
    pose_perblock._CALL["onehot"], pose_perblock._CALL["dist"] = oh, d
    if dr is not None:
        import pose_direction
        pose_direction._CALL["dir"] = dr


def _clear():
    pose_perblock._CALL["onehot"] = pose_perblock._CALL["dist"] = None
    try:
        import pose_direction
        pose_direction._CALL["dir"] = None
    except Exception:
        pass


def main():
    fails = []

    def check(name, ok, detail=""):
        print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    oh, d, dr = _geometry()
    z = torch.randn(N, N, C_Z, generator=torch.Generator().manual_seed(3))

    # ---- 1. refuses without a geometry producer -------------------------------------------
    m = _Model()
    try:
        pose_readout.install(m, loud=False)
        check("refuses with no --pose-perblock/--pose-inblock", False, "it installed anyway")
    except RuntimeError as e:
        check("refuses with no --pose-perblock/--pose-inblock", "silent no-op" in str(e))

    # ---- 2. zero init is bit-identical ----------------------------------------------------
    m = _Model()
    m.confidence_head._pose_perblock_installed = True
    base = m.confidence_head.readout(z).clone()
    info = pose_readout.install(m, loud=False)
    _publish(oh, d)
    got = m.confidence_head.readout(z)
    check("zero-init readout is BIT-identical", bool(torch.equal(base, got)),
          f"max|d| {float((got - base).abs().max()):.3e}")
    check("no orientation arm without --frame-direction", info["orientation"] is False)

    # ---- 3. not gradient-dead at zero init ------------------------------------------------
    m.confidence_head.readout(z).square().mean().backward()
    hd = m.confidence_head
    for nm in ("pose_readout_d", "pose_readout_d_wo_onehot"):
        g = getattr(hd, nm).weight.grad
        check(f"{nm} has a NON-ZERO gradient at zero init",
              g is not None and bool(g.abs().max() > 0),
              f"max|g| {float(g.abs().max()):.3e}" if g is not None else "grad is None")

    # ---- 3b. the orientation arm, same question -------------------------------------------
    import pose_direction
    m2 = _Model()
    m2.confidence_head._pose_perblock_installed = True
    m2.confidence_head._pose_direction_installed = True
    base2 = m2.confidence_head.readout(z).clone()
    info2 = pose_readout.install(m2, loud=False)
    _publish(oh, d, dr)
    got2 = m2.confidence_head.readout(z)
    check("orientation arm installed with --frame-direction", info2["orientation"] is True)
    check("zero-init WITH orientation is BIT-identical", bool(torch.equal(base2, got2)))
    m2.confidence_head.readout(z).square().mean().backward()
    g = m2.confidence_head.pose_readout_frame_dir.weight.grad
    check("pose_readout_frame_dir has a NON-ZERO gradient at zero init",
          g is not None and bool(g.abs().max() > 0),
          f"max|g| {float(g.abs().max()):.3e}" if g is not None else "grad is None")

    # ---- 4. copy_distance actually changes step 0 -----------------------------------------
    m3 = _Model()
    m3.confidence_head._pose_perblock_installed = True
    base3 = m3.confidence_head.readout(z).clone()
    pose_readout.install(m3, init="pretrained", loud=False)
    _publish(oh, d)
    got3 = m3.confidence_head.readout(z)
    check("copy_distance CHANGES step 0 (the flag is not cosmetic)",
          not torch.equal(base3, got3),
          f"max|d| {float((got3 - base3).abs().max()):.3e}")
    same_d = torch.equal(m3.confidence_head.pose_readout_d.weight,
                         m3.confidence_head.linear_no_bias_d.weight)
    same_w = torch.equal(m3.confidence_head.pose_readout_d_wo_onehot.weight,
                         m3.confidence_head.linear_no_bias_d_wo_onehot.weight)
    check("copy_distance copies BOTH pretrained arms exactly", same_d and same_w)

    # ---- 5. the shape guard ---------------------------------------------------------------
    oh_small, d_small, _ = _geometry(n=N - 2)
    _publish(oh_small, d_small)
    try:
        m.confidence_head.readout(z)
        check("shape guard fires on mismatched geometry", False, "it broadcast silently")
    except RuntimeError as e:
        check("shape guard fires on mismatched geometry", "was not written" in str(e))

    # ---- 6. hook is a no-op when no pose is in flight --------------------------------------
    _clear()
    check("no pose in flight -> readout unchanged",
          bool(torch.equal(m.confidence_head.readout(z), base)))

    # ---- 7. prenorm=True: THE OTHER SITE -------------------------------------------------
    # The whole claim of --pose-readout-prenorm is that it injects into pae_ln's INPUT, which is
    # a DIFFERENT function from injecting into its output -- so the tests that matter are that it
    # still fires (a forward PRE-hook has to return the replacement args tuple, and returning the
    # wrong shape silently does nothing), that it is not gradient-dead, and that it is NOT the
    # same number as the post-LN site at the same weights.
    _clear()
    m4 = _Model()
    m4.confidence_head._pose_perblock_installed = True
    m4.confidence_head._pose_direction_installed = True
    base4 = m4.confidence_head.readout(z).clone()
    info4 = pose_readout.install(m4, loud=False, sites=("pre",))
    check("prenorm installs the same three tensors", info4["orientation"] is True
          and info4["tensors"] == ["pose_readout_d", "pose_readout_d_wo_onehot",
                                   "pose_readout_frame_dir"])
    check("prenorm registers the PRE-hook, not the post-hook",
          m4.confidence_head._pose_readout_prenorm is True
          and len(m4.confidence_head.pae_ln._forward_pre_hooks) == 1
          and len(m4.confidence_head.pae_ln._forward_hooks) == 0)
    _publish(oh, d, dr)
    got4 = m4.confidence_head.readout(z)
    check("prenorm zero-init is BIT-identical", bool(torch.equal(base4, got4)),
          f"max|d| {float((got4 - base4).abs().max()):.3e}")
    m4.confidence_head.readout(z).square().mean().backward()
    for nm in ("pose_readout_d", "pose_readout_d_wo_onehot", "pose_readout_frame_dir"):
        g = getattr(m4.confidence_head, nm).weight.grad
        check(f"prenorm: {nm} has a NON-ZERO gradient at zero init",
              g is not None and bool(g.abs().max() > 0),
              f"max|g| {float(g.abs().max()):.3e}" if g is not None else "grad is None")

    # copy_distance pre-norm: the hook must FIRE (output moves) and must not agree with the
    # post-LN site, which is the only evidence that `prenorm` selects a different operator.
    _clear()
    m5, m6 = _Model(), _Model()
    m6.load_state_dict(m5.state_dict())
    for mm in (m5, m6):
        mm.confidence_head._pose_perblock_installed = True
    base5 = m5.confidence_head.readout(z).clone()
    pose_readout.install(m5, init="pretrained", loud=False, sites=("pre",))
    pose_readout.install(m6, init="pretrained", loud=False, sites=("post",))
    _publish(oh, d)
    got5 = m5.confidence_head.readout(z)
    got6 = m6.confidence_head.readout(z)
    check("prenorm copy_distance CHANGES the readout (the pre-hook fires)",
          not torch.equal(base5, got5),
          f"max|d| {float((got5 - base5).abs().max()):.3e}")
    check("prenorm is a DIFFERENT site from post-LN at identical weights",
          not torch.allclose(got5, got6, atol=1e-5),
          f"max|pre - post| {float((got5 - got6).abs().max()):.3e}")
    # AND THE POINT OF THE SITE: the LN divides the magnitude back out, so however large the
    # injection grows, what linear_no_bias_pae reads stays unit-scaled. Post-LN it does not.
    with torch.no_grad():
        for mm in (m5, m6):
            mm.confidence_head.pose_readout_d.weight.mul_(50.0)
            mm.confidence_head.pose_readout_d_wo_onehot.weight.mul_(50.0)
    n_pre = float(m5.confidence_head.pae_ln(z).std())
    _hpre = m5.confidence_head.pae_ln._forward_pre_hooks
    check("prenorm: pae_ln's OUTPUT stays unit-scaled however big the injection",
          abs(n_pre - 1.0) < 0.2 and len(_hpre) == 1, f"std {n_pre:.3f}")
    n_post = float(m6.confidence_head.pae_ln(z).std())
    check("post-LN: the same injection scales the readout input without limit",
          n_post > 5.0, f"std {n_post:.3f}")

    # the shape guard, pre-norm side
    _publish(oh_small, d_small)
    try:
        m5.confidence_head.readout(z)
        check("prenorm shape guard fires on mismatched geometry", False, "it broadcast silently")
    except RuntimeError as e:
        check("prenorm shape guard fires on mismatched geometry",
              "--pose-readout-prenorm" in str(e) and "was not written" in str(e))

    _clear()
    check("prenorm: no pose in flight -> readout unchanged",
          bool(torch.equal(m5.confidence_head.readout(z), base5)))

    # ---------------------------------------------------------------- the TWO-SITE arm
    # Pre-norm copy-init AND a second, zero-init site after the layernorm. The second set of
    # matrices has to be its own: sharing them would add the pose twice through one weight.
    _clear()
    m7, m8 = _Model(), _Model()
    m8.load_state_dict(m7.state_dict())
    for mm in (m7, m8):
        mm.confidence_head._pose_perblock_installed = True
    pose_readout.install(m7, init="pretrained", loud=False, sites=("pre", "post"))
    pose_readout.install(m8, init="pretrained", loud=False, sites=("pre",))
    h7 = m7.confidence_head
    check("two-site: the post matrices exist and are separate objects",
          hasattr(h7, "pose_readout_post_d")
          and h7.pose_readout_post_d is not h7.pose_readout_d)
    check("two-site: the post matrices are EXACTLY zero at install",
          float(h7.pose_readout_post_d.weight.abs().max()) == 0.0
          and float(h7.pose_readout_post_d_wo_onehot.weight.abs().max()) == 0.0)
    check("two-site: both hooks are registered on pae_ln",
          len(h7.pae_ln._forward_pre_hooks) == 1 and len(h7.pae_ln._forward_hooks) == 1)
    _publish(oh, d)
    got7, got8 = h7.readout(z), m8.confidence_head.readout(z)
    # AT INSTALL the zero post site adds nothing, so the two-site arm must equal the prenorm arm.
    check("two-site: at step 0 it is exactly the prenorm arm (the post site is zero)",
          bool(torch.equal(got7, got8)))
    # AND IT IS NOT DEAD: give the post matrices weight and the output must move.
    with torch.no_grad():
        h7.pose_readout_post_d.weight.normal_(0.0, 0.05)
    got7b = h7.readout(z)
    check("two-site: the post site is live once it has weight",
          not torch.equal(got7, got7b),
          f"max|d| {float((got7b - got7).abs().max()):.3e}")
    # the post site is gradient-reachable too
    h7.pose_readout_post_d.weight.grad = None
    h7.readout(z).sum().backward()
    g7 = h7.pose_readout_post_d.weight.grad
    check("two-site: the post site receives gradient",
          g7 is not None and bool(g7.abs().max() > 0))
    # THE OLD post_zero-WITHOUT-prenorm REFUSAL, now where it belongs: the SITES are named
    # directly here (sites=("post",) is the one-site arm, ("pre","post") the two-site one), so
    # install() has nothing left to refuse. The legacy flag combination it protected is still
    # refused, by the desugaring, which is the only place that spelling exists any more.
    import pose_axes
    try:
        pose_axes.resolve({"pose_perblock": True, "pose_readout": True,
                           "pose_readout_post_zero": True})
        check("two-site: --pose-readout-post-zero without --pose-readout-prenorm is refused",
              False, "it resolved")
    except pose_axes.PoseConfigError as e:
        check("two-site: --pose-readout-post-zero without --pose-readout-prenorm is refused",
              "SECOND site" in str(e))
    # AND A SITE LIST IS A SITE LIST: an unknown site is named, not silently dropped.
    try:
        pose_axes.parse_readout_sites("postnorm")
        check("an unknown readout site is refused", False, "it parsed")
    except pose_axes.PoseConfigError as e:
        check("an unknown readout site is refused", "unknown site" in str(e))

    print()
    if fails:
        print(f"FAILED {len(fails)}: " + ", ".join(fails))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
