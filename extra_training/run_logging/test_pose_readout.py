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
    pose_readout.install(m3, copy_distance=True, loud=False)
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

    print()
    if fails:
        print(f"FAILED {len(fails)}: " + ", ".join(fails))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
