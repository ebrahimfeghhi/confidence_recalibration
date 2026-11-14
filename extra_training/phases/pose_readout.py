"""THE FIFTH POSE INJECTION SITE: after pae_ln, upstream of the PAE readout, usable ALONGSIDE
--pose-perblock.

WHY THIS MODULE EXISTS AT ALL. The readout injection already existed as --pose-inblock-pae, but
the trainer refuses it two ways (phase_k_train.py:1884 and :1896): --pose-inblock excludes
--pose-perblock because both replace pairformer_stack._prep_blocks and whichever installs second
silently wins, and --pose-inblock-pae alone is refused because it reads the geometry the in-block
wrapper stashes and would otherwise never fire. The configuration asked for -- the pose at every
block input AND again at the readout -- is exactly the combination those two refusals exclude, and
the first of them says so: "that is a third design and needs its own module." This is that module.

WHAT IT DOES. One forward hook on head.pae_ln:

    pae = linear_no_bias_pae( pae_ln(z_pair)
                              + pose_readout_d(one_hot(d))
                              + pose_readout_d_wo_onehot(d)
                              + pose_readout_frame_dir(dir) )    # if --frame-direction

WHERE THE GEOMETRY COMES FROM, and why it is not recomputed here. pose_perblock's wrapper computes
the cdist and the one-hot ONCE per forward and publishes them in its own _CALL; its `finally`
clears them on the way out of memory_efficient_forward, and the readout runs INSIDE that call, so
they are live when this hook fires. Recomputing cdist here would be a second [N, N] distance and a
second [N, N, 39] one-hot per pose for a quantity already in hand. pose_inblock's cache is accepted
too, so this module also works in the in-block configuration; it is read second so that when both
somehow exist the stack-level injection in force wins.

INITIALISATION. Zero by default, so step 0 is bit-identical to the un-injected head and anything
that moves afterwards is attributable. copy_distance=True instead copies block 0's pretrained
linear_no_bias_d and linear_no_bias_d_wo_onehot, which is what the request asked for -- and it is
NOT a neutral choice, so install() says so out loud rather than doing it quietly:

  the pretrained injection was fitted to add into z_pair INSIDE the stack, where every downstream
  block re-normalises. This site adds AFTER pae_ln, so an unnormalised pretrained-scale term lands
  on a normalised one with nothing downstream to rescale it, and linear_no_bias_pae sees it
  directly. Shapes permit the copy; semantics do not match. The orientation arm is zero-init in
  both modes.

GRADIENT LIFE AT ZERO INIT. frame_dir_proj and these arms are single LinearNoBias layers, not
MLPs: at W = 0 the output is zero but dL/dW = g (x) x^T with x the non-zero geometry, so the first
step moves them. A zero-init multi-layer branch would be gradient-dead; this one is not. The test
asserts it rather than leaving it to the reader.
"""
import torch

from protenix.model.modules.primitives import LinearNoBias
from protenix.model.utils import one_hot  # noqa: F401  (parity with the other injection modules)


def _geometry():
    """(one_hot, dist, dir) for the pose in flight, from whichever stack injection is installed."""
    oh = dd = None
    try:
        import pose_perblock as _ppb
        oh, dd = _ppb._CALL.get("onehot"), _ppb._CALL.get("dist")
    except Exception:
        pass
    if oh is None:
        try:
            import pose_inblock as _pib
            oh, dd = _pib._CALL.get("onehot"), _pib._CALL.get("dist")
        except Exception:
            pass
    dr = None
    try:
        import pose_direction as _pd
        dr = _pd._CALL.get("dir")
    except Exception:
        pass
    return oh, dd, dr


def _zero_linear(in_features, out_features, ref):
    """A zero-initialised LinearNoBias on ref's device and dtype, VERIFIED exactly zero."""
    m = LinearNoBias(in_features=in_features, out_features=out_features)
    with torch.no_grad():
        m.weight.zero_()
    m = m.to(device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        probe = torch.randn(4, in_features, device=ref.device, dtype=ref.dtype)
        got = m(probe)
    if bool(got.abs().max() > 0):
        raise RuntimeError(
            f"STOP: a zero-init readout projection returned {float(got.abs().max()):.3e}, not "
            f"zero, so installing it would change the head at step 0 and nothing later would be "
            f"attributable.")
    return m


def install(model, copy_distance=False, loud=True):
    head = model.confidence_head
    if getattr(head, "_pose_readout_installed", False):
        raise RuntimeError("STOP: pose_readout.install ran twice; the hook would fire twice and "
                           "the pose would be added twice.")
    # THE GEOMETRY MUST HAVE A PRODUCER. Without one the hook would return `out` unchanged on every
    # call and the flag would be a silent no-op -- the exact failure --pose-inblock-pae is refused
    # for at phase_k_train.py:1896.
    if not (getattr(head, "_pose_perblock_installed", False)
            or getattr(head, "_pose_inblock_installed", False)):
        raise RuntimeError(
            "STOP: --pose-readout needs --pose-perblock (or --pose-inblock) to be installed. It "
            "reads the per-forward pose geometry that wrapper publishes; on its own the hook would "
            "never fire and the flag would be a silent no-op.")

    ref = head.linear_no_bias_d.weight
    d_mod = _zero_linear(head.num_bins, head.c_z, ref)
    w_mod = _zero_linear(1, head.c_z, ref)
    if copy_distance:
        with torch.no_grad():
            d_mod.weight.copy_(head.linear_no_bias_d.weight)
            w_mod.weight.copy_(head.linear_no_bias_d_wo_onehot.weight)
    head.add_module("pose_readout_d", d_mod)
    head.add_module("pose_readout_d_wo_onehot", w_mod)

    has_dir = bool(getattr(head, "_pose_direction_installed", False))
    if has_dir:
        # ZERO IN BOTH MODES. The request asks for the orientation branch zero-init so step 0 is
        # the distance-only function; there is also no pretrained orientation matrix anywhere in
        # this project to copy from -- frame_dir_proj is itself zero at install.
        head.add_module("pose_readout_frame_dir", _zero_linear(3, head.c_z, ref))

    def _hook(_mod, _inp, out):
        oh, dd, dr = _geometry()
        if oh is None:
            # NOT AN ERROR. phase_k_eval builds its own model without this installed, and pae_ln is
            # reached from paths that never wanted an injection; raising would turn "no injection"
            # into a crash in a process that never asked for one.
            return out
        if out.shape[:-1] != oh.shape[:-1]:
            raise RuntimeError(
                f"STOP: --pose-readout: pae_ln's output is {tuple(out.shape)} but the pose "
                f"geometry is {tuple(oh.shape)}. The hook is firing on a call it was not written "
                f"for, and adding across that mismatch would broadcast silently.")
        # ADDED ONE TERM AT A TIME, NOT FUSED. `out = out + A` frees A before B is allocated, so
        # the peak is two pair tensors either way; pose_inblock.py:378-388 records the measurement
        # where fusing these made allocated-at-OOM WORSE (90.85 -> 91.63 GiB).
        out = out + head.pose_readout_d(oh).to(out.dtype)
        out = out + head.pose_readout_d_wo_onehot(dd.unsqueeze(-1)).to(out.dtype)
        if has_dir and dr is not None:
            if dr.shape[:-1] != oh.shape[:-1]:
                raise RuntimeError(
                    f"STOP: --pose-readout: orientation is {tuple(dr.shape)} against geometry "
                    f"{tuple(oh.shape)}; one pose's directions would pair with another's bins.")
            out = out + head.pose_readout_frame_dir(dr).to(out.dtype)
        return out

    head._pose_readout_hook = head.pae_ln.register_forward_hook(_hook)
    head._pose_readout_installed = True
    head._pose_readout_copy_distance = bool(copy_distance)

    names = ["pose_readout_d", "pose_readout_d_wo_onehot"]
    if has_dir:
        names.append("pose_readout_frame_dir")
    n_new = sum(p.numel() for n in names for p in getattr(head, n).parameters())
    if loud:
        print(f"POSE READOUT (5th site): injected after pae_ln, upstream of linear_no_bias_pae, "
              f"with the one-hot and raw-distance terms"
              + (" and the orientation term" if has_dir else " (no orientation: --frame-direction "
                                                             "not installed)")
              + f". {n_new:,} new parameter(s) across {len(names)} tensor(s). Geometry is READ from "
              f"the installed stack injection's per-forward cache, not recomputed. A --train-only "
              f"must match `pose_readout` to reach these.")
        if copy_distance:
            print("  COPY-DISTANCE IS ON, AND IT IS NOT NEUTRAL: pose_readout_d and "
                  "pose_readout_d_wo_onehot start at block 0's PRETRAINED weights, so step 0 is "
                  "NOT bit-identical to the un-injected head. Those weights were fitted to add "
                  "into z_pair inside the stack, where downstream blocks re-normalise; this site "
                  "adds AFTER pae_ln, so an unnormalised pretrained-scale term lands on a "
                  "normalised one with nothing downstream to rescale it. Shapes permit the copy, "
                  "semantics do not match. The orientation arm stays zero.")
        else:
            print("  ZERO-INIT and verified exactly zero, so step 0 is bit-identical to the "
                  "un-injected head.")
    return {"params": n_new, "tensors": names, "orientation": has_dir,
            "copy_distance": bool(copy_distance)}
