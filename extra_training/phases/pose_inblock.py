"""Two pose injections placed so that a FROZEN stack stays frozen: inside block 3, and at the readout.

WHY THE PLACEMENT IS THE WHOLE POINT. Freezing pairformer blocks 0-2 (and, here, block 3's two
triangle multiplications) is only a speedup if autograd BUILDS NO GRAPH through them. Autograd starts
building at the first op whose parameters or inputs require grad -- so where the pose enters decides
how much backward there is:

  * `--pose-perblock` injects at each block's INPUT. With block 3 otherwise frozen, the `z` entering
    `tri_mul_out` would then require grad, BOTH tri-muls would come back into the backward, and the
    saving would evaporate SILENTLY -- no error, no warning, just a slower epoch. That is why this is
    a separate module and not a mode of that one.
  * This module injects AFTER `tri_mul_in` and BEFORE `tri_att_start`, which is the earliest point
    at which the frozen tri-muls are already behind us.

run_logging/test_pose_inblock_graph.py asserts exactly that, with a negative control that moves the
injection to the block's input and checks the tri-muls DO get gradient -- because without the control
the assertion proves nothing.

THE TWO INJECTIONS.

  1  IN-BLOCK, after tri_mul_in:
         z = z + pose_inblock_d(one_hot(d))  +  pose_inblock_d_wo_onehot(d)
     BOTH terms, not either, because that is what the native pre-stack injection is
     (confidence.py:297-305): linear_no_bias_d over a 39-bin one-hot of the pairwise distance AND
     linear_no_bias_d_wo_onehot over the raw distance. Same shapes, (c_z, num_bins) and (c_z, 1).

  2  AT THE READOUT, after pae_ln:
         pae = linear_no_bias_pae(pae_ln(z_pair) + pose_inblock_pae_d(one_hot(d))
                                                 + pose_inblock_pae_d_wo_onehot(d))

     AFTER THE LAYER NORM IS THE OWNER'S CHOICE, MADE KNOWING THE ALTERNATIVE, and it has a
     consequence worth stating: it adds an UNNORMALISED quantity to a NORMALISED one, and nothing
     downstream rescales it -- `linear_no_bias_pae` is the next and last op. So the injection's
     effective influence is set by its initialisation and by whatever the optimiser does to it, not
     by the layer norm. Zero-init means it starts with none; after that its scale is unconstrained
     relative to the normalised activations it is added to. That is the thing for a reviewer to look
     at. Injecting BEFORE the norm would have put it inside the normalisation and bounded it.

     IMPLEMENTED AS A FORWARD HOOK ON pae_ln, NOT AS A WRAPPER MODULE. Wrapping pae_ln in a new
     nn.Module would rename its state-dict key from `pae_ln.weight` to `pae_ln.ln.weight` and break
     phase_k_eval's set-equality check on head tensors. A hook that returns a modified output adds no
     keys of its own; the projections are separate children of the head, exactly as
     pose_perblock_proj is.

BOTH ARE ZERO-INIT AND VERIFIED EXACTLY ZERO, following pose_direction: step 0 must be bit-identical
to the un-injected baseline so that any later divergence is attributable to training rather than to
installation. Verified by probing the modules, not by trusting the initialiser.

A MIRROR, SO IT IS PINNED. Injecting between two ops inside PairformerBlock.forward means
reimplementing that function for one block, and a mirror can go stale: if Protenix reorders the
block, this copy would keep the old order while the rest of the stack moved. The sha256 of
PairformerBlock.forward is pinned and a mismatch is a hard stop, the same contract lean_readouts.py
uses. Only the `inplace_safe=False` branch is mirrored, because that is the one the trainer takes.
"""
import hashlib
import inspect

import torch

from protenix.model.modules.fused_ops import dropout_add_rowwise
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.utils import one_hot

# THE EXACT PairformerBlock.forward THIS MIRRORS, recorded against
# /home/ebrahim/Protenix/protenix/model/modules/pairformer.py on 2026-11-07.
PINNED_BLOCK_SHA = "379f595070ca6467f33506b33869a7a7a5773893d362faa7a7530187ae950d78"
PINNED_BLOCK_WHERE = "PairformerBlock.forward"

# WHERE THE IN-BLOCK INJECTION GOES. "after_tri_mul_in" is the only value the trainer ever passes
# and the only one that keeps the tri-muls out of the backward. "block_input" EXISTS ONLY FOR THE
# NEGATIVE CONTROL in run_logging/test_pose_inblock_graph.py -- it reproduces --pose-perblock's
# placement so the test can prove it detects the failure it is there to prevent. No flag exposes it.
INJECT_AT = ("after_tri_mul_in", "block_input")

# THE PER-CALL POSE GEOMETRY. Set by the memory_efficient_forward wrapper, read by _prep_blocks, and
# cleared on the way out -- including on an exception, since leaving an [N, N] tensor and an
# [N, N, 39] one-hot reachable between poses is how a per-pose leak looks from the outside.
_CALL = {"onehot": None, "dist": None}


class _NoPose(RuntimeError):
    """The stack was prepared without the pose geometry set."""


def _pin(fn, want, where):
    sha = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    if sha != want:
        raise SystemExit(
            f"STOP: --pose-inblock mirrors a specific {where}.\n"
            f"  pinned    {want}\n"
            f"  installed {sha}\n"
            f"  at        {inspect.getsourcefile(fn)}\n"
            f"The mirror in phases/pose_inblock.py is a copy of code that no longer runs, so the "
            f"injection could land at a different point in the block than the one the freeze was "
            f"reasoned about -- which costs the whole saving and raises nothing. Re-read that "
            f"function, confirm the op order, move the injection to the new 'after tri_mul_in' "
            f"position, update the pin, and re-run run_logging/test_pose_inblock_graph.py.")
    return sha


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
            f"STOP: a zero-init pose projection returned {float(got.abs().max()):.3e}, not zero, so "
            f"installing it would change the head at step 0 and nothing later would be attributable.")
    return m


def _inject(head, z, oh, dd):
    """z + d_onehot + d_raw, both terms, dropping each delta before the next op.

    ADD, THEN DROP THE REFERENCE. Holding a live [N_token, N_token, c_z] delta across the rest of the
    block pins it for the block's whole forward and backward -- the mistake review caught in pathway
    C, where it was 5.5 GiB at the 3,400-token crop cap.
    """
    z = z + head.pose_inblock_d(oh).to(z.dtype)
    z = z + head.pose_inblock_d_wo_onehot(dd.unsqueeze(-1)).to(z.dtype)
    return z


def block_forward(head, blk, s, z, oh, dd, inject_at="after_tri_mul_in", pair_mask=None,
                  triangle_multiplicative="torch", triangle_attention="torch",
                  inplace_safe=False, chunk_size=None):
    """PairformerBlock.forward's out-of-place branch, with the pose injected at `inject_at`.

    Every kept line mirrors the original. The ONLY difference is the two added terms.
    """
    if inplace_safe:
        raise SystemExit(
            "STOP: pose_inblock mirrors the inplace_safe=False branch of "
            f"{PINNED_BLOCK_WHERE} only, because that is the one the trainer takes "
            "(phase_k_train passes inplace_safe=False). It was called with inplace_safe=True, so it "
            "would be running a path nothing has checked.")
    if inject_at not in INJECT_AT:
        raise SystemExit(f"STOP: pose_inblock inject_at={inject_at!r}; expected one of {INJECT_AT}.")

    # THE NEGATIVE CONTROL'S PLACEMENT. Reproduces --pose-perblock: the pose enters before the first
    # op, so `z` entering tri_mul_out requires grad and both tri-muls rejoin the backward. Used only
    # by the graph test, to prove it can see that.
    if inject_at == "block_input":
        z = _inject(head, z, oh, dd)

    tmu_update = blk.tri_mul_out(
        z, mask=pair_mask, inplace_safe=False, _add_with_inplace=False,
        triangle_multiplicative=triangle_multiplicative)
    z = dropout_add_rowwise(z, tmu_update, blk.p_drop, blk.training)
    del tmu_update
    tmu_update = blk.tri_mul_in(
        z, mask=pair_mask, inplace_safe=False, _add_with_inplace=False,
        triangle_multiplicative=triangle_multiplicative)
    z = dropout_add_rowwise(z, tmu_update, blk.p_drop, blk.training)
    del tmu_update

    # ---- THE INJECTION, AFTER tri_mul_in AND BEFORE tri_att_start ---------------------------
    # Both frozen tri-muls are behind us, so this is the first op in the block whose input requires
    # grad and therefore where the backward graph begins. One op earlier and the tri-muls are back in
    # it; one op later and tri_att_start is frozen out of the trainable set for no reason.
    if inject_at == "after_tri_mul_in":
        z = _inject(head, z, oh, dd)
    # -----------------------------------------------------------------------------------------

    z = dropout_add_rowwise(
        z,
        blk.tri_att_start(z, mask=pair_mask, triangle_attention=triangle_attention,
                          inplace_safe=False, chunk_size=chunk_size),
        blk.p_drop, blk.training)
    z = z.transpose(-2, -3).contiguous()
    z = dropout_add_rowwise(
        z,
        blk.tri_att_end(z, mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                        triangle_attention=triangle_attention, inplace_safe=False,
                        chunk_size=chunk_size),
        blk.p_drop, blk.training)
    z = z.transpose(-2, -3).contiguous()
    z = z + blk.pair_transition(z)
    # THE SINGLE TRACK, ON THE SAME GUARD THE ORIGINAL USES, so --lean-single still turns it off.
    if blk.c_s > 0:
        s = s + blk.attention_pair_bias(a=s, s=None, z=z)
        s = s + blk.single_transition(s)
    return s, z


def install(model, block_index=3, pae_inject=True, inject_at="after_tri_mul_in", loud=True):
    """Install both injections. Returns a dict describing what was added.

    MUST BE INSTALLED AFTER lean_readouts, because it wraps memory_efficient_forward and
    lean_readouts.install() replaces that attribute outright rather than wrapping it. The marker
    this sets is in lean_readouts' refusal list, so the wrong order is a hard stop rather than a
    silently dropped injection.
    """
    head = getattr(model, "confidence_head", None)
    if head is None:
        raise SystemExit("STOP: --pose-inblock: the model has no confidence_head.")
    stack = getattr(head, "pairformer_stack", None)
    blocks = list(getattr(stack, "blocks", []) or [])
    if not blocks:
        raise SystemExit("STOP: --pose-inblock: the head's pairformer_stack has no blocks.")
    if not 0 <= block_index < len(blocks):
        raise SystemExit(f"STOP: --pose-inblock-block {block_index} is out of range for a stack of "
                         f"{len(blocks)} block(s). The default is the LAST block, {len(blocks) - 1}, "
                         f"because that is the one the freeze leaves trainable.")
    _pin(type(blocks[0]).forward, PINNED_BLOCK_SHA, PINNED_BLOCK_WHERE)
    if getattr(head, "_pose_inblock_installed", False):
        raise SystemExit("STOP: --pose-inblock is already installed. Installing twice would stack "
                         "two injections at the same point and double the capacity silently.")

    ref = head.linear_no_bias_d.weight
    head.add_module("pose_inblock_d", _zero_linear(head.num_bins, head.c_z, ref))
    head.add_module("pose_inblock_d_wo_onehot", _zero_linear(1, head.c_z, ref))
    if pae_inject:
        head.add_module("pose_inblock_pae_d", _zero_linear(head.num_bins, head.c_z, ref))
        head.add_module("pose_inblock_pae_d_wo_onehot", _zero_linear(1, head.c_z, ref))

    orig_mef = head.memory_efficient_forward
    orig_prep = stack._prep_blocks

    def memory_efficient_forward(input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        # THE SAME DISTANCE THE NATIVE INJECTION USES, recomputed here rather than reached for.
        # The inner forward computes its own cdist for its own injection; this is a second cdist on
        # [N_rep, 3], which is O(N^2) against an O(N^3) stack, and is the price of wrapping instead
        # of forking. pose_direction pays exactly the same price for exactly the same reason.
        #
        # OUTSIDE AUTOCAST, in fp32, matching confidence.py:285-287 -- a bf16 cdist would put the
        # one-hot bin edges at a different place than the native injection's.
        with torch.amp.autocast("cuda", enabled=False):
            _x = x_pred_rep_coords.to(torch.float32)
            _d = torch.cdist(_x, _x)
        _CALL["dist"] = _d
        _CALL["onehot"] = one_hot(x=_d, lower_bins=head.lower_bins, upper_bins=head.upper_bins)
        try:
            return orig_mef(input_feature_dict=input_feature_dict, s_trunk=s_trunk, z_pair=z_pair,
                            pair_mask=pair_mask, x_pred_rep_coords=x_pred_rep_coords, **kw)
        finally:
            _CALL["dist"] = None
            _CALL["onehot"] = None

    def _prep_blocks(pair_mask=None, **kw):
        prepped = orig_prep(pair_mask=pair_mask, **kw)
        # CAPTURED INTO LOCALS, NOT READ FROM THE DICT INSIDE THE BLOCK. With --blocks-per-ckpt 1 a
        # block's forward is RE-RUN during backward, by which time the wrapper's `finally` has
        # already cleared the dict -- which is exactly the _NoPose-on-backward bug review
        # reproduced in pose_perblock. A local is held by this closure, which the blocks list
        # holds, which checkpoint_blocks' graph holds until that graph's backward is done.
        oh, dd = _CALL["onehot"], _CALL["dist"]
        if oh is None:
            raise _NoPose(
                "STOP: the pair stack was prepared without the pose geometry set. --pose-inblock is "
                "installed but its memory_efficient_forward wrapper did not run, so the stack was "
                "reached by some other path and the injection would be silently absent.")
        tgt = prepped[block_index]
        # THE BLOCK AND ITS KWARGS COME OUT OF THE partial _prep_blocks BUILT, so the mirror is
        # called with exactly the arguments the original would have had. Reconstructing them by hand
        # is how a mirror drifts from its caller.
        blk = getattr(tgt, "func", tgt)
        kws = dict(getattr(tgt, "keywords", {}) or {})

        def run(s, z):
            return block_forward(head, blk, s, z, oh, dd, inject_at=inject_at, **kws)

        prepped[block_index] = run
        return prepped

    def _pae_hook(_mod, _inp, out):
        """pae_ln's output, plus the pose. Returns the replacement, which is what a hook may do."""
        oh, dd = _CALL["onehot"], _CALL["dist"]
        if oh is None:
            # NOT AN ERROR HERE. pae_ln is reached only from the readout, which is inside the
            # wrapper -- but phase_k_eval builds its own model without this installed, and a hook
            # that raised would turn "no injection" into a crash in a process that never wanted one.
            return out
        if out.shape[:-1] != oh.shape[:-1]:
            raise RuntimeError(
                f"STOP: --pose-inblock-pae: pae_ln's output is {tuple(out.shape)} but the pose "
                f"geometry is {tuple(oh.shape)}. The hook is firing on a call it was not written "
                f"for, and adding across that mismatch would broadcast silently.")
        out = out + head.pose_inblock_pae_d(oh).to(out.dtype)
        out = out + head.pose_inblock_pae_d_wo_onehot(dd.unsqueeze(-1)).to(out.dtype)
        return out

    head.memory_efficient_forward = memory_efficient_forward
    stack._prep_blocks = _prep_blocks
    hook = head.pae_ln.register_forward_hook(_pae_hook) if pae_inject else None
    head._pose_inblock_installed = True
    head._pose_inblock_hook = hook

    # ASSERT IT TOOK. A flag that prints a line and changes nothing is the failure mode
    # --blocks-per-ckpt had for its whole life.
    if head.memory_efficient_forward is not memory_efficient_forward or \
            stack._prep_blocks is not _prep_blocks:
        raise SystemExit("STOP: --pose-inblock did not take: the head's memory_efficient_forward or "
                         "the stack's _prep_blocks is not the patched version after assignment.")
    if pae_inject and hook is None:
        raise SystemExit("STOP: --pose-inblock-pae did not register its hook on pae_ln.")

    names = ["pose_inblock_d", "pose_inblock_d_wo_onehot"]
    if pae_inject:
        names += ["pose_inblock_pae_d", "pose_inblock_pae_d_wo_onehot"]
    info = {"block_index": int(block_index), "inject_at": inject_at,
            "pae_inject": bool(pae_inject), "modules": names,
            "params": int(sum(getattr(head, n).weight.numel() for n in names)),
            "num_bins": int(head.num_bins), "c_z": int(head.c_z)}
    if loud:
        print(f"POSE IN-BLOCK: injected into pairformer block {block_index} at {inject_at} "
              f"(after tri_mul_in, before tri_att_start) with BOTH the one-hot and the raw-distance "
              f"term"
              + (", and after pae_ln in the PAE readout" if pae_inject else "")
              + f". {info['params']:,} new parameter(s) across {len(names)} tensor(s), ZERO-INIT and "
              f"verified exactly zero, so step 0 is bit-identical to the un-injected head. "
              f"PairformerBlock.forward pinned at {PINNED_BLOCK_SHA[:16]}.")
        if pae_inject:
            print("  NOTE the PAE injection is added AFTER pae_ln, so an UNNORMALISED quantity is "
                  "added to a normalised one and nothing downstream rescales it. Its effective "
                  "influence is set by its initialisation and by the optimiser, not by the norm.")
    return info
