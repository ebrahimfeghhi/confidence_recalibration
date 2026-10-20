"""Pathway A: re-inject the pose at EVERY pairformer block, not only before the first.

See docs/pose_pathways.md. The confidence head embeds the predicted pose once -- a one-hot of
`cdist` over the representative atoms, through `linear_no_bias_d` -- adds it into `z_pair`, and
then runs 4 pairformer blocks whose weights know nothing about the pose. A gives each later block
its own injection of the same embedding.

WHAT THIS DOES AND DOES NOT TEST. A adds ACCESS and capacity, not information: the one-hot is the
same tensor every time, so nothing new enters the head. Its hypothesis is that the pose signal
degrades as it crosses the stack. With only 4 blocks that story is thin, and nothing measured so
far shows attenuation -- which is why pathway C (frame-relative direction, phases/pose_direction.py)
was built and run first. A is the follow-up, run alone here so that if it helps, combining it with
C is a separate and interpretable step.

UNTIED WEIGHTS. Each injected block gets its own LinearNoBias(num_bins -> c_z): 3 matrices at
39 x 128 = 14,976 parameters, 0.08% of the head's 19.8M trainable. Tied weights would be the
sharper test of the degradation hypothesis specifically -- the identical embedding re-presented at
each depth -- while untied also allows each depth a different projection of distance. Untied is
the default because it is how every other weight in the stack works and because it is strictly
more expressive; `--pose-perblock-tied` runs the sharper experiment. Note tying is NOT equivalent
to scaling a single injection: each block transforms z before the next add.

BLOCK 0 IS LEFT ALONE. Its injection already exists as the pretrained `linear_no_bias_d`, applied
before the stack. Adding a second one there would duplicate it rather than extend it.

ZERO-INIT, so step 0 is bit-identical to the head without A, and `install()` verifies that rather
than asserting it.
"""
import torch

from protenix.model.modules.primitives import LinearNoBias
from protenix.model.utils import one_hot

# The pose embedding for the sample currently in flight, computed ONCE per
# memory_efficient_forward and shared by every injected block. Module-level rather than threaded
# through PairformerStack.forward, whose signature we do not control.
#
# AT 656 TOKENS THIS ONE-HOT IS 67 MB in float32 and it is held for the duration of the stack,
# which is the point: computing it per block would be four times that for an identical result.
# Under activation checkpointing a tensor captured from an enclosing scope is held rather than
# recomputed, so this is also what the checkpointed blocks see on the backward pass.
_CALL = {"onehot": None}


class _NoPose(Exception):
    pass


def install(model, tied=False, loud=True):
    """Add the per-block projections and route the pose embedding into every block but the first.

    The projections are added as submodules of `confidence_head`, so phase_k_train's
    head_param_names -- which derives the trainable set from the module tree rather than from a
    hand-written list -- picks them up with no change there, provided install() runs first.
    """
    head = model.confidence_head
    if getattr(head, "_pose_perblock_installed", False):
        return {"already": True}
    stack = head.pairformer_stack
    n_blocks = len(stack.blocks)
    if n_blocks < 2:
        raise RuntimeError(f"STOP: the pair stack has {n_blocks} block(s); A has nothing to "
                           f"inject into beyond the existing pre-stack embedding.")

    ref = head.linear_no_bias_d.weight
    n_new = 1 if tied else n_blocks - 1
    projs = torch.nn.ModuleList()
    for _ in range(n_new):
        p = LinearNoBias(in_features=head.num_bins, out_features=head.c_z)
        with torch.no_grad():
            p.weight.zero_()
        projs.append(p.to(device=ref.device, dtype=ref.dtype))
    head.add_module("pose_perblock_proj", projs)
    head._pose_perblock_tied = bool(tied)

    # VERIFIED, NOT ASSUMED. The safety story is that step 0 is bit-identical, which holds only if
    # every one of these returns exactly zero -- not merely something small.
    with torch.no_grad():
        probe = torch.randn(4, head.num_bins, device=ref.device, dtype=ref.dtype)
        worst = max(float(p(probe).abs().max()) for p in projs)
    if worst > 0:
        raise RuntimeError(f"STOP: a zero-init projection returned {worst:.3e}, not zero, so "
                           f"installing A would change the head at step 0.")

    orig_mef = head.memory_efficient_forward
    orig_prep = stack._prep_blocks

    def memory_efficient_forward(input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        # THE SAME ONE-HOT THE EXISTING PATH USES, computed here so every block shares one tensor.
        # float32 and no autocast, matching what upstream already does around its own cdist: the
        # bin edges are compared against a distance, and a bf16 distance straddling an edge lands
        # in the wrong bin.
        with torch.amp.autocast("cuda", enabled=False):
            c = x_pred_rep_coords.to(torch.float32)
            d = torch.cdist(c, c)
            _CALL["onehot"] = one_hot(x=d, lower_bins=head.lower_bins,
                                      upper_bins=head.upper_bins)
        try:
            return orig_mef(input_feature_dict=input_feature_dict, s_trunk=s_trunk,
                            z_pair=z_pair, pair_mask=pair_mask,
                            x_pred_rep_coords=x_pred_rep_coords, **kw)
        finally:
            # RELEASED ON THE WAY OUT, including on an exception. Leaving a 67 MB tensor reachable
            # between poses is how a per-pose leak looks from the outside.
            _CALL["onehot"] = None

    def _prep_blocks(pair_mask=None, **kw):
        blocks = orig_prep(pair_mask=pair_mask, **kw)

        def wrap(i, fn):
            def run(s, z):
                oh = _CALL["onehot"]
                if oh is None:
                    raise _NoPose(
                        "STOP: a pairformer block ran without the pose embedding set. A is "
                        "installed but memory_efficient_forward did not run, so the stack was "
                        "reached by some other path and the pose would be silently absent.")
                proj = (head.pose_perblock_proj[0] if head._pose_perblock_tied
                        else head.pose_perblock_proj[i - 1])
                # ADD, THEN DROP THE REFERENCE, BEFORE THE BLOCK RUNS. Passing `z + delta` as an
                # argument while `delta` stays a live local pins an [N_token, N_token, c_z] tensor
                # for the whole block, forward and backward -- the same mistake review caught in
                # pathway C, where it was 5.5 GiB at the 3,400-token crop cap.
                z = z + proj(oh).to(z.dtype)
                return fn(s, z)
            return run

        # BLOCK 0 IS NOT WRAPPED: its injection is the pretrained linear_no_bias_d, already
        # applied before the stack.
        return [blocks[0]] + [wrap(i, b) for i, b in enumerate(blocks[1:], start=1)]

    head.memory_efficient_forward = memory_efficient_forward
    stack._prep_blocks = _prep_blocks
    head._pose_perblock_installed = True
    info = {"n_blocks": n_blocks, "injected": n_blocks - 1, "tied": bool(tied),
            "params": sum(p.weight.numel() for p in projs),
            "num_bins": int(head.num_bins), "c_z": int(head.c_z)}
    if loud:
        print(f"pose per-block: {info['injected']} block(s) injected "
              f"({'TIED, 1 matrix' if tied else f'untied, {n_new} matrices'}), "
              f"LinearNoBias({info['num_bins']} -> {info['c_z']}), {info['params']} parameter(s), "
              f"zero-init verified exactly zero; block 0 keeps linear_no_bias_d")
    return info
