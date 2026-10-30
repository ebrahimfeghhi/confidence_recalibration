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
_CALL = {"onehot": None, "dist": None}


class _NoPose(Exception):
    pass


def _add_dir(head, i, z, d):
    """Add this depth's own orientation projection to z, or return z unchanged.

    `d` IS PASSED IN, NOT READ FROM pose_direction._CALL. It is captured once per forward in
    _prep_blocks, for the same reason the onehot/dist tensors are: under activation checkpointing
    a block's forward is RE-RUN during backward, by which time the outer forward's `finally` has
    cleared the cache -- so a block that re-read the global saw None (raising from inside backward,
    far from the cause) or, worse, a LATER sample's geometry, pairing one pose's directions with
    another's gradient. A captured tensor is held by this closure, which the blocks list holds,
    which the checkpoint graph holds until its backward is done, so the recomputation sees exactly
    what the original forward saw.

    RETURNS z UNCHANGED ONLY WHEN THERE ARE NO COPIES, i.e. pose_direction was not installed.
    If copies exist the geometry MUST be there, and a missing capture raises instead of quietly
    dropping the orientation term for the rest of the stack -- an arm that trains
    distance-at-four-depths while its logs claim orientation-at-four-depths is the failure this
    guard exists to prevent, and it is the same class as the `--train-only` that created
    per-block copies and never trained them.
    """
    dirs = getattr(head, "pose_perblock_frame_dir", None)
    if not dirs or len(dirs) == 0:
        return z
    if d is None:
        raise RuntimeError(
            "STOP: per-block orientation copies are installed but no direction geometry was "
            "captured for this sample. The pre-stack injection sets pose_direction._CALL['dir'] "
            "and _prep_blocks captures it, so reaching a block without it means "
            "memory_efficient_forward was bypassed -- the orientation term would be silently "
            "absent from blocks 1..n while the run reported otherwise.")
    # ADD, THEN DROP, as everywhere else here: the delta is not held live across the block.
    z = z + dirs[i - 1](d).to(z.dtype)
    return z


def install(model, tied=False, share_pretrained=False, copy_pretrained=False,
            loud=True):
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
    _modes = [k for k, v in (("tied", tied), ("share_pretrained", share_pretrained),
                             ("copy_pretrained", copy_pretrained)) if v]
    if len(_modes) > 1:
        raise RuntimeError(
            f"STOP: {' and '.join(_modes)} are different experiments and at most one may be set. "
            f"share_pretrained reuses block 0's OWN pretrained injection at every depth and adds "
            f"no parameters (weights TIED forever). copy_pretrained gives every depth its own COPY "
            f"of that injection, so step 0 matches share_pretrained exactly and the copies are then "
            f"free to diverge. tied adds one new zero-init matrix shared across blocks 1..n-1.")

    # SHARE_PRETRAINED ADDS NO PARAMETERS AT ALL. Blocks 1..n-1 re-apply block 0's own
    # linear_no_bias_d AND linear_no_bias_d_wo_onehot -- the complete pose injection, both terms,
    # because block 0 adds both (confidence.py:286-305) and re-adding only the one-hot path would
    # be a different operator wearing the same name.
    #
    # STEP 0 IS NOT BIT-IDENTICAL HERE, and that is inherent rather than an oversight: blocks
    # 1..n-1 gain an injection they never had, using weights that are already non-zero. The
    # zero-init safety story belongs to the other two modes and the check below is skipped for
    # this one, loudly. A run in this mode is therefore NOT a one-variable change against the
    # shipped head at step 0 -- it is a different architecture evaluated from the same weights.
    # COPY_PRETRAINED: ONE COMPLETE COPY OF BLOCK 0's INJECTION PER DEPTH.
    #
    # Block 0 adds TWO terms -- linear_no_bias_d(one_hot) AND linear_no_bias_d_wo_onehot(distance)
    # (confidence.py:297-305) -- so a faithful copy needs both. Copying only the one-hot path, as
    # the zero-init untied mode's single matrix would suggest, is a different operator.
    #
    # STEP 0 IS BIT-IDENTICAL TO share_pretrained's STEP 0, not to the shipped head. That is the
    # point: it makes CAPACITY the only variable against a share_pretrained arm, where zero-init
    # untied would instead change both the starting function and the capacity at once. Verified
    # below against block 0's weights rather than against zero.
    copies_d = torch.nn.ModuleList()
    copies_wo = torch.nn.ModuleList()
    if copy_pretrained:
        for _ in range(n_blocks - 1):
            cd = LinearNoBias(in_features=head.num_bins, out_features=head.c_z)
            cw = LinearNoBias(in_features=1, out_features=head.c_z)
            with torch.no_grad():
                cd.weight.copy_(head.linear_no_bias_d.weight)
                cw.weight.copy_(head.linear_no_bias_d_wo_onehot.weight)
            copies_d.append(cd.to(device=ref.device, dtype=ref.dtype))
            copies_wo.append(cw.to(device=ref.device, dtype=ref.dtype))
    head.add_module("pose_perblock_linear_no_bias_d", copies_d)
    head.add_module("pose_perblock_linear_no_bias_d_wo_onehot", copies_wo)

    projs = torch.nn.ModuleList()
    if not share_pretrained and not copy_pretrained:
        n_new = 1 if tied else n_blocks - 1
        for _ in range(n_new):
            p = LinearNoBias(in_features=head.num_bins, out_features=head.c_z)
            with torch.no_grad():
                p.weight.zero_()
            projs.append(p.to(device=ref.device, dtype=ref.dtype))
    head.add_module("pose_perblock_proj", projs)

    # PER-BLOCK ORIENTATION, only when pose_direction is already installed. It installs first in
    # phase_k_train, so its presence is detectable here and its absence means there is no
    # geometry to project.
    #
    # ONE COPY PER EXTRA BLOCK, COPYING frame_dir_proj's WEIGHTS -- which are zero by
    # construction, since pose_direction zero-inits and verifies exactly zero. So these are zero
    # too and STEP 0 IS UNCHANGED: identical to the same arm without per-block orientation, and
    # therefore still identical to --pose-perblock-share's step 0 on the distance terms. Copied
    # rather than zeroed directly so that if frame_dir_proj ever gains a non-zero init, these
    # follow it instead of silently disagreeing.
    #
    # 768 parameters each at c_z 256, so three copies add 2,304 -- 5.6% on top of the 40,960 a
    # copy-mode distance arm already trains. A --train-only must match `frame_dir` to reach BOTH
    # these and frame_dir_proj, exactly as it must match `pose_perblock` for the distance copies.
    dirs = torch.nn.ModuleList()
    if getattr(head, "_pose_direction_installed", False):
        # ACTIVATION CHECKPOINTING USED TO BE REFUSED HERE. It is now supported, because the
        # cause was removed rather than worked around: _prep_blocks captures the direction
        # geometry into a local and _add_dir takes it as an argument, so a block recomputed
        # during backward uses the SAME tensor its original forward used instead of re-reading
        # a global the outer forward has since cleared. test_perblock_ckpt_equiv.py compares
        # forward outputs and the gradients of ALL FOUR orientation projections, with NON-ZERO
        # weights and several poses, between blocks_per_ckpt=1 and checkpointing off.
        #
        # THE ORIGINAL HAZARD, kept because it is the thing the test exists to catch:
        #
        # These copies read pose_direction's _CALL["dir"] cache, which the outer forward clears
        # in its `finally`. Under checkpointing a block's forward is RECOMPUTED during backward,
        # after that clear has run, so the cache is either None -- raising from inside backward,
        # where the message is far from the cause -- or, worse, holding a LATER sample's
        # directions, which would pair one pose's geometry with another's gradient. The distance
        # copies do not have this problem because pose_perblock's own onehot/dist cache is set
        # inside memory_efficient_forward, within the checkpointed region.
        #
        # Caught in review of 03f10456 as a future caution; made a refusal because the silent
        # variant -- wrong sample's directions -- is the failure this project has paid for most
        # often, and a latent hazard documented in a comment is one someone enables later.
        src = head.frame_dir_proj.weight
        for _ in range(n_blocks - 1):
            fd = LinearNoBias(in_features=3, out_features=head.c_z)
            with torch.no_grad():
                fd.weight.copy_(src)
            dirs.append(fd.to(device=ref.device, dtype=ref.dtype))
    head.add_module("pose_perblock_frame_dir", dirs)

    head._pose_perblock_tied = bool(tied)
    head._pose_perblock_share = bool(share_pretrained)
    head._pose_perblock_copy = bool(copy_pretrained)

    # VERIFIED, NOT ASSUMED. The safety story is that step 0 is bit-identical, which holds only if
    # every one of these returns exactly zero -- not merely something small.
    if projs:
        with torch.no_grad():
            probe = torch.randn(4, head.num_bins, device=ref.device, dtype=ref.dtype)
            worst = max(float(p(probe).abs().max()) for p in projs)
        if worst > 0:
            raise RuntimeError(f"STOP: a zero-init projection returned {worst:.3e}, not zero, so "
                               f"installing A would change the head at step 0.")

    # VERIFIED AGAINST BLOCK 0, NOT AGAINST ZERO. For the zero-init modes the safety story is
    # "step 0 is bit-identical to the shipped head" and the check above asserts exact zeros. For
    # copy_pretrained the story is "step 0 is bit-identical to SHARE mode", which holds only if
    # every copy equals block 0's weights exactly -- not merely closely. A copy that drifted by
    # one bit would make this arm's step 0 a third starting point and the comparison meaningless.
    if copy_pretrained:
        with torch.no_grad():
            worst_d = max(float((c.weight - head.linear_no_bias_d.weight).abs().max())
                          for c in copies_d)
            worst_w = max(float((c.weight - head.linear_no_bias_d_wo_onehot.weight).abs().max())
                          for c in copies_wo)
        if worst_d != 0.0 or worst_w != 0.0:
            raise RuntimeError(
                f"STOP: a copied injection differs from block 0's by {worst_d:.3e} (one-hot) / "
                f"{worst_w:.3e} (raw distance), so step 0 would NOT match share_pretrained and "
                f"this arm would differ from its control in the starting function as well as in "
                f"capacity.")

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
            # THE RAW DISTANCE TOO, for share_pretrained: block 0's injection is the one-hot path
            # PLUS linear_no_bias_d_wo_onehot(distance), and sharing the injection means sharing
            # both terms.
            _CALL["dist"] = d
        try:
            return orig_mef(input_feature_dict=input_feature_dict, s_trunk=s_trunk,
                            z_pair=z_pair, pair_mask=pair_mask,
                            x_pred_rep_coords=x_pred_rep_coords, **kw)
        finally:
            # RELEASED ON THE WAY OUT, including on an exception. Leaving a 67 MB tensor reachable
            # between poses is how a per-pose leak looks from the outside.
            _CALL["onehot"] = None
            _CALL["dist"] = None

    def _prep_blocks(pair_mask=None, **kw):
        blocks = orig_prep(pair_mask=pair_mask, **kw)
        # CAPTURED HERE, ONCE PER FORWARD, NOT READ FROM THE GLOBAL INSIDE THE BLOCK.
        #
        # With activation checkpointing -- which this project runs, --blocks-per-ckpt 1 -- a
        # block's forward is RE-RUN during backward. Reading _CALL["onehot"] inside the block
        # therefore read it twice: once in the real forward, and again in the recomputation, by
        # which time the `finally` in memory_efficient_forward had already set it to None. That
        # raised _NoPose on the backward pass. Review reproduced it: backward succeeded with
        # checkpointing off and failed with it on, and the old checker missed it because it ran
        # under no_grad with blocks_per_ckpt=None.
        #
        # Binding it to a local makes each closure hold its own reference, which the blocks list
        # holds, which checkpoint_blocks' graph holds until that graph's backward is done. Clearing
        # the dict slot afterwards is then harmless -- it drops the dict's reference, not this one.
        oh = _CALL["onehot"]
        dd = _CALL["dist"]
        # THE ORIENTATION GEOMETRY, captured on exactly the same contract as oh/dd above.
        # pose_direction sets _CALL["dir"] in its pre-stack injection, which runs BEFORE
        # confidence.memory_efficient_forward prepares these blocks, so it is populated here; it
        # is cleared by the outer forward's `finally`, which is why it cannot be read later.
        import pose_direction as _pd
        dr = _pd._CALL.get("dir")
        if oh is None:
            raise _NoPose(
                "STOP: the pair stack was prepared without the pose embedding set. A is installed "
                "but memory_efficient_forward did not run, so the stack was reached by some other "
                "path and the pose would be silently absent.")

        def wrap(i, fn):
            def run(s, z):
                if head._pose_perblock_share:
                    # BLOCK 0's OWN INJECTION, BOTH TERMS, re-applied verbatim. One matrix pair
                    # for the whole stack and zero new parameters.
                    z = z + head.linear_no_bias_d(oh).to(z.dtype)
                    z = z + head.linear_no_bias_d_wo_onehot(dd.unsqueeze(-1)).to(z.dtype)
                    z = _add_dir(head, i, z, dr)
                    return fn(s, z)
                if head._pose_perblock_copy:
                    # THIS DEPTH's OWN COPY, both terms. Identical to the share branch at step 0
                    # and free to diverge after. Same add-then-drop discipline as below: the
                    # delta is not kept live across the block's forward and backward.
                    z = z + head.pose_perblock_linear_no_bias_d[i - 1](oh).to(z.dtype)
                    z = z + head.pose_perblock_linear_no_bias_d_wo_onehot[i - 1](
                        dd.unsqueeze(-1)).to(z.dtype)
                    z = _add_dir(head, i, z, dr)
                    return fn(s, z)
                proj = (head.pose_perblock_proj[0] if head._pose_perblock_tied
                        else head.pose_perblock_proj[i - 1])
                # ADD, THEN DROP THE REFERENCE, BEFORE THE BLOCK RUNS. Passing `z + delta` as an
                # argument while `delta` stays a live local pins an [N_token, N_token, c_z] tensor
                # for the whole block, forward and backward -- the same mistake review caught in
                # pathway C, where it was 5.5 GiB at the 3,400-token crop cap.
                z = z + proj(oh).to(z.dtype)
                z = _add_dir(head, i, z, dr)
                return fn(s, z)
            return run

        # BLOCK 0 IS NOT WRAPPED: its injection is the pretrained linear_no_bias_d, already
        # applied before the stack.
        return [blocks[0]] + [wrap(i, b) for i, b in enumerate(blocks[1:], start=1)]

    head.memory_efficient_forward = memory_efficient_forward
    stack._prep_blocks = _prep_blocks
    head._pose_perblock_installed = True
    info = {"n_blocks": n_blocks, "injected": n_blocks - 1, "tied": bool(tied),
            "share_pretrained": bool(share_pretrained), "copy_pretrained": bool(copy_pretrained),
            # COUNTS THE ORIENTATION COPIES TOO, for the same reason the distance copies are
            # counted: a mode whose new parameters are not in `projs` must not report 0.
            "dir_copies": int(len(dirs)),
            "dir_params": int(sum(p.weight.numel() for p in dirs)),
            # COUNTS THE COPIES TOO. `projs` is empty in both share and copy mode, so summing
            # only it reported 0 parameters for an arm that adds 30,720 -- and the launcher's note
            # would have recorded the wrong capacity.
            "params": (sum(p.weight.numel() for p in projs)
                       + sum(c.weight.numel() for c in copies_d)
                       + sum(c.weight.numel() for c in copies_wo)
                       + sum(d_.weight.numel() for d_ in dirs)),
            "num_bins": int(head.num_bins), "c_z": int(head.c_z)}
    if loud:
        if dirs:
            print(f"pose per-block ORIENTATION: {len(dirs)} copy/copies of frame_dir_proj "
                  f"LinearNoBias(3 -> {head.c_z}), {info['dir_params']:,} new parameter(s), "
                  f"each copied from frame_dir_proj's weights -- which pose_direction zero-inits "
                  f"and verifies exactly zero, so STEP 0 IS UNCHANGED by these. Orientation now "
                  f"enters at ALL {n_blocks} depths; without them it enters ONCE, before the "
                  f"stack, while the distance terms enter at every depth. A --train-only must "
                  f"match `frame_dir` to reach them.")
        if share_pretrained:
            _shared = (head.linear_no_bias_d.weight.numel()
                       + head.linear_no_bias_d_wo_onehot.weight.numel())
            print(f"pose per-block: {info['injected']} block(s) injected, SHARING BLOCK 0's OWN "
                  f"pretrained injection (linear_no_bias_d {tuple(head.linear_no_bias_d.weight.shape)} "
                  f"+ linear_no_bias_d_wo_onehot "
                  f"{tuple(head.linear_no_bias_d_wo_onehot.weight.shape)} = {_shared:,} shared "
                  f"parameter(s)), ZERO new parameters. "
                  f"STEP 0 IS NOT BIT-IDENTICAL to the shipped head -- blocks 1-"
                  f"{n_blocks - 1} gain an injection they never had, from weights that are "
                  f"already non-zero. That is inherent to this mode, not a bug.")
        elif copy_pretrained:
            print(f"pose per-block: {info['injected']} block(s) injected, each with its OWN COPY "
                  f"of block 0's injection -- linear_no_bias_d "
                  f"{tuple(head.linear_no_bias_d.weight.shape)} + linear_no_bias_d_wo_onehot "
                  f"{tuple(head.linear_no_bias_d_wo_onehot.weight.shape)} per depth, "
                  f"{info['params']:,} NEW parameter(s). UNTIED: the copies are free to diverge. "
                  f"STEP 0 IS BIT-IDENTICAL TO --pose-perblock-share's step 0 (verified against "
                  f"block 0's weights, exactly), NOT to the shipped head -- so capacity is the "
                  f"only variable against a share arm. The copies are named "
                  f"pose_perblock_linear_no_bias_d* and a --train-only that does not match them "
                  f"would create them and never train them.")
        else:
            # n_new EXISTS ONLY IN THE ZERO-INIT BRANCHES. Reading it here raised NameError the
            # moment copy_pretrained was added -- the same shape as the `del it` defect in
            # phase_k_alignment_preflight.py. Guarded rather than assumed.
            print(f"pose per-block: {info['injected']} block(s) injected "
                  f"({'TIED, 1 matrix' if tied else f'untied, {len(projs)} matrices'}), "
                  f"LinearNoBias({info['num_bins']} -> {info['c_z']}), {info['params']} parameter(s), "
                  f"zero-init verified exactly zero; block 0 keeps linear_no_bias_d")
    return info
