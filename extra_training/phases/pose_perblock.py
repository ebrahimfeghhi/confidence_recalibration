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

THE AXES, NOT MODES. install() takes `terms` (which of onehot / dist / orient each site carries),
`init` (zero or a copy of block 0's pretrained injection) and `weights` (untied / tied / shared),
and the forward branches on WHICH TERMS ARE PRESENT. It used to branch on head._pose_perblock_copy
-- an init-named flag deciding the structural question of whether the raw-distance term is added
-- which is why "both terms at zero init" could not be asked for. phases/pose_axes.py holds the
axes, the refusals and the legacy-flag desugaring; see its header for the table.

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

import pose_axes as _PA

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


def _add_mlp(head, i, z, h):
    """Add THIS depth's own second layer of the nonlinear branch to z, or return z unchanged.

    `h` IS PASSED IN, NOT READ FROM pose_mlp._CALL, for exactly the reason `d` is in _add_dir
    above -- and here it matters more. Under activation checkpointing this block's forward is
    RE-RUN during backward, by which time the outer wrapper's `finally` has cleared the cache, so
    a block that re-read the global would see None or a LATER pose's hidden tensor. On top of
    that, `h` is not geometry: it is an activation of the branch's trainable first layer, and it
    is the ONLY path by which that layer receives a gradient. Capturing it in _prep_blocks puts it
    in the closure the checkpoint graph holds, so the recomputation sees the same tensor the
    original forward saw and the gradient reaches layer 1 from all four depths.

    RETURNS z UNCHANGED ONLY WHEN pose_mlp IS NOT INSTALLED. If it is, a missing `h` raises from
    pose_mlp.add_site rather than quietly dropping the branch for the rest of the stack.
    """
    if not getattr(head, "_pose_mlp_installed", False):
        return z
    import pose_mlp
    # ADD, THEN DROP, as everywhere else here: the delta is not held live across the block.
    return pose_mlp.add_site(head, i, z, h)


def _mk(in_features, out_features, ref, src=None):
    """One LinearNoBias on ref's device/dtype: zero-initialised, or a copy of `src`."""
    m = LinearNoBias(in_features=in_features, out_features=out_features)
    with torch.no_grad():
        if src is None:
            m.weight.zero_()
        else:
            m.weight.copy_(src)
    return m.to(device=ref.device, dtype=ref.dtype)


def onehot_module_name(terms):
    """The attribute the one-hot projections live under -- A FUNCTION OF THE TERMS ALONE.

    THE NAMES ARE STRUCTURE, SO THEY FOLLOW THE STRUCTURAL AXIS. A one-hot-only arm's matrix has
    always been `pose_perblock_proj` and a both-terms arm's has always been
    `pose_perblock_linear_no_bias_d` (named after the pretrained layer it mirrors, beside
    `..._wo_onehot`), so keying the choice on the term set reproduces every existing checkpoint's
    key set exactly -- plain and tied arms keep `proj`, copy arms keep `linear_no_bias_d` -- while
    the NEW onehot+dist-at-zero arm gets the both-terms names it structurally is.

    Keying it on the init, which is where the old code effectively had it, would mean a tensor's
    NAME depended on its starting VALUE: the same defect as the forward branching on
    _pose_perblock_copy, one layer down.
    """
    return ("pose_perblock_proj" if tuple(t for t in ("onehot", "dist") if t in terms) == ("onehot",)
            else "pose_perblock_linear_no_bias_d")


def install(model, terms=None, init="zero", weights="untied", loud=True):
    """Add the per-block projections and route the pose embedding into every block but the first.

    THE AXES, NOT MODES (phases/pose_axes.py):
      terms    which projections each injected block carries, a subset of
               {"onehot", "dist", "orient"}. None means the historical default of the plain
               --pose-perblock flag: {"onehot"}, plus "orient" when pose_direction is installed.
      init     "zero" (step 0 bit-identical to the head without this, VERIFIED exactly zero) or
               "pretrained" (each depth starts as a copy of block 0's own injection, VERIFIED
               equal to it exactly). Applies to the distance terms; the orientation copies always
               copy frame_dir_proj, which pose_direction zero-inits and verifies.
      weights  "untied" (one matrix per depth), "tied" (one matrix for blocks 1..n-1) or "shared"
               (block 0's OWN tensors re-applied, zero new parameters). Governs the DISTANCE
               terms only -- orientation is one copy per depth in every mode, as it always was.

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
    has_dir = bool(getattr(head, "_pose_direction_installed", False))
    if terms is None:
        # THE PLAIN FLAG's HISTORICAL MEANING, kept so a caller that passes no axes at all gets
        # exactly the arm --pose-perblock always built. Every caller in the live path goes through
        # pose_axes.resolve and passes the terms explicitly.
        terms = ("onehot",) + (("orient",) if has_dir else ())
    # REFUSED HERE TOO, NOT ONLY AT THE CLI. phase_k_eval, the checkers and the tests call
    # install() directly, so the structural rules live with the structure.
    terms = _PA.check_block_axes(terms, init, weights, has_dir)
    terms = frozenset(terms)
    n_new = 1 if weights == "tied" else n_blocks - 1

    # SHARED ADDS NO PARAMETERS AT ALL. Blocks 1..n-1 re-apply block 0's own linear_no_bias_d AND
    # linear_no_bias_d_wo_onehot -- the complete pose injection, both terms, because block 0 adds
    # both (confidence.py:286-305) and re-adding only the one-hot path would be a different
    # operator wearing the same name. pose_axes refuses any other term set for this mode.
    #
    # STEP 0 IS NOT BIT-IDENTICAL HERE, and that is inherent rather than an oversight: blocks
    # 1..n-1 gain an injection they never had, using weights that are already non-zero. The
    # zero-init safety story belongs to the other modes and the check below is skipped for this
    # one, loudly. A run in this mode is therefore NOT a one-variable change against the shipped
    # head at step 0 -- it is a different architecture evaluated from the same weights.
    #
    # init "pretrained": ONE COMPLETE COPY PER DEPTH OF WHICHEVER TERMS THE ARM CARRIES.
    #
    # With terms onehot+dist that is block 0's whole injection, and STEP 0 IS BIT-IDENTICAL TO
    # THE SHARED MODE's STEP 0, not to the shipped head. That is the point: it makes CAPACITY the
    # only variable against a shared arm, where zero-init would change both the starting function
    # and the capacity at once. Verified below against block 0's weights rather than against zero.
    #
    # AND THE CONVERSE IS NOW EXPRESSIBLE: terms onehot+dist at init zero, which makes the
    # INITIALISATION the only variable against a copy arm. That configuration is the reason this
    # module was rewritten; it was unreachable while the forward keyed the both-terms path off an
    # init-named flag.
    onehot_name = onehot_module_name(terms)
    lists = {}
    for name, present, n_in, src in (
            (onehot_name, "onehot" in terms, head.num_bins, head.linear_no_bias_d.weight),
            ("pose_perblock_linear_no_bias_d_wo_onehot", "dist" in terms, 1,
             head.linear_no_bias_d_wo_onehot.weight)):
        ml = torch.nn.ModuleList()
        if present and weights != "shared":
            for _ in range(n_new):
                ml.append(_mk(n_in, head.c_z, ref, src if init == "pretrained" else None))
        lists[name] = ml
    # EVERY NAME IS REGISTERED, POPULATED OR NOT, exactly as before: an empty ModuleList
    # contributes no state_dict keys, so a configuration that does not use one is byte-identical
    # in the checkpoint to code that never created it -- while `getattr` in the forward and in
    # every helper that counts tensors keeps working without a hasattr dance.
    for name in ("pose_perblock_proj", "pose_perblock_linear_no_bias_d",
                 "pose_perblock_linear_no_bias_d_wo_onehot"):
        head.add_module(name, lists.get(name, torch.nn.ModuleList()))
    copies_wo = lists["pose_perblock_linear_no_bias_d_wo_onehot"]

    # PER-BLOCK ORIENTATION, when the term list asks for it. pose_direction installs first in
    # phase_k_train, so its presence is detectable here, and pose_axes refuses `orient` without
    # it because there would be no geometry to project.
    #
    # ASKED FOR, NOT INFERRED. This used to be installed whenever pose_direction was installed,
    # which fused the orientation TERM to the --frame-direction flag: the pre-stack orientation
    # site could not be run without also putting orientation at every depth. The legacy flags
    # desugar to exactly the old behaviour (pose_axes.legacy_block_axes adds `orient` whenever
    # --frame-direction is on), so no existing arm changes.
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
    if "orient" in terms:
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
            dirs.append(_mk(3, head.c_z, ref, src))
    head.add_module("pose_perblock_frame_dir", dirs)

    head._pose_perblock_terms = terms
    head._pose_perblock_init = str(init)
    head._pose_perblock_weights = str(weights)
    # THE LEGACY ATTRIBUTES, DERIVED AND KEPT FOR READERS, NOT FOR THE FORWARD. The forward below
    # branches on the TERMS and on the sharing mode; nothing in it reads these. They stay because
    # other modules and the logs name the old modes, and because `_pose_perblock_copy` being
    # false while the arm carries both terms at zero init is exactly the distinction the old
    # surface could not draw.
    head._pose_perblock_tied = weights == "tied"
    head._pose_perblock_share = weights == "shared"
    head._pose_perblock_copy = (weights == "untied" and init == "pretrained"
                                and {"onehot", "dist"} <= terms)

    # VERIFIED, NOT ASSUMED, AND FOR EVERY TERM. The safety story is that step 0 is bit-identical,
    # which holds only if every zero-init projection returns exactly zero -- not merely something
    # small. Probed through the module rather than read off .weight, so a non-zero bias or a
    # future wrapper cannot pass this check while changing the head.
    #
    # EXTENDED TO THE RAW-DISTANCE ARM, which the old code never zero-initialised at all: the
    # only both-terms mode was the pretrained one, so the zero probe covered a single list. The
    # new onehot+dist-at-zero arm has two, and an unverified one would be exactly the silent
    # difference the probe exists to catch.
    if init == "zero":
        zero_lists = [(n, ml, (head.num_bins if n != "pose_perblock_linear_no_bias_d_wo_onehot"
                               else 1))
                      for n, ml in lists.items() if len(ml)]
        with torch.no_grad():
            for name, ml, n_in in zero_lists:
                probe = torch.randn(4, n_in, device=ref.device, dtype=ref.dtype)
                worst = max(float(p(probe).abs().max()) for p in ml)
                if worst > 0:
                    raise RuntimeError(
                        f"STOP: a zero-init projection in {name} returned {worst:.3e}, not zero, "
                        f"so installing A would change the head at step 0.")

    # VERIFIED AGAINST BLOCK 0, NOT AGAINST ZERO. For the zero-init arms the safety story is
    # "step 0 is bit-identical to the shipped head" and the check above asserts exact zeros. For
    # the pretrained init the story is "step 0 is bit-identical to the SHARED mode", which holds
    # only if every copy equals block 0's weights exactly -- not merely closely. A copy that
    # drifted by one bit would make this arm's step 0 a third starting point and the comparison
    # meaningless.
    if init == "pretrained":
        with torch.no_grad():
            worst_d = max([float((c.weight - head.linear_no_bias_d.weight).abs().max())
                           for c in lists[onehot_name]] or [0.0])
            worst_w = max([float((c.weight - head.linear_no_bias_d_wo_onehot.weight).abs().max())
                           for c in copies_wo] or [0.0])
        if worst_d != 0.0 or worst_w != 0.0:
            raise RuntimeError(
                f"STOP: a copied injection differs from block 0's by {worst_d:.3e} (one-hot) / "
                f"{worst_w:.3e} (raw distance), so step 0 would NOT match the shared mode and "
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
        # THE NONLINEAR BRANCH'S HIDDEN TENSOR, captured on the same contract and for a stronger
        # reason: unlike oh/dd/dr it is not geometry but an activation of trainable weights, so
        # the capture is what carries layer 1's gradient out of the checkpointed region. pose_mlp
        # builds it at the pre-stack site, which runs before this.
        import pose_mlp as _pm
        hh = _pm._CALL.get("h")
        if oh is None:
            raise _NoPose(
                "STOP: the pair stack was prepared without the pose embedding set. A is installed "
                "but memory_efficient_forward did not run, so the stack was reached by some other "
                "path and the pose would be silently absent.")

        # WHICH TERMS, AND WHICH TENSOR CARRIES EACH, DECIDED ONCE PER FORWARD rather than per
        # block. `shared` is the one mode whose tensors are not this module's: it re-applies
        # block 0's own pair, which is why it is a separate pair of entries rather than a flag the
        # inner loop reads.
        if weights == "shared":
            adds = [(head.linear_no_bias_d, False), (head.linear_no_bias_d_wo_onehot, True)]
        else:
            adds = ([(lists[onehot_name], False)] if "onehot" in terms else []) + \
                   ([(copies_wo, True)] if "dist" in terms else [])
        tied = weights == "tied"

        def wrap(i, fn):
            def run(s, z):
                # BRANCHED ON THE TERMS, NOT ON THE INIT. The old code decided whether to add the
                # raw-distance term by reading head._pose_perblock_copy -- an initialisation flag
                # answering a structural question, which is why "both terms at zero init" was
                # unaskable. Each present term adds once, through this depth's own tensor (or the
                # single tied one, or block 0's own in the shared mode).
                #
                # ADD, THEN DROP THE REFERENCE, BEFORE THE BLOCK RUNS. Passing `z + delta` as an
                # argument while `delta` stays a live local pins an [N_token, N_token, c_z] tensor
                # for the whole block, forward and backward -- the same mistake review caught in
                # pathway C, where it was 5.5 GiB at the 3,400-token crop cap.
                for mod, is_dist in adds:
                    m = mod if weights == "shared" else mod[0 if tied else i - 1]
                    z = z + m(dd.unsqueeze(-1) if is_dist else oh).to(z.dtype)
                z = _add_dir(head, i, z, dr)
                z = _add_mlp(head, i, z, hh)
                return fn(s, z)
            return run

        # BLOCK 0 IS NOT WRAPPED: its injection is the pretrained linear_no_bias_d, already
        # applied before the stack.
        return [blocks[0]] + [wrap(i, b) for i, b in enumerate(blocks[1:], start=1)]

    head.memory_efficient_forward = memory_efficient_forward
    stack._prep_blocks = _prep_blocks
    head._pose_perblock_installed = True
    _names = [f"{n}.{k}.weight" for n, ml in
              list(lists.items()) + [("pose_perblock_frame_dir", dirs)]
              for k in range(len(ml))]
    info = {"n_blocks": n_blocks, "injected": n_blocks - 1,
            # THE AXES, which are what the arm IS.
            "terms": tuple(sorted(terms, key=_PA.TERMS.index)), "init": str(init),
            "weights": str(weights), "tensors": _names, "n_tensors": len(_names),
            # AND THE LEGACY MODE NAMES, derived, because the launchers and the notes read them.
            "tied": weights == "tied", "share_pretrained": weights == "shared",
            "copy_pretrained": bool(head._pose_perblock_copy),
            # COUNTS THE ORIENTATION COPIES TOO, for the same reason the distance copies are
            # counted: a mode whose new parameters are not in `projs` must not report 0.
            "dir_copies": int(len(dirs)),
            "dir_params": int(sum(p.weight.numel() for p in dirs)),
            # COUNTS THE COPIES TOO. `projs` is empty in both share and copy mode, so summing
            # only it reported 0 parameters for an arm that adds 30,720 -- and the launcher's note
            # would have recorded the wrong capacity.
            "params": (sum(p.weight.numel() for ml in lists.values() for p in ml)
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
        if weights == "shared":
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
        else:
            # ONE LINE FOR EVERY POINT IN THE SPACE, reading the axes rather than naming a mode:
            # the old three-branch print had a branch per mode and said nothing about a
            # combination it did not anticipate (and read `n_new`, which existed only in two of
            # them -- a NameError the moment a mode was added, the same shape as the `del it`
            # defect in phase_k_alignment_preflight.py).
            _terms = "+".join(info["terms"])
            print(f"pose per-block: {info['injected']} block(s) injected, "
                  f"TERMS {_terms}, INIT {init}, WEIGHTS {weights} "
                  f"({len(_names)} tensor(s), {info['params']:,} parameter(s)): "
                  + ", ".join(f"{n} x{len(ml)} LinearNoBias("
                              f"{1 if n.endswith('_wo_onehot') else info['num_bins']} -> "
                              f"{info['c_z']})"
                              for n, ml in lists.items() if len(ml))
                  + (f"; plus {len(dirs)} orientation copy/copies" if dirs else "")
                  + ". "
                  + ("STEP 0 IS BIT-IDENTICAL to the head without this -- every new projection "
                     "verified exactly zero through the module, not read off .weight."
                     if init == "zero" else
                     "STEP 0 IS BIT-IDENTICAL TO THE SHARED MODE's step 0 (every copy verified "
                     "equal to block 0's weights exactly), NOT to the shipped head -- so against "
                     "a shared arm the only variable is capacity.")
                  + " Block 0 keeps its own pre-stack linear_no_bias_d. A --train-only must match "
                    "`pose_perblock_` to reach these and `frame_dir` to reach the orientation "
                    "copies; one that does not creates them and never trains them.")
    return info
