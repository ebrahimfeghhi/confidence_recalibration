"""A third pose pathway into the confidence head: frame-relative DIRECTION.

See docs/pose_pathways.md for the argument. In short: the head's only view of the predicted pose
is `torch.cdist` over ONE representative atom per token -- a CA-only distance matrix. A residue's
backbone frame is defined by its N, CA and C atoms, and N and C are not in that matrix, so the
frame's orientation is absent from the head's input rather than merely implicit. PAE is defined
on exactly that orientation: superpose on token i's frame, then ask how far token j ended up from
where it should be.

This module adds, for each ordered pair (i, j), the UNIT direction of j in i's frame -- three
numbers -- through one zero-initialised LinearNoBias(3 -> c_z) added into `z_pair` alongside the
two distance paths that are already there.

Three decisions worth stating, all from review:

- THE UNIT VECTOR, NOT THE RAW DISPLACEMENT. Zero-init makes the layer's OUTPUT zero at step 0.
  It does NOT make the first UPDATE small: the gradient on the weight is the output error times
  the input, so raw Angstrom components of +/-30 give an update ~30x larger than inputs of order
  1, and the weight jumps rather than easing off zero. Normalising to a unit vector bounds the
  input to [-1, 1] with no scale constant to choose, and the magnitude is not lost -- the two
  existing distance paths already supply it.
- IT DOES NOT REPLACE ANYTHING. `linear_no_bias_d` and `linear_no_bias_d_wo_onehot` keep their
  trained weights. This is a third additive term.
- THE PATCH IS NOT A NO-OP BY ASSUMPTION. `install()` checks that the projection's output is
  exactly zero at install time, so "step 0 is bit-identical" is verified rather than asserted.

Installed on the MODEL INSTANCE, not by editing Protenix: /home/ebrahim/Protenix is a separate
checkout and a change there would not be in this repo's history.
"""
import torch

from protenix.model.modules.frames import (
    expressCoordinatesInFrame,
    gather_frame_atom_by_indices,
)
from protenix.model.modules.primitives import LinearNoBias

# The per-unit frame definition, set by the caller before that unit's poses run. Module-level
# rather than threaded through run_confidence_head's signature, matching how the PAE chain-pair
# mask is already installed in phase_k_train.
#
# `frame_atom_index` and `pae_rep_atom_mask` are NOT in the head's own feature dict -- the trunk
# cache carries has_frame but neither of the other two (phase_j_rescore_cache.load_unit builds six
# keys and they are not among them). They live in the LOSS features, as lf_frame_atom_index and
# lf_pae_rep_atom_mask in the label npz, so the caller has to hand them over. Under cropping they
# must be the CROPPED copies, whose indices are atom indices after the crop; crop_feat_loss
# carries exactly that.
_UNIT = {"frame_atom_index": None, "has_frame": None, "pae_rep_atom_mask": None, "pdb": ""}

# Set by the patched forward for the patched memory_efficient_forward to read. `forward` holds the
# full atom coordinates and loops over samples; memory_efficient_forward receives only the
# representative-atom subset of one sample, and the frames need N and C, which are not in it.
_CALL = {"coords": None, "i": 0}

# Length below which a displacement has no direction.
#
# THE TARGET ATOMS ARE THE PAE REPRESENTATIVE ATOMS, NOT THE DISTOGRAM ONES. Protenix has two
# per-token representative atoms and they are different atoms: `distogram_rep_atom_mask` is CB
# (CA for glycine) per AF3 SI 4.4, while `pae_rep_atom_mask` is the token CENTRE atom, CA for
# standard amino acids. They differ on every non-glycine residue -- 1158 of 10fj's atoms.
# `memory_efficient_forward` is handed the DISTOGRAM subset, because that is what cdist wants, but
# the PAE label is built on the CENTRE atoms, so that is what this feeds. Using the distogram
# subset would hand the head directions to a different atom than the label is defined on.
#
# It also makes the self-entry exactly zero: the frame origin is the middle frame atom, CA, and
# the target for j == i is also CA, so the displacement is the zero vector. With the distogram
# subset it was the CA->CB direction, a near-constant local vector carrying no pose information,
# which the self-check caught. The two existing distance paths contribute zero on the diagonal
# too -- one_hot's first lower edge is 3.25 A so a distance of 0 one-hots to all zeros, and
# LinearNoBias has no bias -- so a zero diagonal is also what the head already sees there.
EPS_LEN = 1e-6

# A frame is degenerate when the basis construction in expressCoordinatesInFrame cannot produce
# an orthonormal triple: it builds w1 = norm(a - b), w2 = norm(c - b), e1 = norm(w1 + w2),
# e2 = norm(w2 - w1), e3 = e1 x e2. So it fails when a == b or c == b (zero-length w), when
# w1 == w2 (e2 undefined), or when w1 == -w2 (e1 undefined). Checked against the construction
# rather than by a single collinearity proxy, because each of those three fails differently.
# featurizer.py:206 also writes frame_atom_index = [-1, b_idx, -1] when it cannot build a frame
# at all; has_frame should already be false there, and the index check below does not rely on it.
EPS_BASIS = 1e-4


def set_unit(frame_atom_index, has_frame, pae_rep_atom_mask, pdb=""):
    """Install the frame definition for the unit whose poses are about to run.

    All three arrays must match the coordinates the head will be given: under cropping, the
    cropped copies. `frame_atom_index` and `pae_rep_atom_mask` index ATOMS; `has_frame` is
    per-token.
    """
    if frame_atom_index is None or has_frame is None or pae_rep_atom_mask is None:
        raise ValueError("set_unit needs frame_atom_index, has_frame and pae_rep_atom_mask")
    _UNIT["frame_atom_index"] = torch.as_tensor(frame_atom_index).long()
    _UNIT["has_frame"] = torch.as_tensor(has_frame).bool()
    _UNIT["pae_rep_atom_mask"] = torch.as_tensor(pae_rep_atom_mask).bool()
    _UNIT["pdb"] = pdb


def clear_unit():
    _UNIT["frame_atom_index"] = _UNIT["has_frame"] = None
    _UNIT["pae_rep_atom_mask"] = None
    _UNIT["pdb"] = ""


def derive_frames(feat):
    """(frame_atom_index, pae_rep_atom_mask) from what the TRUNK already carries.

    PROVEN, NOT ASSUMED. Verified exact against the label npz on all 500 training structures:
    every `lf_frame_atom_index` row for a framed token and every `lf_pae_rep_atom_mask` entry
    reproduced with zero mismatches. The derivation is simply that a token's atoms carry a
    canonical within-token index -- `atom_to_tokatom_idx` -- so for a standard residue 0/1/2 are
    N/CA/C, which are exactly the frame atoms, and index 1 (CA) is the token centre the PAE label
    is built on.

    WHY THIS EXISTS. The evaluator used to read these from the label npz, and validation/fold has
    53 trunks and NO labels, so --frame-direction could train but never be scored on held-out
    structures -- the only measurement that matters. Deriving from the trunk removes the
    dependency for every fold rather than backfilling sidecar files for one.

    `has_frame` needs no derivation: the trunk carries it as ci_feat_has_frame, which
    phase_j_rescore_cache.load_unit exposes as feat["has_frame"].
    """
    tok = torch.as_tensor(feat["atom_to_token_idx"]).long().reshape(-1)
    tat = torch.as_tensor(feat["atom_to_tokatom_idx"]).long().reshape(-1)
    n_atom = int(tok.shape[0])
    n_token = int(torch.as_tensor(feat["has_frame"]).reshape(-1).shape[0])
    # ALLOCATED ON THE INDICES' OWN DEVICE. phase_j_rescore_cache.load_unit puts the features on
    # the run's cuda device, so a CPU `torch.full` here was indexed with cuda indices on the very
    # next line -- which raises, and would have raised on C's FIRST validation pass. The earlier
    # test missed it because it loaded the unit onto torch.device("cpu"), where the mismatch
    # cannot occur. Found in review.
    dev = tok.device
    fai = torch.full((n_token, 3), -1, dtype=torch.long, device=dev)
    sel = tat < 3
    fai[tok[sel], tat[sel]] = torch.nonzero(sel, as_tuple=True)[0]
    pae = torch.zeros(n_atom, dtype=torch.bool, device=dev)
    pae[tat == 1] = True
    return fai, pae


def direction_tensor(rep_coords, full_coords, frame_atom_index, has_frame):
    """[N_token, N_token, 3]: the unit direction of token j in token i's frame.

    Pure function of coordinates and the frame definition, so the self-test can call it without a
    model. Row i is zero for every token with no usable frame, and entry [i][i] is zero whenever
    the target atom is the frame origin, which holds for standard amino acids when `rep_coords`
    are the PAE representative (centre) atoms.

    Args:
        rep_coords: [N_token, 3] the PAE representative atom of each token -- the CENTRE atom the
            label is built on, not the distogram CB
        full_coords: [N_atom, 3] every atom, which is where the frame atoms come from
        frame_atom_index: [N_token, 3] atom indices of each token's three frame atoms
        has_frame: [N_token] whether that token has a frame
    """
    if rep_coords.ndim != 2 or full_coords.ndim != 2:
        raise ValueError(f"expected [N, 3] tensors, got rep {tuple(rep_coords.shape)} "
                         f"full {tuple(full_coords.shape)}")
    n_token = rep_coords.shape[0]
    if frame_atom_index.shape[0] != n_token or has_frame.shape[0] != n_token:
        raise ValueError(f"frame arrays are {tuple(frame_atom_index.shape)} / "
                         f"{tuple(has_frame.shape)} for {n_token} token(s)")

    rep = rep_coords.to(torch.float32)
    full = full_coords.to(torch.float32)
    fai = frame_atom_index.to(rep.device).long()
    hf = has_frame.to(rep.device).bool()

    # A row is a candidate only if it claims a frame AND its three indices are real. The second
    # half is not redundant: the -1 placeholder would index the last atom instead of failing.
    cand = hf & (fai >= 0).all(dim=-1) & (fai < full.shape[0]).all(dim=-1)
    out = rep.new_zeros((n_token, n_token, 3))
    if not bool(cand.any()):
        return out

    frames = gather_frame_atom_by_indices(coordinate=full, frame_atom_index=fai[cand], dim=-2)
    a, b, c = torch.unbind(frames, dim=-2)              # [N_frame, 3] each
    w1, w2 = a - b, c - b
    n1 = torch.linalg.vector_norm(w1, dim=-1)
    n2 = torch.linalg.vector_norm(w2, dim=-1)
    u1 = w1 / n1.clamp_min(EPS_BASIS)[..., None]
    u2 = w2 / n2.clamp_min(EPS_BASIS)[..., None]
    ok = ((n1 > EPS_BASIS) & (n2 > EPS_BASIS)
          & (torch.linalg.vector_norm(u1 + u2, dim=-1) > EPS_BASIS)      # e1 undefined
          & (torch.linalg.vector_norm(u2 - u1, dim=-1) > EPS_BASIS))     # e2 undefined

    vec = expressCoordinatesInFrame(rep, frames)        # [N_frame, N_token, 3]
    ln = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    unit = torch.where(ln > EPS_LEN, vec / ln.clamp_min(EPS_LEN), torch.zeros_like(vec))
    unit = unit * ok[:, None, None].to(unit.dtype)
    out[cand] = unit
    return out


def install(model, loud=True):
    """Add the projection to the confidence head and route the direction into z_pair.

    The projection is added as a submodule of `confidence_head`, so phase_k_train's
    head_param_names -- which reads the module tree rather than a hand-written list -- picks it up
    as trainable with no change there.
    """
    head = model.confidence_head
    if getattr(head, "_pose_direction_installed", False):
        return {"already": True}

    ref = head.linear_no_bias_d.weight
    proj = LinearNoBias(in_features=3, out_features=head.c_z)
    with torch.no_grad():
        proj.weight.zero_()
    proj.to(device=ref.device, dtype=ref.dtype)
    head.add_module("frame_dir_proj", proj)

    # VERIFIED, NOT ASSUMED. The whole safety story is that step 0 is bit-identical, which holds
    # only if this output is exactly zero -- not merely small.
    with torch.no_grad():
        probe = torch.randn(4, 3, device=ref.device, dtype=ref.dtype)
        got = proj(probe)
    if bool(got.abs().max() > 0):
        raise RuntimeError(f"STOP: the zero-init projection returned {float(got.abs().max()):.3e}, "
                           f"not zero, so installing it would change the head at step 0.")

    orig_forward = head.forward
    orig_mef = head.memory_efficient_forward

    def forward(input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, **kw):
        # THE FULL ATOM COORDINATES, STASHED FOR THE INNER CALL. memory_efficient_forward is given
        # only x_pred_coords[..., rep_atom_mask, :] for one sample, and the frame atoms (N and C)
        # are not in that subset, so the inner call cannot build frames from what it receives.
        _CALL["coords"], _CALL["i"] = x_pred_coords, 0
        try:
            return orig_forward(input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                                s_trunk=s_trunk, z_trunk=z_trunk, pair_mask=pair_mask,
                                x_pred_coords=x_pred_coords, **kw)
        finally:
            _CALL["coords"], _CALL["i"] = None, 0

    def memory_efficient_forward(input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        # ADD, THEN DROP THE REFERENCE, BEFORE THE ORIGINAL HEAD RUNS. The first version passed
        # `z_pair + delta` as an argument while `delta` stayed a live local in this frame, so a
        # [N_token, N_token, c_z] tensor was pinned for the whole of the pairformer stack's
        # forward AND backward: 210 MiB at 656 tokens and 5.5 GiB at the 3,400-token crop cap,
        # on top of the activations the stack already needs. Found in review.
        z = z_pair + _delta(head, input_feature_dict, x_pred_rep_coords)
        return orig_mef(input_feature_dict=input_feature_dict, s_trunk=s_trunk,
                        z_pair=z, pair_mask=pair_mask,
                        x_pred_rep_coords=x_pred_rep_coords, **kw)

    head.forward = forward
    head.memory_efficient_forward = memory_efficient_forward
    head._pose_direction_installed = True
    info = {"c_z": int(head.c_z), "params": int(proj.weight.numel()),
            "device": str(ref.device), "dtype": str(ref.dtype)}
    if loud:
        print(f"pose direction: frame_dir_proj LinearNoBias(3 -> {info['c_z']}) installed on "
              f"confidence_head, {info['params']} parameter(s), zero-init verified exactly zero")
    return info


def _delta(head, feat, x_pred_rep_coords):
    """The additive z_pair term for the sample memory_efficient_forward was just handed."""
    if _UNIT["frame_atom_index"] is None:
        raise RuntimeError("STOP: pose_direction is installed but no unit frames are set. Call "
                           "pose_direction.set_unit(frame_atom_index, has_frame) before the "
                           "unit's poses run -- running without it would silently train a "
                           "different function than the one being evaluated.")
    full = _CALL["coords"]
    if full is None:
        raise RuntimeError("STOP: memory_efficient_forward was reached without the patched "
                           "forward having run, so the full atom coordinates are not available.")
    i = _CALL["i"]
    _CALL["i"] = i + 1

    if full.ndim != 3:
        raise RuntimeError(f"STOP: x_pred_coords is {tuple(full.shape)}; this patch handles "
                           f"[N_sample, N_atom, 3] only, and a batch dimension would make the "
                           f"sample index below wrong.")
    if i >= full.shape[0]:
        raise RuntimeError(f"STOP: sample index {i} past the {full.shape[0]} sample(s) stashed; "
                           f"forward called memory_efficient_forward more times than it has "
                           f"samples, so the index is not tracking the loop.")
    full_i = full[i]

    rep_mask = feat["distogram_rep_atom_mask"].bool()
    pae_mask = _UNIT["pae_rep_atom_mask"].to(full_i.device)
    # THE INDEX ASSUMPTION, CHECKED EVERY CALL. `forward` loops i in range(N_sample) and slices
    # x_pred_rep_coords[..., i, :, :}; this reconstructs that slice and compares. If upstream ever
    # reorders or parallelises the loop, this fires instead of silently pairing one sample's
    # directions with another sample's distances -- which is exactly the class of bug that paired
    # a pose with another molecule's label earlier in this project.
    expect = full_i[rep_mask]
    if expect.shape != x_pred_rep_coords.shape or not torch.equal(
            expect.to(x_pred_rep_coords.dtype), x_pred_rep_coords):
        raise RuntimeError(f"STOP: sample {i}'s representative atoms do not match the tensor "
                           f"memory_efficient_forward was given ({tuple(expect.shape)} vs "
                           f"{tuple(x_pred_rep_coords.shape)}); the sample index is not tracking "
                           f"forward's loop.")

    # FLOAT32 AND NO AUTOCAST FOR THE GEOMETRY ONLY. The basis is built from normalised cross
    # products and a bf16 norm of a near-degenerate frame is not reliable; the upstream code
    # disables autocast around its own cdist for the same reason. The geometry tensor is
    # [N_token, N_token, 3] -- 5 MiB at 656 tokens, 132 MiB at 3,400 -- so float32 is cheap there.
    #
    # THE PROJECTION IS OUTSIDE THAT BLOCK, so it runs under the caller's autocast and returns
    # bf16 rather than a float32 [N_token, N_token, c_z]: 110 MiB instead of 210 at 656 tokens,
    # and it matches the dtype z_pair is already in.
    with torch.amp.autocast("cuda", enabled=False):
        # THE GEOMETRY USES THE PAE REPRESENTATIVE ATOMS, selected from the full coordinates --
        # NOT x_pred_rep_coords, which is the distogram (CB) subset. See the EPS_LEN note.
        full_f = full_i.to(torch.float32)
        d = direction_tensor(full_f[pae_mask], full_f,
                             _UNIT["frame_atom_index"], _UNIT["has_frame"])
    return head.frame_dir_proj(d)
