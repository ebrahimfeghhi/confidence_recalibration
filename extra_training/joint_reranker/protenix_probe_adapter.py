#!/usr/bin/env python
"""The REAL Protenix confidence head + the joint PAE reranker, wired for the CUDA capacity probe.

This is the `--factory` `gpu_probe_step.py` loads:

    build_probe(crop_size, pose_count, case, device)
        -> (forward_loss, optimizer, head_parameters, reranker_parameters)

`forward_loss()` forwards ALL `pose_count` poses of ONE scored antibody-antigen interface
through the SAME trainable confidence head, keeps every pose's graph alive, and returns

    mean(original Protenix PAE CE over the K poses) + alpha * listwise rank loss

as one differentiable 0-dim tensor, via `joint_reranker.integration.joint_group_loss`. One
`loss.backward()` by the caller therefore pays for K retained head graphs at once, which is the
entire quantity this sweep exists to measure: the legacy trainer backwards EVERY POSE SEPARATELY
(`phase_k_train.py::_one_pose` returns floats and frees the pose's tensors on return), so no
existing measurement says what K live PAE graphs cost.

====================================================================================================
WHAT IS REAL AND WHAT IS SYNTHETIC -- READ THIS BEFORE QUOTING A NUMBER FROM THIS PROBE
====================================================================================================

REAL (the module, its weights, the objective, the optimiser):

  * The confidence head is `protenix.model.protenix.Protenix(configs).confidence_head`, built by
    `phase_j_rescore_cache.build_model` from the SHIPPED CHECKPOINT (`--ckpt-dir`/`--model-name`
    out of the reference run's args.json), with its real c_s=384, c_z=256, c_s_inputs=449,
    four pairformer blocks and 19.8M parameters. Nothing here reimplements or shrinks it.
  * The forward is `model.run_confidence_head(...)` with exactly the trainer's arguments
    (`phase_k_train.py:6919`), inside `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`,
    `inplace_safe=False`, `chunk_size=None`, `pair_mask=None`.
  * Every architecture patch the reference arm installs is installed here, in the trainer's own
    order and through the trainer's own modules: `lean_readouts.install` -> `drop_single_pathway`
    -> `pose_direction.install` (--frame-direction) -> `pose_perblock.install` (the axes
    `pose_axes.resolve` desugars --pose-perblock/--pose-perblock-copy into) -> `dead_ops.install`
    (--skip-dead-ops). Pair dropout is set on the blocks the way the trainer sets it.
  * The TRAINABLE SET is the reference run's: `head_param_names(model)` narrowed by its
    `--train-only` regex list, checked against its `--expect-trainable`, then narrowed again by
    `lean_readouts.dead_head_params` exactly as --lean-single does. NOTHING IS FROZEN THAT THE
    REFERENCE ARM TRAINS, and `requires_grad` is set before `dead_ops.install` runs, because that
    module's central refusal reads `requires_grad` off the tensors.
  * The per-pose PAE cross-entropy is Protenix's own `protenix.model.loss.PAELoss`, constructed by
    the same expression `ProtenixLoss.__init__` uses (`loss.py:1481`, `PAELoss(**configs.loss.pae)`)
    and called with its unchanged signature, labels, mask and denominator.
  * The ipTM residual baseline is `rank_loss.iptm_pooled(..., pool="max", n_pair_tokens=<uncropped
    scored chain-pair token count>)` -- the real fixed-d0 max path -- fed the FULL token-grid,
    graph-attached `pae` logits, never a sliced view. Never `.item()`, never detached.
  * The reranker is `joint_reranker.model.PAEPoseReranker` at its shipped defaults, and the
    optimiser is `torch.optim.AdamW(lr=<reference lr>, betas=(0.9, 0.95), weight_decay=0.0)` --
    the trainer's own (`phase_k_train.py:4602`) -- over the head's trainable tensors AND the
    reranker's together.

SYNTHETIC (shape-faithful stand-ins for the dataset, because a probe must not need the trainer):

  * `s_inputs` [T,449], `s_trunk` [T,384], `z_trunk` [T,T,256] bf16: random, drawn to the
    MEASURED moments of the real cache (40-60 real trunks under training/fold/trunks, see
    _TRUNK_MOMENTS). The head DETACHES all three (`confidence.py:179-182`, stop_gradient=True), so
    their values never enter the graph that backward walks -- only the magnitude of the downstream
    activations, which is why the moments were measured instead of guessed.
  * `x_pred_coords`: a synthetic per-chain 3.8 A CA random walk with chains placed in contact, one
    [1,N_atom,3] tensor per pose, the antigen chain rigidly perturbed per pose. Realistic only in
    the sense that matters here -- the pairwise distances land across the head's 3.25-52 A one-hot
    bins instead of saturating one of them.
  * The PAE LABEL (`true_coordinate`, `coordinate_mask`, `frame_atom_index`, `rep_atom_mask`,
    `has_frame`) is synthetic: the label coordinates are the unperturbed geometry plus 0.6 A noise,
    and `coordinate_mask` is the measured median 98.7% present. The CE is the real function of
    those; the NUMBER it returns is meaningless and this probe never reports it.
  * The DOCKQ LABELS are synthetic and deliberately MIXED (see _dockq_labels). DockQ is a training
    label only and never reaches a feature.
  * `--pae-scored-pairs` is NOT replicated. Its wrapper (`phase_k_train.py:2681-2717`) replaces
    `PAELoss.calculate_label` to zero entries of the [N_frame,N_token] pair mask and rescale the
    denominator. That changes WHICH entries are nonzero, not the shape of any allocation, so it
    cannot move a capacity number -- but it does move the CE value, which is the second reason this
    probe's CE is not a result.
  * One structure, one interface, one trunk, reused for all K poses and all steps. That is what a
    K-pose group IS (README: "different predictions of ONE scored interface"), but it means the
    probe measures steady-state VRAM, not the per-unit trunk host-to-device traffic.

NOT MEASURED, and it is a gap in gpu_probe_step's contract rather than an omission here: the
factory is given no hook between `loss.backward()` and `optimizer.step()`, so a multi-rank trial
cannot all-reduce. No process group is initialised. For THIS arm that costs nothing in VRAM --
the reference run's `dist_backend` is "gloo", so its gradient collective runs on CPU copies -- but
a future NCCL arm would need the hook before its buckets could be measured.

`alpha` here is a PLACEHOLDER (1.0). It scales the rank term and therefore moves no allocation,
so capacity is independent of it; the real value must come from the measured shared-gradient ratio
(`training.shared_gradient_stats` / `alpha_for_shared_gradient_ratio`, r=0.3), and the README is
explicit that the old `alpha_rank=59` does not transfer to this objective.

CPU-only sanity: `python -m compileall` and an import of this module. Everything that needs a card
is inside `build_probe`.
"""

import json
import math
import os
import sys

import torch

from .integration import PosePrediction, joint_group_loss
from .model import PAEPoseReranker, antibody_antigen_pair_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXTRA_TRAINING = os.path.dirname(_HERE)
_PROTENIX = "/home/ebrahim/Protenix"

#: The arm this probe reproduces. READ, NOT RETYPED -- a hand-built recipe is how four launches
#: died in twenty minutes, and a capacity number measured on a slightly different architecture is
#: worse than no number because nothing downstream would notice. Override only to probe a
#: different arm, and say so in the report.
REFERENCE_RUN = os.environ.get(
    "JOINT_PROBE_REFERENCE_RUN", os.path.join(_EXTRA_TRAINING, "training", "runs", "7mo52r"))

#: Rank-loss temperature in score (ipTM) units, the README's starting hypothesis.
PROBE_TAU = float(os.environ.get("JOINT_PROBE_TAU", "0.05"))
#: See the docstring: a placeholder, not a calibrated alpha. Only required to be > 0, so that the
#: rank term produces gradient at all and gpu_probe_step's live-gradient check can pass.
PROBE_ALPHA = float(os.environ.get("JOINT_PROBE_ALPHA", "1.0"))
#: The DockQ acceptability threshold the whole project uses. Not a tunable.
DOCKQ_THRESHOLD = 0.23

#: Moments of the real trunk cache, measured 2026-11-23 over 5 random real trunks (the spread
#: across structures is small: s_trunk std 199-243, z std 16.9-23.4, s_inputs std 0.061-0.064).
#: s_inputs is SPARSE and non-negative in the real cache (min 0.00, max ~1.1, mean 0.007), which is
#: why it is drawn as a sparse product rather than as a Gaussian.
_TRUNK_MOMENTS = {
    "s_inputs_p": 0.0074,        # Bernoulli rate; value ~ U(0.5, 1.1) -> mean ~0.006, std ~0.065
    "s_trunk_std": 220.0,
    "z_trunk_std": 20.5,
}

#: Population shape, measured 2026-11-23 over 60 distinct real structures' uncropped trunks
#: (training/fold/trunks). Tokens: median 599, p90 876, max 1786. Chains: 2 in 14/60, 3 in 41/60,
#: 4 in 5/60. The two largest chains cover a median 82.7% of tokens and up to 100%. Atoms per
#: token: median 7.70, range 7.42-8.24. has_frame was 1.0 and is_ligand 0.0 on every one of the
#: 40 trunks sampled for those two -- this is an antibody-antigen population with no ligands --
#: so the probe uses all-framed, ligand-free tokens and N_frame == N_token, exactly as the real
#: units do. The ligand derivation below is still the full scatter_add one, so the code stays
#: correct if the population ever gains a glycan.
#:
#: `uncropped_pair_tokens` pins d0 and so changes the ipTM VALUE, never an allocation: median
#: (436) for the representative case, the largest measured (1680) for the worst.
#:
#: WHY "worst" IS TWO EQUAL CHAINS. The reranker encodes every scored AB/AG token pair, and that
#: count is maximised when the crop is entirely the scored pair, split evenly: 2*(T/2)^2 = T^2/2
#: pairs, against 2*0.42T*0.41T = 0.34*T^2 for the representative layout. The measured maximum
#: two-chain coverage really is 100%, so this is the population's worst case and not a fiction.
_CASES = {
    "representative": {
        "chain_fracs": (0.42, 0.17, 0.41),   # antibody heavy / antibody light / antigen
        "pair": (0, 2),
        "atoms_per_token": 7.70,
        "uncropped_pair_tokens": 436,
    },
    "worst": {
        "chain_fracs": (0.50, 0.50),         # antibody / antigen, the whole crop is the pair
        "pair": (0, 1),
        "atoms_per_token": 8.24,
        "uncropped_pair_tokens": 1680,
    },
}

#: Measured median fraction of label atoms with true coordinates (8 random real label npz files:
#: median 0.9873, min 0.3016). It sets the PAE CE denominator and nothing else.
_COORD_MASK_FRAC = 0.9873

#: Keys of the reference args.json this probe consumes. A missing one is a refusal, not a default:
#: silently defaulting an architecture flag is how a child process ends up measuring a different
#: network than its parent (phase_k_train.py's most repeated failure, commit 7d9d4d10).
_REQUIRED_RECIPE_KEYS = (
    "ckpt_dir", "model_name", "lr", "pair_dropout", "lean_readouts", "lean_single", "pae_only",
    "skip_dead_ops", "train_only", "expect_trainable", "blocks_per_ckpt", "ckpt_token_threshold",
    "frame_direction", "pose_perblock",
)

#: Flags the reference arm must NOT have set, because this probe does not install them. Measuring
#: an architecture that is missing a pathway the arm will actually run is the one failure mode a
#: capacity sweep cannot recover from -- the recommended K would be too large.
_REFUSED_IF_SET = (
    "pose_inblock", "pose_inblock_pae", "pose_mlp", "pose_readout", "lora_rank",
    "pae_row_frac", "tm_centered", "pae_tm_weight", "l2_sp", "pae_entropy",
)


def _bootstrap_path():
    """Put `phases/` and the Protenix checkout on sys.path, the way the CPU gates do.

    `phases/` is a directory of flat modules, not a package -- `pose_direction`, `rank_loss`,
    `dead_ops` and the rest import each other by bare name -- so it has to be on the path rather
    than imported relative to this package.
    """
    for p in (os.path.join(_EXTRA_TRAINING, "phases"), _PROTENIX):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def reference_recipe(run_dir=None):
    """The architecture/optimiser recipe, read out of a finished run's args.json.

    Returns the dict. Raises with the missing key named, because an absent flag is indistinguishable
    from an off flag in a dict and the two build different networks.
    """
    run_dir = run_dir or REFERENCE_RUN
    path = os.path.join(run_dir, "args.json")
    if not os.path.exists(path):
        raise NotImplementedError(
            f"the probe reproduces an EXISTING arm and reads its recipe from {path}, which does "
            f"not exist. Point JOINT_PROBE_REFERENCE_RUN at the run directory of the cropped "
            f"pose-injection arm this sweep is sizing (the 7mo52r recipe: --pose-perblock "
            f"--pose-perblock-copy --frame-direction). Nothing here will retype one.")
    with open(path) as fh:
        rec = json.load(fh)
    missing = [k for k in _REQUIRED_RECIPE_KEYS if k not in rec]
    if missing:
        raise NotImplementedError(
            f"{path} carries no {missing}. Those are architecture or optimiser settings, and a "
            f"default here would measure a different network than the arm will run. Use a "
            f"reference run whose args.json records them.")
    set_anyway = [k for k in _REFUSED_IF_SET if rec.get(k)]
    if set_anyway:
        raise NotImplementedError(
            f"{path} sets {set_anyway}, and this probe installs none of them. Measuring without a "
            f"pathway the arm will run would OVERSTATE the safe K, which is the one error a "
            f"capacity sweep cannot recover from. Either extend build_probe to install them (each "
            f"has an install() in phases/ and an order constraint documented at "
            f"phase_k_train.py:2497-2542) or choose a reference run that does not use them.")
    if not rec.get("train_only"):
        raise NotImplementedError(
            f"{path} has no --train-only pattern, so its trainable scope is the whole head. That "
            f"is a legitimate arm but a different capacity question (19.8M trainable tensors in "
            f"AdamW instead of 5.8M); pass a reference run with the scope this sweep is sizing.")
    return rec


def _require_cuda(device):
    """The head's forward is autocast on "cuda" and dead_ops probes the cuequivariance kernels.

    gpu_probe_step already refuses a CPU run; this repeats it at the factory so that importing and
    calling this module by hand cannot produce a CPU number that looks like a capacity.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            f"build_probe needs a CUDA device, got {device}. A CPU proxy measures neither the "
            f"bf16 autocast path nor the allocator, so it is not a capacity measurement.")
    return device


# ---------------------------------------------------------------------------------------------
# SYNTHETIC UNIT
# ---------------------------------------------------------------------------------------------

def _chain_sizes(n_tok, fracs):
    """Token counts per chain that sum EXACTLY to n_tok, with every chain non-empty.

    The remainder goes to the last chain rather than being rounded away: a chain rounded to zero
    would make the scored pair's mask empty and `antibody_antigen_pair_mask` would raise -- which
    is the right error but a confusing one to meet at crop 256.
    """
    sizes = [max(1, int(round(f * n_tok))) for f in fracs[:-1]]
    sizes.append(n_tok - sum(sizes))
    if sizes[-1] < 1:
        raise ValueError(f"crop {n_tok} is too small for chain fractions {fracs}")
    return sizes


def _rotation(axis, angle, device):
    """Rodrigues rotation matrix, so a pose perturbation is a rigid body move and not a shear."""
    a = axis / axis.norm()
    k = torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], device=device)
    return torch.eye(3, device=device) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _ca_walk(n, gen, device):
    """n CA positions as a 3.8 A random walk, centred. 3.8 A is the CA-CA virtual bond length."""
    step = torch.randn(n, 3, generator=gen, device=device)
    step = step / step.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 3.8
    xyz = torch.cumsum(step, dim=0)
    return xyz - xyz.mean(dim=0, keepdim=True)


def _atoms_per_token(n_tok, ratio):
    """Per-token atom counts averaging `ratio`, each at least 4.

    AT LEAST FOUR, because `frame_atom_index` takes the token's first three atoms as N, CA, C and
    the representative atom is the second of them. A token with fewer would index another token's
    atoms and score a pose against the wrong residue's frame -- the exact silent failure
    `phase_j_build_cache.slice_cached_unit` guards with its `broken` mask.
    """
    base = max(4, int(ratio))
    counts = [base] * n_tok
    extra = int(round(ratio * n_tok)) - base * n_tok
    for i in range(max(0, extra)):
        counts[i % n_tok] += 1
    return counts


def _synthetic_unit(crop_size, pose_count, case, device, seed=0):
    """One synthetic interface at `crop_size` tokens with `pose_count` poses, on `device`.

    Returns a dict whose keys are named after what consumes them: `feat_head` is what
    `run_confidence_head` and `iptm_pooled` read, the `label_*` entries are what `PAELoss.forward`
    reads, and `poses` is the list of per-pose [1,N_atom,3] coordinate tensors.
    """
    spec = _CASES[case]
    gen = torch.Generator(device=device).manual_seed(seed)
    n_tok = int(crop_size)
    sizes = _chain_sizes(n_tok, spec["chain_fracs"])

    asym = torch.cat([torch.full((n,), c, dtype=torch.int64, device=device)
                      for c, n in enumerate(sizes)])

    # ---- per-token CA geometry, chains placed in contact ------------------------------------
    chains, offset = [], 0.0
    for n in sizes:
        xyz = _ca_walk(n, gen, device)
        extent = float(xyz.norm(dim=-1).max())
        # 0.9 * the mean extent puts the next chain's surface against this one's instead of
        # concentric with it, so cross-chain distances populate the head's bins rather than
        # collapsing into the last one (which is where a wildly separated chain would sit).
        xyz = xyz + torch.tensor([offset, 0.0, 0.0], device=device)
        offset += 0.9 * extent
        chains.append(xyz)
    ca = torch.cat(chains, dim=0)                                    # [T,3]

    # ---- atoms ------------------------------------------------------------------------------
    counts = _atoms_per_token(n_tok, spec["atoms_per_token"])
    n_atom = sum(counts)
    starts = [0]
    for c in counts[:-1]:
        starts.append(starts[-1] + c)
    atom_to_token = torch.cat([torch.full((c,), t, dtype=torch.int64, device=device)
                               for t, c in enumerate(counts)])
    atom_to_tokatom = torch.cat([torch.arange(c, dtype=torch.int64, device=device)
                                 for c in counts])
    rep_atom_index = torch.tensor([s + 1 for s in starts], dtype=torch.int64, device=device)
    # ONE representative atom per token, so rep_atom_mask.sum() == N_token == the PAE grid side.
    # Both the distogram mask (which selects the cdist input inside the head) and the PAE mask
    # (which selects the loss's token coordinates) are the token's CA here. In the real cache they
    # are separate arrays; for an all-protein, all-framed population they select the same atom,
    # and keeping them equal is what makes the [T,T,64] logits and the [F,T] label line up.
    rep_mask = torch.zeros(n_atom, dtype=torch.bool, device=device)
    rep_mask[rep_atom_index] = True
    frame_atom_index = torch.stack(
        [torch.tensor(starts, dtype=torch.int64, device=device),
         rep_atom_index,
         torch.tensor([s + 2 for s in starts], dtype=torch.int64, device=device)], dim=-1)  # [T,3]

    base_atoms = ca[atom_to_token] + 1.4 * torch.randn(
        n_atom, 3, generator=gen, device=device)
    base_atoms[rep_atom_index] = ca                                  # the CA is exactly the CA
    base_atoms[frame_atom_index[:, 0]] = ca + torch.tensor([1.46, 0.0, 0.0], device=device)
    base_atoms[frame_atom_index[:, 2]] = ca + torch.tensor([0.0, 1.52, 0.0], device=device)

    # ---- K poses: the antigen chain rigidly perturbed ---------------------------------------
    ag_chain = spec["pair"][1]
    ag_tok = (asym == ag_chain)
    ag_atom = ag_tok[atom_to_token]
    dockq = _dockq_labels(pose_count, device)
    poses = []
    for j in range(pose_count):
        # The severity tracks the synthetic label only so that the group is not geometrically
        # uniform. It is cosmetic: DockQ here is a made-up number and this probe reports no
        # ranking quality, only memory and step time.
        severity = 0.15 if float(dockq[j]) >= DOCKQ_THRESHOLD else 1.0
        axis = torch.randn(3, generator=gen, device=device)
        rot = _rotation(axis, severity * 0.6, device)
        shift = severity * 8.0 * torch.randn(3, generator=gen, device=device)
        x = base_atoms.clone()
        centre = x[ag_atom].mean(dim=0, keepdim=True)
        x[ag_atom] = (x[ag_atom] - centre) @ rot.T + centre + shift
        poses.append(x.unsqueeze(0).contiguous())                    # [1, N_atom, 3]

    # ---- the PAE label ----------------------------------------------------------------------
    true_coord = base_atoms + 0.6 * torch.randn(n_atom, 3, generator=gen, device=device)
    coord_mask = (torch.rand(n_atom, generator=gen, device=device) < _COORD_MASK_FRAC)
    # THE FRAME ATOMS AND THE REPRESENTATIVE ATOM ARE ALWAYS PRESENT. calculate_label drops a
    # frame unless all three of its atoms have coordinates, and at the measured 1.3% missing rate
    # a uniform draw would silently delete ~4% of the rows -- changing the loss's row count, which
    # is one of the shapes this probe is supposed to hold fixed.
    coord_mask[frame_atom_index.reshape(-1)] = True

    has_frame = torch.ones(n_tok, dtype=torch.float32, device=device)
    is_ligand = torch.zeros(n_atom, dtype=torch.int64, device=device)

    feat_head = {
        "asym_id": asym,
        "has_frame": has_frame,
        "atom_to_token_idx": atom_to_token,
        "atom_to_tokatom_idx": atom_to_tokatom,
        "distogram_rep_atom_mask": rep_mask.to(torch.float32),
        "is_ligand": is_ligand,
    }
    m = _TRUNK_MOMENTS
    s_inputs = ((torch.rand(n_tok, 449, generator=gen, device=device) < m["s_inputs_p"]).float()
                * (0.5 + 0.6 * torch.rand(n_tok, 449, generator=gen, device=device)))
    s_trunk = m["s_trunk_std"] * torch.randn(n_tok, 384, generator=gen, device=device)
    z_trunk = (m["z_trunk_std"] * torch.randn(n_tok, n_tok, 256, generator=gen,
                                              device=device)).to(torch.bfloat16)
    return {
        "feat_head": feat_head,
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "poses": poses,
        "dockq": dockq,
        "label_true_coordinate": true_coord,
        "label_coordinate_mask": coord_mask.to(torch.float32),
        "label_frame_atom_index": frame_atom_index,
        "label_rep_atom_mask": rep_mask,
        "label_has_frame": has_frame,
        "pair": spec["pair"],
        "uncropped_pair_tokens": spec["uncropped_pair_tokens"],
        "n_token": n_tok,
        "n_atom": n_atom,
        "chain_sizes": sizes,
    }


def _dockq_labels(pose_count, device):
    """A MIXED group: alternating acceptable / unacceptable, so the ranking term actually runs.

    THIS IS NOT COSMETIC. `training.acceptable_mass_loss` returns `scores.sum() * 0.0` and
    `ranking_active=False` on an all-acceptable or all-unacceptable group, exactly as the spec
    requires -- and then the reranker receives NO gradient at all, `gpu_probe_step` finds
    `saw_reranker_grad` false and reports
        "missing live gradients to confidence head or reranker"
    which looks like an integration bug and is really a label-sampling one. gpu_probe_step
    already refuses `--poses < 2`, so alternating guarantees both classes are non-empty.

    0.55 and 0.08 straddle the 0.23 threshold with margin; the split does not have to be the
    population's real win-rate because the rank term's COST does not depend on the split, only on
    both sides being non-empty.
    """
    dq = torch.tensor([0.55 if j % 2 == 0 else 0.08 for j in range(pose_count)],
                      dtype=torch.float32, device=device)
    good = dq >= DOCKQ_THRESHOLD
    if not (bool(good.any()) and bool((~good).any())):
        raise ValueError(
            f"a {pose_count}-pose group came out all-acceptable or all-unacceptable; the ranking "
            f"term would be an exact zero and the reranker would get no gradient")
    return dq


# ---------------------------------------------------------------------------------------------
# THE REAL HEAD
# ---------------------------------------------------------------------------------------------

def _build_head(recipe, crop_size, device):
    """Build the shipped head and install the reference arm's pathways. Returns (model, configs).

    The install ORDER is phase_k_train.py's (pair dropout, lean_readouts, lean_single,
    pose_direction, pose_perblock) and each module re-checks its own prerequisites, so a reordering
    here is a hard stop rather than a silently different network.
    """
    import phase_j_rescore_cache as RC
    import pose_axes as PAX

    # build_model refuses anything but "openfold" and the help text says why: without it the
    # coordinates come out silently NaN. Set when absent rather than letting a trial die on an
    # environment variable after the GPUs were reserved; a CONFLICTING value is still build_model's
    # refusal, not something this quietly overwrites.
    if "LAYERNORM_TYPE" not in os.environ:
        os.environ["LAYERNORM_TYPE"] = "openfold"
        print("probe: LAYERNORM_TYPE was unset; set to openfold (build_model requires it)")

    model, configs = RC.build_model(recipe["ckpt_dir"], recipe["model_name"], device)
    # build_model comes from the RESCORING helper, so it ends in model.eval() with every parameter
    # frozen. The head we train must be in training mode: Protenix's confidence head offloads pae
    # and pde to the CPU above PROTENIX_CONF_OFFLOAD_TOKENS when `not self.training`
    # (confidence.py:233), which would hand this probe CPU logits and a loss computed on the host.
    model.confidence_head.train()

    head = model.confidence_head
    pd = float(recipe["pair_dropout"])
    if pd >= 0.0:
        blocks = list(getattr(head.pairformer_stack, "blocks", []))
        if not blocks:
            raise RuntimeError("the confidence head's pairformer_stack has no .blocks")
        for b in blocks:
            b.p_drop = pd
            dr = getattr(b, "dropout_row", None)
            if dr is not None:
                for obj in (dr, getattr(dr, "dropout", None)):
                    for attr in ("p", "r"):
                        if obj is not None and hasattr(obj, attr):
                            setattr(obj, attr, pd)
        now = sorted({float(b.p_drop) for b in blocks})
        if now != [pd]:
            raise RuntimeError(f"pair dropout did not take: p_drop is {now}, not [{pd}]")

    dead_head = []
    if recipe["lean_readouts"]:
        import lean_readouts as LR
        LR.install(model, pae_only=bool(recipe["pae_only"]))
        if recipe["lean_single"]:
            LR.drop_single_pathway(model)
            dead_head = LR.dead_head_params(head, single_dropped=True)
    elif recipe["lean_single"]:
        raise NotImplementedError("--lean-single requires --lean-readouts; the reference run's "
                                  "args.json has the first without the second")

    if recipe["frame_direction"]:
        import pose_direction
        pose_direction.install(model)
    blk, rd = PAX.resolve(recipe)
    if rd is not None:
        raise NotImplementedError("the reference run installs a pose READOUT site; this probe "
                                  "does not. See _REFUSED_IF_SET.")
    if blk is not None:
        import pose_perblock
        pose_perblock.install(model, terms=blk["terms"], init=blk["init"], weights=blk["weights"])

    # SIZE-CONDITIONAL ACTIVATION CHECKPOINTING, the trainer's own rule (phase_k_train.py:6852).
    # This is the single most memory-relevant switch in the probe, so it is printed: at
    # --ckpt-token-threshold 1250 every crop this sweep tests (256/384/512) is BELOW the threshold
    # and runs with checkpointing OFF, which is the expensive branch and the correct one to size.
    thr = int(recipe["ckpt_token_threshold"])
    bpc = None if (thr > 0 and 0 <= crop_size <= thr) else (int(recipe["blocks_per_ckpt"]) or None)
    head.pairformer_stack.blocks_per_ckpt = bpc
    print(f"probe: crop {crop_size} tokens, ckpt_token_threshold {thr} -> blocks_per_ckpt="
          f"{bpc} ({'NO activation checkpointing' if bpc is None else 'checkpointed'})")
    return model, configs, dead_head


def _trainable_names(model, recipe, dead_head):
    """The reference arm's trainable tensor names, by its own selection procedure.

    --expect-trainable is checked where the trainer checks it: AFTER --train-only and BEFORE
    --lean-single's drop. That ordering is the whole value of the number -- it is a gate on the
    REGEX, and a run whose pattern silently collapsed to a negative lookahead (19 tensors instead
    of 113) was caught only because a human read the startup line.
    """
    import re
    from phase_k_train import head_param_names

    prefixes, trainable = head_param_names(model)
    pfx = prefixes[0]
    rel = {n: n[len(pfx) + 1:] for n in trainable}
    pats = [t.strip() for t in str(recipe["train_only"]).split(",") if t.strip()]
    kept = [n for n in trainable if any(re.search(p, rel[n]) for p in pats)]
    if not kept:
        raise RuntimeError(f"--train-only {pats} matched no head parameter; the optimiser would "
                           f"be empty and the probe would report success while training nothing")
    expect = int(recipe["expect_trainable"] or 0)
    if expect and len(kept) != expect:
        raise RuntimeError(
            f"--train-only left {len(kept)} trainable tensor(s), but the reference run records "
            f"--expect-trainable {expect}. The pattern matched something, so nothing else would "
            f"have caught this, and the probe would size a different trainable scope than the arm "
            f"will train.")
    if dead_head:
        dead_abs = {f"{pfx}.{n}" for n in dead_head}
        kept = [n for n in kept if n not in dead_abs]
        if not kept:
            raise RuntimeError("--lean-single left the trainable set empty")
    return kept


# ---------------------------------------------------------------------------------------------
# THE FACTORY
# ---------------------------------------------------------------------------------------------

def build_probe(crop_size: int, pose_count: int, case: str, device: torch.device):
    """(forward_loss, optimizer, head_parameters, reranker_parameters) -- see the module docstring.

    `forward_loss()` may be called repeatedly; it rebuilds all K graphs each time and returns the
    scalar `mean(PAE CE) + alpha * rank`. The caller owns backward and the optimiser step.
    """
    device = _require_cuda(device)
    if case not in _CASES:
        raise ValueError(f"unknown case {case!r}; this adapter defines {sorted(_CASES)} "
                         f"(see _CASES for what each one's shapes are and why)")
    if int(pose_count) < 2:
        raise ValueError("a ranking group needs at least 2 poses")
    _bootstrap_path()
    recipe = reference_recipe()

    import dead_ops as DOPS
    import pose_direction
    import rank_loss as RKL
    from protenix.model.loss import PAELoss

    model, configs, dead_head = _build_head(recipe, int(crop_size), device)
    trainable = _trainable_names(model, recipe, dead_head)
    tset = set(trainable)
    for _n, p in model.named_parameters():
        p.requires_grad_(_n in tset)
    head_params = [p for n, p in model.named_parameters() if n in tset]

    # AFTER requires_grad, exactly as in the trainer: --skip-dead-ops refuses to stub a module
    # that is still TRAINABLE, and it reads that off the tensors rather than off the flag string.
    if recipe["skip_dead_ops"]:
        DOPS.install(model, autocast_dtype=torch.bfloat16,
                     triangle_multiplicative=configs.triangle_multiplicative,
                     triangle_attention=configs.triangle_attention)

    # The real PAE loss object, built by the same expression ProtenixLoss uses (loss.py:1481), so
    # its bin boundaries, eps and reduction are the configs' and not this file's. Left in training
    # mode (nn.Module's default, and what the trainer's ProtenixLoss is in) because
    # PAELoss.forward's `not self.training` branch calls torch.cuda.empty_cache(), which would
    # make the probe's peak-reserved reading optimistic and unlike the real run.
    pae_loss = PAELoss(**configs.loss.pae)

    reranker = PAEPoseReranker().to(device)
    reranker.train()
    reranker_params = [p for p in reranker.parameters()]

    unit = _synthetic_unit(int(crop_size), int(pose_count), case, device)
    feat = unit["feat_head"]
    pair = unit["pair"]
    npt = int(unit["uncropped_pair_tokens"])

    # THE LIGAND-AWARE ROW SET, derived the way iptm_pooled derives it (rank_loss.py:395-397) and
    # not from a chain annotation: scatter_add of the per-ATOM is_ligand onto tokens, then > 0.
    # `live` is what row_t pools over, so the [F,T,64] logits handed to the reranker must be sliced
    # with the SAME boolean -- model.antibody_antigen_pair_mask's docstring is explicit that the
    # row axis is `has_frame & ~ligand` while the TARGET axis keeps ligand columns. In this
    # ligand-free population live == has_frame, so F == T; the derivation is kept general anyway.
    lig_tok = (torch.zeros_like(feat["asym_id"]).scatter_add(
        0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0)
    live = feat["has_frame"].reshape(-1).bool() & (~lig_tok)
    pair_mask = antibody_antigen_pair_mask(
        feat["asym_id"], [pair[0]], [pair[1]],
        has_frame=feat["has_frame"].reshape(-1), is_ligand_token=lig_tok)

    print(f"probe: case={case} crop={crop_size} K={pose_count} chains={unit['chain_sizes']} "
          f"n_atom={unit['n_atom']} scored pair={pair} uncropped pair tokens={npt} "
          f"AB/AG pairs/pose={int(pair_mask.sum())}")
    print(f"probe: trainable head tensors={len(head_params)} "
          f"({sum(p.numel() for p in head_params)/1e6:.3f}M params), reranker tensors="
          f"{len(reranker_params)} ({sum(p.numel() for p in reranker_params)/1e6:.3f}M params), "
          f"alpha={PROBE_ALPHA:g} tau={PROBE_TAU:g} (alpha is a PLACEHOLDER, see the docstring)")

    opt = torch.optim.AdamW(head_params + reranker_params, lr=float(recipe["lr"]),
                            betas=(0.9, 0.95), weight_decay=0.0)

    def predict_one(j):
        """One pose: the real head forward, then the three live quantities the bridge wants."""
        x = unit["poses"][j]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _plddt, pae, _pde, _res = model.run_confidence_head(
                input_feature_dict=feat, s_inputs=unit["s_inputs"], s_trunk=unit["s_trunk"],
                z_trunk=unit["z_trunk"], pair_mask=None, x_pred_coords=x,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention, inplace_safe=False,
                chunk_size=None)
        # .float() is what the trainer puts in its `pred` dict; the loss and iptm_pooled both read
        # fp32 logits. The tensor stays attached -- detaching here is the failure integration.py's
        # requires_grad checks exist to catch.
        pae = pae.float()
        if pae.device.type != "cuda":
            raise RuntimeError(
                f"the confidence head returned pae on {pae.device}; Protenix offloads pae/pde to "
                f"the CPU above PROTENIX_CONF_OFFLOAD_TOKENS when the module is NOT in training "
                f"mode (confidence_head.training={model.confidence_head.training})")

        # ipTM GETS THE FULL [N_sample,T,T,64] GRID, not the frame-sliced view: iptm_pooled builds
        # its own row selector and target mask from feat, and a pre-sliced tensor would be indexed
        # a second time. pool="max" WITH n_pair_tokens is the fixed-d0 path added in 60020ce5 --
        # the max is what selection deploys, and pinning d0 to the UNCROPPED chain-pair length is
        # what keeps one interface's crops comparable (at 600 -> 256 pair tokens the per-bin weight
        # w(10 A) moves 0.423 -> 0.259, which is the bug the old refusal existed to prevent).
        iptm = RKL.iptm_pooled(pae, feat, configs, pair, pool="max", n_pair_tokens=npt)
        if iptm is None:
            raise RuntimeError(
                f"iptm_pooled returned None for pair {pair}: no scored row survived. The crop or "
                f"the chain layout left the AB/AG pair empty, so there is nothing to rank.")

        ce = pae_loss(
            logits=pae,
            pred_coordinate=x,
            true_coordinate=unit["label_true_coordinate"],
            coordinate_mask=unit["label_coordinate_mask"],
            frame_atom_index=unit["label_frame_atom_index"],
            rep_atom_mask=unit["label_rep_atom_mask"],
            has_frame=unit["label_has_frame"],
        )
        # [N_sample=1,T,T,64] -> the [F,T,64] the reranker's pair encoder indexes. Sliced from the
        # SAME tensor ipTM came from, so one head forward feeds both terms.
        return PosePrediction(pae_logits=pae[0][live], pair_mask=pair_mask, iptm=iptm, pae_ce=ce)

    def forward_loss():
        # ONCE PER STEP, BEFORE ANY POSE RUNS, which is where the trainer calls it
        # (phase_k_train.py:6866). pose_direction publishes the unit's frame definition through a
        # module-level dict; an unset one makes its patched forward raise rather than silently
        # inject nothing, so re-arming it every step is cheap insurance against anything that
        # cleared it between steps.
        if recipe["frame_direction"]:
            pose_direction.set_unit(unit["label_frame_atom_index"], unit["label_has_frame"],
                                    unit["label_rep_atom_mask"].to(torch.float32),
                                    pdb=f"probe{crop_size}")
        group = joint_group_loss(
            reranker, list(range(int(pose_count))), predict_one, unit["dockq"],
            alpha=PROBE_ALPHA, temperature=PROBE_TAU, threshold=DOCKQ_THRESHOLD)
        if not group.ranking_active:
            raise RuntimeError(
                "the ranking term was inactive on this group, so the reranker receives no "
                "gradient and gpu_probe_step's live-gradient check will fail. _dockq_labels is "
                "supposed to make that impossible -- the labels are no longer mixed.")
        return group.total

    return forward_loss, opt, head_params, reranker_params
