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
    module's central refusal reads `requires_grad` off the tensors. Run against 7mo52r this comes
    out at 113 tensors from the regex (its `--expect-trainable`, so the gate passes), 102 tensors
    and 5,347,406 parameters after --lean-single drops 75 provably gradient-free names -- measured
    on CPU, see the verification note at the end.
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
  * `x_pred_coords`: a synthetic per-chain 3.8 A CA random walk, rescaled to a globular radius of
    gyration and with the SCORED chain pair slid into 4.5 A contact; one [1,N_atom,3] tensor per
    pose, the antigen chain rigidly perturbed per pose. Realistic only in the three senses that
    matter: 88-98% of pairwise distances land inside the head's 3.25-52 A one-hot bins instead of
    piling into the last one, the scored interface actually exists, and the within-token atom
    indexing is the real one -- the frame is atoms 0/1/2 (N, CA, C) and the DISTOGRAM
    representative atom is atom 4 (CB) on 92.6% of tokens while the PAE representative atom is the
    CA, which is the split `pose_direction._delta` depends on and which `build_probe` checks
    against `pose_direction.derive_frames`.
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

WHAT WAS VERIFIED WITHOUT A CARD (the three live GPUs were off limits while this was written):

  * `_build_head` and `_trainable_names` ran end to end against the REAL checkpoint on CPU (150 s):
    lean_readouts' pinned-SHA check passed, pose_direction installed its zero-verified
    frame_dir_proj, pose_perblock installed 9 tensors at terms onehot+dist+orient / init
    pretrained / weights untied plus 3 orientation copies, blocks_per_ckpt resolved to None at
    crop 256, and the trainable scope came out 113 -> 102 tensors / 5,347,406 parameters.
  * A CPU rehearsal of the whole objective with a one-tensor stand-in in place of
    `run_confidence_head` -- three AdamW steps, both cases, 96-512 tokens: `pae_ce` a 0-dim
    tensor, `iptm_pooled` differentiable, `ranking_active` True on every step, nonzero gradient to
    the head stand-in on every step and to 2 of the reranker's 24 tensors at step 1, all 24 from
    step 2.
  * `PAELoss(**configs.loss.pae)` and `sample_confidence.get_bin_params` against the real parsed
    configs; `pose_direction.derive_frames` and `direction_tensor` against the synthetic unit;
    `compileall`; import; `reference_recipe()` against 7mo52r.

The head forward itself, `dead_ops.install`'s cuequivariance probe and every VRAM figure are
UNMEASURED here, by construction -- that is what the sweep is for.
"""

import json
import math
import os
import sys

import torch

from .integration import PosePrediction, joint_group_loss
from .model import PAEPoseReranker
# THE MASK AND ROW-SET DERIVATIONS COME FROM THE LIVE BRIDGE, NOT FROM A SECOND COPY HERE. The
# probe's whole claim is that it measures the arm the trainer will run, so the one place where the
# probe could silently describe a different pair population than the real predict_one is this
# derivation -- and that is exactly the drift `protenix_head_bridge`'s own header warns about.
# `scored_pair_mask` also carries a fix a chain-list mask does not have: iptm_pooled scores ONE
# asym pair and restricts its anchor rows to that pair, so a mask unioning every antibody chain
# against every antigen chain would pool H-AG and L-AG rows while the baseline ipTM is the single
# pair -- on nearly every H+L structure in the set.
from .protenix_head_bridge import ligand_token_flag, live_row_mask, scored_pair_mask

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

#: Globular-protein radius of gyration, Rg ~ 2.2 * N^0.38 A for N residues (the standard empirical
#: scaling). A 3.8 A CA random walk of N steps has Rg = 3.8*sqrt(N/6), which at N=210 is 22.5 A
#: against the 16.7 A a real 210-residue domain has -- 35% too extended, enough to push most of
#: the distance histogram past the head's last one-hot bin at 52 A. Each synthetic chain is
#: rescaled to this Rg. It changes no allocation (`one_hot` emits [T,T,num_bins] whichever bin a
#: pair lands in) and no reported number; it exists so the head is not fed a degenerate distance
#: matrix in which nearly every pair is in the same bin.
_RG_COEF, _RG_EXP = 2.2, 0.38
#: Closest CA-CA approach across a contacting chain pair. Real antibody-antigen interfaces sit
#: here; the first version of this file slid chains apart by 0.9x their own extent instead and
#: produced a "scored interface" whose closest approach was 33-42 A -- no interface at all, and a
#: claim the module docstring was making in good faith. The assertion in _synthetic_unit exists
#: because of that.
_INTERFACE_GAP_A = 4.5

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
    """n CA positions: a 3.8 A random walk, centred, then rescaled to a globular Rg.

    3.8 A is the CA-CA virtual bond length; the rescaling is _RG_COEF/_RG_EXP, see there.
    """
    step = torch.randn(n, 3, generator=gen, device=device)
    step = step / step.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 3.8
    xyz = torch.cumsum(step, dim=0)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    rg = float(xyz.square().sum(dim=-1).mean().sqrt())
    want = _RG_COEF * (max(n, 1) ** _RG_EXP)
    return xyz * (want / max(rg, 1e-6))


def _place_chains(sizes, placement, gen, device):
    """Per-chain CA coordinates with each chain slid into CONTACT with the ones already placed.

    `placement` is [(chain_index, direction), ...] in placement order; the first entry is the
    anchor and every later chain is slid along `direction * x_hat` until its minimum CA-CA
    distance to the whole placed set reaches _INTERFACE_GAP_A. Bisection on that slide, which
    assumes the minimum distance is non-decreasing in it -- true once the two blobs are disjoint,
    and the achieved interface distance is ASSERTED by the caller rather than trusted.

    Returns a list indexed by CHAIN INDEX (not placement order), because the token axis is ordered
    by chain id and `asym_id` has to agree with it.
    """
    out = [None] * len(sizes)
    placed = None
    for chain, direction in placement:
        xyz = _ca_walk(sizes[chain], gen, device)
        if placed is None:
            out[chain] = xyz
            placed = xyz
            continue
        axis = torch.tensor([float(direction), 0.0, 0.0], device=device)
        hi = 2.0 * (float(placed.norm(dim=-1).max()) + float(xyz.norm(dim=-1).max())) + 20.0
        lo = 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if float(torch.cdist(placed, xyz + mid * axis).min()) >= _INTERFACE_GAP_A:
                hi = mid
            else:
                lo = mid
        out[chain] = xyz + hi * axis
        placed = torch.cat((placed, out[chain]), dim=0)
    return out


def _atoms_per_token(n_tok, ratio):
    """Per-token atom counts averaging `ratio`, each at least 5.

    AT LEAST FIVE, because the within-token atom indices are canonical and this construction uses
    four of them: `frame_atom_index` is the token's atoms 0/1/2 (N, CA, C -- `pose_direction
    .derive_frames` proves that derivation exact against the label npz on all 500 training
    structures) and the distogram representative atom is index 4 (CB). A token with fewer atoms
    would index the NEXT token's atoms and score a pose against the wrong residue's frame -- the
    silent failure `phase_j_build_cache.slice_cached_unit` guards with its `broken` mask.
    """
    base = max(5, int(ratio))
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

    # ---- per-token CA geometry, the SCORED PAIR placed in contact ---------------------------
    # The placement order puts the two chains of the scored pair next to each other, which a
    # left-to-right order does NOT do for the representative layout: there the scored pair is
    # (heavy, antigen) = (0, 2) with the light chain in between, and placing 0,1,2 in sequence left
    # the scored pair 33-42 A apart -- i.e. no interface, so no AB/AG contacts, so a reranker
    # encoding pairs that touch nothing. The antigen is therefore placed against the heavy chain on
    # +x and the light chain against it on -x, which is also the real topology.
    ab, ag = spec["pair"]
    placement = [(ab, +1), (ag, +1)] + [(c, -1) for c in range(len(sizes)) if c not in (ab, ag)]
    ca = torch.cat(_place_chains(sizes, placement, gen, device), dim=0)   # [T,3]
    # ASSERTED, NOT ASSUMED, because the bisection above relies on a monotonicity that is only
    # practically true. An interface that failed to form would not raise anywhere else: the pair
    # mask would still be full, the ipTM would still compute, and the probe would still report a
    # capacity -- measured on a geometry no structure has.
    _ab_ca, _ag_ca = ca[asym == ab], ca[asym == ag]
    _gap = float(torch.cdist(_ab_ca, _ag_ca).min())
    if not (2.5 <= _gap <= 12.0):
        raise RuntimeError(
            f"the synthetic scored pair {spec['pair']} came out {_gap:.1f} A apart at its closest "
            f"CA-CA approach; a real antibody-antigen interface is ~{_INTERFACE_GAP_A} A, so the "
            f"chain placement failed and this geometry has no interface to score")

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
    # TWO DIFFERENT REPRESENTATIVE ATOMS, which is what the real cache has and what a first
    # version of this file got wrong. MEASURED on 10fj (625 tokens, 4,795 atoms):
    #   * the PAE representative atom is exactly frame_atom_index[:, 1] -- the CA, within-token
    #     index 1 -- on every token (checked: equal for all 625).
    #   * the DISTOGRAM representative atom is within-token index 4 (CB) on 579 tokens and index 1
    #     on 46 (7.4%, the glycines, which have no CB). The two masks differ on 1,158 of the atoms.
    # They both sum to N_token, so the [T,T,64] logit grid and the [F,T] label line up either way,
    # but the head's `torch.cdist` runs on the DISTOGRAM atoms while the PAE label is built on the
    # CA frames -- so making them the same atom would feed the head a distance matrix the real run
    # never sees. Every 14th token takes CA here, reproducing the measured 7.1% glycine share.
    pae_rep_index = torch.tensor([s + 1 for s in starts], dtype=torch.int64, device=device)
    disto_rep_index = torch.tensor([s + (1 if t % 14 == 0 else 4) for t, s in enumerate(starts)],
                                   dtype=torch.int64, device=device)
    pae_rep_mask = torch.zeros(n_atom, dtype=torch.bool, device=device)
    pae_rep_mask[pae_rep_index] = True
    disto_rep_mask = torch.zeros(n_atom, dtype=torch.bool, device=device)
    disto_rep_mask[disto_rep_index] = True
    frame_atom_index = torch.stack(
        [torch.tensor(starts, dtype=torch.int64, device=device),
         pae_rep_index,
         torch.tensor([s + 2 for s in starts], dtype=torch.int64, device=device)], dim=-1)  # [T,3]

    base_atoms = ca[atom_to_token] + 1.4 * torch.randn(
        n_atom, 3, generator=gen, device=device)
    base_atoms[pae_rep_index] = ca                                   # the CA is exactly the CA
    # N AND C AT THEIR REAL BOND LENGTHS AND IN A DIFFERENT DIRECTION PER RESIDUE. A fixed offset
    # (+x for N, +y for C) was the first version and it gives every token the SAME frame
    # orientation, which would make --frame-direction's whole input degenerate: the frame-relative
    # unit direction of j in i's frame would then be a pure function of the displacement, carrying
    # none of the orientation information that pathway exists to supply. Two independent random
    # directions are almost surely non-collinear, so `expressCoordinatesInFrame`'s Gram-Schmidt
    # never sees a degenerate frame.
    def _unit(k):
        v = torch.randn(k, 3, generator=gen, device=device)
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    base_atoms[frame_atom_index[:, 0]] = ca + 1.46 * _unit(n_tok)    # N, the N-CA bond
    base_atoms[frame_atom_index[:, 2]] = ca + 1.52 * _unit(n_tok)    # C, the CA-C bond
    # CB at the real 1.53 A CA-CB bond length, in a random direction. This is the atom the head's
    # cdist -- and therefore its one-hot distance bins -- actually sees. ONLY on the tokens that
    # HAVE a CB: on a glycine token the distogram atom IS the CA, and writing a displaced position
    # there would move the CA that the frame and the PAE label are both built on.
    has_cb = torch.tensor([t % 14 != 0 for t in range(n_tok)], dtype=torch.bool, device=device)
    base_atoms[disto_rep_index[has_cb]] = (ca + 1.53 * _unit(n_tok))[has_cb]

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
    # The PAE label's token coordinates are the representative atoms, and `calculate_label` builds
    # its [N_frame,N_token] mask from `coordinate_mask[rep_atom_mask]`; a missing one would delete
    # that token's whole COLUMN.
    coord_mask[pae_rep_index] = True
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
        "distogram_rep_atom_mask": disto_rep_mask.to(torch.float32),
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
        "label_rep_atom_mask": pae_rep_mask,
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

    # WHY THIS IS PAELoss AND NOT protenix_head_bridge.make_predict_one, which is the live
    # trainer's predict_one for this same objective. That bridge takes the CE from
    # `loss_fn.calculate_losses` and requires two things the probe has no way to hold: a
    # ProtenixLoss whose plddt/pde/resolved are the trainer's `_Zero` stubs (it checks the class
    # NAME), and a `set_pae_mask` callable reaching the trainer's `_pae_keep` closure. Faking
    # either would be a reimplementation wearing the real path's name. For CAPACITY the two agree
    # anyway: under --pae-only + --lean-readouts pae_loss is the only term that computes anything,
    # the other three are 1-element placeholders, and calculate_losses adds no allocation of a
    # size that matters. For the CE VALUE they do not necessarily agree, which is one more reason
    # this probe's CE is not a result. The mask/row-set helpers ARE imported from the bridge
    # above, because those are the parts where disagreeing would change the measured shapes.
    #
    # Built by the same expression ProtenixLoss uses (loss.py:1481), so its bin boundaries, eps and
    # reduction are the configs' and not this file's. Left in training mode (nn.Module's default,
    # and what the trainer's ProtenixLoss is in) because PAELoss.forward's `not self.training`
    # branch calls torch.cuda.empty_cache(), which would make the probe's peak-reserved reading
    # optimistic and unlike the real run.
    pae_loss = PAELoss(**configs.loss.pae)

    # AT STEP 1 ONLY score_head[-1].weight GETS REAL GRADIENT, and that is by design rather than a
    # half-wired reranker. `score_head[-1]` is zero-initialised so the scores start exactly equal
    # to the ipTM baseline, which makes d(delta)/d(anything upstream) exactly zero on the first
    # step. Measured on CPU at K=5: 22 of the 24 reranker tensors get exactly zero,
    # `score_head.2.weight` gets |g|max 1.6e+01, and `score_head.2.bias` gets 3.6e-06 -- which is
    # float noise around an analytic zero, because `logsumexp(s/tau) - logsumexp(s_good/tau)` is
    # invariant to a uniform shift of the scores and a bias is exactly a uniform shift.
    # gpu_probe_step's check is `any(...)` over the group, so the weight alone satisfies it; from
    # step 2 the last layer is nonzero and the pair encoder and pose attention get gradient too,
    # which is why the sweep is run with --steps >= 2 (it refuses fewer). DO NOT read a step-0
    # gradient ratio as a calibrated alpha for the same reason.
    reranker = PAEPoseReranker().to(device)
    reranker.train()
    reranker_params = [p for p in reranker.parameters()]

    unit = _synthetic_unit(int(crop_size), int(pose_count), case, device)
    feat = unit["feat_head"]
    pair = unit["pair"]
    npt = int(unit["uncropped_pair_tokens"])

    # THE SYNTHETIC UNIT'S ATOM INDEXING IS CHECKED AGAINST THE REAL DERIVATION, not asserted.
    # `pose_direction.derive_frames` recovers (frame_atom_index, pae_rep_atom_mask) from
    # atom_to_tokatom_idx alone, and its docstring records that derivation as verified exact
    # against the label npz on all 500 training structures. If it reproduces what _synthetic_unit
    # built, the synthetic unit obeys the same canonical within-token atom convention the real
    # cache does -- which is the one property of the fake data the real pose pathway depends on.
    _fai, _prep = pose_direction.derive_frames(feat)
    if not (bool(torch.equal(_fai, unit["label_frame_atom_index"]))
            and bool(torch.equal(_prep, unit["label_rep_atom_mask"]))):
        raise RuntimeError(
            "the synthetic unit's frame atoms / PAE representative atoms disagree with "
            "pose_direction.derive_frames, so its within-token atom indexing is not the real "
            "convention and --frame-direction would be fed a different geometry than it will see")

    # THE ROW SET AND THE PAIR MASK ARE THE LIVE BRIDGE'S, imported rather than rebuilt. `live` is
    # `has_frame & ~ligand` with the ligand scattered from atoms onto tokens, which is what row_t
    # pools over -- so the [F,T,64] logits handed to the reranker must be sliced with the SAME
    # boolean, and are. In this measured ligand-free population live == has_frame and F == T; the
    # derivation stays general because the population is an observation, not a guarantee.
    lig_tok = ligand_token_flag(feat)
    live = live_row_mask(feat)
    pair_mask = scored_pair_mask(feat, pair)
    if int(lig_tok.sum()) or int(live.sum()) != unit["n_token"]:
        raise RuntimeError(
            f"the synthetic unit produced {int(lig_tok.sum())} ligand token(s) and "
            f"{int(live.sum())} live row(s) of {unit['n_token']}; it is built ligand-free and "
            f"all-framed because every one of the 40 real trunks sampled is, so this means the "
            f"construction drifted from what the docstring claims")

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
