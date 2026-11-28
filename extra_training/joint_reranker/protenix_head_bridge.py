"""The live Protenix forward, turned into the `predict_one` the joint reranker needs.

This is the adapter `integration.joint_group_loss` asks for. It owns ONE job: given the training
state the trainer already has in hand for a pose, produce a `PosePrediction` whose four fields all
carry a live graph back to the trainable confidence head -- the head's PAE logits, the AB/AG pair
mask, the differentiable ipTM, and the ORIGINAL Protenix PAE cross-entropy.

NOTHING HERE REIMPLEMENTS A LOSS OR A KERNEL. Every number comes from the function the existing
arms already use:
  * the head            `model.run_confidence_head(...)`            phase_k_train.py:6919
  * the PAE CE          `loss_fn.calculate_losses(...)`             phase_k_train.py:7171
  * the ipTM            `rank_loss.iptm_pooled(...)`                rank_loss.py:356
A reimplementation that drifted would answer "does a learned readout beat the fixed one" with a
different quantity on each side, which is the failure mode this project has paid for twice.

THE ipTM IS MAX-POOLED WITH A PINNED UNCROPPED d0. Decided in issue #16 on 2026-11-24: the
reranker scores a pose as `ipTM + correction`, so the baseline has to be the ipTM that selection
deploys, which is the max. `iptm_pooled(pool="max", n_pair_tokens=<uncropped count>)` is the
fixed-d0 path added in 60020ce5 -- without the pin, the vendor kernel recomputes d0 from the
CROPPED asym_id and one interface's poses get different TM kernels at 256/384/512. Measured on a
synthetic fixture: 0.0163 against 0.1614 at the smallest crop, a ~10x error.

THE ROW SET IS iptm_pooled's OWN. `live = has_frame & ~is_ligand`, the ligand derived by
scatter_add over `atom_to_token_idx` exactly as rank_loss.py does it. The reranker pools over the
pairs it is correcting, not a different set -- otherwise the correction and the baseline describe
different pair populations.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .integration import PosePrediction


@dataclass
class PoseInput:
    """Everything the head needs for ONE pose, taken from the trainer's own pose loop.

    These are the names the trainer already uses, so a caller inside phase_k_train can pass its
    local variables through without renaming and without a second source of truth:
      feat_head   the head's input_feature_dict (CROPPED token set under --crop-pool)
      feat_cache  the feature dict calculate_losses reads (the trainer's `feat_cache`)
      labels      the label dict calculate_losses reads (the trainer's `ld`)
      s_inputs/s_trunk/z_trunk/coords   the trunk tensors and this pose's coordinates
      pose_id     (pdb, seed, sample) -- the SAMPLE index, not the pose index; see below
    """
    feat_head: dict
    feat_cache: dict
    labels: dict
    s_inputs: Tensor
    s_trunk: Tensor
    z_trunk: Tensor
    coords: Tensor
    pose_id: tuple


def live_row_mask(feat: dict) -> Tensor:
    """iptm_pooled's `live`: has_frame AND NOT ligand, ligand scattered from atoms onto tokens.

    rank_loss.py:395-397 computes this inline and does not expose it, so the derivation is
    repeated here -- but ONLY the derivation. The mask itself comes from
    model.antibody_antigen_pair_mask, which now takes the flag; keeping a second copy of the mask
    logic is the drift this file's own header warns about.
    """
    asym = feat["asym_id"].reshape(-1)
    hf = feat["has_frame"].reshape(-1).bool()
    lig = (torch.zeros_like(feat["asym_id"]).scatter_add(
        0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0).reshape(-1)
    if hf.shape != lig.shape or hf.shape != asym.shape:
        raise ValueError(
            f"feat fields disagree on token count: asym_id {tuple(asym.shape)}, has_frame "
            f"{tuple(hf.shape)}, ligand {tuple(lig.shape)}")
    return hf & (~lig)


def ligand_token_flag(feat: dict) -> Tensor:
    """The [T] token-level ligand flag, for model.antibody_antigen_pair_mask's parameter."""
    return (torch.zeros_like(feat["asym_id"]).scatter_add(
        0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0).reshape(-1)


def scored_pair_mask(feat: dict, iptm_pair: tuple) -> Tensor:
    """[F,T] mask over EXACTLY the chains of the scored pair, matching iptm_pooled row for row.

    THE PAIR, NOT THE CHAIN LISTS, AND THIS IS A REAL BUG THIS AVOIDS. iptm_pooled scores ONE
    asym pair (i, j) and restricts its anchor rows to `(arow == i) | (arow == j)`
    (rank_loss.py:404), with d0 pinned to THAT pair's uncropped token count. An AB/AG mask built
    from chain LISTS unions every antibody chain against every antigen chain, so on an ordinary
    H+L antibody it would pool H-AG and L-AG rows while the baseline ipTM is the single pair -- the
    correction and the baseline would then describe different pair populations, on nearly every
    structure in the set, which is the exact failure the ligand fix was made to close. Found by
    review, 2026-11-24.

    So the mask is built from the pair itself: rows on iptm_pooled's live set restricted to the
    pair's two chains, columns the pair's two chains, cross-chain only.
    """
    asym = feat["asym_id"].reshape(-1)
    live = live_row_mask(feat)
    i, j = int(iptm_pair[0]), int(iptm_pair[1])
    if i == j:
        raise ValueError(f"iptm_pair must name two different chains, got ({i}, {j})")
    arow = asym[live]
    in_pair_col = (asym == i) | (asym == j)
    in_pair_row = (arow == i) | (arow == j)
    mask = (in_pair_row.unsqueeze(1) & in_pair_col.unsqueeze(0)
            & (arow.unsqueeze(1) != asym.unsqueeze(0)))
    if not bool(mask.any()):
        raise ValueError(
            f"no cross-chain pairs remain for scored pair ({i}, {j}) on this crop's live rows")
    return mask


def make_predict_one(
    model: Any,
    configs: Any,
    loss_fn: Any,
    *,
    iptm_pair: tuple,
    n_pair_tokens: int,
    antibody_asym_ids,
    antigen_asym_ids,
    pool: str = "max",
    autocast: bool = True,
    set_pae_mask: Callable[[PoseInput], None] | None = None,
    require_pae_only: bool = True,
) -> Callable[[PoseInput], PosePrediction]:
    """Build the `predict_one` closure for ONE scored interface.

    `iptm_pair` is that interface's scored asym pair and `n_pair_tokens` its UNCROPPED chain-pair
    token count -- the trainer gets the latter from `_uncrop_npt(pdb, pair)`. Both are properties
    of the interface, not of the pose, which is why they are bound here once rather than passed per
    pose: binding them per pose is how a group could silently mix two interfaces.

    pool defaults to "max" per the issue-#16 decision. "mean" is accepted so the control arm can
    be run on the kernel the cropped 7mo52r chain trains on without a second adapter.
    """
    if pool not in ("max", "mean"):
        raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
    if not n_pair_tokens or int(n_pair_tokens) <= 0:
        raise ValueError(
            "n_pair_tokens must be the UNCROPPED scored chain-pair token count. Passing 0/None "
            "would let iptm_pooled fall back to the cropped count, which is the crop-dependent d0 "
            "the fixed-d0 path exists to prevent.")

    # THE PAIR MUST NAME ONE ANTIBODY AND ONE ANTIGEN CHAIN. Checked once here rather than per
    # pose: a pair that does not straddle the interface would still produce a mask and an ipTM,
    # both describing an intra-side pair, and nothing downstream would notice.
    _ab = {int(x) for x in antibody_asym_ids}
    _ag = {int(x) for x in antigen_asym_ids}
    if not _ab or not _ag:
        raise ValueError("antibody and antigen chain ID lists must be nonempty")
    if _ab & _ag:
        raise ValueError(f"antibody and antigen chain IDs overlap: {sorted(_ab & _ag)}")
    _i, _j = int(iptm_pair[0]), int(iptm_pair[1])
    if not ((_i in _ab and _j in _ag) or (_i in _ag and _j in _ab)):
        raise ValueError(
            f"iptm_pair {(_i, _j)} does not straddle the interface: antibody chains {sorted(_ab)}, "
            f"antigen chains {sorted(_ag)}. The reranker would then correct an intra-side ipTM.")

    # --pae-only IS NOT OPTIONAL HERE, AND THE FIELD NAME IS WHY. calculate_losses returns the
    # WHOLE confidence loss; it is the PAE cross-entropy alone only because --pae-only replaced
    # plddt/pde/resolved with _Zero stubs (phase_k_train.py:2600-2622). Without it, a field called
    # pae_ce would silently carry four terms and the alpha calibration would be measuring four
    # gradients against the ranking term. Checked against the real module, not a flag string.
    if require_pae_only:
        _stubbed = [n for n in ("plddt_loss", "pde_loss", "resolved_loss")
                    if type(getattr(loss_fn, n, None)).__name__ == "_Zero"]
        if len(_stubbed) != 3:
            raise ValueError(
                f"the joint objective needs --pae-only: of plddt/pde/resolved only {_stubbed} are "
                f"stubbed, so calculate_losses would return the full confidence loss and the field "
                f"named pae_ce would carry four terms. Pass require_pae_only=False only for an arm "
                f"that DELIBERATELY trains all four, and rename what you report.")

    # THE PER-POSE PAE MASK MUST BE SET BY THE CALLER, PER POSE. pae_loss.forward and
    # calculate_label are monkeypatched to read the closure dict _pae_keep (phase_k_train.py:2627),
    # which only _set_pae_mask fills (:3800). This bridge cannot reach that closure, so a launcher
    # that forgets the call gets whichever pose's mask was left behind -- a stale [F,T] of the
    # wrong crop either shape-errors or, at equal shapes, silently masks the wrong chains.
    if set_pae_mask is None:
        raise ValueError(
            "set_pae_mask is required: pass a callable that invokes the trainer's _set_pae_mask "
            "for THIS pose before the loss is computed. Passing None would reuse the previous "
            "pose's scored-pair mask, which at equal crop shapes fails silently.")

    import rank_loss as RKL

    def predict_one(pose: PoseInput) -> PosePrediction:
        if not isinstance(pose, PoseInput):
            raise TypeError("predict_one expects a PoseInput carrying the trainer's live tensors")
        # THE HEAD, EXACTLY AS THE TRAINER CALLS IT. Same keyword names, same kernels from
        # configs, inplace_safe False, chunk_size None -- Protenix's own config says chunk_size
        # "should set to null for normal training" and --chunk-size 64 measured 5.8x slower here
        # for no memory saving.
        set_pae_mask(pose)
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast
               else torch.autocast(device_type="cuda", enabled=False))
        with ctx:
            plddt, pae, pde, resolved = model.run_confidence_head(
                input_feature_dict=pose.feat_head, s_inputs=pose.s_inputs,
                s_trunk=pose.s_trunk, z_trunk=pose.z_trunk, pair_mask=None,
                x_pred_coords=pose.coords,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention,
                inplace_safe=False, chunk_size=None)
        # FLOAT, AS THE TRAINER DOES. pred["pae"] = pae.float() at phase_k_train.py:6946; the bins
        # are weighted and summed over N_token^2 and bf16 accumulation there is not what the
        # reference path does.
        pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                "resolved": resolved.float(), "coordinate_mini": pose.coords}

        # THE ORIGINAL PAE CE, FROM THE ORIGINAL FUNCTION. `cum` is the differentiable scalar the
        # existing arms descend; under --pae-only it IS the PAE cross-entropy. Nothing here
        # touches its mask, denominator or weighting -- the whole point of calling it.
        cum, met = loss_fn.calculate_losses(pose.feat_cache, pred, pose.labels, mode="train")
        if not isinstance(cum, torch.Tensor) or cum.ndim != 0:
            raise ValueError(f"calculate_losses returned {type(cum).__name__} for the loss, "
                             f"expected a 0-dim tensor")
        if not cum.requires_grad:
            raise ValueError(
                "the PAE CE came back detached. The joint objective needs it live: it is half the "
                "gradient reaching the head, and a detached scalar would train the reranker alone.")

        # THE ipTM, FROM THE FULL-TOKEN-GRID LOGITS. iptm_pooled indexes frame rows itself via
        # has_frame, so it must receive the whole [.., T, T, 64] tensor -- the reranker's [F,T,64]
        # view is NOT a drop-in substitute for it, and handing it the sliced view would silently
        # score a different grid. Raised in review on issue #16.
        iptm = RKL.iptm_pooled(pred["pae"], pose.feat_head, configs, iptm_pair,
                               pool=pool, n_pair_tokens=int(n_pair_tokens))
        if iptm is None:
            raise ValueError(
                f"{pose.pose_id}: iptm_pooled found no scored chain pair for {iptm_pair} on this "
                f"crop. A pose with no ipTM has no residual baseline, so the group is invalid; the "
                f"crop pool guarantees every selected crop keeps a scorable cross-chain pair, so "
                f"this means the crop or the pair is wrong rather than merely unlucky.")

        # THE RERANKER'S VIEW: [F,T,64] on iptm_pooled's own live rows.
        live = live_row_mask(pose.feat_head)
        logits = pred["pae"]
        while logits.ndim > 3:
            if logits.shape[0] != 1:
                raise ValueError(
                    f"{pose.pose_id}: the head returned {tuple(logits.shape)}; this adapter takes "
                    f"ONE pose per forward, so a leading axis > 1 means several samples were "
                    f"batched and the group would attribute one pose's logits to another.")
            logits = logits.squeeze(0)
        if logits.ndim != 3 or logits.shape[-1] != 64:
            raise ValueError(f"{pose.pose_id}: expected [T,T,64] PAE logits, got "
                             f"{tuple(logits.shape)}")
        rows = logits[live]
        mask = scored_pair_mask(pose.feat_head, iptm_pair)
        if mask.shape != rows.shape[:2]:
            raise ValueError(f"{pose.pose_id}: mask {tuple(mask.shape)} does not match the live "
                             f"logit grid {tuple(rows.shape[:2])}")
        return PosePrediction(pae_logits=rows, pair_mask=mask, iptm=iptm, pae_ce=cum)

    return predict_one
